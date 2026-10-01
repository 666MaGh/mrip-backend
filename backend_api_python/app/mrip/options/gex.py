"""Modeled dealer gamma exposure (GEX) from an observed options chain.

EVERYTHING here is MODELED / ESTIMATED. Public options data does not show market
makers' net positions, so dealer GEX, the gamma flip and the call/put walls rest
on assumptions (``GexAssumptions``) that always travel with the result.

Dollar gamma per contract (per 1 % move in the underlying):
    gex = sign * gamma_per_share * open_interest * contract_multiplier * spot**2 * 0.01
with ``sign = call_sign`` for calls and ``put_sign`` for puts. The default
convention (calls +1, puts -1) is the common market model that assumes dealers are
long calls and short puts; it is an assumption, not an observation.

Per-contract gamma is the provider's gamma when present, otherwise Black-Scholes
from the contract's IV. The gamma-flip curve re-evaluates Black-Scholes gamma at
hypothetical spots from each contract's IV, so only contracts with IV take part in it.
Contracts without open interest, or without usable gamma, are excluded and counted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np

from app.mrip.data.models import OptionContract, OptionsChainSnapshot
from app.mrip.options.bs import bs_gamma_array
from app.mrip.options.vol_surface import snapshot_date

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")
_SECONDS_PER_YEAR = 365.0 * 24 * 3600


@dataclass(frozen=True, slots=True)
class GexAssumptions:
    version: str = "gex-v0-uncalibrated"
    call_sign: int = 1
    put_sign: int = -1
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    min_time_to_expiry_hours: float = 0.25  # floors T so 0DTE gamma stays finite
    prefer_provider_gamma: bool = True
    flip_range_pct: float = 0.15
    flip_grid_points: int = 61
    top_n_strikes: int = 3

    def __post_init__(self) -> None:
        if self.call_sign not in (-1, 1) or self.put_sign not in (-1, 1):
            raise ValueError("call_sign and put_sign must be -1 or +1")
        if self.flip_grid_points < 3 or not 0 < self.flip_range_pct < 1:
            raise ValueError("invalid flip grid")
        if self.min_time_to_expiry_hours <= 0:
            raise ValueError("min_time_to_expiry_hours must be positive")

    def describe(self) -> dict[str, object]:
        return {
            "version": self.version,
            "dealer_convention": "dealers long calls / short puts" if (self.call_sign, self.put_sign) == (1, -1)
            else f"call_sign={self.call_sign}, put_sign={self.put_sign}",
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "expiry_time": "16:00 America/New_York",
            "min_time_to_expiry_hours": self.min_time_to_expiry_hours,
            "gamma_source": "provider gamma when present, else Black-Scholes from IV",
            "units": "dollar gamma per 1% underlying move",
        }


@dataclass(frozen=True, slots=True)
class GammaFlip:
    """Estimated spot level where modeled net GEX changes sign."""

    level: float | None
    distance_pct: float | None  # (level - spot) / spot
    status: str  # "found" | "none_positive_in_range" | "none_negative_in_range" | "unavailable"


@dataclass(frozen=True, slots=True)
class Concentration:
    strike: float
    gross_gex: float
    net_gex: float
    distance_pct: float  # (strike - spot) / spot


@dataclass(frozen=True, slots=True)
class Walls:
    call_wall: float | None  # strike >= spot with the largest call GEX
    put_wall: float | None  # strike <= spot with the largest |put GEX|
    call_wall_distance_pct: float | None
    put_wall_distance_pct: float | None


@dataclass(frozen=True, slots=True)
class GexProfile:
    spot: float
    net_gex: float
    call_gex: float
    put_gex: float
    gross_gex: float
    by_strike: dict[float, float]
    by_expiry: dict[date, float]
    concentrations: tuple[Concentration, ...]
    top_strikes_share: float | None  # share of gross GEX in the top-N strikes
    hhi: float | None  # Herfindahl index of gross GEX across strikes (0..1)
    zero_dte_share: float | None
    short_dated_share: float | None  # dte <= 7 (includes 0DTE)
    walls: Walls
    flip: GammaFlip
    contracts_total: int
    contracts_used: int
    excluded: dict[str, int]
    gamma_source: dict[str, int]

    @property
    def tilt(self) -> float | None:
        """Net / gross GEX in [-1, 1]; None when there is no gamma at all."""
        return self.net_gex / self.gross_gex if self.gross_gex > 0 else None


def seconds_to_expiry(expiry: date, snapshot_timestamp: datetime) -> float:
    """Seconds from the snapshot to 16:00 New York on the expiry date (<= 0 once expired)."""
    if snapshot_timestamp.tzinfo is None:
        raise ValueError("snapshot_timestamp must be timezone-aware")
    close = datetime(expiry.year, expiry.month, expiry.day, 16, 0, tzinfo=_ET).astimezone(_UTC)
    return (close - snapshot_timestamp.astimezone(_UTC)).total_seconds()


def years_to_expiry(expiry: date, snapshot_timestamp: datetime, min_hours: float) -> float:
    """Time to expiry in years, floored at ``min_hours`` so 0DTE gamma stays finite."""
    return max(seconds_to_expiry(expiry, snapshot_timestamp), min_hours * 3600.0) / _SECONDS_PER_YEAR


@dataclass(slots=True)
class _Arrays:
    contracts: list[OptionContract]
    sign: np.ndarray
    strike: np.ndarray
    t: np.ndarray
    iv: np.ndarray  # NaN when missing
    provider_gamma: np.ndarray  # NaN when missing
    oi: np.ndarray
    multiplier: np.ndarray
    dte: np.ndarray


def _prepare(snapshot: OptionsChainSnapshot, a: GexAssumptions) -> tuple[_Arrays, dict[str, int]]:
    excluded = {"no_open_interest": 0, "no_gamma_source": 0, "expired": 0}
    kept: list[OptionContract] = []
    snap_day = snapshot_date(snapshot)
    for c in snapshot.contracts:
        if seconds_to_expiry(c.expiry, snapshot.snapshot_timestamp) <= 0:
            excluded["expired"] += 1
            continue
        if c.open_interest is None or c.open_interest <= 0:
            excluded["no_open_interest"] += 1
            continue
        if c.gamma is None and c.implied_volatility is None:
            excluded["no_gamma_source"] += 1
            continue
        kept.append(c)
    nan = float("nan")
    arrays = _Arrays(
        contracts=kept,
        sign=np.array([a.call_sign if c.option_type == "call" else a.put_sign for c in kept], dtype=float),
        strike=np.array([c.strike for c in kept], dtype=float),
        t=np.array([years_to_expiry(c.expiry, snapshot.snapshot_timestamp, a.min_time_to_expiry_hours) for c in kept]),
        iv=np.array([c.implied_volatility if c.implied_volatility is not None else nan for c in kept], dtype=float),
        provider_gamma=np.array([c.gamma if c.gamma is not None else nan for c in kept], dtype=float),
        oi=np.array([c.open_interest for c in kept], dtype=float),
        multiplier=np.array([c.contract_multiplier for c in kept], dtype=float),
        dte=np.array([(c.expiry - snap_day).days for c in kept], dtype=int),
    )
    return arrays, excluded


def _gamma_at(arr: _Arrays, spot: float, a: GexAssumptions) -> tuple[np.ndarray, np.ndarray]:
    """Per-share gamma at ``spot``: provider value when preferred and present, else Black-Scholes."""
    bs = bs_gamma_array(spot, arr.strike, arr.t, arr.iv, a.risk_free_rate, a.dividend_yield)
    if a.prefer_provider_gamma:
        use_provider = ~np.isnan(arr.provider_gamma)
        return np.where(use_provider, arr.provider_gamma, bs), use_provider
    return np.where(np.isnan(bs), arr.provider_gamma, bs), np.zeros(len(bs), dtype=bool)


def _flip(arr: _Arrays, spot: float, a: GexAssumptions) -> GammaFlip:
    has_iv = ~np.isnan(arr.iv)
    if has_iv.sum() == 0:
        return GammaFlip(None, None, "unavailable")
    sub = {name: getattr(arr, name)[has_iv] for name in ("sign", "strike", "t", "iv", "oi", "multiplier")}
    grid = spot * np.linspace(1 - a.flip_range_pct, 1 + a.flip_range_pct, a.flip_grid_points)
    net = np.empty(len(grid))
    for i, s in enumerate(grid):
        g = bs_gamma_array(float(s), sub["strike"], sub["t"], sub["iv"], a.risk_free_rate, a.dividend_yield)
        net[i] = float(np.nansum(sub["sign"] * g * sub["oi"] * sub["multiplier"] * s * s * 0.01))
    crossings = [
        (grid[i] - net[i] * (grid[i + 1] - grid[i]) / (net[i + 1] - net[i]))
        for i in range(len(grid) - 1)
        if net[i] != net[i + 1] and (net[i] <= 0 <= net[i + 1] or net[i] >= 0 >= net[i + 1])
    ]
    if not crossings:
        return GammaFlip(None, None, "none_positive_in_range" if net[len(net) // 2] > 0 else "none_negative_in_range")
    level = float(min(crossings, key=lambda x: abs(x - spot)))
    return GammaFlip(level, (level - spot) / spot, "found")


def compute_gex(snapshot: OptionsChainSnapshot, assumptions: GexAssumptions = GexAssumptions()) -> GexProfile:
    spot = snapshot.underlying_price
    if spot is None or spot <= 0:
        raise ValueError("snapshot has no usable underlying_price")
    arr, excluded = _prepare(snapshot, assumptions)
    gamma, from_provider = _gamma_at(arr, spot, assumptions)
    usable = ~np.isnan(gamma)
    excluded["no_gamma_source"] += int((~usable).sum())
    gex = np.where(usable, arr.sign * gamma * arr.oi * arr.multiplier * spot * spot * 0.01, 0.0)

    by_strike: dict[float, float] = {}
    gross_by_strike: dict[float, float] = {}
    call_by_strike: dict[float, float] = {}
    put_by_strike: dict[float, float] = {}
    by_expiry: dict[date, float] = {}
    for c, g, ok in zip(arr.contracts, gex, usable):
        if not ok:
            continue
        by_strike[c.strike] = by_strike.get(c.strike, 0.0) + float(g)
        gross_by_strike[c.strike] = gross_by_strike.get(c.strike, 0.0) + abs(float(g))
        by_expiry[c.expiry] = by_expiry.get(c.expiry, 0.0) + float(g)
        target = call_by_strike if c.option_type == "call" else put_by_strike
        target[c.strike] = target.get(c.strike, 0.0) + float(g)

    gross = float(np.abs(gex).sum())
    is_call = np.array([c.option_type == "call" for c in arr.contracts], dtype=bool)
    call_gex = float(gex[is_call].sum())
    put_gex = float(gex[~is_call].sum())

    ordered = sorted(gross_by_strike, key=lambda k: (-gross_by_strike[k], k))
    concentrations = tuple(
        Concentration(k, gross_by_strike[k], by_strike[k], (k - spot) / spot) for k in ordered[: assumptions.top_n_strikes]
    )
    if gross > 0:
        shares = np.array(list(gross_by_strike.values())) / gross
        top_share = float(sum(gross_by_strike[k] for k in ordered[: assumptions.top_n_strikes]) / gross)
        hhi = float((shares**2).sum())
    else:
        top_share = hhi = None

    def share_of(mask: np.ndarray) -> float | None:
        return float(np.abs(gex[mask]).sum() / gross) if gross > 0 else None

    calls_above = {k: v for k, v in call_by_strike.items() if k >= spot}
    puts_below = {k: v for k, v in put_by_strike.items() if k <= spot}
    call_wall = max(calls_above, key=lambda k: (abs(calls_above[k]), -k)) if calls_above else None
    put_wall = max(puts_below, key=lambda k: (abs(puts_below[k]), k)) if puts_below else None
    walls = Walls(
        call_wall, put_wall,
        None if call_wall is None else (call_wall - spot) / spot,
        None if put_wall is None else (put_wall - spot) / spot,
    )
    return GexProfile(
        spot=float(spot),
        net_gex=float(gex.sum()),
        call_gex=call_gex,
        put_gex=put_gex,
        gross_gex=gross,
        by_strike=dict(sorted(by_strike.items())),
        by_expiry=dict(sorted(by_expiry.items())),
        concentrations=concentrations,
        top_strikes_share=top_share,
        hhi=hhi,
        zero_dte_share=share_of(arr.dte == 0),
        short_dated_share=share_of(arr.dte <= 7),
        walls=walls,
        flip=_flip(arr, float(spot), assumptions),
        contracts_total=len(snapshot.contracts),
        contracts_used=int(usable.sum()),
        excluded=excluded,
        gamma_source={
            "provider": int((from_provider & usable).sum()),
            "black_scholes": int((~from_provider & usable).sum()),
        },
    )
