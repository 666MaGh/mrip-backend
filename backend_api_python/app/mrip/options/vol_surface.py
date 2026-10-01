"""Observed-data volatility and activity features from an options chain snapshot.

Only observed provider fields are used (IV, delta, volume, open interest);
nothing here is modeled. Days to expiry are counted in calendar days from the
snapshot's New York date.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timezone
from zoneinfo import ZoneInfo

import numpy as np

from app.mrip.data.models import OptionContract, OptionsChainSnapshot

_NY = ZoneInfo("America/New_York")


def snapshot_date(snapshot: OptionsChainSnapshot) -> date:
    """Snapshot date in America/New_York (naive timestamps are read as UTC)."""
    ts = snapshot.snapshot_timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(_NY).date()


def _dte(snapshot: OptionsChainSnapshot, expiry: date) -> int:
    return (expiry - snapshot_date(snapshot)).days


def _has_iv(c: OptionContract) -> bool:
    return c.implied_volatility is not None and math.isfinite(c.implied_volatility)


@dataclass(frozen=True, slots=True)
class TermPoint:
    expiry: date
    dte: int
    atm_iv: float
    n_contracts: int  # contracts with IV in this expiry


def atm_iv_term_structure(snapshot: OptionsChainSnapshot, min_dte: int = 0) -> list[TermPoint]:
    """ATM implied vol per expiry, sorted by expiry.

    ATM strike is the strike nearest the underlying price (ties: lower strike)
    among strikes with IV; the IV is the call/put mean when both exist.
    """
    spot = snapshot.underlying_price
    if spot is None:
        raise ValueError("underlying_price is required for the ATM term structure")
    by_expiry: dict[date, list[OptionContract]] = defaultdict(list)
    for c in snapshot.contracts:
        if _has_iv(c):
            by_expiry[c.expiry].append(c)
    points: list[TermPoint] = []
    for expiry in sorted(by_expiry):
        dte = _dte(snapshot, expiry)
        if dte < min_dte:
            continue
        contracts = by_expiry[expiry]
        atm_strike = min({c.strike for c in contracts}, key=lambda s: (abs(s - spot), s))
        ivs = [float(c.implied_volatility) for c in contracts if c.strike == atm_strike]  # type: ignore[arg-type]
        points.append(
            TermPoint(expiry=expiry, dte=dte, atm_iv=sum(ivs) / len(ivs), n_contracts=len(contracts))
        )
    return points


def term_structure_slope(
    points: Sequence[TermPoint], short_dte: int = 7, long_dte: int = 60
) -> float | None:
    """IV(largest dte <= long_dte) - IV(smallest dte >= short_dte); None if undefined."""
    shorts = [p for p in points if p.dte >= short_dte]
    longs = [p for p in points if p.dte <= long_dte]
    if not shorts or not longs:
        return None
    short = min(shorts, key=lambda p: p.dte)
    long = max(longs, key=lambda p: p.dte)
    if short.expiry == long.expiry and short.dte == long.dte:
        return None
    return long.atm_iv - short.atm_iv


@dataclass(frozen=True, slots=True)
class Skew:
    expiry: date
    dte: int
    put_strike: float
    call_strike: float
    put_delta: float
    call_delta: float
    put_iv: float
    call_iv: float
    skew: float  # put_iv - call_iv


def iv_skew(
    snapshot: OptionsChainSnapshot, target_dte: int = 30, target_delta: float = 0.25
) -> Skew | None:
    """Put-minus-call IV at +-target_delta for the expiry nearest target_dte (dte >= 1).

    Provider deltas are used as given. Returns None if that expiry lacks a put
    or a call with both IV and delta.
    """
    expiries = {c.expiry for c in snapshot.contracts if _dte(snapshot, c.expiry) >= 1}
    if not expiries:
        return None
    expiry = min(expiries, key=lambda e: (abs(_dte(snapshot, e) - target_dte), _dte(snapshot, e)))
    usable = [
        c for c in snapshot.contracts
        if c.expiry == expiry and _has_iv(c) and c.delta is not None and math.isfinite(c.delta)
    ]
    puts = [c for c in usable if c.option_type == "put"]
    calls = [c for c in usable if c.option_type == "call"]
    if not puts or not calls:
        return None
    put = min(puts, key=lambda c: (abs(float(c.delta) + target_delta), c.strike))  # type: ignore[arg-type]
    call = min(calls, key=lambda c: (abs(float(c.delta) - target_delta), c.strike))  # type: ignore[arg-type]
    put_iv, call_iv = float(put.implied_volatility), float(call.implied_volatility)  # type: ignore[arg-type]
    return Skew(
        expiry=expiry,
        dte=_dte(snapshot, expiry),
        put_strike=put.strike,
        call_strike=call.strike,
        put_delta=float(put.delta),  # type: ignore[arg-type]
        call_delta=float(call.delta),  # type: ignore[arg-type]
        put_iv=put_iv,
        call_iv=call_iv,
        skew=put_iv - call_iv,
    )


@dataclass(frozen=True, slots=True)
class PutCallRatios:
    volume: float | None
    open_interest: float | None
    call_volume: float
    put_volume: float
    call_open_interest: float
    put_open_interest: float


def put_call_ratios(snapshot: OptionsChainSnapshot, max_dte: int | None = None) -> PutCallRatios:
    """Put/call volume and open-interest ratios (None when the call total is 0)."""
    cv = pv = coi = poi = 0.0
    for c in snapshot.contracts:
        if max_dte is not None and _dte(snapshot, c.expiry) > max_dte:
            continue
        vol = c.volume or 0.0
        oi = c.open_interest or 0.0
        if c.option_type == "call":
            cv += vol
            coi += oi
        else:
            pv += vol
            poi += oi
    return PutCallRatios(
        volume=pv / cv if cv > 0 else None,
        open_interest=poi / coi if coi > 0 else None,
        call_volume=cv,
        put_volume=pv,
        call_open_interest=coi,
        put_open_interest=poi,
    )


@dataclass(frozen=True, slots=True)
class Anomaly:
    contract_symbol: str | None
    expiry: date
    strike: float
    option_type: str
    volume: float
    open_interest: float | None
    volume_oi_ratio: float | None


def volume_oi_anomalies(
    snapshot: OptionsChainSnapshot,
    min_volume: float = 100,
    ratio_threshold: float = 1.0,
    top_n: int = 10,
) -> list[Anomaly]:
    """Contracts with unusual volume relative to open interest.

    Qualifies when volume >= min_volume and open interest is missing/zero
    (ratio None) or volume / open interest > ratio_threshold. Ratio-None
    contracts come first (largest volume first), then ratio descending; ties by
    (expiry, strike, option_type).
    """
    found: list[Anomaly] = []
    for c in snapshot.contracts:
        if c.volume is None or not c.volume >= min_volume:
            continue
        if not c.open_interest:
            ratio: float | None = None
        else:
            ratio = c.volume / c.open_interest
            if not ratio > ratio_threshold:
                continue
        found.append(
            Anomaly(
                contract_symbol=c.contract_symbol,
                expiry=c.expiry,
                strike=c.strike,
                option_type=c.option_type,
                volume=c.volume,
                open_interest=c.open_interest,
                volume_oi_ratio=ratio,
            )
        )

    def key(a: Anomaly) -> tuple[int, float, date, float, str]:
        if a.volume_oi_ratio is None:
            return (0, -a.volume, a.expiry, a.strike, a.option_type)
        return (1, -a.volume_oi_ratio, a.expiry, a.strike, a.option_type)

    found.sort(key=key)
    return found[: max(top_n, 0)]


def volume_zscore(value: float, history: Sequence[float], min_history: int = 20) -> float | None:
    """(value - mean(history)) / std(history, ddof=1); None if undefined."""
    if len(history) < min_history or len(history) < 2:
        return None
    arr = np.asarray(history, dtype=float)
    std = float(np.std(arr, ddof=1))
    if not math.isfinite(std) or std == 0:
        return None
    z = (value - float(np.mean(arr))) / std
    return z if math.isfinite(z) else None
