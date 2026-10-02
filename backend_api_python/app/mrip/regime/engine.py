"""Market Regime Engine: a deterministic, point-in-time view of market conditions.

Inputs (all via the data gateway): VIX (required), VIX3M, a broad index (SPX), the
10-year yield (TNX, quoted as yield x 10) and a US-dollar proxy (UUP). Optional inputs
that are unavailable are listed in ``unavailable`` and their fields are None; the
regime itself needs only the VIX. Every feature is strictly trailing, so ``at(as_of)``
never sees later data and can be recomputed identically for any past date.

The internal regime LOW_VOL..CRISIS comes from VIX level thresholds
(``RegimePolicy.vix_thresholds``), uncalibrated initial values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any

import pandas as pd

from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.regime import features as f
from app.mrip.stats.regimes import Regime, RegimeThresholds, classify_vix
from app.mrip.stats.service import prices_to_series


@dataclass(frozen=True, slots=True)
class RegimePolicy:
    version: str = "regime-v0-uncalibrated"
    vix_thresholds: RegimeThresholds = field(default_factory=RegimeThresholds)
    percentile_window: int = 252
    vix_change_periods: int = 5
    realized_window: int = 20
    trend_fast: int = 50
    trend_slow: int = 200
    rates_periods: int = 60
    rates_band_yield_points: float = 0.25  # absolute change in the 10y yield (percentage points)
    usd_periods: int = 60
    usd_band: float = 0.02  # relative change of the dollar proxy
    ffill_limit: int = 3  # carry an optional series' last value across at most this many VIX dates


@dataclass(frozen=True, slots=True)
class MarketRegime:
    as_of: date
    regime: Regime
    vix: float
    vix_change: float | None
    vix_percentile: float | None
    realized_vol: float | None
    implied_to_realized: float | None
    vix_term_ratio: float | None  # VIX / VIX3M; above 1 is backwardation (stress)
    index_trend: f.Trend | None
    index_drawdown: float | None
    rates_yield: float | None  # 10y yield in percent
    rates_direction: f.Direction | None
    usd_direction: f.Direction | None
    unavailable: tuple[str, ...]
    policy_version: str
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def term_backwardation(self) -> bool | None:
        return None if self.vix_term_ratio is None else bool(self.vix_term_ratio > 1.0)

    def attrs(self) -> dict[str, str]:
        """Segment attributes for calibration and validation."""
        return {"vix_regime": self.regime.value}

    def context(self) -> dict[str, Any]:
        """JSON-friendly snapshot stored on predictions (lineage and later segmentation)."""
        return {
            "as_of": self.as_of.isoformat(), "regime": self.regime.value, "vix": self.vix,
            "vix_percentile": self.vix_percentile, "term_backwardation": self.term_backwardation,
            "index_trend": self.index_trend.value if self.index_trend else None,
            "rates_direction": self.rates_direction.value if self.rates_direction else None,
            "usd_direction": self.usd_direction.value if self.usd_direction else None,
            "policy_version": self.policy_version,
        }


def _none(v: Any) -> Any:
    """NaN -> None; numpy scalars -> plain Python values."""
    if v is None or (not isinstance(v, (str, Enum)) and pd.isna(v)):
        return None
    return v.item() if hasattr(v, "item") else v


class MarketRegimeEngine:
    def __init__(
        self,
        gateway: FinancialDataGateway,
        policy: RegimePolicy = RegimePolicy(),
        *,
        index_symbol: str = "SPX",
        vix3m_symbol: str = "VIX3M",
        rates_symbol: str = "TNX",
        usd_symbol: str = "UUP",
    ) -> None:
        self._gw, self._p = gateway, policy
        self._index, self._vix3m, self._rates, self._usd = index_symbol, vix3m_symbol, rates_symbol, usd_symbol

    # -- public ---------------------------------------------------------------

    def history(self, end: date | None = None) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, Any]]]:
        """Daily regime features per VIX date (up to ``end``), the unavailable inputs and source provenance."""
        p = self._p
        try:
            vix_ps = self._gw.index_history("VIX")
        except DataUnavailable as exc:
            raise DataUnavailable(f"VIX history is required for the market regime: {exc}") from exc
        vix = prices_to_series(vix_ps, end)
        if vix.empty:
            raise DataUnavailable(f"no VIX data on or before {end}" if end else "VIX history is empty")
        sources = {"vix": _prov(vix_ps)}
        unavailable: list[str] = []
        frame = pd.DataFrame(index=vix.index)
        frame["vix"] = vix
        frame["regime"] = classify_vix(vix, p.vix_thresholds)
        frame["vix_change"] = f.change(vix, p.vix_change_periods)
        frame["vix_percentile"] = f.trailing_percentile(vix, p.percentile_window)

        def optional(name: str, symbol: str, kind: str):
            try:
                ps = self._gw.index_history(symbol) if kind == "index" else self._gw.price_history(symbol)
            except DataUnavailable:
                unavailable.append(name)
                return None
            sources[name] = _prov(ps)
            return prices_to_series(ps, end)

        def onto_vix(series: pd.Series) -> pd.Series:
            return series.reindex(vix.index.union(series.index)).ffill(limit=p.ffill_limit).reindex(vix.index)

        idx = optional("index", self._index, "index")
        if idx is not None:
            rv = f.realized_volatility(idx, p.realized_window)
            frame["realized_vol"] = onto_vix(rv)
            frame["implied_to_realized"] = f.implied_to_realized(vix, frame["realized_vol"])
            frame["index_trend"] = onto_vix(f.trend_state(idx, p.trend_fast, p.trend_slow))
            frame["index_drawdown"] = onto_vix(f.drawdown(idx))
        vix3m = optional("vix3m", self._vix3m, "index")
        if vix3m is not None:
            frame["vix_term_ratio"] = f.term_ratio(vix, onto_vix(vix3m))
        rates = optional("rates", self._rates, "index")
        if rates is not None:
            y = rates / 10.0  # TNX is quoted as yield x 10
            frame["rates_yield"] = onto_vix(y)
            frame["rates_direction"] = onto_vix(
                f.direction_state(y, p.rates_periods, p.rates_band_yield_points, mode="absolute")
            )
        usd = optional("usd", self._usd, "equity")
        if usd is not None:
            frame["usd_direction"] = onto_vix(f.direction_state(usd, p.usd_periods, p.usd_band, mode="relative"))
        return frame, unavailable, sources

    def at(self, as_of: date | None = None) -> MarketRegime:
        """The regime as known at the close of ``as_of`` (default: the latest available day)."""
        frame, unavailable, sources = self.history(as_of)
        if as_of is not None:
            frame = frame.loc[: pd.Timestamp(as_of)]
        if frame.empty:
            raise DataUnavailable(f"no VIX data on or before {as_of}")
        row = frame.iloc[-1]
        col = lambda name: _none(row[name]) if name in frame.columns else None  # noqa: E731
        return MarketRegime(
            as_of=frame.index[-1].date(),
            regime=row["regime"],
            vix=float(row["vix"]),
            vix_change=col("vix_change"),
            vix_percentile=col("vix_percentile"),
            realized_vol=col("realized_vol"),
            implied_to_realized=col("implied_to_realized"),
            vix_term_ratio=col("vix_term_ratio"),
            index_trend=col("index_trend"),
            index_drawdown=col("index_drawdown"),
            rates_yield=col("rates_yield"),
            rates_direction=col("rates_direction"),
            usd_direction=col("usd_direction"),
            unavailable=tuple(unavailable),
            policy_version=self._p.version,
            sources=sources,
        )


def _prov(ps) -> dict[str, Any]:
    pr = ps.provenance
    return {"symbol": ps.symbol, "provider": pr.provider, "endpoint": pr.endpoint, "fetched_at": pr.fetched_at.isoformat()}
