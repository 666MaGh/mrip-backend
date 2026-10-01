"""OutcomeService: resolve logged predictions against observed prices.

Pulls daily bars through the data gateway, locates each prediction's window,
computes path statistics and type-specific measures, and stores the outcome.
A prediction whose window is not yet fully observed stays pending. Outcomes are
immutable: resolving again returns the stored outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.data.models import PriceSeries
from app.mrip.outcomes import resolve
from app.mrip.outcomes.store import PredictionStore
from app.mrip.outcomes.types import HorizonKind, Outcome, Prediction, PredictionType

_ET = ZoneInfo("America/New_York")


def bars_from_price_series(series: PriceSeries) -> pd.DataFrame:
    """Daily bars (open/high/low/close) indexed by date; rows without a close are dropped."""
    rows = {}
    for b in series.bars:
        day = b.ts.date() if isinstance(b.ts, datetime) else b.ts
        if b.close is not None:
            rows[pd.Timestamp(day)] = {"open": b.open, "high": b.high, "low": b.low, "close": b.close}
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=["open", "high", "low", "close"]).sort_index()
    return frame.astype(float)


@dataclass(slots=True)
class ResolutionReport:
    resolved: list[Outcome] = field(default_factory=list)
    still_pending: list[tuple[int, str]] = field(default_factory=list)
    unavailable: list[tuple[int, str]] = field(default_factory=list)


class OutcomeService:
    def __init__(self, store: PredictionStore, gateway: FinancialDataGateway) -> None:
        self._store, self._gateway = store, gateway

    def resolve_pending(self, *, as_of: date | None = None, subject: str | None = None) -> ResolutionReport:
        """Resolve every pending prediction whose window is observable in data up to ``as_of``."""
        report = ResolutionReport()
        cache: dict[str, tuple[PriceSeries, pd.DataFrame] | None] = {}

        def bars_for(symbol: str) -> tuple[PriceSeries, pd.DataFrame] | None:
            if symbol not in cache:
                try:
                    series = self._gateway.price_history(symbol)
                    frame = bars_from_price_series(series)
                    cache[symbol] = (series, frame.loc[: pd.Timestamp(as_of)] if as_of else frame)
                except DataUnavailable:
                    cache[symbol] = None
            return cache[symbol]

        for prediction in self._store.pending(subject=subject):
            asset = bars_for(prediction.subject)
            if asset is None or asset[1].empty:
                report.unavailable.append((prediction.id, f"no price data for {prediction.subject}"))
                continue
            made_on = prediction.made_at.astimezone(_ET).date()
            try:
                window = resolve.locate_window(
                    asset[1], made_on, prediction.horizon_kind.value,
                    expiry=prediction.expiry_date, entry_price=prediction.entry_price,
                )
            except ValueError as exc:
                report.unavailable.append((prediction.id, str(exc)))
                continue
            if window is None:
                report.still_pending.append((prediction.id, "target date not yet observed in the data"))
                continue
            report.resolved.append(self._resolve(prediction, window, asset, bars_for))
        return report

    # -- internals --------------------------------------------------------

    def _resolve(self, p: Prediction, window: resolve.Window, asset: tuple[PriceSeries, pd.DataFrame], bars_for: Any) -> Outcome:
        series, frame = asset
        stats = resolve.path_stats(frame, window)
        bench_return = None
        provenance: dict[str, Any] = {"subject": _prov(series, frame)}
        if p.benchmark:
            bench = bars_for(p.benchmark)
            if bench is not None and not bench[1].empty:
                bench_return = resolve.benchmark_return(bench[1], window)
                provenance["benchmark"] = _prov(bench[0], bench[1])
        measures = self._measures(p, window, frame, stats)
        measures["path"] = {"n_bars": stats.n_bars, "excursions_from": stats.excursions_from, "kind": p.horizon_kind.value}
        return self._store.record_outcome(
            p.id,
            resolver_version=resolve.OUTCOME_VERSION,
            window_start=window.start_date,
            window_end=window.end_date,
            actual_return=stats.actual_return,
            benchmark_return=bench_return,
            realized_volatility=stats.realized_volatility,
            max_adverse_excursion=stats.max_adverse_excursion,
            max_favorable_excursion=stats.max_favorable_excursion,
            measures=measures,
            data_provenance=provenance,
        )

    def _measures(self, p: Prediction, window: resolve.Window, frame: pd.DataFrame, stats: resolve.PathStats) -> dict[str, Any]:
        if p.prediction_type is PredictionType.FORECAST:
            quantiles = {float(q): float(v) for q, v in p.payload["quantiles"].items()}
            return {
                "quantile_hits": {str(q): stats.actual_return <= v for q, v in sorted(quantiles.items())},
                "predicted_median": quantiles.get(0.5),
            }
        walls: dict[str, str | None] = {}
        details: dict[str, Any] = {}
        for side in ("call", "put"):
            level = p.payload.get(f"{side}_wall")
            if level is None:
                walls[side] = None
                continue
            w = resolve.wall_outcome(frame, window, float(level), side)
            walls[side] = w.result.value
            details[side] = {
                "level": w.level,
                "first_touch_date": w.first_touch_date.isoformat() if w.first_touch_date else None,
                "first_break_date": w.first_break_date.isoformat() if w.first_break_date else None,
                "max_penetration": w.max_penetration,
            }
        measures: dict[str, Any] = {"wall_hold_or_break": walls, "wall_details": details}
        atm_iv = p.payload.get("atm_iv")
        if atm_iv:
            # EOD is counted as one full session: an upper bound for the expected move.
            bars = max(1, resolve.horizon_bars_for(p.horizon_kind.value, window))
            implied = resolve.implied_move_outcome(stats, float(atm_iv), bars)
            measures["implied_move"] = {
                "expected_move": implied.expected_move,
                "abs_return_to_expected": implied.abs_return_to_expected,
                "excursion_to_expected": implied.excursion_to_expected,
                "realized_to_implied_vol": implied.realized_to_implied_vol,
                "amplified": implied.amplified,
                "version": implied.version,
            }
            measures["amplification_realized"] = {
                "predicted_direction": p.payload.get("direction"),
                "predicted_level": p.payload.get("amplification"),
                "amplified": implied.amplified,
            }
            measures["gamma_regime_outcome"] = {
                "regime": p.payload.get("regime"),
                "realized_to_implied_vol": implied.realized_to_implied_vol,
                "excursion_to_expected": implied.excursion_to_expected,
            }
        else:
            measures["gamma_regime_outcome"] = {"regime": p.payload.get("regime"), "note": "no ATM IV logged"}
        return measures


def _prov(series: PriceSeries, frame: pd.DataFrame) -> dict[str, Any]:
    pr = series.provenance
    return {
        "symbol": series.symbol, "provider": pr.provider, "endpoint": pr.endpoint,
        "fetched_at": pr.fetched_at.isoformat(), "last_bar": frame.index[-1].date().isoformat(),
    }
