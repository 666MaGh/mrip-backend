"""Build loggable predictions from forecast and options-analysis results."""
from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Sequence

from app.mrip.forecast.types import ForecastResult
from app.mrip.options.analysis import OptionsAnalysis
from app.mrip.outcomes.types import HorizonKind, NewPrediction, PredictionType

_FORECAST_HORIZONS = {5: HorizonKind.W1, 21: HorizonKind.M1, 63: HorizonKind.M3, 126: HorizonKind.M6, 252: HorizonKind.M12}
_DEFAULT_OPTION_HORIZONS = (HorizonKind.EOD, HorizonKind.NEXT_SESSION, HorizonKind.W1)


def horizon_kind_for(horizon_days: int) -> HorizonKind:
    kind = _FORECAST_HORIZONS.get(horizon_days)
    if kind is None:
        raise ValueError(f"horizon_days {horizon_days} is not a resolvable horizon {sorted(_FORECAST_HORIZONS)}")
    return kind


def forecast_prediction(
    result: ForecastResult, *, benchmark: str | None = "SPY", market_regime: Mapping[str, Any] | None = None
) -> NewPrediction:
    """A forecast made after the close of ``as_of`` (23:00 UTC that day); entry is that day's close."""
    kind = horizon_kind_for(result.horizon_days)
    quantiles = {str(q): v for q, v in result.return_quantiles.items()}
    s = result.scenarios
    return NewPrediction(
        prediction_type=PredictionType.FORECAST,
        subject=result.symbol,
        benchmark=benchmark,
        made_at=datetime.combine(result.as_of, time(23, 0), tzinfo=timezone.utc),
        horizon_kind=kind,
        model_version=result.provider_version,
        payload={
            "quantiles": quantiles,
            "scenarios": {"bear": s.bear, "base": s.base, "bull": s.bull, "version": s.version},
            "last_price": result.last_price,
            "n_obs": result.n_obs,
            "warnings": list(result.warnings),
            **({"market_regime": dict(market_regime)} if market_regime else {}),
        },
    )


def _atm_iv_near_30d(analysis: OptionsAnalysis) -> float | None:
    points = [p for p in analysis.observed.term_structure if p.dte >= 1]
    if not points:
        return None
    return min(points, key=lambda p: (abs(p.dte - 30), p.dte)).atm_iv


def options_event_predictions(
    analysis: OptionsAnalysis,
    *,
    horizons: Sequence[HorizonKind] = _DEFAULT_OPTION_HORIZONS,
    expiry: date | None = None,
    benchmark: str | None = "SPY",
    lineage: Mapping[str, Any] | None = None,
    market_regime: Mapping[str, Any] | None = None,
) -> list[NewPrediction]:
    """One prediction per horizon from a modeled options analysis (made at the snapshot time, entry = spot)."""
    spot = analysis.observed.underlying_price
    if spot is None:
        raise ValueError("analysis has no underlying price")
    if HorizonKind.EXPIRY in horizons and expiry is None:
        raise ValueError("an EXPIRY horizon needs the expiry date")
    gex, modeled = analysis.modeled.gex, analysis.modeled
    amp = modeled.amplification
    version = f"{modeled.assumptions['version']}|{modeled.regime_method_version}|{amp.method_version}"
    payload = {
        "spot": spot,
        "regime": modeled.regime.value,
        "amplification": amp.level.value,
        "direction": amp.direction.value,
        "tilt": gex.tilt,
        "net_gex": gex.net_gex,
        "gross_gex": gex.gross_gex,
        "flip": gex.flip.level,
        "call_wall": gex.walls.call_wall,
        "put_wall": gex.walls.put_wall,
        "zero_dte_share": gex.zero_dte_share,
        "atm_iv": _atm_iv_near_30d(analysis),
        "oi_status": analysis.data_quality.oi_status,
        "modeled": True,
        **({"market_regime": dict(market_regime)} if market_regime else {}),
    }
    return [
        NewPrediction(
            prediction_type=PredictionType.OPTIONS_EVENT,
            subject=analysis.underlying,
            benchmark=benchmark,
            made_at=analysis.data_quality.snapshot_timestamp,
            entry_price=spot,
            horizon_kind=kind,
            expiry_date=expiry if kind is HorizonKind.EXPIRY else None,
            model_version=version,
            payload=payload,
            lineage=dict(lineage or {}),
        )
        for kind in horizons
    ]
