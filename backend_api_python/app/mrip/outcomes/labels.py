"""Labels for calibration and learning, built ONLY from observed outcomes.

A label is never taken from a prediction's own content (and in particular never
from a LAYA/model output): the inputs here are (prediction, resolved outcome)
pairs, and unresolved outcomes are refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from app.mrip.outcomes.types import HorizonKind, Outcome, OutcomeError, OutcomeStatus, Prediction, PredictionType


def require_observed(outcome: Outcome) -> float:
    """The realised return of a resolved outcome; anything else cannot produce a label."""
    if outcome.status is not OutcomeStatus.RESOLVED or outcome.actual_return is None:
        raise OutcomeError(f"outcome {outcome.id} is not a resolved observation; no label can be built from it")
    return outcome.actual_return


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    prediction_id: int
    subject: str
    model_version: str
    horizon_kind: HorizonKind
    made_at: datetime
    actual_return: float
    hits: Mapping[float, bool]  # quantile level -> realised return <= predicted quantile


def forecast_calibration_rows(pairs: Sequence[tuple[Prediction, Outcome]]) -> list[CalibrationRow]:
    rows = []
    for prediction, outcome in pairs:
        if prediction.prediction_type is not PredictionType.FORECAST:
            raise OutcomeError(f"prediction {prediction.id} is not a forecast")
        actual = require_observed(outcome)
        quantiles = {float(q): float(v) for q, v in prediction.payload["quantiles"].items()}
        rows.append(
            CalibrationRow(
                prediction.id, prediction.subject, prediction.model_version, prediction.horizon_kind,
                prediction.made_at, actual, {q: actual <= v for q, v in sorted(quantiles.items())},
            )
        )
    return rows


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    n: int
    nominal: float
    observed: float  # share of outcomes at or below the predicted quantile; equals nominal when calibrated


def reliability_table(rows: Sequence[CalibrationRow]) -> dict[float, ReliabilityBin]:
    """Observed hit frequency per nominal quantile level (the basis for segmented calibration)."""
    levels = sorted({q for r in rows for q in r.hits})
    table = {}
    for q in levels:
        flags = [r.hits[q] for r in rows if q in r.hits]
        table[q] = ReliabilityBin(n=len(flags), nominal=q, observed=sum(flags) / len(flags))
    return table


@dataclass(frozen=True, slots=True)
class OptionsEventRow:
    prediction_id: int
    subject: str
    horizon_kind: HorizonKind
    regime: str
    predicted_direction: str
    predicted_level: str
    amplified: bool | None
    call_wall: str | None  # untested / held / broke
    put_wall: str | None
    realized_to_implied_vol: float | None
    actual_return: float
    max_adverse_excursion: float | None
    max_favorable_excursion: float | None


def options_event_rows(pairs: Sequence[tuple[Prediction, Outcome]]) -> list[OptionsEventRow]:
    rows = []
    for prediction, outcome in pairs:
        if prediction.prediction_type is not PredictionType.OPTIONS_EVENT:
            raise OutcomeError(f"prediction {prediction.id} is not an options event")
        actual = require_observed(outcome)
        m = outcome.measures
        walls = m.get("wall_hold_or_break") or {}
        implied = m.get("implied_move") or {}
        rows.append(
            OptionsEventRow(
                prediction.id, prediction.subject, prediction.horizon_kind,
                prediction.payload["regime"], prediction.payload["direction"], prediction.payload["amplification"],
                (m.get("amplification_realized") or {}).get("amplified"),
                walls.get("call"), walls.get("put"), implied.get("realized_to_implied_vol"),
                actual, outcome.max_adverse_excursion, outcome.max_favorable_excursion,
            )
        )
    return rows
