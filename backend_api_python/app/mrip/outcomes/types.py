"""Prediction and outcome records (work 009).

Predictions are immutable logs of what the system said; outcomes are what the
market then did, resolved from observed data. They live in separate tables so a
prediction can never be rewritten to fit its outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping


class PredictionType(str, Enum):
    FORECAST = "forecast"  # logged as FORECAST_SCENARIO by the automatic logger
    OPTIONS_EVENT = "options_event"
    DISCOVER_ITEM = "discover_item"  # automatic logger; attention event with optional direction
    RELATED_SIGNAL = "related_signal"  # automatic logger; up/down signal of the related service


# Types whose outcome is judged against a benchmark (excess return), so resolution needs its prices.
BENCHMARK_REQUIRED_TYPES = frozenset({PredictionType.DISCOVER_ITEM, PredictionType.RELATED_SIGNAL})


class HorizonKind(str, Enum):
    """1w, 1m, 3m, 6m, 12m plus the shorter analytical horizons for options events."""

    W1 = "W1"
    M1 = "M1"
    M3 = "M3"
    M6 = "M6"
    M12 = "M12"
    EOD = "EOD"
    NEXT_SESSION = "NEXT_SESSION"
    EXPIRY = "EXPIRY"


class OutcomeStatus(str, Enum):
    RESOLVED = "resolved"
    UNRESOLVABLE = "unresolvable"


class OutcomeError(Exception):
    """Invalid prediction/outcome operation."""


@dataclass(frozen=True, slots=True)
class NewPrediction:
    prediction_type: PredictionType
    subject: str
    made_at: datetime
    horizon_kind: HorizonKind
    model_version: str
    payload: Mapping[str, Any]
    benchmark: str | None = None
    entry_price: float | None = None  # None: use the close of the made_at session
    expiry_date: date | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Prediction:
    id: int
    prediction_type: PredictionType
    subject: str
    made_at: datetime
    horizon_kind: HorizonKind
    model_version: str
    payload: Mapping[str, Any]
    benchmark: str | None = None
    entry_price: float | None = None
    expiry_date: date | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Outcome:
    id: int
    prediction_id: int
    status: OutcomeStatus
    resolved_at: datetime
    resolver_version: str
    window_start: date | None = None
    window_end: date | None = None
    actual_return: float | None = None
    benchmark_return: float | None = None
    realized_volatility: float | None = None
    max_adverse_excursion: float | None = None
    max_favorable_excursion: float | None = None
    measures: Mapping[str, Any] = field(default_factory=dict)
    data_provenance: Mapping[str, Any] = field(default_factory=dict)
    note: str | None = None
