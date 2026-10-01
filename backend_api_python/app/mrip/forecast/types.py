"""Forecast engine contract (work 008).

A forecast is a distribution of the SIMPLE RETURN over a horizon, expressed as
quantiles; scenarios and price levels are derived from it. A forecast is never a
price target. Providers may only use data up to and including ``as_of``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Mapping, Protocol

import pandas as pd

# Standard quantile grid. P10/P25/P50/P75/P90 are the specified outputs; P20/P80 define the scenarios.
QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9)
SCENARIO_QUANTILES = {"bear": 0.2, "base": 0.5, "bull": 0.8}  # scenario-v0-uncalibrated
SCENARIO_VERSION = "scenario-v0-uncalibrated"


class Horizon(int, Enum):
    """Trading-day horizons matching the outcome-resolution horizons (1w, 1m, 3m, 6m, 12m)."""

    W1 = 5
    M1 = 21
    M3 = 63
    M6 = 126
    M12 = 252


class ForecastUnavailable(Exception):
    """The provider cannot produce a forecast for this request (too little data, service down...)."""


@dataclass(frozen=True, slots=True)
class ForecastRequest:
    """``prices`` are positive closes indexed by date; the last index is ``as_of``.

    ``covariates`` (optional) must be aligned to dates <= as_of only; providers that
    do not use covariates ignore them.
    """

    symbol: str
    prices: pd.Series
    horizon_days: int
    covariates: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        if len(self.prices) == 0 or (self.prices <= 0).any():
            raise ValueError("prices must be non-empty and positive")
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("prices must be sorted by date")
        if self.covariates is not None and len(self.covariates) and self.covariates.index.max() > self.prices.index[-1]:
            raise ValueError("covariates extend past as_of (look-ahead)")

    @property
    def as_of(self) -> date:
        return pd.Timestamp(self.prices.index[-1]).date()

    @property
    def last_price(self) -> float:
        return float(self.prices.iloc[-1])


@dataclass(frozen=True, slots=True)
class Scenarios:
    """Bear / Base / Bull returns (P20 / P50 / P80, ``SCENARIO_VERSION``)."""

    bear: float
    base: float
    bull: float
    version: str = SCENARIO_VERSION


@dataclass(frozen=True, slots=True)
class ForecastResult:
    provider: str
    provider_version: str
    symbol: str
    as_of: date
    horizon_days: int
    last_price: float
    return_quantiles: Mapping[float, float]  # simple-return quantiles over the horizon
    n_obs: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if set(self.return_quantiles) != set(QUANTILES):
            raise ValueError("return_quantiles must cover exactly the standard quantile grid")
        values = [self.return_quantiles[q] for q in QUANTILES]
        if any(b < a for a, b in zip(values, values[1:])):
            raise ValueError("quantiles must be non-decreasing")

    def price_quantiles(self) -> dict[float, float]:
        return {q: self.last_price * (1.0 + r) for q, r in self.return_quantiles.items()}

    @property
    def scenarios(self) -> Scenarios:
        q = self.return_quantiles
        return Scenarios(bear=q[SCENARIO_QUANTILES["bear"]], base=q[SCENARIO_QUANTILES["base"]], bull=q[SCENARIO_QUANTILES["bull"]])


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Walk-forward evaluation of a provider at one horizon (see ``evaluation.py``)."""

    provider: str
    provider_version: str
    horizon_days: int
    n_forecasts: int
    mean_pinball_loss: float  # average over the standard quantile grid; lower is better
    coverage_p10_p90: float  # share of outcomes inside [P10, P90]; ideal 0.80
    coverage_p25_p75: float  # ideal 0.50
    median_abs_error: float
    directional_accuracy: float | None  # share where sign(P50) matched the realised sign
    first_origin: date | None
    last_origin: date | None
    notes: tuple[str, ...] = field(default_factory=tuple)


class ForecastProvider(Protocol):
    def forecast(self, request: ForecastRequest) -> ForecastResult: ...

    def evaluate(
        self,
        prices: pd.Series,
        horizon_days: int,
        *,
        min_history: int = 250,
        step: int | None = None,
        covariates: pd.DataFrame | None = None,
    ) -> EvaluationReport: ...

    def version(self) -> str: ...
