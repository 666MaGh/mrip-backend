"""Shared helpers for forecast providers."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from app.mrip.forecast.types import QUANTILES, EvaluationReport, ForecastRequest, ForecastResult, ForecastUnavailable


def rearrange(values: Sequence[float]) -> list[float]:
    """Monotone rearrangement: sort quantile estimates so they never cross."""
    return [float(v) for v in np.sort(np.asarray(values, dtype=float))]


def returns_from_log_quantiles(log_returns_by_q: Mapping[float, float]) -> dict[float, float]:
    """Convert log-return quantiles to simple-return quantiles on the standard grid (rearranged)."""
    simple = [float(np.expm1(log_returns_by_q[q])) for q in QUANTILES]
    return dict(zip(QUANTILES, rearrange(simple)))


class BaseForecastProvider:
    """Gives every provider its ``evaluate`` via the shared walk-forward backtest."""

    name = "base"

    def forecast(self, request: ForecastRequest) -> ForecastResult:  # pragma: no cover - interface
        raise NotImplementedError

    def version(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def evaluate(
        self,
        prices: pd.Series,
        horizon_days: int,
        *,
        min_history: int = 250,
        step: int | None = None,
        covariates: pd.DataFrame | None = None,
    ) -> EvaluationReport:
        from app.mrip.forecast.evaluation import evaluate_provider

        return evaluate_provider(
            self, prices, horizon_days, min_history=min_history, step=step, covariates=covariates
        )

    @staticmethod
    def require(request: ForecastRequest, minimum: int, what: str) -> None:
        if len(request.prices) < minimum:
            raise ForecastUnavailable(f"{what} needs at least {minimum} prices, got {len(request.prices)}")
