"""Walk-forward evaluation of a forecast provider at one horizon (work 008).

At every origin the provider only receives data up to and including the origin
date (prices and covariates are truncated), so the evaluation has no look-ahead.
The realised outcome is the simple return from the origin close to the close
``horizon_days`` rows later. By default origins are spaced ``horizon_days``
apart so outcomes do not overlap.
"""
from __future__ import annotations

from datetime import date
from typing import Mapping, Protocol

import numpy as np
import pandas as pd

from app.mrip.forecast.types import EvaluationReport, ForecastRequest, ForecastResult, ForecastUnavailable


class _EvaluableProvider(Protocol):
    def forecast(self, request: ForecastRequest) -> ForecastResult: ...

    def version(self) -> str: ...


def pinball_loss(actual: float, quantile_forecasts: Mapping[float, float]) -> float:
    """Mean pinball (quantile) loss over the given quantile levels."""
    if not quantile_forecasts:
        raise ValueError("quantile_forecasts must not be empty")
    losses = []
    for tau, q in quantile_forecasts.items():
        diff = actual - q
        losses.append(max(tau * diff, (tau - 1.0) * diff))
    return float(sum(losses) / len(losses))


def interval_covered(actual: float, lower: float, upper: float) -> bool:
    """True when ``actual`` lies inside the closed interval [lower, upper]."""
    return lower <= actual <= upper


def _validate(prices: pd.Series, horizon_days: int, min_history: int, step: int | None) -> int:
    """Validate arguments and return the effective stride."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if min_history < 2:
        raise ValueError("min_history must be >= 2")
    if step is not None and step < 1:
        raise ValueError("step must be >= 1")
    if len(prices) == 0 or not bool((prices > 0).all()):
        raise ValueError("prices must be non-empty and positive")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be sorted by date")
    return step if step is not None else horizon_days


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def evaluate_provider(
    provider: _EvaluableProvider,
    prices: pd.Series,
    horizon_days: int,
    *,
    min_history: int = 250,
    step: int | None = None,
    covariates: pd.DataFrame | None = None,
) -> EvaluationReport:
    """Walk-forward evaluation of ``provider`` at one horizon (see module docstring)."""
    stride = _validate(prices, horizon_days, min_history, step)
    last_pos = len(prices) - 1

    pinballs: list[float] = []
    covered_80: list[bool] = []
    covered_50: list[bool] = []
    abs_errors: list[float] = []
    direction_hits: list[bool] = []
    origins: list[date] = []
    skipped = 0

    i = min_history - 1
    while i + horizon_days <= last_pos:
        origin_ts = prices.index[i]
        request = ForecastRequest(
            symbol="evaluation",
            prices=prices.iloc[: i + 1],
            horizon_days=horizon_days,
            covariates=covariates.loc[:origin_ts] if covariates is not None else None,
        )
        try:
            result = provider.forecast(request)
        except ForecastUnavailable:
            skipped += 1
            i += stride
            continue

        actual = float(prices.iloc[i + horizon_days] / prices.iloc[i] - 1.0)
        q = result.return_quantiles
        pinballs.append(pinball_loss(actual, q))
        covered_80.append(interval_covered(actual, q[0.1], q[0.9]))
        covered_50.append(interval_covered(actual, q[0.25], q[0.75]))
        abs_errors.append(abs(actual - q[0.5]))
        s_pred, s_act = _sign(q[0.5]), _sign(actual)
        if s_pred != 0 and s_act != 0:
            direction_hits.append(s_pred == s_act)
        origins.append(pd.Timestamp(origin_ts).date())
        i += stride

    if not origins:
        raise ValueError("no evaluable origins")

    notes: list[str] = []
    if skipped > 0:
        notes.append(f"{skipped} origin(s) skipped: provider unavailable")
    if stride < horizon_days:
        notes.append("overlapping outcomes: step < horizon")

    return EvaluationReport(
        provider=str(getattr(provider, "name", type(provider).__name__)),
        provider_version=provider.version(),
        horizon_days=horizon_days,
        n_forecasts=len(origins),
        mean_pinball_loss=float(np.mean(pinballs)),
        coverage_p10_p90=float(np.mean(covered_80)),
        coverage_p25_p75=float(np.mean(covered_50)),
        median_abs_error=float(np.mean(abs_errors)),
        directional_accuracy=float(np.mean(direction_hits)) if direction_hits else None,
        first_origin=origins[0],
        last_origin=origins[-1],
        notes=tuple(notes),
    )
