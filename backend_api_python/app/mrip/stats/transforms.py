"""Pure series transforms for the statistics layer.

Deterministic and free of look-ahead unless a function says otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Standard deviations at or below this (relative to the mean magnitude) count as zero.
_ZERO_STD_REL = 1e-12


def log_returns(prices: pd.Series) -> pd.Series:
    """Natural-log returns; the first (undefined) observation is dropped."""
    if not prices.index.is_monotonic_increasing:
        raise ValueError("index must be monotonic increasing")
    if bool((prices <= 0).any()):
        raise ValueError("prices must be strictly positive")
    return np.log(prices.astype(float)).diff().iloc[1:]


def align(*series: pd.Series) -> list[pd.Series]:
    """Inner-join series on their index and drop rows where any is NaN."""
    if len(series) < 2:
        raise ValueError("align needs at least 2 series")
    frame = pd.concat(list(series), axis=1, join="inner", keys=range(len(series)))
    frame = frame.dropna().sort_index()
    return [frame[i].rename(series[i].name) for i in range(len(series))]


def _safe_divide(numerator: pd.Series, std: pd.Series, mean: pd.Series) -> pd.Series:
    """Divide by std, returning NaN where std is (numerically) zero."""
    floor = _ZERO_STD_REL * np.maximum(mean.abs(), 1.0)
    return numerator / std.where(std > floor)


def zscore(series: pd.Series, window: int | None = None) -> pd.Series:
    """Z-score (ddof=1). With `window`, a trailing window including the current value."""
    values = series.astype(float)
    if window is None:
        mean = pd.Series(values.mean(), index=values.index)
        std = pd.Series(values.std(ddof=1), index=values.index)
    else:
        if window < 2:
            raise ValueError("window must be >= 2")
        roll = values.rolling(window, min_periods=window)
        mean = roll.mean()
        std = roll.std(ddof=1)
    return _safe_divide(values - mean, std, mean)


def volatility_normalize(returns: pd.Series, window: int = 20) -> pd.Series:
    """Divide each return by the std of the previous `window` returns (no look-ahead)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    values = returns.astype(float)
    roll = values.rolling(window, min_periods=window)
    std = roll.std(ddof=1).shift(1)
    mean = roll.mean().shift(1)
    return _safe_divide(values, std, mean)


def relative_strength(a_prices: pd.Series, b_prices: pd.Series) -> pd.Series:
    """Ratio a/b on the aligned index, rescaled so the first value is 1.0."""
    a, b = align(a_prices, b_prices)
    if a.empty:
        raise ValueError("no overlapping observations")
    ratio = a.astype(float) / b.astype(float)
    return ratio / ratio.iloc[0]
