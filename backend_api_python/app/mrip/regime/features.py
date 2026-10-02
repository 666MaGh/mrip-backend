"""Trailing regime features for the regime layer.

Every function is strictly trailing: the value at time t uses only observations at or
before t. Inputs are pandas Series with an ascending, unique DatetimeIndex; anything
else raises ValueError. Warm-up positions are NaN, never back-filled.
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

# Trading days per year used to annualise daily volatility.
_TRADING_DAYS = 252


class Trend(str, Enum):
    """Price trend relative to the slow and fast moving averages."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    MIXED = "MIXED"


class Direction(str, Enum):
    """Direction of a series over a lookback, outside a dead band."""

    RISING = "RISING"
    FALLING = "FALLING"
    FLAT = "FLAT"


def _check_index(series: pd.Series, name: str = "series") -> None:
    """Require an ascending, unique DatetimeIndex."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must have a DatetimeIndex")
    if not series.index.is_unique:
        raise ValueError(f"{name} index must be unique")
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be monotonic increasing")


def _check_min(value: int, minimum: int, name: str) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _common(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Both series as floats on their shared index (no NaN dropping)."""
    _check_index(a, "first series")
    _check_index(b, "second series")
    index = a.index.intersection(b.index)
    return a.reindex(index).astype(float), b.reindex(index).astype(float)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN where the denominator is 0 or NaN."""
    return numerator / denominator.where(denominator != 0)


def _last_rank(window_values: np.ndarray) -> float:
    """Share of the window that is <= its last value."""
    return float(np.mean(window_values <= window_values[-1]))


def trailing_percentile(series: pd.Series, window: int) -> pd.Series:
    """Share of the last `window` values (current included) that are <= the current one.

    Lies in (0, 1]; NaN until `window` observations exist.
    """
    _check_index(series)
    _check_min(window, 1, "window")
    values = series.astype(float)
    return values.rolling(window, min_periods=window).apply(_last_rank, raw=True)


def change(series: pd.Series, periods: int) -> pd.Series:
    """Absolute change versus `periods` observations earlier."""
    _check_index(series)
    _check_min(periods, 1, "periods")
    values = series.astype(float)
    return values - values.shift(periods)


def pct_change(series: pd.Series, periods: int) -> pd.Series:
    """Relative change versus `periods` observations earlier; NaN where that value is 0 or NaN."""
    _check_index(series)
    _check_min(periods, 1, "periods")
    values = series.astype(float)
    return _ratio(values, values.shift(periods)) - 1.0


def realized_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Annualised (sqrt(252)) sample std (ddof=1) of the trailing `window` daily log returns.

    NaN until `window` returns exist. Prices must be strictly positive.
    """
    _check_index(prices, "prices")
    _check_min(window, 2, "window")
    values = prices.astype(float)
    if bool((values <= 0).any()):
        raise ValueError("prices must be strictly positive")
    returns = np.log(values).diff()
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(_TRADING_DAYS)


def implied_to_realized(implied_pct: pd.Series, realized: pd.Series) -> pd.Series:
    """(implied_pct / 100) / realized on the common index; NaN where realized is 0 or NaN.

    `implied_pct` is in volatility points (18.5 means 18.5 %).
    """
    implied, real = _common(implied_pct, realized)
    return _ratio(implied / 100.0, real)


def term_ratio(front: pd.Series, back: pd.Series) -> pd.Series:
    """front / back on the common index; NaN where back is 0 or NaN."""
    front_values, back_values = _common(front, back)
    return _ratio(front_values, back_values)


def trend_state(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """Trend label per observation; NaN until `slow` observations exist.

    UPTREND: close > SMA(slow) and SMA(fast) > SMA(slow). DOWNTREND: both below.
    Everything else (including exact equality) is MIXED.
    """
    _check_index(prices, "prices")
    _check_min(fast, 1, "fast")
    if fast >= slow:
        raise ValueError("fast must be < slow")
    close = prices.astype(float)
    sma_fast = close.rolling(fast, min_periods=fast).mean()
    sma_slow = close.rolling(slow, min_periods=slow).mean()
    valid = (close.notna() & sma_fast.notna() & sma_slow.notna()).to_numpy()
    up = ((close > sma_slow) & (sma_fast > sma_slow)).to_numpy()
    down = ((close < sma_slow) & (sma_fast < sma_slow)).to_numpy()
    labels = np.full(len(close), np.nan, dtype=object)
    for i in np.flatnonzero(valid):
        labels[i] = Trend.UPTREND if up[i] else Trend.DOWNTREND if down[i] else Trend.MIXED
    return pd.Series(labels, index=close.index, dtype=object)


def direction_state(
    series: pd.Series,
    periods: int = 60,
    band: float = 0.02,
    mode: str = "relative",
) -> pd.Series:
    """Direction of the change over `periods` observations; NaN where it is undefined.

    `mode` is "relative" (pct change) or "absolute" (difference). RISING if the change
    is > band, FALLING if < -band, otherwise FLAT (exactly at the band is FLAT).
    """
    if mode not in ("relative", "absolute"):
        raise ValueError("mode must be 'relative' or 'absolute'")
    if band < 0:
        raise ValueError("band must be >= 0")
    delta = pct_change(series, periods) if mode == "relative" else change(series, periods)
    valid = delta.notna().to_numpy()
    values = delta.to_numpy()
    labels = np.full(len(delta), np.nan, dtype=object)
    for i in np.flatnonzero(valid):
        labels[i] = (
            Direction.RISING if values[i] > band
            else Direction.FALLING if values[i] < -band
            else Direction.FLAT
        )
    return pd.Series(labels, index=delta.index, dtype=object)


def drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
    """Close over the rolling max of the trailing `window` closes (current included) minus 1.

    Always <= 0. Prices must be strictly positive.
    """
    _check_index(prices, "prices")
    _check_min(window, 1, "window")
    close = prices.astype(float)
    if bool((close <= 0).any()):
        raise ValueError("prices must be strictly positive")
    peak = close.rolling(window, min_periods=1).max()
    return close / peak - 1.0
