"""Correlation family: Pearson, rolling, partial, cross-correlation and lead/lag.

Pure functions on aligned pandas Series. Lag convention: ``lag = k > 0`` pairs
``x[t-k]`` with ``y[t]``, i.e. x LEADS y by k observations; ``k < 0`` means y
leads x. All p-values are two-sided.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True, slots=True)
class CorrResult:
    r: float
    n: int
    p_value: float


@dataclass(frozen=True, slots=True)
class LeadLagResult:
    best_lag: int
    best: CorrResult
    corrected_p_value: float  # Bonferroni over all lags tried
    lags_tested: int

    @property
    def direction(self) -> str:
        if self.best_lag > 0:
            return "x_leads"
        if self.best_lag < 0:
            return "y_leads"
        return "contemporaneous"


def _p_from_r(r: float, dof: int) -> float:
    if dof <= 0 or not np.isfinite(r):
        return float("nan")
    if abs(r) >= 1.0:
        return 0.0
    t = r * np.sqrt(dof / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), dof))


def pearson(x: pd.Series, y: pd.Series) -> CorrResult:
    pair = pd.concat([x, y], axis=1, join="inner").dropna()
    n = len(pair)
    if n < 3:
        raise ValueError("need at least 3 paired observations")
    a, b = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return CorrResult(r=float("nan"), n=n, p_value=float("nan"))
    r = float(np.corrcoef(a, b)[0, 1])
    return CorrResult(r=r, n=n, p_value=_p_from_r(r, n - 2))


def rolling_correlation(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    if window < 3:
        raise ValueError("window must be at least 3")
    pair = pd.concat([x, y], axis=1, join="inner")
    return pair.iloc[:, 0].rolling(window).corr(pair.iloc[:, 1])


def partial_correlation(x: pd.Series, y: pd.Series, controls: pd.DataFrame) -> CorrResult:
    """Correlation of x and y after removing the linear effect of ``controls`` from both."""
    frame = pd.concat([x.rename("x"), y.rename("y"), controls], axis=1, join="inner").dropna()
    k = controls.shape[1]
    n = len(frame)
    if n < k + 4:
        raise ValueError("too few observations for the number of controls")
    design = np.column_stack([np.ones(n), frame[list(controls.columns)].to_numpy()])

    def residual(col: str) -> np.ndarray:
        values = frame[col].to_numpy()
        coef, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coef

    rx, ry = residual("x"), residual("y")
    if np.std(rx) == 0 or np.std(ry) == 0:
        return CorrResult(r=float("nan"), n=n, p_value=float("nan"))
    r = float(np.corrcoef(rx, ry)[0, 1])
    return CorrResult(r=r, n=n, p_value=_p_from_r(r, n - 2 - k))


def lagged_pair(x: pd.Series, y: pd.Series, lag: int) -> tuple[pd.Series, pd.Series]:
    pair = pd.concat([x, y], axis=1, join="inner")
    xs, ys = pair.iloc[:, 0], pair.iloc[:, 1]
    return (xs.shift(lag), ys) if lag >= 0 else (xs, ys.shift(-lag))


def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> dict[int, CorrResult]:
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    return {lag: pearson(*lagged_pair(x, y, lag)) for lag in range(-max_lag, max_lag + 1)}


def lead_lag(x: pd.Series, y: pd.Series, max_lag: int) -> LeadLagResult:
    """Lag with the largest |r|; its p-value is Bonferroni-corrected for the lags searched."""
    table = {lag: res for lag, res in cross_correlation(x, y, max_lag).items() if np.isfinite(res.r)}
    if not table:
        raise ValueError("no valid lag (constant series?)")
    best_lag = max(table, key=lambda lag: (abs(table[lag].r), -abs(lag)))
    best = table[best_lag]
    tested = len(table)
    return LeadLagResult(
        best_lag=best_lag,
        best=best,
        corrected_p_value=min(1.0, best.p_value * tested),
        lags_tested=tested,
    )
