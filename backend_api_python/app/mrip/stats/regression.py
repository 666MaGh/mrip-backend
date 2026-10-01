"""OLS regression, beta and factor exposure with HAC (Newey-West) standard errors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True, slots=True)
class OlsResult:
    n: int
    r_squared: float
    params: dict[str, float]
    t_values: dict[str, float]
    p_values: dict[str, float]
    hac_lags: int


def default_hac_lags(n: int) -> int:
    """Newey-West rule of thumb: floor(4 * (n / 100) ** (2 / 9))."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def ols(y: pd.Series, X: pd.DataFrame, hac_lags: int | None = None) -> OlsResult:
    frame = pd.concat([y.rename("__y"), X], axis=1, join="inner").dropna()
    n, k = len(frame), X.shape[1]
    if n < k + 3:
        raise ValueError("too few observations for the number of regressors")
    lags = default_hac_lags(n) if hac_lags is None else hac_lags
    design = sm.add_constant(frame[list(X.columns)], has_constant="add")
    fit = sm.OLS(frame["__y"], design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return OlsResult(
        n=n,
        r_squared=float(fit.rsquared),
        params={k_: float(v) for k_, v in fit.params.items()},
        t_values={k_: float(v) for k_, v in fit.tvalues.items()},
        p_values={k_: float(v) for k_, v in fit.pvalues.items()},
        hac_lags=lags,
    )


def beta(asset: pd.Series, market: pd.Series) -> OlsResult:
    """Market beta: slope of asset returns on market returns (name ``market``)."""
    return ols(asset, market.rename("market").to_frame())


def factor_exposure(asset: pd.Series, factors: pd.DataFrame) -> OlsResult:
    return ols(asset, factors)
