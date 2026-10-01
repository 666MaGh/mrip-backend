"""Baseline forecast providers: Naive, Statistical (empirical) and FactorRegression.

All three work on log returns and only use data up to ``as_of``. They are the
benchmarks every other provider (TimesFM, ensembles) must beat in walk-forward
evaluation before it earns weight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from app.mrip.forecast.base import BaseForecastProvider, returns_from_log_quantiles
from app.mrip.forecast.types import QUANTILES, ForecastRequest, ForecastResult, ForecastUnavailable


def _result(
    provider: "BaseForecastProvider", request: ForecastRequest, log_q: dict[float, float], n_obs: int,
    warnings: tuple[str, ...] = (),
) -> ForecastResult:
    return ForecastResult(
        provider=provider.name,
        provider_version=provider.version(),
        symbol=request.symbol,
        as_of=request.as_of,
        horizon_days=request.horizon_days,
        last_price=request.last_price,
        return_quantiles=returns_from_log_quantiles(log_q),
        n_obs=n_obs,
        warnings=warnings,
    )


class NaiveBaseline(BaseForecastProvider):
    """Random walk: zero expected log return, spread from trailing volatility scaled by sqrt(horizon)."""

    name = "naive"

    def __init__(self, vol_window: int = 60) -> None:
        if vol_window < 10:
            raise ValueError("vol_window must be >= 10")
        self._window = vol_window

    def version(self) -> str:
        return f"naive-v1(vol_window={self._window})"

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        self.require(request, self._window + 1, "naive baseline")
        log_ret = np.diff(np.log(request.prices.to_numpy()))[-self._window :]
        sigma = float(np.std(log_ret, ddof=1)) * np.sqrt(request.horizon_days)
        log_q = {q: float(stats.norm.ppf(q)) * sigma for q in QUANTILES}
        return _result(self, request, log_q, len(log_ret))


class StatisticalBaseline(BaseForecastProvider):
    """Empirical distribution of historical overlapping horizon returns (last ``lookback`` days)."""

    name = "statistical"

    def __init__(self, lookback: int = 1260, min_samples: int = 120) -> None:
        if lookback < min_samples or min_samples < 30:
            raise ValueError("need lookback >= min_samples >= 30")
        self._lookback, self._min = lookback, min_samples

    def version(self) -> str:
        return f"statistical-v1(lookback={self._lookback},min_samples={self._min})"

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        h = request.horizon_days
        log_p = np.log(request.prices.to_numpy())[-(self._lookback + h + 1) :]
        horizon_returns = log_p[h:] - log_p[:-h]
        if len(horizon_returns) < self._min:
            raise ForecastUnavailable(
                f"statistical baseline needs {self._min} overlapping {h}-day returns, got {len(horizon_returns)}"
            )
        log_q = {q: float(np.quantile(horizon_returns, q)) for q in QUANTILES}
        warnings = ("overlapping horizon returns: effective sample is smaller than n_obs",) if h > 1 else ()
        return _result(self, request, log_q, len(horizon_returns), warnings)


class FactorRegression(BaseForecastProvider):
    """Predictive regression of the forward horizon return on covariates known at the forecast date.

    Features at date t are the covariate columns at t (callers pass them already
    transformed, e.g. trailing returns, VIX level, rate changes) and, optionally,
    the asset's own trailing horizon return. The target is log(P[t+h]/P[t]),
    fitted only where P[t+h] is already known. The forecast adds the empirical
    residual quantiles to the predicted mean. It is a benchmark; whether it adds
    information is decided by walk-forward evaluation, never assumed.
    """

    name = "factor_regression"

    def __init__(self, lookback: int = 1260, include_own_momentum: bool = True, weak_fit_p: float = 0.10) -> None:
        self._lookback, self._momentum, self._weak_p = lookback, include_own_momentum, weak_fit_p

    def version(self) -> str:
        return f"factor-regression-v1(lookback={self._lookback},own_momentum={self._momentum})"

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        if request.covariates is None or request.covariates.shape[1] == 0:
            raise ForecastUnavailable("factor regression needs covariates")
        h = request.horizon_days
        log_p = np.log(request.prices)
        features = request.covariates.reindex(request.prices.index).ffill(limit=5).copy()
        if self._momentum:
            features["own_momentum"] = log_p - log_p.shift(h)
        target = log_p.shift(-h) - log_p  # known only where t + h <= as_of
        frame = pd.concat([target.rename("__y"), features], axis=1)
        x_now = features.iloc[[-1]]
        if x_now.isna().any(axis=None):
            raise ForecastUnavailable("covariates are missing at as_of")
        train = frame.dropna().iloc[-self._lookback :]
        k = features.shape[1]
        if len(train) < max(5 * k + 60, 100):
            raise ForecastUnavailable(f"factor regression needs more usable rows (have {len(train)}, {k} features)")
        design = sm.add_constant(train[list(features.columns)], has_constant="add")
        fit = sm.OLS(train["__y"], design).fit(cov_type="HAC", cov_kwds={"maxlags": h})
        mean = float(fit.predict(sm.add_constant(x_now, has_constant="add")).iloc[0])
        residuals = np.asarray(fit.resid)
        log_q = {q: mean + float(np.quantile(residuals, q)) for q in QUANTILES}
        warnings: list[str] = ["overlapping horizon returns: effective sample is smaller than n_obs"] if h > 1 else []
        joint_p = float(fit.f_pvalue) if np.isfinite(fit.f_pvalue) else float("nan")
        if not joint_p < self._weak_p:
            warnings.append(f"weak fit: joint p-value {joint_p:.3f} (R2 {fit.rsquared:.3f}); factors add little")
        return _result(self, request, log_q, len(train), tuple(warnings))
