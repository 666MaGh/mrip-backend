"""Baseline providers and ensemble on synthetic series with known structure."""
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.mrip.forecast.baselines import FactorRegression, NaiveBaseline, StatisticalBaseline
from app.mrip.forecast.ensemble import EnsembleProvider, inverse_loss_weights
from app.mrip.forecast.types import (
    QUANTILES, EvaluationReport, ForecastRequest, ForecastResult, ForecastUnavailable, Horizon,
)


def idx(n):
    return pd.bdate_range("2018-01-01", periods=n)


def gbm(seed, n=1500, mu=0.0003, sigma=0.01):
    r = np.random.default_rng(seed).normal(mu, sigma, n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx(n))


def request(prices, h=21, covariates=None):
    return ForecastRequest("TEST", prices, h, covariates)


def test_request_validation():
    p = gbm(1, 100)
    with pytest.raises(ValueError):
        request(p, h=0)
    with pytest.raises(ValueError):
        ForecastRequest("T", p * -1, 5)
    with pytest.raises(ValueError):
        ForecastRequest("T", p.iloc[::-1], 5)
    future = pd.DataFrame({"x": 1.0}, index=pd.bdate_range(p.index[-1], periods=3))
    with pytest.raises(ValueError, match="look-ahead"):
        ForecastRequest("T", p, 5, future)
    r = request(p)
    assert r.as_of == p.index[-1].date() and r.last_price == float(p.iloc[-1])
    assert [h.value for h in Horizon] == [5, 21, 63, 126, 252]


def test_naive_matches_the_closed_form_and_is_symmetric_in_log_space():
    p = gbm(2)
    res = NaiveBaseline(vol_window=60).forecast(request(p, h=21))
    sigma = np.std(np.diff(np.log(p.to_numpy()))[-60:], ddof=1) * np.sqrt(21)
    for q in QUANTILES:
        assert res.return_quantiles[q] == pytest.approx(np.expm1(stats.norm.ppf(q) * sigma), rel=1e-12)
    assert res.return_quantiles[0.5] == pytest.approx(0.0, abs=1e-12)
    s = res.scenarios
    assert s.bear < s.base < s.bull and s.version == "scenario-v0-uncalibrated"
    assert res.price_quantiles()[0.5] == pytest.approx(res.last_price)
    assert res.as_of == p.index[-1].date() and res.provider == "naive"


def test_naive_needs_enough_history_and_wider_horizon_means_wider_spread():
    with pytest.raises(ForecastUnavailable):
        NaiveBaseline().forecast(request(gbm(2, 30)))
    p = gbm(3)
    short, long = NaiveBaseline().forecast(request(p, 5)), NaiveBaseline().forecast(request(p, 63))
    assert long.return_quantiles[0.9] - long.return_quantiles[0.1] > short.return_quantiles[0.9] - short.return_quantiles[0.1]


def test_statistical_constant_growth_collapses_to_one_value():
    p = pd.Series(100 * 1.001 ** np.arange(600), index=idx(600))
    res = StatisticalBaseline().forecast(request(p, h=21))
    assert all(v == pytest.approx(1.001**21 - 1, rel=1e-9) for v in res.return_quantiles.values())
    assert res.n_obs == 600 - 21 and res.warnings  # overlap warning


def test_statistical_uses_only_the_lookback_window_and_enforces_minimum_samples():
    base = gbm(4, 1500)
    altered = base.copy()
    altered.iloc[:100] = altered.iloc[:100] * 3  # far outside the 500-day lookback
    a = StatisticalBaseline(lookback=500, min_samples=120).forecast(request(base, 21))
    b = StatisticalBaseline(lookback=500, min_samples=120).forecast(request(altered, 21))
    assert a.return_quantiles == b.return_quantiles
    with pytest.raises(ForecastUnavailable, match="overlapping"):
        StatisticalBaseline().forecast(request(gbm(4, 100), 21))
    with pytest.raises(ValueError):
        StatisticalBaseline(lookback=50, min_samples=100)


def predictive_case(seed=10, n=1500, coef=0.001, relevant=True):
    """Daily log returns driven by yesterday's value of a persistent AR(1) factor x (phi=0.97)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 0.3, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.97 * x[t - 1] + e[t]
    drift = coef * x if relevant else np.zeros(n)
    r = np.concatenate([[0.0], drift[:-1]]) + rng.normal(0, 0.005, n)
    prices = pd.Series(100 * np.exp(np.cumsum(r)), index=idx(n))
    return prices, pd.Series(x, index=idx(n)).to_frame("x")


def test_factor_regression_recovers_a_real_predictive_relationship():
    prices, cov = predictive_case()
    hi, lo = cov.copy(), cov.copy()
    hi.iloc[-1, 0], lo.iloc[-1, 0] = 2.0, -2.0  # same history, different factor reading at as_of
    res_hi = FactorRegression().forecast(request(prices, 21, hi))
    res_lo = FactorRegression().forecast(request(prices, 21, lo))
    assert res_hi.return_quantiles[0.5] > res_lo.return_quantiles[0.5] + 0.03  # true effect ~ 0.0156 per unit of x
    assert not any("weak fit" in w for w in res_hi.warnings)


def test_factor_regression_flags_an_irrelevant_factor_and_handles_missing_inputs():
    prices, cov = predictive_case(relevant=False, seed=11)
    res = FactorRegression().forecast(request(prices, 21, cov))
    assert any("weak fit" in w for w in res.warnings)
    with pytest.raises(ForecastUnavailable, match="needs covariates"):
        FactorRegression().forecast(request(prices, 21))
    with_gap = cov.copy()
    with_gap.iloc[-10:, 0] = np.nan  # beyond the 5-day forward-fill limit
    with pytest.raises(ForecastUnavailable, match="missing at as_of"):
        FactorRegression().forecast(request(prices, 21, with_gap))
    with pytest.raises(ForecastUnavailable, match="more usable rows"):
        FactorRegression().forecast(request(prices.iloc[:110], 21, cov.iloc[:110]))


def test_factor_regression_does_not_use_outcomes_not_yet_known():
    prices, cov = predictive_case(seed=12, n=800)
    a = FactorRegression().forecast(request(prices, 21, cov))
    assert a == FactorRegression().forecast(request(prices, 21, cov))  # deterministic
    # Targets need t + h <= as_of, so training rows = usable rows - h; fewer prices -> fewer rows.
    shorter = FactorRegression().forecast(request(prices.iloc[:-21], 21, cov.iloc[:-21]))
    assert shorter.as_of == prices.index[-22].date() and shorter.n_obs == a.n_obs - 21


class Fixed:
    """Stub provider with fixed return quantiles (monotone by construction)."""

    def __init__(self, name, shift, available=True):
        self.name, self._shift, self._available = name, shift, available

    def version(self):
        return f"{self.name}-v1"

    def forecast(self, req):
        if not self._available:
            raise ForecastUnavailable("nope")
        q = {k: self._shift + 0.1 * (k - 0.5) for k in QUANTILES}
        return ForecastResult(self.name, self.version(), req.symbol, req.as_of, req.horizon_days, req.last_price, q, 100, ("w",))


def test_ensemble_averages_quantiles_with_normalised_weights_and_keeps_order():
    p = gbm(5, 100)
    ens = EnsembleProvider([(Fixed("a", 0.0), 1.0), (Fixed("b", 0.10), 3.0)])
    res = ens.forecast(request(p))
    assert res.return_quantiles[0.5] == pytest.approx(0.075) and res.provider == "ensemble"
    vals = [res.return_quantiles[q] for q in QUANTILES]
    assert vals == sorted(vals)
    assert sum(ens.weights.values()) == pytest.approx(1.0) and ens.weights["b-v1"] == pytest.approx(0.75)
    assert "a-v1:0.2500" in ens.version() and "a: w" in res.warnings


def test_ensemble_skips_unavailable_members_renormalises_and_enforces_min_members():
    p = gbm(6, 100)
    res = EnsembleProvider([(Fixed("a", 0.0), 1.0), (Fixed("b", 0.2, available=False), 1.0)]).forecast(request(p))
    assert res.return_quantiles[0.5] == pytest.approx(0.0) and any("b-v1 unavailable" in w for w in res.warnings)
    with pytest.raises(ForecastUnavailable, match="only 1 of 2"):
        EnsembleProvider([(Fixed("a", 0.0), 1.0), (Fixed("b", 0.2, available=False), 1.0)], min_members=2).forecast(request(p))
    with pytest.raises(ForecastUnavailable):
        EnsembleProvider([(Fixed("a", 0.0, available=False), 1.0)]).forecast(request(p))


def test_ensemble_validation_and_inverse_loss_weights():
    with pytest.raises(ValueError):
        EnsembleProvider([])
    with pytest.raises(ValueError):
        EnsembleProvider([(Fixed("a", 0), 0.0)])
    with pytest.raises(ValueError):
        EnsembleProvider([(Fixed("a", 0), 1.0)], min_members=2)

    def report(loss):
        return EvaluationReport("x", "v", 21, 50, loss, 0.8, 0.5, 0.05, 0.5, date(2020, 1, 1), date(2021, 1, 1))

    w = inverse_loss_weights({"good": report(0.01), "bad": report(0.03)})
    assert w["good"] == pytest.approx(0.75) and w["bad"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        inverse_loss_weights({})
    with pytest.raises(ValueError):
        inverse_loss_weights({"z": report(0.0)})


def test_result_rejects_crossing_or_incomplete_quantiles():
    base = dict(provider="p", provider_version="v", symbol="T", as_of=date(2026, 1, 1), horizon_days=5, last_price=1.0, n_obs=1)
    with pytest.raises(ValueError, match="non-decreasing"):
        ForecastResult(return_quantiles={q: -q for q in QUANTILES}, **base)
    with pytest.raises(ValueError, match="standard quantile grid"):
        ForecastResult(return_quantiles={0.5: 0.0}, **base)


def test_baselines_are_calibrated_on_iid_returns():
    p = gbm(77, 3000, mu=0.0003, sigma=0.01)
    for provider in (NaiveBaseline(), StatisticalBaseline(lookback=1000)):
        report = provider.evaluate(p, 21, min_history=500, step=5)
        assert report.n_forecasts > 400
        assert 0.70 <= report.coverage_p10_p90 <= 0.90, (provider.name, report.coverage_p10_p90)
        assert 0.38 <= report.coverage_p25_p75 <= 0.62, (provider.name, report.coverage_p25_p75)
        assert "overlapping outcomes: step < horizon" in report.notes


def test_factor_regression_beats_the_naive_baseline_only_when_the_factor_is_real():
    prices, cov = predictive_case(seed=21, n=2200)
    naive = NaiveBaseline().evaluate(prices, 21, min_history=800, step=21)
    factor = FactorRegression().evaluate(prices, 21, min_history=800, step=21, covariates=cov)
    assert factor.mean_pinball_loss < 0.9 * naive.mean_pinball_loss

    noise_prices, noise_cov = predictive_case(seed=22, n=2200, relevant=False)
    naive_n = NaiveBaseline().evaluate(noise_prices, 21, min_history=800, step=21)
    factor_n = FactorRegression().evaluate(noise_prices, 21, min_history=800, step=21, covariates=noise_cov)
    assert factor_n.mean_pinball_loss > 0.93 * naive_n.mean_pinball_loss  # no real edge without a real factor


def test_evaluation_derived_ensemble_weights_favour_the_better_provider():
    prices, cov = predictive_case(seed=23, n=2200)
    members = {"naive": NaiveBaseline(), "factor": FactorRegression()}
    reports = {n: m.evaluate(prices, 21, min_history=800, step=21, covariates=cov) for n, m in members.items()}
    weights = inverse_loss_weights(reports)
    assert weights["factor"] > weights["naive"]
    ens = EnsembleProvider([(members[n], w) for n, w in weights.items()])
    report = ens.evaluate(prices, 21, min_history=800, step=21, covariates=cov)
    assert report.mean_pinball_loss <= reports["naive"].mean_pinball_loss
