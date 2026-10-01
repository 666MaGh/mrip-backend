"""Correlation, regression, cointegration and walk-forward on synthetic data with known structure."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from app.mrip.stats.cointegration import engle_granger
from app.mrip.stats.correlation import (
    cross_correlation, lead_lag, partial_correlation, pearson, rolling_correlation,
)
from app.mrip.stats.regression import beta, default_hac_lags, factor_exposure, ols
from app.mrip.stats.walkforward import lead_lag_walk_forward, sign_consistency, walk_forward


def idx(n):
    return pd.bdate_range("2020-01-01", periods=n)


def noise(seed, n, scale=1.0):
    return pd.Series(np.random.default_rng(seed).normal(0, scale, n), index=idx(n))


def test_pearson_matches_scipy():
    x, y = noise(1, 200), noise(2, 200)
    y = y + 0.3 * x
    res = pearson(x, y)
    ref = sps.pearsonr(x, y)
    assert res.n == 200
    assert res.r == pytest.approx(ref.statistic, abs=1e-12) and res.p_value == pytest.approx(ref.pvalue, rel=1e-9)


def test_pearson_constant_series_is_nan_and_small_sample_raises():
    const = pd.Series(1.0, index=idx(50))
    assert np.isnan(pearson(const, noise(1, 50)).r)
    with pytest.raises(ValueError):
        pearson(noise(1, 2), noise(2, 2))


def test_rolling_correlation_window_and_validation():
    x = noise(1, 100)
    rc = rolling_correlation(x, 2 * x, 20)
    assert rc.iloc[:19].isna().all() and rc.iloc[19:].round(10).eq(1.0).all()
    with pytest.raises(ValueError):
        rolling_correlation(x, x, 2)


def test_partial_correlation_removes_common_driver():
    z = noise(3, 600)
    x, y = z + noise(4, 600, 0.5), z + noise(5, 600, 0.5)
    raw = pearson(x, y)
    partial = partial_correlation(x, y, z.to_frame("z"))
    assert raw.r > 0.7 and raw.p_value < 1e-10
    assert abs(partial.r) < 0.15 and partial.p_value > 0.001


def test_partial_correlation_keeps_direct_link():
    z = noise(3, 600)
    x = z + noise(4, 600, 0.5)
    y = z + 0.8 * x + noise(5, 600, 0.5)
    assert partial_correlation(x, y, z.to_frame("z")).r > 0.5


def test_lead_lag_finds_x_leading_by_three():
    x = noise(10, 800)
    y = 0.6 * x.shift(3) + noise(11, 800, 0.8)
    res = lead_lag(x.iloc[3:], y.iloc[3:], 6)
    assert res.best_lag == 3 and res.direction == "x_leads" and res.best.r > 0.3
    assert res.corrected_p_value < 1e-6 and res.lags_tested == 13
    flipped = lead_lag(y.iloc[3:], x.iloc[3:], 6)
    assert flipped.best_lag == -3 and flipped.direction == "y_leads"


def test_lead_lag_bonferroni_blunts_a_chance_peak():
    res = lead_lag(noise(20, 120), noise(21, 120), 10)
    assert res.corrected_p_value == pytest.approx(min(1.0, res.best.p_value * res.lags_tested))
    assert res.corrected_p_value > 0.05  # independent noise should not look significant after correction


def test_cross_correlation_covers_all_lags_and_rejects_negative_max():
    table = cross_correlation(noise(1, 100), noise(2, 100), 4)
    assert sorted(table) == list(range(-4, 5))
    with pytest.raises(ValueError):
        cross_correlation(noise(1, 100), noise(2, 100), -1)


def test_ols_recovers_beta_with_hac():
    m = noise(30, 1000, 0.01)
    a = 0.0002 + 1.2 * m + noise(31, 1000, 0.005)
    res = beta(a, m)
    assert res.params["market"] == pytest.approx(1.2, abs=0.05)
    assert res.p_values["market"] < 1e-10 and res.r_squared > 0.7
    assert res.n == 1000 and res.hac_lags == default_hac_lags(1000) == 6  # floor(4 * 10 ** (2/9))


def test_factor_exposure_two_factors():
    f = pd.DataFrame({"mkt": noise(40, 1000, 0.01), "rates": noise(41, 1000, 0.01)})
    a = 0.9 * f["mkt"] - 0.5 * f["rates"] + noise(42, 1000, 0.003)
    res = factor_exposure(a, f)
    assert res.params["mkt"] == pytest.approx(0.9, abs=0.05) and res.params["rates"] == pytest.approx(-0.5, abs=0.05)
    with pytest.raises(ValueError):
        ols(a.iloc[:3], f.iloc[:3])


def random_walk(seed, n, drift=0.0):
    steps = np.random.default_rng(seed).normal(drift, 0.01, n)
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=idx(n))


def test_engle_granger_detects_cointegration_and_not_independent_walks():
    x = random_walk(50, 600)
    y_coint = np.exp(0.8 * np.log(x) + 0.5 + noise(51, 600, 0.01))
    res = engle_granger(y_coint, x)
    assert res.p_value < 0.01 and res.hedge_ratio == pytest.approx(0.8, abs=0.05)
    independent = engle_granger(random_walk(60, 600), random_walk(61, 600))
    assert independent.p_value > 0.05


def test_engle_granger_validates_inputs():
    x = random_walk(1, 100)
    with pytest.raises(ValueError):
        engle_granger(x * -1, x)
    with pytest.raises(ValueError):
        engle_granger(x.iloc[:30], x.iloc[:30])


def test_walk_forward_fold_geometry():
    x, y = noise(1, 100), noise(2, 100)
    folds = walk_forward(
        x, y, fit=lambda a, b: pearson(a, b).r, train_stat=lambda p: p,
        score=lambda p, a, b: pearson(a, b).r, train_size=40, test_size=20,
    )
    assert len(folds) == 3
    assert [f.n_train for f in folds] == [40, 40, 40] and [f.n_test for f in folds] == [20, 20, 20]
    assert folds[0].test_start == x.index[40] and folds[1].train_start == x.index[20]
    with pytest.raises(ValueError):
        walk_forward(x, y, lambda a, b: 0, lambda p: 0, lambda p, a, b: 0, train_size=5, test_size=5)


def test_walk_forward_has_no_lookahead():
    x, y = noise(1, 200), noise(2, 200)
    kwargs = dict(fit=lambda a, b: pearson(a, b).r, train_stat=lambda p: p,
                  score=lambda p, a, b: pearson(a, b).r, train_size=60, test_size=30)
    base = walk_forward(x, y, **kwargs)
    # Rewriting data after the first fold's test window must not change that fold.
    y2 = y.copy()
    y2.iloc[100:] = noise(99, 100)
    changed = walk_forward(x, y2, **kwargs)
    assert changed[0] == base[0]
    assert any(c != b for c, b in zip(changed[1:], base[1:]))


def test_lead_lag_walk_forward_is_consistent_for_a_stable_relationship_only():
    x = noise(70, 900)
    stable = 0.6 * x.shift(2) + noise(71, 900, 0.8)
    folds = lead_lag_walk_forward(x.iloc[2:], stable.iloc[2:], 5, train_size=150, test_size=50)
    assert len(folds) >= 10 and sign_consistency(folds) >= 0.9
    # Relationship flips sign halfway: out-of-sample consistency must drop.
    flipped = stable.copy()
    flipped.iloc[450:] = (-0.6 * x.shift(2).iloc[450:]).to_numpy() + noise(72, 450, 0.8).to_numpy()
    mixed = lead_lag_walk_forward(x.iloc[2:], flipped.iloc[2:], 5, train_size=150, test_size=50)
    assert sign_consistency(mixed) < sign_consistency(folds)


def test_sign_consistency_edge_cases():
    assert sign_consistency([]) is None
