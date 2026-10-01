"""Stats transforms: known-answer and no-look-ahead checks."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.mrip.stats.transforms import (
    align,
    log_returns,
    relative_strength,
    volatility_normalize,
    zscore,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def test_log_returns_known_answer() -> None:
    prices = pd.Series([100.0, 110.0, 121.0], index=_dates(3))
    out = log_returns(prices)
    assert len(out) == 2
    assert out.index.equals(prices.index[1:])
    np.testing.assert_allclose(out.to_numpy(), [math.log(1.1), math.log(1.1)])


def test_log_returns_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError):
        log_returns(pd.Series([100.0, 0.0, 90.0], index=_dates(3)))
    with pytest.raises(ValueError):
        log_returns(pd.Series([100.0, -1.0, 90.0], index=_dates(3)))


def test_log_returns_rejects_unsorted_index() -> None:
    idx = _dates(3)[[0, 2, 1]]
    with pytest.raises(ValueError):
        log_returns(pd.Series([100.0, 101.0, 102.0], index=idx))


def test_align_inner_join_drops_nan_and_sorts() -> None:
    idx = _dates(6)
    a = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], index=idx)
    b = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx[[5, 3, 2, 1, 0]])  # unsorted, no idx[4]
    out_a, out_b = align(a, b)
    assert list(out_a.index) == [idx[0], idx[1], idx[3], idx[5]]
    assert out_a.index.equals(out_b.index)
    assert out_a.tolist() == [1.0, 2.0, 4.0, 6.0]
    assert out_b.tolist() == [50.0, 40.0, 20.0, 10.0]


def test_align_three_series_keeps_order() -> None:
    idx = _dates(4)
    s1 = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    s2 = pd.Series([5.0, 6.0, 7.0, 8.0], index=idx)
    s3 = pd.Series([9.0, np.nan, 11.0, 12.0], index=idx)
    out = align(s1, s2, s3)
    assert len(out) == 3
    assert out[0].tolist() == [1.0, 3.0, 4.0]
    assert out[1].tolist() == [5.0, 7.0, 8.0]
    assert out[2].tolist() == [9.0, 11.0, 12.0]


def test_align_needs_two_series() -> None:
    with pytest.raises(ValueError):
        align(pd.Series([1.0]))
    with pytest.raises(ValueError):
        align()


def test_zscore_whole_sample_has_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(1)
    s = pd.Series(rng.normal(5.0, 3.0, 200), index=_dates(200))
    z = zscore(s)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-12)


def test_zscore_whole_sample_known_answer() -> None:
    z = zscore(pd.Series([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(z.to_numpy(), [-1.0, 0.0, 1.0])


def test_zscore_constant_series_is_nan_not_inf() -> None:
    z = zscore(pd.Series([2.5] * 10))
    assert z.isna().all()
    zw = zscore(pd.Series([2.5] * 10), window=4)
    assert zw.isna().all()


def test_zscore_trailing_window_uses_only_window() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0, 6.0]
    s = pd.Series(values, index=_dates(6))
    z = zscore(s, window=3)
    assert z.iloc[:2].isna().all()
    for t in range(2, 6):
        w = np.array(values[t - 2 : t + 1])
        expected = (w[-1] - w.mean()) / w.std(ddof=1)
        assert z.iloc[t] == pytest.approx(expected)
    # First full window [1, 2, 3] -> exactly 1.0 for the current value 3.
    assert z.iloc[2] == pytest.approx(1.0)


def test_zscore_window_is_causal() -> None:
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(size=60), index=_dates(60))
    base = zscore(s, window=10)
    mutated = s.copy()
    mutated.iloc[40:] += 50.0
    changed = zscore(mutated, window=10)
    pd.testing.assert_series_equal(base.iloc[:40], changed.iloc[:40])


def test_volatility_normalize_known_answer() -> None:
    values = [0.01, -0.02, 0.03, 0.04, -0.01, 0.02]
    r = pd.Series(values, index=_dates(6))
    out = volatility_normalize(r, window=3)
    assert out.iloc[:3].isna().all()
    for t in range(3, 6):
        prev = np.array(values[t - 3 : t])
        assert out.iloc[t] == pytest.approx(values[t] / prev.std(ddof=1))


def test_volatility_normalize_has_no_lookahead() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 100), index=_dates(100))
    base = volatility_normalize(r, window=20)
    t = 60
    mutated = r.copy()
    mutated.iloc[t + 1 :] = rng.normal(0, 0.5, 100 - t - 1)
    changed = volatility_normalize(mutated, window=20)
    pd.testing.assert_series_equal(base.iloc[: t + 1], changed.iloc[: t + 1])
    # The value at t itself must not depend on its own history after t either.
    assert base.iloc[t] == changed.iloc[t]


def test_volatility_normalize_excludes_current_observation_from_std() -> None:
    r = pd.Series([0.01, 0.02, 0.03, 0.5], index=_dates(4))
    out = volatility_normalize(r, window=3)
    assert out.iloc[3] == pytest.approx(0.5 / np.std([0.01, 0.02, 0.03], ddof=1))


def test_volatility_normalize_zero_std_is_nan() -> None:
    r = pd.Series([0.01] * 6 + [0.02], index=_dates(7))
    out = volatility_normalize(r, window=5)
    assert out.isna().all()


def test_relative_strength_starts_at_one_and_known_ratio() -> None:
    idx = _dates(4)
    a = pd.Series([50.0, 55.0, 60.0, 45.0], index=idx)
    b = pd.Series([10.0, 10.0, 20.0, 15.0], index=idx)
    rs = relative_strength(a, b)
    assert rs.iloc[0] == 1.0
    np.testing.assert_allclose(rs.to_numpy(), [1.0, 1.1, 0.6, 0.6])


def test_relative_strength_uses_common_index_only() -> None:
    idx = _dates(5)
    a = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
    b = pd.Series([1.0, 2.0, 3.0], index=idx[2:])
    rs = relative_strength(a, b)
    assert list(rs.index) == list(idx[2:])
    assert rs.iloc[0] == 1.0
    np.testing.assert_allclose(rs.to_numpy(), [1.0, (40 / 2) / (30 / 1), (50 / 3) / (30 / 1)])
