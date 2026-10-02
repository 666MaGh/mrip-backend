"""Regime features: known-answer, numpy cross-checks and no-look-ahead checks."""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from app.mrip.regime.features import (
    Direction,
    Trend,
    change,
    direction_state,
    drawdown,
    implied_to_realized,
    pct_change,
    realized_volatility,
    term_ratio,
    trailing_percentile,
    trend_state,
)

NAN = float("nan")


def _dates(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="B")


def _series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=_dates(len(values), start), dtype=float)


def _random_prices(n: int, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))), index=_dates(n))


def _random_levels(n: int, seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.uniform(5.0, 40.0, n), index=_dates(n))


def _labels(series: pd.Series) -> list[object]:
    return [None if (isinstance(v, float) and math.isnan(v)) else v for v in series]


# --- no-look-ahead helper ----------------------------------------------------------


def _assert_no_look_ahead(fn: Callable[..., pd.Series], *inputs: pd.Series) -> None:
    """Past outputs must not change when the series is truncated or its future altered."""
    full = fn(*inputs)
    k = len(inputs[0]) // 2
    truncated = fn(*[s.iloc[:k] for s in inputs])
    pd.testing.assert_series_equal(full.iloc[:k], truncated, check_exact=True)

    altered_inputs = []
    for s in inputs:
        altered = s.copy()
        altered.iloc[k:] = altered.iloc[k:] * 3.0 + 7.0
        altered_inputs.append(altered)
    altered_out = fn(*altered_inputs)
    pd.testing.assert_series_equal(full.iloc[:k], altered_out.iloc[:k], check_exact=True)


def test_helper_detects_look_ahead() -> None:
    def centered_mean(s: pd.Series) -> pd.Series:
        return s.rolling(5, center=True, min_periods=1).mean()

    with pytest.raises(AssertionError):
        _assert_no_look_ahead(centered_mean, _random_levels(40))


# --- trailing_percentile -----------------------------------------------------------


def test_trailing_percentile_known_answer_with_warmup() -> None:
    out = trailing_percentile(_series([3, 1, 2, 2, 5]), 3)
    assert out.iloc[:2].isna().all()
    # windows [3,1,2] -> 2/3, [1,2,2] -> 1, [2,2,5] -> 1
    assert out.iloc[2] == pytest.approx(2 / 3)
    assert out.iloc[3] == pytest.approx(1.0)
    assert out.iloc[4] == pytest.approx(1.0)


def test_trailing_percentile_lowest_value_is_one_over_window() -> None:
    out = trailing_percentile(_series([5, 4, 3]), 3)
    assert out.iloc[2] == pytest.approx(1 / 3)


def test_trailing_percentile_ties_count_as_less_or_equal() -> None:
    out = trailing_percentile(_series([2, 2, 2, 2]), 3)
    assert out.iloc[2] == 1.0
    assert out.iloc[3] == 1.0


def test_trailing_percentile_window_one_is_always_one() -> None:
    out = trailing_percentile(_series([4, 1, 9]), 1)
    assert out.tolist() == [1.0, 1.0, 1.0]


def test_trailing_percentile_matches_numpy_and_stays_in_unit_interval() -> None:
    s = _random_levels(60)
    window = 10
    out = trailing_percentile(s, window)
    values = s.to_numpy()
    for t in range(window - 1, len(s)):
        chunk = values[t - window + 1 : t + 1]
        assert out.iloc[t] == pytest.approx(np.sum(chunk <= values[t]) / window)
    valid = out.dropna()
    assert ((valid > 0) & (valid <= 1)).all()
    assert out.iloc[: window - 1].isna().all()


def test_trailing_percentile_nan_in_window_gives_nan() -> None:
    out = trailing_percentile(_series([1, NAN, 3, 4]), 2)
    assert math.isnan(out.iloc[1])
    assert math.isnan(out.iloc[2])
    assert out.iloc[3] == 1.0


def test_trailing_percentile_no_look_ahead() -> None:
    _assert_no_look_ahead(lambda s: trailing_percentile(s, 8), _random_levels(50))


# --- change / pct_change -----------------------------------------------------------


def test_change_known_answer() -> None:
    out = change(_series([1, 3, 6, 10]), 2)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [5.0, 7.0]


def test_change_propagates_nan() -> None:
    out = change(_series([1, NAN, 6, 10]), 1)
    assert out.isna().tolist() == [True, True, True, False]
    assert out.iloc[3] == 4.0


def test_pct_change_known_answer() -> None:
    out = pct_change(_series([2, 4, 0, 3, 6]), 1)
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(1.0)
    assert out.iloc[2] == pytest.approx(-1.0)
    assert math.isnan(out.iloc[3])  # earlier value is 0
    assert out.iloc[4] == pytest.approx(1.0)


def test_pct_change_two_periods_and_zero_denominator() -> None:
    out = pct_change(_series([2, 4, 0, 3, 6]), 2)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(-1.0)
    assert out.iloc[3] == pytest.approx(-0.25)
    assert math.isnan(out.iloc[4])  # earlier value is 0


def test_pct_change_nan_denominator_is_nan() -> None:
    out = pct_change(_series([NAN, 5, 7]), 1)
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(0.4)


def test_change_and_pct_change_match_numpy() -> None:
    s = _random_levels(40)
    values = s.to_numpy()
    np.testing.assert_allclose(change(s, 5).to_numpy()[5:], values[5:] - values[:-5])
    np.testing.assert_allclose(pct_change(s, 5).to_numpy()[5:], values[5:] / values[:-5] - 1)


def test_change_no_look_ahead() -> None:
    _assert_no_look_ahead(lambda s: change(s, 4), _random_levels(50))


def test_pct_change_no_look_ahead() -> None:
    _assert_no_look_ahead(lambda s: pct_change(s, 4), _random_levels(50))


# --- realized_volatility -----------------------------------------------------------


def test_realized_volatility_known_answer() -> None:
    prices = _series([1.0, math.e, math.e**3, math.e**4])  # log returns 1, 2, 1
    out = realized_volatility(prices, window=2)
    assert out.iloc[:2].isna().all()
    expected = math.sqrt(0.5) * math.sqrt(252)
    assert out.iloc[2] == pytest.approx(expected)
    assert out.iloc[3] == pytest.approx(expected)


def test_realized_volatility_matches_numpy() -> None:
    prices = _random_prices(80)
    window = 20
    out = realized_volatility(prices, window)
    log_ret = np.diff(np.log(prices.to_numpy()))
    assert out.iloc[:window].isna().all()
    for t in range(window, len(prices)):
        chunk = log_ret[t - window : t]
        assert out.iloc[t] == pytest.approx(np.std(chunk, ddof=1) * np.sqrt(252), rel=1e-9)


def test_realized_volatility_constant_series_is_zero() -> None:
    out = realized_volatility(_series([50.0] * 30), window=20)
    assert out.iloc[:20].isna().all()
    assert (out.iloc[20:] == 0.0).all()


def test_realized_volatility_default_window_is_twenty() -> None:
    out = realized_volatility(_random_prices(40))
    assert out.iloc[:20].isna().all()
    assert out.iloc[20:].notna().all()


def test_realized_volatility_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError):
        realized_volatility(_series([10.0, 0.0, 11.0, 12.0]), window=2)
    with pytest.raises(ValueError):
        realized_volatility(_series([10.0, -1.0, 11.0, 12.0]), window=2)


def test_realized_volatility_no_look_ahead() -> None:
    _assert_no_look_ahead(lambda s: realized_volatility(s, 10), _random_prices(60))


# --- implied_to_realized -----------------------------------------------------------


def test_implied_to_realized_known_answer_in_vol_points() -> None:
    implied = _series([18.5, 20.0, 30.0, 25.0])
    realized = _series([0.185, 0.0, NAN, 0.5])
    out = implied_to_realized(implied, realized)
    assert out.iloc[0] == pytest.approx(1.0)
    assert math.isnan(out.iloc[1])  # realized 0
    assert math.isnan(out.iloc[2])  # realized NaN
    assert out.iloc[3] == pytest.approx(0.5)


def test_implied_to_realized_uses_common_index() -> None:
    implied = _series([10.0, 20.0, 30.0, 40.0])
    realized = pd.Series([0.1, 0.4], index=implied.index[1:3])
    out = implied_to_realized(implied, realized)
    assert out.index.equals(implied.index[1:3])
    assert out.tolist() == pytest.approx([2.0, 0.75])


def test_implied_to_realized_nan_implied_stays_nan() -> None:
    out = implied_to_realized(_series([NAN, 20.0]), _series([0.2, 0.2]))
    assert math.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(1.0)


def test_implied_to_realized_no_look_ahead() -> None:
    _assert_no_look_ahead(
        implied_to_realized,
        _random_levels(50),
        _random_levels(50, seed=3) / 100.0,
    )


# --- term_ratio --------------------------------------------------------------------


def test_term_ratio_known_answer_with_zero_and_nan_back() -> None:
    front = _series([2.0, 4.0, 6.0, 8.0])
    back = _series([1.0, 0.0, NAN, 4.0])
    out = term_ratio(front, back)
    assert out.iloc[0] == 2.0
    assert math.isnan(out.iloc[1])
    assert math.isnan(out.iloc[2])
    assert out.iloc[3] == 2.0


def test_term_ratio_uses_common_index() -> None:
    front = _series([2.0, 4.0, 6.0])
    back = pd.Series([2.0, 3.0, 5.0], index=front.index[1:].append(_dates(1, "2030-01-01")))
    out = term_ratio(front, back)
    assert out.index.equals(front.index[1:])
    assert out.tolist() == pytest.approx([2.0, 6.0 / 3.0])


def test_term_ratio_matches_numpy() -> None:
    front, back = _random_levels(30), _random_levels(30, seed=5)
    np.testing.assert_allclose(term_ratio(front, back).to_numpy(), front.to_numpy() / back.to_numpy())


def test_term_ratio_no_look_ahead() -> None:
    _assert_no_look_ahead(term_ratio, _random_levels(50), _random_levels(50, seed=5))


# --- trend_state -------------------------------------------------------------------


def test_trend_state_rising_series_is_uptrend_after_warmup() -> None:
    out = trend_state(_series([float(i) for i in range(1, 11)]), fast=3, slow=5)
    assert out.iloc[:4].isna().all()
    assert all(v is Trend.UPTREND for v in out.iloc[4:])


def test_trend_state_falling_series_is_downtrend_after_warmup() -> None:
    out = trend_state(_series([float(i) for i in range(10, 0, -1)]), fast=3, slow=5)
    assert out.iloc[:4].isna().all()
    assert all(v is Trend.DOWNTREND for v in out.iloc[4:])


def test_trend_state_v_shape_known_answer() -> None:
    prices = _series([10, 8, 6, 4, 2, 4, 6, 8, 10, 12])
    out = trend_state(prices, fast=3, slow=6)
    assert _labels(out) == [
        None, None, None, None, None,
        Trend.DOWNTREND, Trend.MIXED, Trend.UPTREND, Trend.UPTREND, Trend.UPTREND,
    ]


def test_trend_state_close_equal_to_slow_average_is_mixed() -> None:
    # close == SMA(slow) at t=5 for fast=2, slow=4 (see hand calculation in the V series)
    prices = _series([10, 8, 6, 4, 2, 4, 6, 8, 10, 12])
    out = trend_state(prices, fast=2, slow=4)
    assert out.iloc[4] is Trend.DOWNTREND
    assert out.iloc[5] is Trend.MIXED
    assert out.iloc[6] is Trend.UPTREND


def test_trend_state_matches_numpy_on_random_series() -> None:
    prices = _random_prices(120)
    fast, slow = 10, 30
    out = trend_state(prices, fast=fast, slow=slow)
    values = prices.to_numpy()
    assert out.iloc[: slow - 1].isna().all()
    for t in range(slow - 1, len(values)):
        sma_fast = values[t - fast + 1 : t + 1].mean()
        sma_slow = values[t - slow + 1 : t + 1].mean()
        if values[t] > sma_slow and sma_fast > sma_slow:
            expected = Trend.UPTREND
        elif values[t] < sma_slow and sma_fast < sma_slow:
            expected = Trend.DOWNTREND
        else:
            expected = Trend.MIXED
        assert out.iloc[t] is expected


def test_trend_state_labels_are_strings() -> None:
    out = trend_state(_series([float(i) for i in range(1, 8)]), fast=2, slow=4)
    assert out.dtype == object
    assert out.iloc[-1] == "UPTREND"


def test_trend_state_requires_fast_below_slow() -> None:
    prices = _series([1.0] * 10)
    with pytest.raises(ValueError):
        trend_state(prices, fast=5, slow=5)
    with pytest.raises(ValueError):
        trend_state(prices, fast=6, slow=5)


def test_trend_state_no_look_ahead() -> None:
    _assert_no_look_ahead(lambda s: trend_state(s, fast=5, slow=15), _random_prices(80))


# --- direction_state ---------------------------------------------------------------


def test_direction_state_relative_boundaries() -> None:
    s = _series([100, 150, 300, 100, 40, 60, 30])
    out = direction_state(s, periods=1, band=0.5, mode="relative")
    assert _labels(out) == [
        None,
        Direction.FLAT,     # +0.5 exactly at the band
        Direction.RISING,   # +1.0
        Direction.FALLING,  # -0.667
        Direction.FALLING,  # -0.6
        Direction.FLAT,     # +0.5 exactly at the band
        Direction.FLAT,     # -0.5 exactly at the band
    ]


def test_direction_state_absolute_boundaries() -> None:
    s = _series([10, 12, 11, 10, 8, 9, 13])
    out = direction_state(s, periods=2, band=1.0, mode="absolute")
    assert _labels(out) == [
        None,
        None,
        Direction.FLAT,     # +1 exactly at the band
        Direction.FALLING,  # -2
        Direction.FALLING,  # -3
        Direction.FLAT,     # -1 exactly at the band
        Direction.RISING,   # +5
    ]


def test_direction_state_zero_band_is_strict() -> None:
    out = direction_state(_series([5, 5, 6, 4]), periods=1, band=0.0, mode="absolute")
    assert _labels(out) == [None, Direction.FLAT, Direction.RISING, Direction.FALLING]


def test_direction_state_default_mode_is_relative() -> None:
    s = _series([100, 150, 90])
    default = direction_state(s, periods=1, band=0.2)
    explicit = direction_state(s, periods=1, band=0.2, mode="relative")
    assert _labels(default) == _labels(explicit) == [None, Direction.RISING, Direction.FALLING]


def test_direction_state_relative_zero_base_is_nan() -> None:
    out = direction_state(_series([0, 5, 6]), periods=1, band=0.1)
    assert _labels(out)[:2] == [None, None]
    assert out.iloc[2] is Direction.RISING


def test_direction_state_warmup_length_and_dtype() -> None:
    out = direction_state(_series([float(i + 1) for i in range(10)]), periods=4, band=0.0)
    assert out.dtype == object
    assert out.iloc[:4].isna().all()
    assert out.iloc[4:].notna().all()
    assert out.iloc[-1] == "RISING"


def test_direction_state_rejects_bad_arguments() -> None:
    s = _series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        direction_state(s, periods=1, mode="log")
    with pytest.raises(ValueError):
        direction_state(s, periods=1, band=-0.01)
    with pytest.raises(ValueError):
        direction_state(s, periods=0)


@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_direction_state_no_look_ahead(mode: str) -> None:
    _assert_no_look_ahead(
        lambda s: direction_state(s, periods=5, band=0.05 if mode == "relative" else 1.0, mode=mode),
        _random_levels(60),
    )


# --- drawdown ----------------------------------------------------------------------


def test_drawdown_hand_built_path() -> None:
    out = drawdown(_series([100, 110, 99, 120, 90, 95]), window=3)
    expected = [0.0, 0.0, 99 / 110 - 1, 0.0, 90 / 120 - 1, 95 / 120 - 1]
    assert out.tolist() == pytest.approx(expected)


def test_drawdown_window_limits_the_peak() -> None:
    prices = _series([100, 110, 99, 120, 90, 95])
    narrow = drawdown(prices, window=2)
    wide = drawdown(prices, window=10)
    assert narrow.iloc[5] == pytest.approx(0.0)  # peak of [90, 95] is 95
    assert wide.iloc[5] == pytest.approx(95 / 120 - 1)
    assert drawdown(prices, window=1).tolist() == [0.0] * 6


def test_drawdown_has_no_warmup_nan_and_is_non_positive() -> None:
    out = drawdown(_random_prices(100), window=30)
    assert out.notna().all()
    assert (out <= 0).all()
    assert out.iloc[0] == 0.0


def test_drawdown_matches_numpy() -> None:
    prices = _random_prices(100)
    window = 25
    values = prices.to_numpy()
    out = drawdown(prices, window)
    for t in range(len(values)):
        peak = values[max(0, t - window + 1) : t + 1].max()
        assert out.iloc[t] == pytest.approx(values[t] / peak - 1)


def test_drawdown_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError):
        drawdown(_series([10.0, 0.0, 5.0]))


def test_drawdown_no_look_ahead() -> None:
    _assert_no_look_ahead(lambda s: drawdown(s, 20), _random_prices(80))


# --- argument and index validation -------------------------------------------------


def _unsorted(values: list[float]) -> pd.Series:
    idx = _dates(len(values))
    return pd.Series(values, index=idx[::-1])


def _duplicated(values: list[float]) -> pd.Series:
    idx = _dates(len(values)).tolist()
    idx[1] = idx[0]
    return pd.Series(values, index=pd.DatetimeIndex(idx))


def _range_indexed(values: list[float]) -> pd.Series:
    return pd.Series(values)


_VALS = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]

_SINGLE: dict[str, Callable[[pd.Series], pd.Series]] = {
    "trailing_percentile": lambda s: trailing_percentile(s, 2),
    "change": lambda s: change(s, 1),
    "pct_change": lambda s: pct_change(s, 1),
    "realized_volatility": lambda s: realized_volatility(s, 2),
    "trend_state": lambda s: trend_state(s, 2, 3),
    "direction_state": lambda s: direction_state(s, 1),
    "drawdown": lambda s: drawdown(s, 2),
}


@pytest.mark.parametrize("name", list(_SINGLE))
@pytest.mark.parametrize("make_bad", [_unsorted, _duplicated, _range_indexed])
def test_single_series_functions_reject_bad_index(
    name: str, make_bad: Callable[[list[float]], pd.Series]
) -> None:
    fn = _SINGLE[name]
    fn(_series(_VALS))  # the good index works
    with pytest.raises(ValueError):
        fn(make_bad(_VALS))


@pytest.mark.parametrize("make_bad", [_unsorted, _duplicated, _range_indexed])
def test_two_series_functions_reject_bad_index_on_either_side(
    make_bad: Callable[[list[float]], pd.Series]
) -> None:
    good, bad = _series(_VALS), make_bad(_VALS)
    for fn in (implied_to_realized, term_ratio):
        with pytest.raises(ValueError):
            fn(bad, good)
        with pytest.raises(ValueError):
            fn(good, bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_window_and_period_arguments_below_one_raise(bad: int) -> None:
    s = _series(_VALS)
    with pytest.raises(ValueError):
        trailing_percentile(s, bad)
    with pytest.raises(ValueError):
        change(s, bad)
    with pytest.raises(ValueError):
        pct_change(s, bad)
    with pytest.raises(ValueError):
        realized_volatility(s, bad)
    with pytest.raises(ValueError):
        direction_state(s, periods=bad)
    with pytest.raises(ValueError):
        drawdown(s, bad)
    with pytest.raises(ValueError):
        trend_state(s, fast=bad, slow=3)


def test_realized_volatility_window_one_raises_because_std_needs_two() -> None:
    with pytest.raises(ValueError):
        realized_volatility(_series(_VALS), window=1)
