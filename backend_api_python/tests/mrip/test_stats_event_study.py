"""Market-model event study: known answers, skipping, no look-ahead."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.mrip.stats.event_study import EventStudyResult, event_study

N = 1200
ALPHA = 0.0002
BETA = 1.2
# Spaced > estimation_window + gap + pre + post apart so earlier jumps never enter a later fit.
EVENT_POSITIONS = [200 + 150 * k for k in range(7)]


def _market(seed: int = 42, n: int = N) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.01, n), index=pd.bdate_range("2018-01-01", periods=n))


def _stock(market: pd.Series, noise_sd: float, seed: int = 7, jump: float = 0.03,
           positions: list[int] | None = None) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = ALPHA + BETA * market + pd.Series(rng.normal(0.0, noise_sd, len(market)), index=market.index)
    for p in positions if positions is not None else EVENT_POSITIONS:
        r.iloc[p] += jump
    return r


def _dates(market: pd.Series, positions: list[int]) -> list[pd.Timestamp]:
    return [market.index[p] for p in positions]


def test_noisy_data_recovers_injected_abnormal_return() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002)
    res = event_study(r, m, _dates(m, EVENT_POSITIONS))
    assert isinstance(res, EventStudyResult)
    assert res.n_events == len(EVENT_POSITIONS)
    assert res.n_skipped == 0
    assert res.relative_days == tuple(range(-5, 6))
    day0 = res.relative_days.index(0)
    assert res.aar[day0] == pytest.approx(0.03, abs=0.003)
    for i, aar in enumerate(res.aar):
        if i != day0:
            assert abs(aar) < 0.003
    assert res.car_mean == pytest.approx(0.03, abs=0.006)
    assert res.car_mean > 0
    assert res.car_tstat is not None and res.car_tstat > 5.0
    assert len(res.per_event_car) == res.n_events
    assert res.caar[-1] == pytest.approx(res.car_mean)
    assert res.caar[-1] == pytest.approx(sum(res.aar))


def test_noise_free_model_gives_exact_known_answer() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.0)
    res = event_study(r, m, _dates(m, EVENT_POSITIONS))
    day0 = res.relative_days.index(0)
    np.testing.assert_allclose(res.aar[day0], 0.03, atol=1e-10)
    for i, aar in enumerate(res.aar):
        if i != day0:
            assert abs(aar) < 1e-10
    np.testing.assert_allclose(res.per_event_car, 0.03, atol=1e-10)
    np.testing.assert_allclose(res.car_mean, 0.03, atol=1e-10)
    # All CARs identical -> zero cross-sectional std -> no t-stat.
    assert res.car_tstat is None
    np.testing.assert_allclose(res.caar[: day0], 0.0, atol=1e-10)
    np.testing.assert_allclose(res.caar[day0:], 0.03, atol=1e-10)


def test_no_event_effect_gives_small_car_and_small_tstat() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002, jump=0.0)
    res = event_study(r, m, _dates(m, EVENT_POSITIONS))
    assert abs(res.car_mean) < 0.005
    assert res.car_tstat is not None and abs(res.car_tstat) < 4.0


def test_event_date_on_non_trading_day_uses_next_label() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.0, positions=[300])
    label = m.index[300]
    # Day before is a Sunday/Saturday-safe offset: use 1 calendar day before the label,
    # which still lies after label 299 only if 299 is >= 1 day earlier (always true).
    before = label - pd.Timedelta(hours=12)
    res = event_study(r, m, [before])
    day0 = res.relative_days.index(0)
    assert res.aar[day0] == pytest.approx(0.03, abs=1e-10)


def test_string_dates_are_accepted() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.0, positions=[300])
    res = event_study(r, m, [str(m.index[300].date())])
    assert res.n_events == 1
    assert res.car_tstat is None  # single event


def test_events_too_close_to_start_or_end_are_skipped() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002)
    # Needs start - gap - est >= 0 -> t0 >= 5 + 5 + 120 = 130 for defaults.
    positions = [10, 129, 130, 500, N - 6, N - 5, N - 1]
    res = event_study(r, m, _dates(m, positions))
    # usable: 130, 500, N-6 (end = N-1 fits); skipped: 10, 129, N-5, N-1
    assert res.n_events == 3
    assert res.n_skipped == 4


def test_event_after_last_date_is_skipped() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002)
    res = event_study(r, m, [m.index[500], m.index[-1] + pd.Timedelta(days=30)])
    assert res.n_events == 1
    assert res.n_skipped == 1


def test_all_events_skipped_raises() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002)
    with pytest.raises(ValueError, match="no usable events"):
        event_study(r, m, _dates(m, [3, 50]))
    with pytest.raises(ValueError, match="no usable events"):
        event_study(r, m, [])


def test_results_unchanged_when_data_after_event_window_changes() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002, positions=[400])
    event = [m.index[400]]
    base = event_study(r, m, event)
    r2 = r.copy()
    m2 = m.copy()
    rng = np.random.default_rng(99)
    r2.iloc[406:] = rng.normal(0.0, 0.2, N - 406)  # strictly after t0+post = 405
    m2.iloc[406:] = rng.normal(0.0, 0.2, N - 406)
    assert event_study(r2, m2, event) == base


def test_gap_region_does_not_influence_results() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002, positions=[400])
    base = event_study(r, m, event_dates=[m.index[400]])
    # event window = 395..405, gap = 5 -> estimation ends before position 390;
    # positions 390..394 are the gap and must be ignored.
    r2 = r.copy()
    r2.iloc[390:395] += 1.0
    assert event_study(r2, m, [m.index[400]]) == base


def test_event_window_changes_move_abnormal_return_one_for_one() -> None:
    """Event-window data is not used to fit alpha/beta."""
    m = _market()
    r = _stock(m, noise_sd=0.002, positions=[400])
    base = event_study(r, m, [m.index[400]])
    r2 = r.copy()
    r2.iloc[400] += 0.05  # change inside the event window only
    changed = event_study(r2, m, [m.index[400]])
    day0 = base.relative_days.index(0)
    assert changed.aar[day0] - base.aar[day0] == pytest.approx(0.05, abs=1e-12)
    for i, (a, b) in enumerate(zip(base.aar, changed.aar)):
        if i != day0:
            assert a == pytest.approx(b, abs=1e-12)


def test_estimation_window_data_does_change_the_fit() -> None:
    """Sanity: the estimation window is actually used."""
    m = _market()
    r = _stock(m, noise_sd=0.002, positions=[400])
    base = event_study(r, m, [m.index[400]])
    r2 = r.copy()
    r2.iloc[300:380] += 0.05  # inside the estimation window (270..389)
    assert event_study(r2, m, [m.index[400]]).aar != base.aar


def test_custom_window_sizes() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.0, positions=[300])
    res = event_study(r, m, [m.index[300]], pre=2, post=10, estimation_window=60, gap=0)
    assert res.relative_days == tuple(range(-2, 11))
    assert len(res.aar) == len(res.caar) == 13
    assert res.aar[res.relative_days.index(0)] == pytest.approx(0.03, abs=1e-10)
    # gap=0, est=60, pre=2 -> earliest usable t0 = 62
    res2 = event_study(r, m, [m.index[62], m.index[61]], pre=2, post=10, estimation_window=60, gap=0)
    assert res2.n_events == 1 and res2.n_skipped == 1


def test_misaligned_inputs_are_aligned_on_common_index() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.0, positions=[500])
    res = event_study(r.iloc[100:], m.iloc[:-100], [m.index[500]])
    assert res.aar[res.relative_days.index(0)] == pytest.approx(0.03, abs=1e-10)


def test_tstat_formula() -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002)
    res = event_study(r, m, _dates(m, EVENT_POSITIONS))
    cars = np.array(res.per_event_car)
    expected = cars.mean() / (cars.std(ddof=1) / np.sqrt(len(cars)))
    assert res.car_tstat == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [{"pre": -1}, {"post": -1}, {"estimation_window": 29}, {"gap": -1}],
)
def test_invalid_arguments_raise(kwargs: dict[str, int]) -> None:
    m = _market()
    r = _stock(m, noise_sd=0.002)
    with pytest.raises(ValueError):
        event_study(r, m, [m.index[500]], **kwargs)
