"""COT positioning features: known-answer, numpy cross-checks and no-look-ahead checks."""
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.mrip.cot.features import (
    COT_PUBLICATION_LAG_DAYS,
    add_crowding,
    add_percentiles,
    point_in_time,
    positioning_frame,
    price_position_divergence,
)
from app.mrip.data.models import CotRecord, CotSeries, Provenance

# 2024-01-02 is a Tuesday.
FIRST_TUESDAY = date(2024, 1, 2)

PROVENANCE = Provenance(
    provider="t", gateway="t", endpoint="t",
    fetched_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
)


def _tuesday(i: int) -> date:
    return (pd.Timestamp(FIRST_TUESDAY) + pd.Timedelta(weeks=i)).date()


def _record(
    i: int,
    long_: dict[str, float | None],
    short: dict[str, float | None],
    oi: float | None = 1000.0,
) -> CotRecord:
    return CotRecord(
        report_date=_tuesday(i), market="ES", open_interest=oi,
        long_positions=long_, short_positions=short,
    )


def _series(records: Sequence[CotRecord]) -> CotSeries:
    return CotSeries(market="ES", records=tuple(records), provenance=PROVENANCE)


def _mm_series(nets: Sequence[float], oi: float | None = 1000.0) -> CotSeries:
    """Managed-money series whose net position equals `nets` (short is a fixed 100)."""
    return _series([
        _record(i, {"managed_money": 100.0 + n}, {"managed_money": 100.0}, oi)
        for i, n in enumerate(nets)
    ])


def _frame(nets: Sequence[float]) -> pd.DataFrame:
    return positioning_frame(_mm_series(nets))[0]


def _random_nets(n: int, seed: int = 5) -> list[float]:
    return list(np.random.default_rng(seed).normal(0.0, 1000.0, n).round(0))


def _report_index(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(_tuesday(i)) for i in range(n)])


def _weekly_prices(values: Sequence[float]) -> pd.Series:
    return pd.Series(list(values), index=_report_index(len(values)), dtype=float)


def _assert_series_equal(a: pd.Series, b: pd.Series) -> None:
    assert len(a) == len(b)
    np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


# ---------------------------------------------------------------- positioning_frame


def test_publication_lag_is_three_days() -> None:
    assert COT_PUBLICATION_LAG_DAYS == 3


def test_group_prefers_managed_money_when_available() -> None:
    series = _series([
        _record(i, {"managed_money": 50.0, "non_commercial": 900.0},
                {"managed_money": 20.0, "non_commercial": 100.0})
        for i in range(3)
    ])
    frame, group = positioning_frame(series)
    assert group == "managed_money"
    assert frame["net_position"].tolist() == [30.0, 30.0, 30.0]


def test_group_falls_back_to_non_commercial_when_managed_money_is_sparse() -> None:
    # Only 1 of 4 records has both sides for managed money -> fewer than half.
    records = [_record(0, {"managed_money": 50.0, "non_commercial": 500.0},
                       {"managed_money": 20.0, "non_commercial": 200.0})]
    records += [
        _record(i, {"managed_money": None, "non_commercial": 500.0},
                {"managed_money": 20.0, "non_commercial": 200.0})
        for i in range(1, 4)
    ]
    frame, group = positioning_frame(_series(records))
    assert group == "non_commercial"
    assert len(frame) == 4
    assert frame["net_position"].tolist() == [300.0] * 4


def test_group_managed_money_chosen_at_exactly_half_coverage() -> None:
    records = [
        _record(0, {"managed_money": 50.0, "non_commercial": 500.0},
                {"managed_money": 20.0, "non_commercial": 200.0}),
        _record(1, {"managed_money": 60.0, "non_commercial": 500.0},
                {"managed_money": 20.0, "non_commercial": 200.0}),
        _record(2, {"managed_money": 70.0, "non_commercial": 500.0},
                {"managed_money": None, "non_commercial": 200.0}),
        _record(3, {"managed_money": None, "non_commercial": 500.0},
                {"managed_money": 20.0, "non_commercial": 200.0}),
    ]
    frame, group = positioning_frame(_series(records))
    assert group == "managed_money"
    # Records with a None side are dropped.
    assert frame["net_position"].tolist() == [30.0, 40.0]
    assert [ts.date() for ts in frame.index] == [_tuesday(0), _tuesday(1)]


def test_explicit_group_overrides_automatic_choice() -> None:
    series = _series([
        _record(i, {"managed_money": 50.0, "non_commercial": 500.0},
                {"managed_money": 20.0, "non_commercial": 200.0})
        for i in range(2)
    ])
    frame, group = positioning_frame(series, group="non_commercial")
    assert group == "non_commercial"
    assert frame["net_position"].tolist() == [300.0, 300.0]


def test_explicit_group_missing_from_data_raises() -> None:
    series = _series([_record(0, {"managed_money": 5.0}, {"managed_money": 1.0})])
    with pytest.raises(ValueError):
        positioning_frame(series, group="swap_dealers")


def test_empty_series_raises() -> None:
    with pytest.raises(ValueError):
        positioning_frame(_series([]))


def test_nothing_remaining_after_dropping_none_raises() -> None:
    series = _series([
        _record(0, {"managed_money": None}, {"managed_money": 1.0}),
        _record(1, {"managed_money": 5.0}, {"managed_money": None}),
    ])
    with pytest.raises(ValueError):
        positioning_frame(series, group="managed_money")


def test_frame_values_known_answer() -> None:
    nets = [10.0, 25.0, 15.0, 40.0, 30.0, 70.0]
    ois = [100.0, 200.0, 150.0, 400.0, 300.0, 700.0]
    series = _series([
        _record(i, {"managed_money": 100.0 + n}, {"managed_money": 100.0}, oi)
        for i, (n, oi) in enumerate(zip(nets, ois))
    ])
    frame, _ = positioning_frame(series)
    assert list(frame.columns) == [
        "available_date", "gross_long", "gross_short", "net_position",
        "open_interest", "net_pct_oi", "net_change", "net_change_4w",
    ]
    assert frame.index.name == "report_date"
    assert frame["gross_long"].tolist() == [110.0, 125.0, 115.0, 140.0, 130.0, 170.0]
    assert frame["gross_short"].tolist() == [100.0] * 6
    assert frame["net_position"].tolist() == nets
    assert frame["open_interest"].tolist() == ois
    np.testing.assert_allclose(
        frame["net_pct_oi"], [0.1, 0.125, 0.1, 0.1, 0.1, 0.1], rtol=1e-12
    )
    np.testing.assert_allclose(
        frame["net_change"].to_numpy(), [np.nan, 15.0, -10.0, 25.0, -10.0, 40.0]
    )
    # 4-report difference: 30-10, 70-25.
    np.testing.assert_allclose(
        frame["net_change_4w"].to_numpy(), [np.nan, np.nan, np.nan, np.nan, 20.0, 45.0]
    )


def test_net_pct_oi_is_nan_for_missing_or_zero_open_interest() -> None:
    series = _series([
        _record(0, {"managed_money": 150.0}, {"managed_money": 100.0}, oi=None),
        _record(1, {"managed_money": 150.0}, {"managed_money": 100.0}, oi=0.0),
        _record(2, {"managed_money": 150.0}, {"managed_money": 100.0}, oi=500.0),
    ])
    frame, _ = positioning_frame(series)
    assert math.isnan(frame["net_pct_oi"].iloc[0])
    assert math.isnan(frame["net_pct_oi"].iloc[1])
    assert frame["net_pct_oi"].iloc[2] == pytest.approx(0.1)


def test_available_date_is_report_date_plus_three_days() -> None:
    frame = _frame([1.0, 2.0, 3.0])
    for report_ts, available_ts in zip(frame.index, frame["available_date"]):
        assert available_ts - report_ts == pd.Timedelta(days=3)
        assert report_ts.dayofweek == 1  # Tuesday
        assert available_ts.dayofweek == 4  # Friday


def test_index_is_sorted_and_duplicates_keep_the_last_record() -> None:
    records = [
        _record(2, {"managed_money": 130.0}, {"managed_money": 100.0}),
        _record(0, {"managed_money": 110.0}, {"managed_money": 100.0}),
        _record(1, {"managed_money": 120.0}, {"managed_money": 100.0}),
        _record(1, {"managed_money": 190.0}, {"managed_money": 100.0}),  # restated
    ]
    frame, _ = positioning_frame(_series(records))
    assert frame.index.is_monotonic_increasing and frame.index.is_unique
    assert [ts.date() for ts in frame.index] == [_tuesday(0), _tuesday(1), _tuesday(2)]
    assert frame["net_position"].tolist() == [10.0, 90.0, 30.0]
    assert isinstance(frame.index, pd.DatetimeIndex)


# ---------------------------------------------------------------- add_percentiles


def test_percentiles_on_rising_ramp_and_warmup_lengths() -> None:
    frame = add_percentiles(
        _frame([float(i) for i in range(30)]),
        weeks_1y=4, weeks_3y=6, weeks_5y=8, zscore_weeks=5,
    )
    for column, window in (("percentile_1y", 4), ("percentile_3y", 6), ("percentile_5y", 8)):
        values = frame[column]
        assert int(values.isna().sum()) == window - 1
        assert values.iloc[: window - 1].isna().all()
        # The latest value of a rising ramp is the window maximum.
        assert values.iloc[window - 1 :].eq(1.0).all()
    assert int(frame["zscore"].isna().sum()) == 4


def test_percentiles_on_falling_ramp_equal_one_over_window() -> None:
    frame = add_percentiles(
        _frame([float(-i) for i in range(12)]),
        weeks_1y=4, weeks_3y=5, weeks_5y=6, zscore_weeks=4,
    )
    assert frame["percentile_1y"].dropna().tolist() == pytest.approx([0.25] * 9)
    assert frame["percentile_3y"].dropna().tolist() == pytest.approx([0.2] * 8)


def test_percentiles_first_full_window_known_answer() -> None:
    nets = [5.0, 3.0, 8.0, 1.0, 9.0, 2.0, 7.0, 4.0, 6.0, 0.0]
    frame = add_percentiles(
        _frame(nets), weeks_1y=4, weeks_3y=5, weeks_5y=6, zscore_weeks=4
    )
    p1 = frame["percentile_1y"].tolist()
    assert all(math.isnan(v) for v in p1[:3])
    # idx3: window [5,3,8,1], current 1 -> 1/4. idx4: [3,8,1,9] -> 4/4.
    # idx5: [8,1,9,2], current 2 -> {1,2} -> 2/4. idx9: [7,4,6,0] -> 1/4.
    assert p1[3] == 0.25
    assert p1[4] == 1.0
    assert p1[5] == 0.5
    assert p1[9] == 0.25
    p3 = frame["percentile_3y"].tolist()
    assert all(math.isnan(v) for v in p3[:4])
    # idx4: [5,3,8,1,9], current 9 -> 5/5. idx5: [3,8,1,9,2], current 2 -> {1,2} -> 2/5.
    assert p3[4] == 1.0
    assert p3[5] == 0.4
    p5 = frame["percentile_5y"].tolist()
    assert all(math.isnan(v) for v in p5[:5])
    # idx5: [5,3,8,1,9,2], current 2 -> {1,2} -> 2/6.
    assert p5[5] == pytest.approx(2 / 6)


def test_percentile_ties_count_as_less_or_equal() -> None:
    frame = add_percentiles(
        _frame([4.0, 4.0, 4.0, 4.0, 1.0]),
        weeks_1y=4, weeks_3y=4, weeks_5y=4, zscore_weeks=4,
    )
    # Window [4,4,4,4] -> all <= 4 -> 1.0; window [4,4,4,1] -> only the 1 -> 0.25.
    assert frame["percentile_1y"].iloc[3] == 1.0
    assert frame["percentile_1y"].iloc[4] == 0.25


def test_zscore_matches_numpy_with_warmup() -> None:
    nets = _random_nets(40)
    window = 8
    frame = add_percentiles(_frame(nets), weeks_1y=4, weeks_3y=5, weeks_5y=6, zscore_weeks=window)
    z = frame["zscore"].to_numpy()
    assert np.isnan(z[: window - 1]).all()
    for i in range(window - 1, len(nets)):
        w = np.asarray(nets[i - window + 1 : i + 1])
        expected = (w[-1] - w.mean()) / w.std(ddof=1)
        assert z[i] == pytest.approx(expected, rel=1e-10)


def test_zscore_hand_computed_value() -> None:
    # Window [1,2,3]: mean 2, sample std 1 -> z of 3 is 1; of [2,3,7]: mean 4, std sqrt(7) -> 3/sqrt(7).
    frame = add_percentiles(
        _frame([1.0, 2.0, 3.0, 7.0]), weeks_1y=2, weeks_3y=2, weeks_5y=2, zscore_weeks=3
    )
    assert math.isnan(frame["zscore"].iloc[1])
    assert frame["zscore"].iloc[2] == pytest.approx(1.0)
    assert frame["zscore"].iloc[3] == pytest.approx(3.0 / math.sqrt(7.0))


def test_zscore_is_nan_when_window_is_constant() -> None:
    frame = add_percentiles(
        _frame([3.0] * 6 + [4.0]), weeks_1y=2, weeks_3y=2, weeks_5y=2, zscore_weeks=4
    )
    assert frame["zscore"].iloc[3:6].isna().all()
    assert not math.isnan(frame["zscore"].iloc[6])


def test_add_percentiles_returns_a_copy_and_keeps_columns() -> None:
    original = _frame([1.0, 2.0, 3.0, 4.0])
    before = original.copy()
    out = add_percentiles(original, weeks_1y=2, weeks_3y=2, weeks_5y=2, zscore_weeks=2)
    pd.testing.assert_frame_equal(original, before)
    assert "percentile_1y" not in original.columns
    for column in original.columns:
        assert column in out.columns
    assert out is not original


def test_add_percentiles_default_windows_are_one_three_five_years() -> None:
    frame = add_percentiles(_frame([float(i) for i in range(270)]))
    assert int(frame["percentile_1y"].isna().sum()) == 51
    assert int(frame["percentile_3y"].isna().sum()) == 155
    assert int(frame["percentile_5y"].isna().sum()) == 259
    assert int(frame["zscore"].isna().sum()) == 155


@pytest.mark.parametrize("name", ["weeks_1y", "weeks_3y", "weeks_5y", "zscore_weeks"])
def test_add_percentiles_rejects_windows_below_two(name: str) -> None:
    kwargs = {"weeks_1y": 4, "weeks_3y": 4, "weeks_5y": 4, "zscore_weeks": 4}
    kwargs[name] = 1
    with pytest.raises(ValueError):
        add_percentiles(_frame([1.0, 2.0, 3.0, 4.0, 5.0]), **kwargs)


def test_add_percentiles_requires_net_position_and_ordered_index() -> None:
    frame = _frame([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        add_percentiles(frame.drop(columns="net_position"))
    with pytest.raises(ValueError):
        add_percentiles(frame.iloc[::-1])
    with pytest.raises(ValueError):
        add_percentiles(frame.reset_index(drop=True))


# ---------------------------------------------------------------- add_crowding


def _pct_frame(values: Sequence[float]) -> pd.DataFrame:
    return pd.DataFrame({"percentile_3y": list(values)}, index=_report_index(len(values)))


def test_crowding_score_side_and_flag_known_answer() -> None:
    out = add_crowding(_pct_frame([1.0, 0.0, 0.9, 0.1, 0.5, 0.75, 0.25, 0.85]))
    np.testing.assert_allclose(
        out["crowding_score"].to_numpy(), [1.0, 1.0, 0.8, 0.8, 0.0, 0.5, 0.5, 0.7]
    )
    assert out["crowding_side"].tolist() == [
        "LONG", "SHORT", "LONG", "SHORT", "NONE", "LONG", "SHORT", "LONG",
    ]
    assert out["crowded"].tolist() == [True, True, True, True, False, False, False, False]
    assert out["crowded"].dtype == bool


def test_crowding_exactly_at_extreme_is_crowded_and_just_below_is_not() -> None:
    out = add_crowding(_pct_frame([0.9, 0.1, 0.8999, 0.1001]), extreme=0.8)
    assert out["crowded"].tolist() == [True, True, False, False]
    out_half = add_crowding(_pct_frame([0.75, 0.25, 0.7499]), extreme=0.5)
    assert out_half["crowded"].tolist() == [True, True, False]
    # extreme = 1 only flags a percentile at the very edge.
    out_one = add_crowding(_pct_frame([1.0, 0.0, 0.99]), extreme=1.0)
    assert out_one["crowded"].tolist() == [True, True, False]


def test_crowding_is_nan_and_not_crowded_where_percentile_is_nan() -> None:
    out = add_crowding(_pct_frame([np.nan, 0.95, np.nan]))
    assert math.isnan(out["crowding_score"].iloc[0])
    assert math.isnan(out["crowding_score"].iloc[2])
    assert pd.isna(out["crowding_side"].iloc[0])
    assert pd.isna(out["crowding_side"].iloc[2])
    assert out["crowding_side"].iloc[1] == "LONG"
    assert out["crowded"].tolist() == [False, True, False]
    assert out["crowded"].dtype == bool


def test_crowding_score_stays_in_unit_interval_and_frame_is_copied() -> None:
    frame = add_percentiles(
        _frame(_random_nets(60)), weeks_1y=4, weeks_3y=10, weeks_5y=12, zscore_weeks=10
    )
    before = frame.copy()
    out = add_crowding(frame)
    pd.testing.assert_frame_equal(frame, before)
    score = out["crowding_score"].dropna()
    assert ((score >= 0.0) & (score <= 1.0)).all()
    assert int(out["crowding_score"].isna().sum()) == 9


@pytest.mark.parametrize("extreme", [0.0, -0.1, 1.0001, 2.0])
def test_crowding_rejects_invalid_extreme(extreme: float) -> None:
    with pytest.raises(ValueError):
        add_crowding(_pct_frame([0.5]), extreme=extreme)


def test_crowding_requires_percentile_3y() -> None:
    with pytest.raises(ValueError):
        add_crowding(_frame([1.0, 2.0]))


# ---------------------------------------------------------------- point_in_time


def test_point_in_time_hides_report_until_its_friday() -> None:
    frame = _frame([1.0, 2.0, 3.0])
    tuesday = _tuesday(1)  # 2024-01-09
    assert pd.Timestamp(tuesday).dayofweek == 1
    first_visible = pd.Timestamp(tuesday) + pd.Timedelta(days=3)  # Friday 2024-01-12
    for offset in (0, 1, 2):  # Tuesday, Wednesday, Thursday
        visible = point_in_time(frame, (pd.Timestamp(tuesday) + pd.Timedelta(days=offset)).date())
        assert pd.Timestamp(tuesday) not in visible.index
    friday = point_in_time(frame, first_visible.date())
    assert pd.Timestamp(tuesday) in friday.index
    assert len(friday) == 2  # the first report (published the Friday before) is visible too


def test_point_in_time_known_answer_and_edges() -> None:
    frame = _frame([1.0, 2.0, 3.0, 4.0])
    assert point_in_time(frame, date(2024, 1, 4)).empty  # Thursday before the first Friday
    assert len(point_in_time(frame, date(2024, 1, 5))) == 1  # first Friday
    assert len(point_in_time(frame, date(2024, 1, 11))) == 1  # Thursday of week 2
    assert len(point_in_time(frame, date(2024, 1, 12))) == 2
    assert len(point_in_time(frame, date(2030, 1, 1))) == 4
    # Result keeps the columns and is a copy.
    sub = point_in_time(frame, date(2024, 1, 12))
    assert list(sub.columns) == list(frame.columns)
    sub.loc[sub.index[0], "net_position"] = -999.0
    assert frame["net_position"].iloc[0] == 1.0


def test_point_in_time_requires_available_date() -> None:
    with pytest.raises(ValueError):
        point_in_time(_frame([1.0, 2.0]).drop(columns="available_date"), date(2024, 1, 5))


# ---------------------------------------------------------------- price_position_divergence


def test_divergence_hand_computed_example() -> None:
    nets = [0.0, 1.0, 2.0, 3.0, 10.0, 12.0, 11.0, 20.0]
    prices = _weekly_prices([100.0, 110.0, 105.0, 120.0, 130.0, 125.0, 140.0, 150.0])
    frame = _frame(nets)
    out = price_position_divergence(frame, prices, price_weeks=1, zscore_weeks=2)
    assert out.index.equals(frame.index)
    # net_change_4w = [nan x4, 10, 11, 9, 17]; a 2-point z-score is sign(change)/sqrt(2).
    # Price returns rise at idx 1, fall at 2, rise 3, fall 4 (130/120 vs 120/105 smaller), ...
    r = 1.0 / math.sqrt(2.0)
    # idx5: price z = -r (return fell), net z = +r (change rose) -> -sqrt2.
    # idx6: price z = +r, net z = -r -> +sqrt2. idx7: price z = -r, net z = +r -> -sqrt2.
    assert out.iloc[:5].isna().all()
    assert out.iloc[5] == pytest.approx(-2 * r)
    assert out.iloc[6] == pytest.approx(2 * r)
    assert out.iloc[7] == pytest.approx(-2 * r)


def test_divergence_sign_convention() -> None:
    n = 14
    # Price return keeps rising (convex growth) while the 4-report positioning change keeps
    # falling: price moved more than positioning -> positive divergence.
    accel_prices = _weekly_prices([100.0 * math.exp(0.02 * i * i) for i in range(n)])
    falling_pos = [-float(i * i) for i in range(n)]
    out = price_position_divergence(_frame(falling_pos), accel_prices, price_weeks=1, zscore_weeks=3)
    assert out.notna().sum() > 0
    assert (out.dropna() > 0).all()
    # Mirror image: positioning change keeps rising while the price return keeps falling
    # -> negative divergence.
    decel_prices = _weekly_prices([100.0 * math.exp(-0.02 * i * i) for i in range(n)])
    rising_pos = [float(i * i) for i in range(n)]
    out2 = price_position_divergence(_frame(rising_pos), decel_prices, price_weeks=1, zscore_weeks=3)
    assert out2.notna().sum() > 0
    assert (out2.dropna() < 0).all()


def _reference_divergence(
    nets: Sequence[float], prices: pd.Series, price_weeks: int, window: int
) -> list[float]:
    """Loop-based reference: last observation <= t, ddof=1 z-scores, NaN until full."""
    n = len(nets)
    returns: list[float] = []
    for i in range(n):
        t = pd.Timestamp(_tuesday(i))
        then = t - pd.Timedelta(days=7 * price_weeks)
        now_px = prices[prices.index <= t]
        then_px = prices[prices.index <= then]
        if now_px.empty or then_px.empty:
            returns.append(float("nan"))
        else:
            returns.append(float(now_px.iloc[-1] / then_px.iloc[-1] - 1.0))
    change4 = [float("nan")] * 4 + [nets[i] - nets[i - 4] for i in range(4, n)]

    def z_at(values: list[float], i: int) -> float:
        if i < window - 1:
            return float("nan")
        w = np.asarray(values[i - window + 1 : i + 1])
        if np.isnan(w).any() or w.std(ddof=1) == 0:
            return float("nan")
        return float((w[-1] - w.mean()) / w.std(ddof=1))

    return [z_at(returns, i) - z_at(change4, i) for i in range(n)]


def test_divergence_matches_reference_with_daily_prices_and_gaps() -> None:
    nets = _random_nets(30, seed=21)
    rng = np.random.default_rng(3)
    # Daily weekday prices starting a few weeks before the first report; some Tuesdays missing.
    days = pd.bdate_range("2023-12-04", periods=200)
    days = days[~days.isin([pd.Timestamp("2024-01-16"), pd.Timestamp("2024-02-13")])]
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)
    for price_weeks in (1, 4):
        out = price_position_divergence(_frame(nets), prices, price_weeks=price_weeks, zscore_weeks=6)
        expected = _reference_divergence(nets, prices, price_weeks, 6)
        np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-9, equal_nan=True)
        assert out.notna().sum() > 10


def test_divergence_warmup_lengths() -> None:
    nets = _random_nets(20, seed=2)
    prices = _weekly_prices([100.0 + i * i for i in range(20)])
    out = price_position_divergence(_frame(nets), prices, price_weeks=4, zscore_weeks=5)
    # net_change_4w is valid from idx 4 -> z valid from idx 8; price return valid from idx 4
    # (needs a price 4 weeks back) -> z valid from idx 8.
    assert out.iloc[:8].isna().all()
    assert out.iloc[8:].notna().all()


def test_divergence_nan_when_prices_do_not_reach_back_far_enough() -> None:
    nets = _random_nets(12, seed=4)
    # Prices only start at the 5th report (idx 4); return needs a price 1 week earlier.
    prices = pd.Series(
        [100.0, 101.0, 103.0, 102.0, 105.0, 107.0, 106.0, 110.0],
        index=_report_index(12)[4:],
    )
    out = price_position_divergence(_frame(nets), prices, price_weeks=1, zscore_weeks=2)
    # Price return exists from idx 5, its z-score from idx 6; net z from idx 5 -> valid from idx 6.
    assert out.iloc[:6].isna().all()
    assert out.iloc[6:].notna().all()


def test_divergence_nan_when_std_is_zero() -> None:
    nets = [float(i) for i in range(12)]  # net_change_4w constant 4 -> std 0
    prices = _weekly_prices([100.0 + (i % 3) * 5 for i in range(12)])
    out = price_position_divergence(_frame(nets), prices, price_weeks=1, zscore_weeks=3)
    assert out.isna().all()


def test_divergence_rejects_invalid_arguments() -> None:
    frame = _frame(_random_nets(12))
    prices = _weekly_prices([100.0 + i for i in range(12)])
    with pytest.raises(ValueError):
        price_position_divergence(frame, prices, price_weeks=0)
    with pytest.raises(ValueError):
        price_position_divergence(frame, prices, zscore_weeks=1)
    with pytest.raises(ValueError):
        price_position_divergence(frame, prices.iloc[::-1])
    with pytest.raises(ValueError):
        price_position_divergence(frame, prices.reset_index(drop=True))
    bad = prices.copy()
    bad.iloc[3] = 0.0
    with pytest.raises(ValueError):
        price_position_divergence(frame, bad)
    bad.iloc[3] = -5.0
    with pytest.raises(ValueError):
        price_position_divergence(frame, bad)
    with pytest.raises(ValueError):
        price_position_divergence(frame.drop(columns="net_change_4w"), prices)


# ---------------------------------------------------------------- no look-ahead


def _alter_future(nets: list[float], cut: int) -> list[float]:
    return nets[:cut] + [n * 7.0 + 12345.0 for n in nets[cut:]]


def test_percentiles_and_zscore_have_no_look_ahead() -> None:
    nets = _random_nets(50, seed=8)
    cut = 30
    kwargs = {"weeks_1y": 5, "weeks_3y": 9, "weeks_5y": 12, "zscore_weeks": 10}
    base = add_percentiles(_frame(nets), **kwargs)
    altered = add_percentiles(_frame(_alter_future(nets, cut)), **kwargs)
    # Sanity: the future really did change.
    assert not np.allclose(
        base["zscore"].iloc[cut:].to_numpy(), altered["zscore"].iloc[cut:].to_numpy(), equal_nan=True
    )
    for column in ("percentile_1y", "percentile_3y", "percentile_5y", "zscore"):
        _assert_series_equal(base[column].iloc[:cut], altered[column].iloc[:cut])
    # Cropping the history gives the same past values as computing on the full history.
    cropped = add_percentiles(_frame(nets[:cut]), **kwargs)
    for column in ("percentile_1y", "percentile_3y", "percentile_5y", "zscore"):
        _assert_series_equal(base[column].iloc[:cut], cropped[column])


def test_crowding_has_no_look_ahead() -> None:
    nets = _random_nets(50, seed=9)
    cut = 30
    kwargs = {"weeks_1y": 5, "weeks_3y": 9, "weeks_5y": 12, "zscore_weeks": 10}
    base = add_crowding(add_percentiles(_frame(nets), **kwargs))
    altered = add_crowding(add_percentiles(_frame(_alter_future(nets, cut)), **kwargs))
    _assert_series_equal(base["crowding_score"].iloc[:cut], altered["crowding_score"].iloc[:cut])
    assert base["crowding_side"].iloc[:cut].equals(altered["crowding_side"].iloc[:cut])
    assert base["crowded"].iloc[:cut].equals(altered["crowded"].iloc[:cut])


def test_divergence_has_no_look_ahead() -> None:
    n, cut = 50, 30
    nets = _random_nets(n, seed=12)
    rng = np.random.default_rng(13)
    days = pd.bdate_range("2023-12-04", periods=320)
    prices = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)

    last_known = pd.Timestamp(_tuesday(cut - 1))
    altered_prices = prices.copy()
    altered_prices[altered_prices.index > last_known] *= 3.0

    base = price_position_divergence(_frame(nets), prices, price_weeks=4, zscore_weeks=8)
    altered = price_position_divergence(
        _frame(_alter_future(nets, cut)), altered_prices, price_weeks=4, zscore_weeks=8
    )
    assert not np.allclose(
        base.iloc[cut:].to_numpy(), altered.iloc[cut:].to_numpy(), equal_nan=True
    )
    _assert_series_equal(base.iloc[:cut], altered.iloc[:cut])
    assert base.iloc[:cut].notna().sum() > 10


def test_point_in_time_then_features_equals_features_then_point_in_time() -> None:
    nets = _random_nets(40, seed=17)
    kwargs = {"weeks_1y": 5, "weeks_3y": 9, "weeks_5y": 12, "zscore_weeks": 10}
    as_of = date(2024, 6, 14)
    full = point_in_time(add_percentiles(_frame(nets), **kwargs), as_of)
    visible_only = add_percentiles(point_in_time(_frame(nets), as_of), **kwargs)
    assert len(full) == len(visible_only) > 0
    for column in ("percentile_1y", "percentile_3y", "percentile_5y", "zscore"):
        _assert_series_equal(full[column], visible_only[column])
