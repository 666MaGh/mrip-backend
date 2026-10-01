"""Outcome resolution against observed bars (work 009): windows, path, walls, implied move."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.mrip.outcomes.resolve import (
    HORIZON_BARS,
    HORIZON_KINDS,
    OUTCOME_VERSION,
    PathStats,
    WallResult,
    benchmark_return,
    horizon_bars_for,
    implied_move_outcome,
    locate_window,
    path_stats,
    wall_outcome,
)

START = "2024-01-01"  # a Monday; bar i is the i-th business day


def make_bars(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    start: str = START,
) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes))
    data: dict[str, list[float]] = {"close": closes}
    if highs is not None:
        data["high"] = highs
    if lows is not None:
        data["low"] = lows
    return pd.DataFrame(data, index=idx)


def ramp(n: int = 40, base: float = 100.0) -> pd.DataFrame:
    return make_bars([base + i for i in range(n)])


def day(pos: int) -> date:
    return pd.bdate_range(START, periods=pos + 1)[pos].date()


# ---------------------------------------------------------------- locate_window


def test_horizon_kinds_listing() -> None:
    assert HORIZON_KINDS == ("W1", "M1", "M3", "M6", "M12", "EOD", "NEXT_SESSION", "EXPIRY")
    assert HORIZON_BARS == {"W1": 5, "M1": 21, "M3": 63, "M6": 126, "M12": 252}


@pytest.mark.parametrize("kind,n", list(HORIZON_BARS.items()))
def test_locate_horizon_kinds(kind: str, n: int) -> None:
    bars = ramp(n + 10)
    w = locate_window(bars, day(2), kind)
    assert w is not None
    assert (w.start_pos, w.end_pos) == (2, 2 + n)
    assert w.start_date == day(2) and w.end_date == day(2 + n)
    assert w.entry_price == 102.0
    assert w.kind == kind


def test_locate_entry_price_override() -> None:
    w = locate_window(ramp(), day(2), "W1", entry_price=99.5)
    assert w is not None and w.entry_price == 99.5


def test_locate_weekend_made_on_uses_last_earlier_bar() -> None:
    saturday = date(2024, 1, 6)
    w = locate_window(ramp(), saturday, "W1")
    assert w is not None
    assert w.start_pos == 4 and w.start_date == date(2024, 1, 5)
    assert w.entry_price == 104.0


def test_locate_beyond_last_bar_is_none() -> None:
    bars = ramp(40)
    assert locate_window(bars, day(34), "W1") is not None  # end_pos == 39 == last
    assert locate_window(bars, day(35), "W1") is None
    assert locate_window(bars, day(30), "M1") is None


def test_locate_made_on_after_data_uses_last_bar_and_is_unresolved() -> None:
    assert locate_window(ramp(10), date(2024, 3, 1), "W1") is None


def test_locate_no_data_before_made_on() -> None:
    with pytest.raises(ValueError, match="no data on or before made_on"):
        locate_window(ramp(), date(2023, 12, 29), "W1")


def test_locate_unknown_kind() -> None:
    with pytest.raises(ValueError):
        locate_window(ramp(), day(2), "W2")


def test_locate_entry_must_be_positive() -> None:
    with pytest.raises(ValueError):
        locate_window(ramp(), day(2), "W1", entry_price=0.0)
    with pytest.raises(ValueError):
        locate_window(ramp(), day(2), "W1", entry_price=-1.0)
    with pytest.raises(ValueError):
        locate_window(make_bars([0.0, 1, 2, 3, 4, 5, 6]), day(0), "W1")


def test_locate_eod() -> None:
    w = locate_window(ramp(), day(2), "EOD", entry_price=101.5)
    assert w is not None
    assert w.start_pos == w.end_pos == 2
    assert w.entry_price == 101.5


def test_locate_eod_requires_entry_price() -> None:
    with pytest.raises(ValueError):
        locate_window(ramp(), day(2), "EOD")


def test_locate_eod_requires_observed_made_on_session() -> None:
    assert locate_window(ramp(), date(2024, 1, 6), "EOD", entry_price=100.0) is None
    assert locate_window(ramp(10), date(2024, 2, 1), "EOD", entry_price=100.0) is None


def test_locate_next_session() -> None:
    w = locate_window(ramp(), day(2), "NEXT_SESSION", entry_price=101.5)
    assert w is not None
    assert (w.start_pos, w.end_pos) == (2, 3)


def test_locate_next_session_requirements() -> None:
    with pytest.raises(ValueError):
        locate_window(ramp(), day(2), "NEXT_SESSION")
    assert locate_window(ramp(), date(2024, 1, 6), "NEXT_SESSION", entry_price=100.0) is None
    bars = ramp(5)
    assert locate_window(bars, day(4), "NEXT_SESSION", entry_price=100.0) is None  # no next bar


def test_locate_expiry() -> None:
    bars = ramp(10)  # last bar 2024-01-12
    w = locate_window(bars, day(1), "EXPIRY", expiry=date(2024, 1, 10))
    assert w is not None
    assert (w.start_pos, w.end_pos) == (1, 7)
    assert w.end_date == date(2024, 1, 10)


def test_locate_expiry_requires_expiry() -> None:
    with pytest.raises(ValueError):
        locate_window(ramp(), day(1), "EXPIRY")


def test_locate_expiry_not_resolvable_until_data_reaches_it() -> None:
    bars = ramp(10)  # last bar 2024-01-12 (Fri)
    assert locate_window(bars, day(1), "EXPIRY", expiry=date(2024, 1, 15)) is None
    # A weekend expiry is only reached once a later bar exists.
    assert locate_window(bars, day(1), "EXPIRY", expiry=date(2024, 1, 13)) is None
    assert locate_window(bars, day(1), "EXPIRY", expiry=date(2024, 1, 12)) is not None


def test_locate_expiry_weekend_uses_last_earlier_bar_once_reached() -> None:
    bars = ramp(12)  # reaches Tue 2024-01-16
    w = locate_window(bars, day(1), "EXPIRY", expiry=date(2024, 1, 13))
    assert w is not None
    assert w.end_pos == 9 and w.end_date == date(2024, 1, 12)


def test_locate_expiry_must_follow_start() -> None:
    bars = ramp(10)
    assert locate_window(bars, day(5), "EXPIRY", expiry=day(5)) is None
    assert locate_window(bars, day(5), "EXPIRY", expiry=day(3)) is None


# ------------------------------------------------------------------- path_stats


def test_path_stats_ramp_closes_only() -> None:
    bars = ramp(40)
    w = locate_window(bars, day(0), "W1")
    assert w is not None
    s = path_stats(bars, w)
    assert s.actual_return == pytest.approx(105 / 100 - 1)
    closes = np.array([100, 101, 102, 103, 104, 105], dtype=float)
    expected_vol = np.std(np.diff(np.log(closes)), ddof=1) * math.sqrt(252)
    assert s.realized_volatility == pytest.approx(expected_vol)
    assert s.max_adverse_excursion == 0.0
    assert s.max_favorable_excursion == pytest.approx(0.05)
    assert s.n_bars == 5
    assert s.excursions_from == "closes"


def test_path_stats_vol_matches_numpy_on_long_window() -> None:
    rng = np.random.default_rng(7)
    closes = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))))
    bars = make_bars(closes)
    w = locate_window(bars, day(3), "M1")
    assert w is not None
    s = path_stats(bars, w)
    seg = np.array(closes[3 : 3 + 21 + 1])
    assert s.realized_volatility == pytest.approx(np.std(np.diff(np.log(seg)), ddof=1) * math.sqrt(252))
    assert s.actual_return == pytest.approx(closes[24] / closes[3] - 1)


CLOSES = [100.0, 98.0, 103.0, 97.0, 101.0, 102.0]
HIGHS = [120.0, 99.0, 105.0, 99.0, 102.0, 103.0]  # bar 0 is deliberately extreme
LOWS = [80.0, 96.0, 100.0, 95.0, 99.0, 100.0]


def test_path_stats_excursions_closes_only() -> None:
    bars = make_bars(CLOSES)
    w = locate_window(bars, day(0), "W1")
    assert w is not None
    s = path_stats(bars, w)
    assert s.actual_return == pytest.approx(0.02)
    assert s.max_adverse_excursion == pytest.approx(-0.03)
    assert s.max_favorable_excursion == pytest.approx(0.03)
    assert s.excursions_from == "closes"


def test_path_stats_excursions_intraday_range_excludes_start_bar() -> None:
    bars = make_bars(CLOSES, HIGHS, LOWS)
    w = locate_window(bars, day(0), "W1")
    assert w is not None
    s = path_stats(bars, w)
    assert s.max_adverse_excursion == pytest.approx(-0.05)  # low 95, not bar 0's 80
    assert s.max_favorable_excursion == pytest.approx(0.05)  # high 105, not bar 0's 120
    assert s.excursions_from == "intraday_range"


def test_path_stats_needs_both_high_and_low_for_intraday_label() -> None:
    bars = make_bars(CLOSES, highs=HIGHS)
    w = locate_window(bars, day(0), "W1")
    assert w is not None
    s = path_stats(bars, w)
    assert s.excursions_from == "closes"
    assert s.max_adverse_excursion == pytest.approx(-0.03)
    assert s.max_favorable_excursion == pytest.approx(0.03)


def test_path_stats_final_close_always_an_observed_point() -> None:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 95.0]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    lows[5] = 96.0  # inconsistent on purpose: low above the close
    bars = make_bars(closes, highs, lows)
    w = locate_window(bars, day(0), "W1")
    assert w is not None
    s = path_stats(bars, w)
    assert s.max_adverse_excursion == pytest.approx(-0.05)
    assert s.max_favorable_excursion == pytest.approx(0.045)  # high 104.5


def test_path_stats_sign_convention_on_one_way_paths() -> None:
    down = make_bars([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    w = locate_window(down, day(0), "W1")
    assert w is not None
    s = path_stats(down, w)
    assert s.max_favorable_excursion == 0.0
    assert s.max_adverse_excursion == pytest.approx(-0.05)
    assert s.actual_return == pytest.approx(-0.05)


def test_path_stats_entry_override() -> None:
    bars = make_bars(CLOSES)
    w = locate_window(bars, day(0), "W1", entry_price=99.0)
    assert w is not None
    s = path_stats(bars, w)
    assert s.actual_return == pytest.approx(102 / 99 - 1)
    assert s.max_adverse_excursion == pytest.approx(97 / 99 - 1)
    assert s.max_favorable_excursion == pytest.approx(103 / 99 - 1)


def test_path_stats_eod_has_no_excursion_bars() -> None:
    closes = [100.0, 101.0, 102.0, 103.0]
    highs = [101.0, 102.0, 200.0, 104.0]
    lows = [99.0, 100.0, 1.0, 102.0]
    bars = make_bars(closes, highs, lows)
    w = locate_window(bars, day(2), "EOD", entry_price=101.0)
    assert w is not None
    s = path_stats(bars, w)
    assert s.actual_return == pytest.approx(102 / 101 - 1)
    assert s.realized_volatility is None
    assert s.max_adverse_excursion == 0.0  # the bar's low (1.0) pre-dates the entry: ignored
    assert s.max_favorable_excursion == pytest.approx(102 / 101 - 1)
    assert s.excursions_from == "closes"
    assert s.n_bars == 0


def test_path_stats_eod_adverse_from_end_close() -> None:
    bars = make_bars([100.0, 101.0, 102.0], [101.0, 102.0, 300.0], [99.0, 100.0, 1.0])
    w = locate_window(bars, day(2), "EOD", entry_price=104.0)
    assert w is not None
    s = path_stats(bars, w)
    assert s.max_adverse_excursion == pytest.approx(102 / 104 - 1)
    assert s.max_favorable_excursion == 0.0


def test_path_stats_next_session_excludes_made_on_bar() -> None:
    closes = [100.0, 101.0, 102.0, 103.0]
    highs = [101.0, 102.0, 300.0, 104.0]
    lows = [99.0, 100.0, 50.0, 101.0]
    bars = make_bars(closes, highs, lows)
    w = locate_window(bars, day(2), "NEXT_SESSION", entry_price=101.5)
    assert w is not None
    s = path_stats(bars, w)
    assert s.actual_return == pytest.approx(103 / 101.5 - 1)
    assert s.realized_volatility is None
    assert s.max_adverse_excursion == pytest.approx(101 / 101.5 - 1)
    assert s.max_favorable_excursion == pytest.approx(104 / 101.5 - 1)
    assert s.excursions_from == "intraday_range"
    assert s.n_bars == 1


def test_path_stats_vol_needs_three_daily_returns() -> None:
    bars = ramp(20)
    short = locate_window(bars, day(0), "EXPIRY", expiry=day(2))  # 2 returns
    ok = locate_window(bars, day(0), "EXPIRY", expiry=day(3))  # 3 returns
    assert short is not None and ok is not None
    assert path_stats(bars, short).realized_volatility is None
    vol = path_stats(bars, ok).realized_volatility
    expected = np.std(np.diff(np.log([100.0, 101.0, 102.0, 103.0])), ddof=1) * math.sqrt(252)
    assert vol == pytest.approx(expected)


# ------------------------------------------------------------- benchmark_return


def test_benchmark_return_basic() -> None:
    bars = ramp(40)
    w = locate_window(bars, day(0), "W1")
    assert w is not None
    bench = make_bars([200.0 + 2 * i for i in range(40)])
    assert benchmark_return(bench, w) == pytest.approx(210 / 200 - 1)


def test_benchmark_uses_last_bar_on_or_before_dates() -> None:
    bars = ramp(40)
    w = locate_window(bars, day(0), "W1")
    assert w is not None
    bench = make_bars([200.0 + 2 * i for i in range(12)]).drop(pd.Timestamp(day(5)))
    assert benchmark_return(bench, w) == pytest.approx(208 / 200 - 1)  # bar 4 stands in for bar 5


def test_benchmark_missing_start_data_is_none() -> None:
    bars = ramp(40)
    w = locate_window(bars, day(3), "W1")
    assert w is not None
    late = make_bars([200.0 + i for i in range(20)], start="2024-01-08")
    assert benchmark_return(late, w) is None


def test_benchmark_not_reaching_end_date_is_none() -> None:
    bars = ramp(40)
    w = locate_window(bars, day(0), "W1")
    assert w is not None
    short = make_bars([200.0 + i for i in range(5)])  # last bar is day(4) < day(5)
    assert benchmark_return(short, w) is None
    exact = make_bars([200.0 + i for i in range(6)])
    assert benchmark_return(exact, w) == pytest.approx(205 / 200 - 1)


# ----------------------------------------------------------------- wall_outcome

TOL = 0.005


def wall_bars(highs: list[float], lows: list[float], closes: list[float]) -> pd.DataFrame:
    return make_bars(closes, highs, lows)


def w1(bars: pd.DataFrame, entry: float = 100.0):  # type: ignore[no-untyped-def]
    w = locate_window(bars, day(0), "W1", entry_price=entry)
    assert w is not None
    return w


def test_call_wall_untested() -> None:
    bars = wall_bars(
        highs=[200.0, 101, 102, 105, 104, 103],  # bar 0 is outside the excursion bars
        lows=[99.0, 99, 99, 100, 100, 100],
        closes=[100.0, 100, 101, 102, 101, 101],
    )
    r = wall_outcome(bars, w1(bars), 110.0, "call")
    assert r.result is WallResult.UNTESTED
    assert r.first_touch_date is None and r.first_break_date is None
    assert r.max_penetration == 0.0
    assert r.side == "call" and r.level == 110.0


def test_call_wall_touch_boundary_is_inclusive() -> None:
    threshold = 110.0 * (1 - TOL)
    base_lows = [99.0] * 6
    base_closes = [100.0, 100, 101, 102, 101, 101]
    at = wall_bars([101.0, 101, 101, threshold, 101, 101], base_lows, base_closes)
    r = wall_outcome(at, w1(at), 110.0, "call", TOL)
    assert r.result is WallResult.HELD
    assert r.first_touch_date == day(3)
    assert r.max_penetration == 0.0  # never above the level
    below = wall_bars([101.0, 101, 101, threshold - 0.01, 101, 101], base_lows, base_closes)
    assert wall_outcome(below, w1(below), 110.0, "call", TOL).result is WallResult.UNTESTED


def test_call_wall_held_with_wick_above_level() -> None:
    bars = wall_bars(
        highs=[100.0, 101, 111, 105, 104, 103],
        lows=[99.0] * 6,
        closes=[100.0, 100, 109, 102, 101, 101],
    )
    r = wall_outcome(bars, w1(bars), 110.0, "call")
    assert r.result is WallResult.HELD
    assert r.first_touch_date == day(2)
    assert r.first_break_date is None
    assert r.max_penetration == pytest.approx(111 / 110 - 1)


def test_call_wall_close_equal_to_level_is_not_a_break() -> None:
    bars = wall_bars(
        highs=[100.0, 101, 110, 105, 104, 103],
        lows=[99.0] * 6,
        closes=[100.0, 100, 110, 102, 101, 101],
    )
    r = wall_outcome(bars, w1(bars), 110.0, "call")
    assert r.result is WallResult.HELD
    assert r.first_break_date is None


def test_call_wall_broke() -> None:
    bars = wall_bars(
        highs=[100.0, 101, 109.5, 112, 113, 105],
        lows=[99.0] * 6,
        closes=[100.0, 100, 108, 110.5, 111, 105],
    )
    r = wall_outcome(bars, w1(bars), 110.0, "call")
    assert r.result is WallResult.BROKE
    assert r.first_touch_date == day(2)  # high 109.5 >= 109.45
    assert r.first_break_date == day(3)
    assert r.max_penetration == pytest.approx(113 / 110 - 1)


def test_call_wall_closes_only_fallback() -> None:
    bars = make_bars([100.0, 100, 109.45, 110.2, 101, 101])
    r = wall_outcome(bars, w1(bars), 110.0, "call")
    assert r.result is WallResult.BROKE
    assert r.first_touch_date == day(2)
    assert r.first_break_date == day(3)
    assert r.max_penetration == pytest.approx(110.2 / 110 - 1)


def test_put_wall_untested() -> None:
    bars = wall_bars(
        highs=[101.0] * 6,
        lows=[10.0, 99, 98, 97, 98, 99],  # bar 0 is outside the excursion bars
        closes=[100.0, 100, 99, 98, 99, 100],
    )
    r = wall_outcome(bars, w1(bars), 90.0, "put")
    assert r.result is WallResult.UNTESTED
    assert r.max_penetration == 0.0


def test_put_wall_touch_boundary_is_inclusive() -> None:
    threshold = 90.0 * (1 + TOL)
    closes = [100.0, 100, 99, 98, 99, 100]
    highs = [101.0] * 6
    at = wall_bars(highs, [99.0, 99, 99, threshold, 99, 99], closes)
    r = wall_outcome(at, w1(at), 90.0, "put", TOL)
    assert r.result is WallResult.HELD
    assert r.first_touch_date == day(3)
    assert r.max_penetration == 0.0
    above = wall_bars(highs, [99.0, 99, 99, threshold + 0.01, 99, 99], closes)
    assert wall_outcome(above, w1(above), 90.0, "put", TOL).result is WallResult.UNTESTED


def test_put_wall_held_and_close_equal_not_a_break() -> None:
    bars = wall_bars(
        highs=[101.0] * 6,
        lows=[99.0, 99, 88, 95, 96, 97],
        closes=[100.0, 100, 90, 95, 96, 97],
    )
    r = wall_outcome(bars, w1(bars), 90.0, "put")
    assert r.result is WallResult.HELD
    assert r.first_touch_date == day(2)
    assert r.first_break_date is None
    assert r.max_penetration == pytest.approx(1 - 88 / 90)


def test_put_wall_broke() -> None:
    bars = wall_bars(
        highs=[101.0] * 6,
        lows=[99.0, 99, 90.3, 88, 85, 95],
        closes=[100.0, 100, 92, 89.5, 86, 95],
    )
    r = wall_outcome(bars, w1(bars), 90.0, "put")
    assert r.result is WallResult.BROKE
    assert r.first_touch_date == day(2)  # low 90.3 <= 90.45
    assert r.first_break_date == day(3)
    assert r.max_penetration == pytest.approx(1 - 85 / 90)


def test_put_wall_closes_only_fallback() -> None:
    bars = make_bars([100.0, 100, 90.4, 89.0, 95, 95])
    r = wall_outcome(bars, w1(bars), 90.0, "put")
    assert r.result is WallResult.BROKE
    assert r.first_touch_date == day(2)
    assert r.first_break_date == day(3)
    assert r.max_penetration == pytest.approx(1 - 89 / 90)


def test_wall_custom_tolerance() -> None:
    bars = make_bars([100.0, 100, 107.0, 100, 100, 100])
    w = w1(bars)
    assert wall_outcome(bars, w, 110.0, "call", 0.005).result is WallResult.UNTESTED
    assert wall_outcome(bars, w, 110.0, "call", 0.03).result is WallResult.HELD  # 106.7


def test_wall_eod_uses_only_end_close() -> None:
    bars = make_bars([100.0, 101.0, 102.0, 103.0], [101.0, 102.0, 999.0, 104.0], [99.0, 100.0, 1.0, 102.0])
    w = locate_window(bars, day(2), "EOD", entry_price=101.0)
    assert w is not None
    # The bar's wild high/low are ignored: high = low = close = 102.
    assert wall_outcome(bars, w, 110.0, "call").result is WallResult.UNTESTED
    assert wall_outcome(bars, w, 90.0, "put").result is WallResult.UNTESTED
    held = wall_outcome(bars, w, 102.0, "call")
    assert held.result is WallResult.HELD
    assert held.first_touch_date == day(2) and held.first_break_date is None
    broke = wall_outcome(bars, w, 101.0, "call")
    assert broke.result is WallResult.BROKE
    assert broke.first_break_date == day(2)
    assert broke.max_penetration == pytest.approx(102 / 101 - 1)
    put_broke = wall_outcome(bars, w, 103.0, "put")
    assert put_broke.result is WallResult.BROKE
    assert put_broke.max_penetration == pytest.approx(1 - 102 / 103)


def test_wall_next_session_considers_next_bar_only() -> None:
    bars = make_bars(
        [100.0, 101.0, 102.0, 103.0],
        [101.0, 102.0, 300.0, 104.0],
        [99.0, 100.0, 50.0, 102.0],
    )
    w = locate_window(bars, day(2), "NEXT_SESSION", entry_price=101.5)
    assert w is not None
    assert wall_outcome(bars, w, 110.0, "call").result is WallResult.UNTESTED


def test_wall_validation() -> None:
    bars = ramp()
    w = w1(bars)
    with pytest.raises(ValueError):
        wall_outcome(bars, w, 0.0, "call")
    with pytest.raises(ValueError):
        wall_outcome(bars, w, -5.0, "put")
    with pytest.raises(ValueError):
        wall_outcome(bars, w, 100.0, "middle")  # type: ignore[arg-type]


# ----------------------------------------------------------- implied_move_outcome


def stats(
    ret: float = -0.05,
    vol: float | None = 0.30,
    mae: float = -0.08,
    mfe: float = 0.02,
) -> PathStats:
    return PathStats(ret, vol, mae, mfe, 21, "closes")


def test_implied_move_numbers() -> None:
    o = implied_move_outcome(stats(), atm_iv=0.25, horizon_bars=21)
    expected = 0.25 * math.sqrt(21 / 252)
    assert o.expected_move == pytest.approx(expected)
    assert o.abs_return_to_expected == pytest.approx(0.05 / expected)
    assert o.excursion_to_expected == pytest.approx(0.08 / expected)
    assert o.realized_to_implied_vol == pytest.approx(1.2)
    assert o.amplified is False
    assert o.version == OUTCOME_VERSION == "outcome-v0-uncalibrated"


def test_implied_move_realized_vol_missing() -> None:
    o = implied_move_outcome(stats(vol=None), 0.25, 21)
    assert o.realized_to_implied_vol is None


def test_implied_move_amplified_boundary_counts() -> None:
    # iv 0.5 over 252 bars -> expected move 0.5 exactly; 0.75 / 0.5 == 1.5 exactly.
    at = implied_move_outcome(stats(mae=-0.1, mfe=0.75), 0.5, 252)
    assert at.excursion_to_expected == 1.5 and at.amplified is True
    at_adverse = implied_move_outcome(stats(mae=-0.75, mfe=0.1), 0.5, 252)
    assert at_adverse.amplified is True
    below = implied_move_outcome(stats(mae=-0.1, mfe=0.74), 0.5, 252)
    assert below.amplified is False


def test_implied_move_custom_ratio() -> None:
    assert implied_move_outcome(stats(mae=-0.1, mfe=0.5), 0.5, 252, amplification_ratio=1.0).amplified is True
    assert implied_move_outcome(stats(mae=-0.1, mfe=0.5), 0.5, 252, amplification_ratio=1.5).amplified is False


def test_implied_move_validation() -> None:
    with pytest.raises(ValueError):
        implied_move_outcome(stats(), 0.0, 21)
    with pytest.raises(ValueError):
        implied_move_outcome(stats(), -0.2, 21)
    with pytest.raises(ValueError):
        implied_move_outcome(stats(), 0.2, 0)


# --------------------------------------------------------------- horizon_bars_for


def test_horizon_bars_for() -> None:
    bars = ramp(20)
    w = locate_window(bars, day(1), "EXPIRY", expiry=day(8))
    assert w is not None
    assert horizon_bars_for("EXPIRY", w) == 7
    for kind, n in HORIZON_BARS.items():
        assert horizon_bars_for(kind, w) == n
    assert horizon_bars_for("EOD", w) == 0
    assert horizon_bars_for("NEXT_SESSION", w) == 1
    with pytest.raises(ValueError):
        horizon_bars_for("BOGUS", w)


# ------------------------------------------------------------------ no look-ahead


def test_no_look_ahead_after_end_pos() -> None:
    n = 40
    closes = [100.0 + i + (i % 3) for i in range(n)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    bars = make_bars(closes, highs, lows)
    bench = make_bars([300.0 + 2 * i for i in range(n)])
    w = locate_window(bars, day(2), "W1")
    assert w is not None

    altered = bars.copy()
    altered.iloc[w.end_pos + 1 :, :] = altered.iloc[w.end_pos + 1 :, :] * 7.0
    altered_bench = bench.copy()
    altered_bench.iloc[w.end_pos + 1 :, :] = 1.0

    assert path_stats(altered, w) == path_stats(bars, w)
    for side, level in (("call", 106.0), ("put", 99.0)):
        assert wall_outcome(altered, w, level, side) == wall_outcome(bars, w, level, side)  # type: ignore[arg-type]
    assert benchmark_return(altered_bench, w) == benchmark_return(bench, w)

    # Truncating the data right at end_pos gives the same answers as well.
    cut = bars.iloc[: w.end_pos + 1]
    assert path_stats(cut, w) == path_stats(bars, w)
    assert wall_outcome(cut, w, 106.0, "call") == wall_outcome(bars, w, 106.0, "call")
    assert benchmark_return(bench.iloc[: w.end_pos + 1], w) == benchmark_return(bench, w)
