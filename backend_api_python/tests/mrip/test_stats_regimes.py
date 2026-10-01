"""VIX regimes: boundaries, threshold validation, segmentation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.mrip.stats.regimes import (
    Regime,
    RegimeThresholds,
    by_regime,
    classify_vix,
    segment,
)


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def test_regime_values_equal_names() -> None:
    assert [r.value for r in Regime] == [r.name for r in Regime]
    assert [r.name for r in Regime] == ["LOW_VOL", "NORMAL", "ELEVATED", "STRESS", "CRISIS"]


def test_classify_vix_boundaries() -> None:
    levels = [10.0, 14.99, 15.0, 19.99, 20.0, 29.99, 30.0, 39.99, 40.0, 80.0]
    expected = [
        Regime.LOW_VOL, Regime.LOW_VOL, Regime.NORMAL, Regime.NORMAL, Regime.ELEVATED,
        Regime.ELEVATED, Regime.STRESS, Regime.STRESS, Regime.CRISIS, Regime.CRISIS,
    ]
    out = classify_vix(pd.Series(levels, index=_dates(len(levels))))
    assert out.dtype == object
    assert list(out) == expected


def test_classify_vix_nan_stays_nan_and_index_kept() -> None:
    idx = _dates(3)
    out = classify_vix(pd.Series([12.0, np.nan, 25.0], index=idx))
    assert out.index.equals(idx)
    assert out.iloc[0] is Regime.LOW_VOL
    assert pd.isna(out.iloc[1])
    assert out.iloc[2] is Regime.ELEVATED


def test_classify_vix_custom_thresholds() -> None:
    t = RegimeThresholds(low_normal=10.0, normal_elevated=12.0, elevated_stress=14.0, stress_crisis=16.0)
    out = classify_vix(pd.Series([9.0, 10.0, 12.0, 14.0, 16.0], index=_dates(5)), t)
    assert list(out) == [Regime.LOW_VOL, Regime.NORMAL, Regime.ELEVATED, Regime.STRESS, Regime.CRISIS]


def test_thresholds_defaults_and_version() -> None:
    t = RegimeThresholds()
    assert (t.low_normal, t.normal_elevated, t.elevated_stress, t.stress_crisis) == (15.0, 20.0, 30.0, 40.0)
    assert t.version == "v0-uncalibrated"


@pytest.mark.parametrize(
    "cuts",
    [
        (20.0, 20.0, 30.0, 40.0),  # equal
        (25.0, 20.0, 30.0, 40.0),  # decreasing
        (15.0, 20.0, 40.0, 30.0),
        (15.0, 20.0, 30.0, 30.0),
    ],
)
def test_thresholds_must_be_strictly_increasing(cuts: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        RegimeThresholds(*cuts)


def test_thresholds_are_frozen() -> None:
    with pytest.raises(AttributeError):
        RegimeThresholds().low_normal = 1.0  # type: ignore[misc]


def _frame_and_regimes() -> tuple[pd.DataFrame, pd.Series]:
    idx = _dates(10)
    data = pd.DataFrame({"x": np.arange(10, dtype=float)}, index=idx)
    regimes = pd.Series(
        [Regime.NORMAL, Regime.NORMAL, Regime.STRESS, np.nan, Regime.LOW_VOL,
         Regime.NORMAL, Regime.STRESS, Regime.LOW_VOL, Regime.NORMAL, Regime.NORMAL],
        index=idx,
        dtype=object,
    )
    return data, regimes


def test_segment_groups_rows_and_drops_nan_regime() -> None:
    data, regimes = _frame_and_regimes()
    seg = segment(data, regimes)
    assert list(seg) == [Regime.LOW_VOL, Regime.NORMAL, Regime.STRESS]  # declaration order
    assert seg[Regime.LOW_VOL]["x"].tolist() == [4.0, 7.0]
    assert seg[Regime.NORMAL]["x"].tolist() == [0.0, 1.0, 5.0, 8.0, 9.0]
    assert seg[Regime.STRESS]["x"].tolist() == [2.0, 6.0]
    assert sum(len(v) for v in seg.values()) == 9  # row 3 (NaN regime) dropped


def test_segment_uses_common_index_only() -> None:
    data, regimes = _frame_and_regimes()
    seg = segment(data.iloc[:5], regimes.iloc[2:])  # common rows 2..4
    assert seg[Regime.STRESS]["x"].tolist() == [2.0]
    assert seg[Regime.LOW_VOL]["x"].tolist() == [4.0]
    assert Regime.NORMAL not in seg


def test_by_regime_min_obs_and_all_members_present() -> None:
    data, regimes = _frame_and_regimes()
    out = by_regime(data, regimes, lambda d: float(d["x"].sum()), min_obs=3)
    assert list(out) == list(Regime)
    assert out[Regime.NORMAL] == 0.0 + 1.0 + 5.0 + 8.0 + 9.0
    assert out[Regime.LOW_VOL] is None  # 2 rows < 3
    assert out[Regime.STRESS] is None
    assert out[Regime.ELEVATED] is None  # no rows at all
    assert out[Regime.CRISIS] is None


def test_by_regime_min_obs_boundary_is_inclusive() -> None:
    data, regimes = _frame_and_regimes()
    out = by_regime(data, regimes, lambda d: float(d["x"].mean()), min_obs=2)
    assert out[Regime.LOW_VOL] == pytest.approx(5.5)
    assert out[Regime.STRESS] == pytest.approx(4.0)
    assert out[Regime.NORMAL] == pytest.approx(4.6)


def test_by_regime_default_min_obs_is_30() -> None:
    idx = _dates(60)
    data = pd.DataFrame({"x": np.ones(60)}, index=idx)
    regimes = pd.Series([Regime.NORMAL] * 30 + [Regime.STRESS] * 29 + [Regime.CRISIS], index=idx, dtype=object)
    out = by_regime(data, regimes, lambda d: float(len(d)))
    assert out[Regime.NORMAL] == 30.0
    assert out[Regime.STRESS] is None  # 29 < 30
    assert out[Regime.CRISIS] is None
