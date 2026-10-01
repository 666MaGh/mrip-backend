"""Walk-forward forecast evaluation: metrics, origin geometry, skipping, no look-ahead (work 008)."""
from __future__ import annotations

from datetime import date
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import pytest

from app.mrip.forecast.evaluation import evaluate_provider, interval_covered, pinball_loss
from app.mrip.forecast.types import QUANTILES, ForecastRequest, ForecastResult, ForecastUnavailable

H = 5
N = 30
MIN_HISTORY = 10


# --------------------------------------------------------------------------- helpers


def _index(n: int = N) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _growth_prices(n: int = N, daily: float = 0.01) -> pd.Series:
    return pd.Series(100.0 * (1.0 + daily) ** np.arange(n), index=_index(n))


def _flat_quantiles(value: float) -> dict[float, float]:
    return {q: value for q in QUANTILES}


def _grid(median: float, wide: float = 0.05, narrow: float = 0.01) -> dict[float, float]:
    """Symmetric quantile grid around ``median``: P10/P90 = +-wide, P25/P75 = +-narrow."""
    mid = (wide + narrow) / 2.0
    return {
        0.1: median - wide,
        0.2: median - mid,
        0.25: median - narrow,
        0.5: median,
        0.75: median + narrow,
        0.8: median + mid,
        0.9: median + wide,
    }


def _result(request: ForecastRequest, quantiles: Mapping[float, float]) -> ForecastResult:
    return ForecastResult(
        provider="stub",
        provider_version="stub-1",
        symbol=request.symbol,
        as_of=request.as_of,
        horizon_days=request.horizon_days,
        last_price=request.last_price,
        return_quantiles=dict(quantiles),
        n_obs=len(request.prices),
    )


class ConstantProvider:
    name = "constant"

    def __init__(self, quantiles: Mapping[float, float]) -> None:
        self._quantiles = dict(quantiles)

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        return _result(request, self._quantiles)

    def version(self) -> str:
        return "constant-1"


class ScriptedProvider:
    """Returns the n-th scripted quantile grid on the n-th call."""

    def __init__(self, script: list[Mapping[float, float]]) -> None:
        self._script = script
        self.calls = 0

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        quantiles = self._script[self.calls]
        self.calls += 1
        return _result(request, quantiles)

    def version(self) -> str:
        return "scripted-1"


class SpyProvider:
    """Records what it was shown at every origin."""

    name = "spy"

    def __init__(self) -> None:
        self.max_dates: list[pd.Timestamp] = []
        self.min_dates: list[pd.Timestamp] = []
        self.n_prices: list[int] = []
        self.covariate_lens: list[int | None] = []
        self.covariate_max_dates: list[pd.Timestamp | None] = []
        self.last_prices: list[float] = []

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        self.max_dates.append(request.prices.index.max())
        self.min_dates.append(request.prices.index.min())
        self.n_prices.append(len(request.prices))
        self.last_prices.append(request.last_price)
        if request.covariates is None:
            self.covariate_lens.append(None)
            self.covariate_max_dates.append(None)
        else:
            self.covariate_lens.append(len(request.covariates))
            self.covariate_max_dates.append(request.covariates.index.max())
        return _result(request, _flat_quantiles(0.0))

    def version(self) -> str:
        return "spy-1"


class FlakyProvider:
    """Raises ForecastUnavailable whenever ``unavailable(len(prices))`` is true."""

    name = "flaky"

    def __init__(self, unavailable: Callable[[int], bool]) -> None:
        self._unavailable = unavailable

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        if self._unavailable(len(request.prices)):
            raise ForecastUnavailable("scripted outage")
        return _result(request, _grid(0.0))

    def version(self) -> str:
        return "flaky-1"


class NamelessProvider:
    def forecast(self, request: ForecastRequest) -> ForecastResult:
        return _result(request, _grid(0.0))

    def version(self) -> str:
        return "nameless-7"


# --------------------------------------------------------------------------- pinball loss


def test_pinball_loss_known_case() -> None:
    # actual 0.05; q0.1=-0.02 -> 0.1*0.07 = 0.007; q0.5=0.01 -> 0.5*0.04 = 0.02; q0.9=0.04 -> 0.9*0.01 = 0.009
    loss = pinball_loss(0.05, {0.1: -0.02, 0.5: 0.01, 0.9: 0.04})
    assert loss == pytest.approx((0.007 + 0.02 + 0.009) / 3.0)
    assert loss == pytest.approx(0.012)


def test_pinball_loss_actual_below_forecast() -> None:
    # actual -0.03; q0.9=0.04 -> (0.9-1)*(-0.07) = 0.007; q0.5=0.0 -> 0.5*0.03 = 0.015
    loss = pinball_loss(-0.03, {0.5: 0.0, 0.9: 0.04})
    assert loss == pytest.approx((0.015 + 0.007) / 2.0)


def test_pinball_loss_is_zero_when_forecast_is_exact() -> None:
    assert pinball_loss(0.02, {0.1: 0.02, 0.5: 0.02, 0.9: 0.02}) == 0.0


def test_pinball_loss_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError):
        pinball_loss(0.0, {})


# --------------------------------------------------------------------------- coverage


def test_interval_covered_boundaries_are_inclusive() -> None:
    assert interval_covered(0.1, 0.1, 0.2) is True
    assert interval_covered(0.2, 0.1, 0.2) is True
    assert interval_covered(0.15, 0.1, 0.2) is True


def test_interval_covered_outside() -> None:
    assert interval_covered(0.0999, 0.1, 0.2) is False
    assert interval_covered(0.2001, 0.1, 0.2) is False


def test_interval_covered_degenerate_interval() -> None:
    assert interval_covered(0.05, 0.05, 0.05) is True
    assert interval_covered(0.06, 0.05, 0.05) is False


# --------------------------------------------------------------------------- scripted metrics

# prices -> actual simple returns at h=1, min_history=2 are [0.1, -0.1, 0.0, 0.1] for origins 1..4
_SCRIPTED_PRICES = pd.Series([100.0, 110.0, 121.0, 108.9, 108.9, 119.79], index=_index(6))
_SCRIPTED_ACTUALS = [0.1, -0.1, 0.0, 0.1]
_SCRIPTED_MEDIANS = [0.08, 0.0, 0.0, -0.05]


def _scripted_report():  # type: ignore[no-untyped-def]
    provider = ScriptedProvider([_grid(m) for m in _SCRIPTED_MEDIANS])
    return evaluate_provider(provider, _SCRIPTED_PRICES, 1, min_history=2)


def test_scripted_origin_count_and_dates() -> None:
    report = _scripted_report()
    idx = _SCRIPTED_PRICES.index
    assert report.n_forecasts == 4
    assert report.first_origin == idx[1].date()
    assert report.last_origin == idx[4].date()
    assert report.horizon_days == 1


def test_scripted_coverage() -> None:
    report = _scripted_report()
    # P10-P90 covers origin 0 (0.1 in [0.03, 0.13]) and origin 2 (0.0 in [-0.05, 0.05]) -> 2/4
    assert report.coverage_p10_p90 == pytest.approx(0.5)
    # P25-P75 covers only origin 2 (0.0 in [-0.01, 0.01]) -> 1/4
    assert report.coverage_p25_p75 == pytest.approx(0.25)


def test_scripted_median_abs_error() -> None:
    report = _scripted_report()
    # |0.1-0.08| + |-0.1-0| + |0-0| + |0.1+0.05| = 0.02 + 0.1 + 0 + 0.15 = 0.27 -> /4
    assert report.median_abs_error == pytest.approx(0.0675)


def test_scripted_directional_accuracy_ignores_zeros() -> None:
    report = _scripted_report()
    # origin 0: pred + / actual + hit; origin 1: pred 0 ignored; origin 2: both 0 ignored;
    # origin 3: pred - / actual + miss -> 1 of 2
    assert report.directional_accuracy == pytest.approx(0.5)


def test_scripted_mean_pinball_loss() -> None:
    report = _scripted_report()
    expected = np.mean([pinball_loss(a, _grid(m)) for a, m in zip(_SCRIPTED_ACTUALS, _SCRIPTED_MEDIANS)])
    assert report.mean_pinball_loss == pytest.approx(float(expected))


def test_directional_accuracy_none_when_no_comparable_origin() -> None:
    # median forecast is always exactly zero -> no origin has a non-zero predicted sign
    prices = _growth_prices()
    report = evaluate_provider(ConstantProvider(_grid(0.0)), prices, H, min_history=MIN_HISTORY)
    assert report.directional_accuracy is None


def test_directional_accuracy_none_when_actuals_are_all_zero() -> None:
    prices = pd.Series(100.0, index=_index())
    report = evaluate_provider(ConstantProvider(_grid(0.01)), prices, H, min_history=MIN_HISTORY)
    assert report.directional_accuracy is None


def test_directional_accuracy_perfect_and_zero() -> None:
    up = _growth_prices(daily=0.01)
    down = _growth_prices(daily=-0.01)
    assert evaluate_provider(ConstantProvider(_grid(0.02)), up, H, min_history=MIN_HISTORY).directional_accuracy == 1.0
    assert evaluate_provider(ConstantProvider(_grid(0.02)), down, H, min_history=MIN_HISTORY).directional_accuracy == 0.0


# --------------------------------------------------------------------------- realised outcome


def test_realised_outcome_on_exact_growth_series() -> None:
    # Every quantile is 0.01, so for actual a = 1.01**5 - 1 (> 0.01):
    #   pinball = mean(tau) * (a - 0.01) = 0.5 * (a - 0.01); the degenerate interval never covers a;
    #   median abs error = a - 0.01; the median sign (+) always matches the realised sign (+).
    actual = 1.01**H - 1.0
    report = evaluate_provider(ConstantProvider(_flat_quantiles(0.01)), _growth_prices(), H, min_history=MIN_HISTORY)
    assert report.n_forecasts == 4
    assert report.mean_pinball_loss == pytest.approx(0.5 * (actual - 0.01), abs=1e-12)
    assert report.median_abs_error == pytest.approx(actual - 0.01, abs=1e-12)
    assert report.coverage_p10_p90 == 0.0
    assert report.coverage_p25_p75 == 0.0
    assert report.directional_accuracy == 1.0


def test_realised_outcome_equals_growth_when_forecast_is_exact() -> None:
    actual = 1.01**H - 1.0
    report = evaluate_provider(ConstantProvider(_flat_quantiles(actual)), _growth_prices(), H, min_history=MIN_HISTORY)
    assert report.mean_pinball_loss == pytest.approx(0.0, abs=1e-12)
    assert report.median_abs_error == pytest.approx(0.0, abs=1e-12)


def test_coverage_boundaries_inclusive_end_to_end() -> None:
    # Flat prices -> realised return is exactly 0.0, which sits on both edges of the degenerate interval.
    prices = pd.Series(100.0, index=_index())
    report = evaluate_provider(ConstantProvider(_flat_quantiles(0.0)), prices, H, min_history=MIN_HISTORY)
    assert report.coverage_p10_p90 == 1.0
    assert report.coverage_p25_p75 == 1.0
    assert report.mean_pinball_loss == 0.0


def test_realised_outcome_uses_horizon_rows_after_origin() -> None:
    # Single origin at position 1 with h=3: outcome = prices[4] / prices[1] - 1 = 150 / 100 - 1 = 0.5
    prices = pd.Series([90.0, 100.0, 7.0, 9.0, 150.0], index=_index(5))
    report = evaluate_provider(ConstantProvider(_flat_quantiles(0.2)), prices, 3, min_history=2)
    assert report.n_forecasts == 1
    assert report.median_abs_error == pytest.approx(0.3)


# --------------------------------------------------------------------------- geometry


def test_geometry_default_stride_is_horizon() -> None:
    idx = _index()
    report = evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY)
    # origins at positions 9, 14, 19, 24 (24 + 5 = 29 = last position)
    assert report.n_forecasts == 4
    assert report.first_origin == idx[9].date()
    assert report.last_origin == idx[24].date()
    assert report.notes == ()


def test_geometry_step_one() -> None:
    idx = _index()
    report = evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY, step=1)
    # origins 9..24 inclusive
    assert report.n_forecasts == 16
    assert report.first_origin == idx[9].date()
    assert report.last_origin == idx[24].date()
    assert "overlapping outcomes: step < horizon" in report.notes


def test_geometry_step_smaller_than_horizon_but_larger_than_one() -> None:
    idx = _index()
    report = evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY, step=3)
    # origins 9, 12, 15, 18, 21, 24
    assert report.n_forecasts == 6
    assert report.last_origin == idx[24].date()
    assert report.notes == ("overlapping outcomes: step < horizon",)


def test_geometry_step_larger_than_horizon_has_no_overlap_note() -> None:
    idx = _index()
    report = evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY, step=10)
    # origins 9, 19 (29 + 5 > last position)
    assert report.n_forecasts == 2
    assert report.last_origin == idx[19].date()
    assert report.notes == ()


def test_geometry_step_equal_to_horizon_matches_default() -> None:
    default = evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY)
    explicit = evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY, step=H)
    assert default == explicit


def test_geometry_exact_fit_last_origin_is_included() -> None:
    # 10 prices, min_history 5, h=5: only origin 4 (4 + 5 = 9 = last position)
    prices = _growth_prices(10)
    report = evaluate_provider(ConstantProvider(_grid(0.0)), prices, 5, min_history=5)
    assert report.n_forecasts == 1
    assert report.first_origin == report.last_origin == prices.index[4].date()


def test_origin_dates_are_datetime_dates() -> None:
    report = evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY)
    assert type(report.first_origin) is date
    assert type(report.last_origin) is date


# --------------------------------------------------------------------------- report metadata


def test_report_uses_provider_name_and_version() -> None:
    report = evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY)
    assert report.provider == "constant"
    assert report.provider_version == "constant-1"


def test_report_falls_back_to_class_name_without_name_attribute() -> None:
    report = evaluate_provider(NamelessProvider(), _growth_prices(), H, min_history=MIN_HISTORY)
    assert report.provider == "NamelessProvider"
    assert report.provider_version == "nameless-7"


# --------------------------------------------------------------------------- skipping


def test_unavailable_origins_are_skipped_and_noted() -> None:
    idx = _index()
    # origins at positions 9, 14, 19, 24 see 10, 15, 20, 25 prices; make 15 and 25 unavailable
    provider = FlakyProvider(lambda n_prices: n_prices in (15, 25))
    report = evaluate_provider(provider, _growth_prices(), H, min_history=MIN_HISTORY)
    assert report.n_forecasts == 2
    assert report.first_origin == idx[9].date()
    assert report.last_origin == idx[19].date()
    assert report.notes == ("2 origin(s) skipped: provider unavailable",)


def test_single_skipped_origin_note_and_overlap_note_combined() -> None:
    provider = FlakyProvider(lambda n_prices: n_prices == 10)
    report = evaluate_provider(provider, _growth_prices(), H, min_history=MIN_HISTORY, step=5 - 1)
    # stride 4 < 5; origins 9, 13, 17, 21, 25 (25 + 5 = 30 > 29, so 9..21): first skipped
    assert report.n_forecasts == 3
    assert report.notes == (
        "1 origin(s) skipped: provider unavailable",
        "overlapping outcomes: step < horizon",
    )


def test_all_origins_unavailable_raises() -> None:
    provider = FlakyProvider(lambda n_prices: True)
    with pytest.raises(ValueError, match="no evaluable origins"):
        evaluate_provider(provider, _growth_prices(), H, min_history=MIN_HISTORY)


def test_other_provider_exceptions_propagate() -> None:
    class Broken(ConstantProvider):
        def forecast(self, request: ForecastRequest) -> ForecastResult:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        evaluate_provider(Broken(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY)


# --------------------------------------------------------------------------- no look-ahead


def test_spy_never_sees_data_after_origin() -> None:
    idx = _index()
    prices = _growth_prices()
    spy = SpyProvider()
    evaluate_provider(spy, prices, H, min_history=MIN_HISTORY, step=1)
    origins = [idx[i] for i in range(9, 25)]
    assert spy.max_dates == origins  # the newest date seen is exactly the origin date, never later
    assert all(seen <= origin for seen, origin in zip(spy.max_dates, origins))
    assert all(start == idx[0] for start in spy.min_dates)
    assert spy.n_prices == [i + 1 for i in range(9, 25)]
    # the last visible price is the origin close, never the future outcome close
    assert spy.last_prices == [float(prices.iloc[i]) for i in range(9, 25)]


def test_spy_max_date_never_exceeds_last_evaluated_origin_with_default_stride() -> None:
    idx = _index()
    spy = SpyProvider()
    report = evaluate_provider(spy, _growth_prices(), H, min_history=MIN_HISTORY)
    assert report.last_origin == idx[24].date()
    assert max(spy.max_dates) == idx[24]
    assert max(spy.max_dates) < idx[-1]  # the final rows are only ever used as outcomes


def test_spy_covariates_are_truncated_to_origin() -> None:
    idx = _index()
    covariates = pd.DataFrame({"x": np.arange(N, dtype=float)}, index=idx)
    spy = SpyProvider()
    evaluate_provider(spy, _growth_prices(), H, min_history=MIN_HISTORY, step=1, covariates=covariates)
    assert spy.covariate_lens == [i + 1 for i in range(9, 25)]
    assert spy.covariate_max_dates == [idx[i] for i in range(9, 25)]


def test_spy_covariates_longer_than_prices_are_still_truncated() -> None:
    idx = _index()
    long_idx = pd.bdate_range(idx[0], periods=N + 10)
    covariates = pd.DataFrame({"x": np.arange(N + 10, dtype=float)}, index=long_idx)
    spy = SpyProvider()
    evaluate_provider(spy, _growth_prices(), H, min_history=MIN_HISTORY, covariates=covariates)
    assert spy.covariate_lens == [10, 15, 20, 25]
    assert all(c is not None and c <= m for c, m in zip(spy.covariate_max_dates, spy.max_dates))


def test_spy_without_covariates_sees_none() -> None:
    spy = SpyProvider()
    evaluate_provider(spy, _growth_prices(), H, min_history=MIN_HISTORY)
    assert spy.covariate_lens == [None] * 4


def test_future_prices_do_not_change_earlier_forecast_inputs() -> None:
    # Two series identical up to position 24 but different afterwards: the provider's view at every
    # origin <= 24 must be identical.
    base = _growth_prices()
    altered = base.copy()
    altered.iloc[25:] = altered.iloc[25:] * 3.0
    spy_a, spy_b = SpyProvider(), SpyProvider()
    evaluate_provider(spy_a, base, H, min_history=MIN_HISTORY, step=1)
    evaluate_provider(spy_b, altered, H, min_history=MIN_HISTORY, step=1)
    assert spy_a.last_prices[:16] == spy_b.last_prices[:16]
    assert spy_a.n_prices == spy_b.n_prices


# --------------------------------------------------------------------------- argument validation


def test_invalid_horizon_days() -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), 0, min_history=MIN_HISTORY)


def test_invalid_min_history() -> None:
    with pytest.raises(ValueError, match="min_history"):
        evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=1)


def test_invalid_step() -> None:
    with pytest.raises(ValueError, match="step"):
        evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=MIN_HISTORY, step=0)


def test_non_positive_prices() -> None:
    prices = _growth_prices()
    prices.iloc[3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        evaluate_provider(ConstantProvider(_grid(0.0)), prices, H, min_history=MIN_HISTORY)
    prices.iloc[3] = -1.0
    with pytest.raises(ValueError, match="positive"):
        evaluate_provider(ConstantProvider(_grid(0.0)), prices, H, min_history=MIN_HISTORY)


def test_empty_prices() -> None:
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    with pytest.raises(ValueError):
        evaluate_provider(ConstantProvider(_grid(0.0)), empty, H, min_history=MIN_HISTORY)


def test_unsorted_prices() -> None:
    prices = _growth_prices().iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        evaluate_provider(ConstantProvider(_grid(0.0)), prices, H, min_history=MIN_HISTORY)


def test_series_too_short_has_no_evaluable_origins() -> None:
    # min_history 10 -> first origin at position 9; 9 + 5 > 13
    with pytest.raises(ValueError, match="no evaluable origins"):
        evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(14), H, min_history=MIN_HISTORY)


def test_min_history_longer_than_series_has_no_evaluable_origins() -> None:
    with pytest.raises(ValueError, match="no evaluable origins"):
        evaluate_provider(ConstantProvider(_grid(0.0)), _growth_prices(), H, min_history=500)


def test_validation_errors_do_not_call_provider() -> None:
    spy = SpyProvider()
    with pytest.raises(ValueError):
        evaluate_provider(spy, _growth_prices(), 0, min_history=MIN_HISTORY)
    assert spy.n_prices == []
