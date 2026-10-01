"""Live forecast checks against a running TimesFM service and real SPY history.

Opt in with MRIP_LIVE_TIMESFM_URL (service) and MRIP_LIVE_OPENBB=1 (SPY history).
The comparison is printed (run with -s); it is evidence, not a pass/fail claim about skill.
"""
import os
from datetime import date

import pandas as pd
import pytest

from app.mrip.data.openbb_adapter import OpenBBAdapter
from app.mrip.forecast.baselines import NaiveBaseline, StatisticalBaseline
from app.mrip.forecast.ensemble import EnsembleProvider
from app.mrip.forecast.timesfm import TimesFMProvider
from app.mrip.forecast.types import ForecastRequest
from app.mrip.stats.service import prices_to_series

URL = os.getenv("MRIP_LIVE_TIMESFM_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not URL or os.getenv("MRIP_LIVE_OPENBB") != "1", reason="needs MRIP_LIVE_TIMESFM_URL and MRIP_LIVE_OPENBB=1"),
]


@pytest.fixture(scope="module")
def spy() -> pd.Series:
    return prices_to_series(OpenBBAdapter().price_history("SPY", start=date(2016, 1, 1)))


def test_timesfm_forecasts_real_spy_prices(spy):
    provider = TimesFMProvider(URL, timeout=600)
    result = provider.forecast(ForecastRequest("SPY", spy, 21))
    q = result.return_quantiles
    assert q[0.1] < q[0.5] < q[0.9] and -0.5 < q[0.1] and q[0.9] < 0.5
    print(f"\nSPY as_of {result.as_of} last {result.last_price:.2f}: P10 {q[0.1]:+.3%} P50 {q[0.5]:+.3%} P90 {q[0.9]:+.3%}")


def test_walk_forward_comparison_on_spy(spy):
    print(f"\nSPY history: {len(spy)} days, {spy.index[0].date()} .. {spy.index[-1].date()}")
    timesfm = TimesFMProvider(URL, timeout=600)
    naive, stat = NaiveBaseline(), StatisticalBaseline(lookback=1000)
    members = {"naive": naive, "statistical": stat, "timesfm": timesfm}
    members["ensemble(equal)"] = EnsembleProvider([(naive, 1), (stat, 1), (timesfm, 1)])
    rows = []
    for name, provider in members.items():
        r = provider.evaluate(spy, 21, min_history=750, step=21)
        rows.append((name, r.n_forecasts, r.mean_pinball_loss, r.coverage_p10_p90, r.coverage_p25_p75, r.median_abs_error, r.directional_accuracy))
    print(f"{'provider':18}{'n':>5}{'pinball':>10}{'cov80':>8}{'cov50':>8}{'MAE50':>9}{'dir':>7}")
    for n, k, pl, c80, c50, mae, d in rows:
        print(f"{n:18}{k:>5}{pl:>10.5f}{c80:>8.2f}{c50:>8.2f}{mae:>9.4f}{(d if d is not None else float('nan')):>7.2f}")
    assert all(r[1] >= 10 for r in rows)
