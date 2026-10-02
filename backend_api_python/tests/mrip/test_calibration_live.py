"""Live: calibrate monthly SPY forecasts from resolved outcomes and score the effect out of sample.

Opt in with MRIP_LIVE_OPENBB=1 and MRIP_TEST_DB=1 (throwaway DATABASE_URL). Prints before/after.
"""
import os
from datetime import date
from pathlib import Path

import pytest

from app.mrip.calibration.forecast import evaluate_forecast_calibration
from app.mrip.data.openbb_adapter import OpenBBAdapter
from app.mrip.forecast.baselines import NaiveBaseline, StatisticalBaseline
from app.mrip.forecast.types import ForecastRequest, QUANTILES
from app.mrip.outcomes.predictions import forecast_prediction
from app.mrip.outcomes.service import OutcomeService
from app.mrip.outcomes.store import PredictionStore
from app.mrip.outcomes.types import PredictionType
from app.mrip.stats.service import prices_to_series

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_LIVE_OPENBB") != "1" or os.getenv("MRIP_TEST_DB") != "1", reason="needs live OpenBB and a test DB"),
]


def test_calibration_effect_on_real_spy_forecasts_out_of_sample():
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute((Path(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_outcomes.sql").read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_outcomes, mrip_predictions RESTART IDENTITY CASCADE")
        conn.commit()
        cur.close()
    store, gateway = PredictionStore(db.get_db_connection), OpenBBAdapter()
    series = gateway.price_history("SPY", start=date(2016, 1, 1))
    prices = prices_to_series(series)
    providers = [NaiveBaseline(), StatisticalBaseline(lookback=750)]
    origins = list(range(749, len(prices) - 22, 21))
    for i in origins:
        for provider in providers:
            store.log(forecast_prediction(provider.forecast(ForecastRequest("SPY", prices.iloc[: i + 1], 21))))
    OutcomeService(store, gateway).resolve_pending()
    pairs = store.resolved_pairs(PredictionType.FORECAST)
    print(f"\nresolved monthly SPY forecasts: {len(pairs)} ({len(origins)} origins x {len(providers)} models)")
    for model in sorted({p.model_version for p, _ in pairs}):
        subset = [(p, o) for p, o in pairs if p.model_version == model]
        ev = evaluate_forecast_calibration(subset, min_samples=25, train_fraction=0.6)
        print(f"{model[:40]:40} train {ev.n_train} test {ev.n_test}  pinball {ev.pinball_before:.5f} -> {ev.pinball_after:.5f}"
              f"  cov80 {ev.coverage_p10_p90_before:.2f} -> {ev.coverage_p10_p90_after:.2f}"
              f"  cov50 {ev.coverage_p25_p75_before:.2f} -> {ev.coverage_p25_p75_after:.2f}")
        print("   hit freq raw  :", " ".join(f"P{int(q*100)}={ev.hit_before[q]:.2f}" for q in QUANTILES))
        print("   hit freq calib:", " ".join(f"P{int(q*100)}={ev.hit_after[q]:.2f}" for q in QUANTILES))
        assert ev.n_calibrated_test > 10
