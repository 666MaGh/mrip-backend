"""Live end-to-end: historical forecasts on real SPY -> log -> resolve from CBOE data -> calibration rows.

Opt in with MRIP_LIVE_OPENBB=1 and MRIP_TEST_DB=1 (throwaway DATABASE_URL). Prints the reliability table.
"""
import os
from datetime import date

import pytest

from app.mrip.data.openbb_adapter import OpenBBAdapter
from app.mrip.forecast.baselines import NaiveBaseline
from app.mrip.forecast.types import ForecastRequest
from app.mrip.outcomes.labels import forecast_calibration_rows, reliability_table
from app.mrip.outcomes.predictions import forecast_prediction
from app.mrip.outcomes.service import OutcomeService, bars_from_price_series
from app.mrip.outcomes.store import PredictionStore
from app.mrip.outcomes.types import PredictionType
from app.mrip.stats.service import prices_to_series

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_LIVE_OPENBB") != "1" or os.getenv("MRIP_TEST_DB") != "1", reason="needs live OpenBB and a test DB"),
]


def test_historical_forecasts_resolve_against_live_spy_data():
    from app.utils import db
    from pathlib import Path

    migration = Path(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_outcomes.sql"
    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(migration.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_outcomes, mrip_predictions RESTART IDENTITY CASCADE")
        conn.commit()
        cur.close()
    store = PredictionStore(db.get_db_connection)
    gateway = OpenBBAdapter()
    series = gateway.price_history("SPY", start=date(2018, 1, 1))
    prices = prices_to_series(series)
    provider = NaiveBaseline()

    origins = list(range(len(prices) - 21 * 40 - 1, len(prices) - 22, 21))  # ~40 monthly origins, the last fully observable
    for i in origins:
        history = prices.iloc[: i + 1]
        store.log(forecast_prediction(provider.forecast(ForecastRequest("SPY", history, 21))))

    report = OutcomeService(store, gateway).resolve_pending()
    assert len(report.resolved) == len(origins) and report.still_pending == []

    bars = bars_from_price_series(series)
    for (pred, out) in store.resolved_pairs(PredictionType.FORECAST)[:3]:
        start = prices.index.get_loc(__import__("pandas").Timestamp(out.window_start))
        assert out.actual_return == pytest.approx(prices.iloc[start + 21] / prices.iloc[start] - 1)
        assert out.window_end == prices.index[start + 21].date() and out.benchmark_return == pytest.approx(out.actual_return)
        assert out.max_adverse_excursion <= 0 <= out.max_favorable_excursion and out.realized_volatility > 0

    rows = forecast_calibration_rows(store.resolved_pairs(PredictionType.FORECAST))
    table = reliability_table(rows)
    print(f"\nNaive monthly SPY forecasts resolved: {len(rows)} ({rows[0].made_at.date()} .. {rows[-1].made_at.date()}); bars {len(bars)}")
    for q, b in table.items():
        print(f"  P{int(q * 100):>2}: nominal {b.nominal:.2f}  observed {b.observed:.2f}  (n={b.n})")
    assert all(0.0 <= b.observed <= 1.0 for b in table.values())
