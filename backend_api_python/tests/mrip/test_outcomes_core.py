"""Outcome Resolver: contract, predictions, labels (no DB) and end-to-end resolution (PostgreSQL, opt-in)."""
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path as FsPath

import numpy as np
import pandas as pd
import pytest

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import Latency, OptionContract, OptionsChainSnapshot, PriceBar, PriceSeries, Provenance
from app.mrip.forecast.types import QUANTILES, ForecastResult
from app.mrip.options.analysis import analyze_options
from app.mrip.outcomes import resolve
from app.mrip.outcomes.labels import (
    forecast_calibration_rows, options_event_rows, reliability_table, require_observed,
)
from app.mrip.outcomes.predictions import forecast_prediction, options_event_predictions
from app.mrip.outcomes.store import PredictionStore
from app.mrip.outcomes.types import (
    HorizonKind, NewPrediction, Outcome, OutcomeError, OutcomeStatus, Prediction, PredictionType,
)

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_outcomes.sql"
AUTOLOG_MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261009_prediction_log.sql"
PROV = Provenance("fake", "test", "fake.history", datetime(2026, 10, 1, tzinfo=timezone.utc), Latency.UNKNOWN)


def sql_values(column):
    sql = MIGRATION.read_text(encoding="utf-8")
    m = re.search(rf"{column}\s+VARCHAR\(\d+\)\s+NOT NULL\s+CHECK\s*\(\s*{column}\s+IN\s*\((.*?)\)\s*\)", sql, re.S)
    assert m, column
    return set(re.findall(r"'([^']+)'", m.group(1)))


def test_sql_check_constraints_match_python_enums():
    # the original migration created the first two types; mrip_20261009_prediction_log.sql extends the list
    assert sql_values("prediction_type") == {"forecast", "options_event"}
    extended = re.search(r"CHECK \(prediction_type IN \((.*?)\)\)", AUTOLOG_MIGRATION.read_text(encoding="utf-8"), re.S)
    assert extended, "prediction_type check not extended"
    assert set(re.findall(r"'([^']+)'", extended.group(1))) == {t.value for t in PredictionType}
    assert sql_values("horizon_kind") == {k.value for k in HorizonKind}
    assert sql_values("status") == {s.value for s in OutcomeStatus}
    assert {k.value for k in HorizonKind} == set(resolve.HORIZON_BARS) | {"EOD", "NEXT_SESSION", "EXPIRY"}


# -- predictions -------------------------------------------------------------

def forecast_result(horizon=21, as_of=date(2026, 9, 30)):
    q = {k: -0.05 + 0.1 * i / (len(QUANTILES) - 1) for i, k in enumerate(QUANTILES)}
    return ForecastResult("naive", "naive-v1", "SPY", as_of, horizon, 700.0, q, 60, ("w",))


def test_forecast_prediction_maps_horizon_and_made_at_after_the_close():
    p = forecast_prediction(forecast_result(63))
    assert p.horizon_kind is HorizonKind.M3 and p.prediction_type is PredictionType.FORECAST
    assert p.made_at == datetime(2026, 9, 30, 23, 0, tzinfo=timezone.utc) and p.entry_price is None
    assert set(p.payload["quantiles"]) == {str(q) for q in QUANTILES} and p.payload["scenarios"]["version"]
    assert p.model_version == "naive-v1" and p.benchmark == "SPY"
    with pytest.raises(ValueError, match="resolvable horizon"):
        forecast_prediction(forecast_result(10))


def options_analysis():
    t = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)
    prov = Provenance("cboe", "openbb", "cboe.options.chains", t, Latency.DELAYED)
    exp = date(2026, 10, 30)
    cs = []
    for i in range(60):
        k = 80 + i
        cs.append(OptionContract(f"C{k}", exp, float(k), "call", 1000, 5, 0.2, 0.5, 0.03))
        cs.append(OptionContract(f"P{k}", exp, float(k), "put", 5000, 5, 0.2, -0.5, 0.03))
    snap = OptionsChainSnapshot("SPY", 100.0, datetime(2026, 10, 1, 14, 58), t, None, tuple(cs), prov)
    return analyze_options(snap), exp


def test_options_event_predictions_carry_the_modeled_state_and_entry_price():
    analysis, exp = options_analysis()
    preds = options_event_predictions(
        analysis, horizons=(HorizonKind.EOD, HorizonKind.W1, HorizonKind.EXPIRY), expiry=exp, lineage={"snapshot_id": 7}
    )
    assert [p.horizon_kind for p in preds] == [HorizonKind.EOD, HorizonKind.W1, HorizonKind.EXPIRY]
    first = preds[0]
    assert first.entry_price == 100.0 and first.subject == "SPY" and first.made_at == analysis.data_quality.snapshot_timestamp
    assert first.payload["regime"] == analysis.modeled.regime.value and first.payload["modeled"] is True
    assert first.payload["call_wall"] == analysis.modeled.gex.walls.call_wall
    assert first.payload["atm_iv"] == pytest.approx(0.2) and first.lineage == {"snapshot_id": 7}
    assert preds[2].expiry_date == exp and preds[0].expiry_date is None
    assert "gex-v0-uncalibrated|gamma-regime-v0-uncalibrated|amplification-v0-uncalibrated" == first.model_version
    with pytest.raises(ValueError, match="EXPIRY"):
        options_event_predictions(analysis, horizons=(HorizonKind.EXPIRY,))


# -- labels ------------------------------------------------------------------

NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def pred(i, quantiles, kind=HorizonKind.M1, ptype=PredictionType.FORECAST, payload=None):
    body = payload if payload is not None else {"quantiles": quantiles}
    return Prediction(i, ptype, "SPY", NOW, kind, "m-1", body)


def outcome(i, actual, status=OutcomeStatus.RESOLVED, measures=None):
    return Outcome(i, i, status, NOW, "outcome-v0", date(2026, 1, 2), date(2026, 2, 2), actual_return=actual, measures=measures or {})


def test_calibration_rows_and_reliability_table_from_observed_outcomes():
    q = {"0.1": -0.04, "0.5": 0.0, "0.9": 0.04}
    pairs = [(pred(1, q), outcome(1, -0.05)), (pred(2, q), outcome(2, 0.01)), (pred(3, q), outcome(3, 0.06)),
             (pred(4, q), outcome(4, 0.0))]
    rows = forecast_calibration_rows(pairs)
    assert rows[0].hits == {0.1: True, 0.5: True, 0.9: True}
    assert rows[1].hits == {0.1: False, 0.5: False, 0.9: True}
    assert rows[3].hits[0.5] is True  # equal to the quantile counts as at-or-below
    table = reliability_table(rows)
    assert table[0.1].observed == pytest.approx(0.25) and table[0.5].observed == pytest.approx(0.5)
    assert table[0.9].observed == pytest.approx(0.75) and table[0.9].n == 4 and table[0.9].nominal == 0.9


def test_labels_refuse_unresolved_outcomes_and_wrong_prediction_types():
    unresolved = Outcome(9, 9, OutcomeStatus.UNRESOLVABLE, NOW, "v", note="delisted")
    with pytest.raises(OutcomeError, match="not a resolved observation"):
        require_observed(unresolved)
    with pytest.raises(OutcomeError):
        forecast_calibration_rows([(pred(9, {"0.5": 0.0}), unresolved)])
    with pytest.raises(OutcomeError, match="not a forecast"):
        forecast_calibration_rows([(pred(1, {}, ptype=PredictionType.OPTIONS_EVENT), outcome(1, 0.0))])
    with pytest.raises(OutcomeError, match="not an options event"):
        options_event_rows([(pred(1, {"0.5": 0.0}), outcome(1, 0.0))])


def test_options_event_rows_flatten_the_measures():
    payload = {"regime": "NEGATIVE_GAMMA", "direction": "AMPLIFYING", "amplification": "HIGH"}
    measures = {
        "wall_hold_or_break": {"call": "broke", "put": "untested"},
        "implied_move": {"realized_to_implied_vol": 1.4},
        "amplification_realized": {"amplified": True},
    }
    p = pred(1, {}, HorizonKind.W1, PredictionType.OPTIONS_EVENT, payload)
    o = Outcome(1, 1, OutcomeStatus.RESOLVED, NOW, "v", actual_return=0.03, max_adverse_excursion=-0.01,
                max_favorable_excursion=0.05, measures=measures)
    (row,) = options_event_rows([(p, o)])
    assert (row.regime, row.predicted_direction, row.predicted_level) == ("NEGATIVE_GAMMA", "AMPLIFYING", "HIGH")
    assert (row.call_wall, row.put_wall, row.amplified, row.realized_to_implied_vol) == ("broke", "untested", True, 1.4)
    assert (row.actual_return, row.max_adverse_excursion, row.max_favorable_excursion) == (0.03, -0.01, 0.05)


# -- store validation (no DB touched) ----------------------------------------

class _NoDb:
    def __call__(self):
        raise AssertionError("must not touch the database for invalid arguments")


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"made_at": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"subject": " "}, "required"),
        ({"horizon_kind": HorizonKind.EXPIRY}, "expiry_date"),
        ({"horizon_kind": HorizonKind.EOD}, "entry_price"),
        ({"entry_price": -1.0}, "positive"),
    ],
)
def test_log_validates_before_touching_the_database(overrides, message):
    base = dict(prediction_type=PredictionType.FORECAST, subject="SPY", made_at=NOW, horizon_kind=HorizonKind.M1,
                model_version="m", payload={})
    with pytest.raises(OutcomeError, match=message):
        PredictionStore(_NoDb()).log(NewPrediction(**{**base, **overrides}))


# -- end to end on PostgreSQL --------------------------------------------------

def make_prices(symbol, closes, start=date(2026, 6, 1), highs=None, lows=None):
    days = pd.bdate_range(start, periods=len(closes))
    bars = tuple(
        PriceBar(d.date(), c, (highs or closes)[i], (lows or closes)[i], c, 1000) for i, (d, c) in enumerate(zip(days, closes))
    )
    return PriceSeries(symbol, "1d", bars, PROV)


class FakeGateway:
    def __init__(self, series):
        self.series = series

    def price_history(self, symbol, start=None, end=None, interval="1d"):
        if symbol not in self.series:
            raise DataUnavailable(symbol)
        return self.series[symbol]


@pytest.fixture()
def store():
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_outcomes, mrip_predictions RESTART IDENTITY CASCADE")
        conn.commit()
        cur.close()
    return PredictionStore(db.get_db_connection)


def spy_closes(n, growth=0.001):
    return [100.0 * (1 + growth) ** i for i in range(n)]


@DB
def test_logging_is_idempotent_and_pending_lists_unresolved_only(store):
    p = forecast_prediction(forecast_result(5, as_of=date(2026, 6, 10)))
    a, b = store.log(p), store.log(p)
    assert a.id == b.id and [x.id for x in store.pending()] == [a.id]
    assert store.get(a.id).payload["quantiles"] == a.payload["quantiles"]
    with pytest.raises(OutcomeError):
        store.get(999)


@DB
def test_forecast_resolves_when_observable_with_exact_numbers_and_stays_immutable(store):
    from app.mrip.outcomes.service import OutcomeService

    closes = spy_closes(80)
    days = pd.bdate_range(date(2026, 6, 1), periods=80)
    as_of = days[20].date()
    pred_row = store.log(forecast_prediction(forecast_result(5, as_of=as_of)))
    gateway = FakeGateway({"SPY": make_prices("SPY", closes)})
    service = OutcomeService(store, gateway)

    early = service.resolve_pending(as_of=days[24].date())  # data ends one bar short of the 5-bar horizon
    assert early.resolved == [] and early.still_pending == [(pred_row.id, "target date not yet observed in the data")]

    report = service.resolve_pending(as_of=days[25].date())
    (out,) = report.resolved
    assert out.window_start == as_of and out.window_end == days[25].date()
    assert out.actual_return == pytest.approx(closes[25] / closes[20] - 1)
    assert out.benchmark_return == pytest.approx(out.actual_return)  # benchmark is SPY itself
    assert out.max_adverse_excursion == 0.0 and out.max_favorable_excursion == pytest.approx(closes[25] / closes[20] - 1)
    hits = out.measures["quantile_hits"]
    assert hits["0.5"] is (out.actual_return <= pred_row.payload["quantiles"]["0.5"])
    assert out.data_provenance["subject"]["provider"] == "fake" and out.data_provenance["subject"]["last_bar"] == days[25].date().isoformat()
    assert out.measures["path"]["excursions_from"] == "intraday_range"

    later = service.resolve_pending()
    assert later.resolved == [] and store.outcome_for(pred_row.id).id == out.id  # nothing left; outcome unchanged
    assert [x.id for x in store.pending()] == []
    (pair,) = store.resolved_pairs(PredictionType.FORECAST)
    assert pair[0].id == pred_row.id and pair[1].actual_return == out.actual_return
    (row,) = forecast_calibration_rows([pair])
    assert row.actual_return == out.actual_return


@DB
def test_options_event_resolves_walls_implied_move_and_amplification(store):
    from app.mrip.outcomes.service import OutcomeService

    analysis, exp = options_analysis()  # made 2026-10-01 15:00 ET, spot 100
    preds = options_event_predictions(analysis, horizons=(HorizonKind.EOD, HorizonKind.NEXT_SESSION, HorizonKind.W1))
    logged = [store.log(p) for p in preds]
    payload = logged[0].payload
    call_wall, put_wall = payload["call_wall"], payload["put_wall"]
    assert call_wall is not None and put_wall is not None

    # Daily bars for SPY ending 2026-10-09: the made_on session (10-01) closes 101, then a spike through the call wall.
    days = list(pd.bdate_range(date(2026, 9, 1), date(2026, 10, 9)))
    closes = {d.date(): 100.0 for d in days}
    closes[date(2026, 10, 1)] = 101.0
    closes[date(2026, 10, 2)] = call_wall * 1.03
    for d in days:
        if d.date() > date(2026, 10, 2):
            closes[d.date()] = call_wall * 1.03
    bars = tuple(PriceBar(d.date(), closes[d.date()], closes[d.date()] * 1.01, closes[d.date()] * 0.99, closes[d.date()], 1) for d in days)
    gateway = FakeGateway({"SPY": PriceSeries("SPY", "1d", bars, PROV)})
    report = OutcomeService(store, gateway).resolve_pending()
    assert len(report.resolved) == 3 and report.still_pending == []

    by_kind = {store.get(o.prediction_id).horizon_kind: o for o in report.resolved}
    eod, nxt, w1 = by_kind[HorizonKind.EOD], by_kind[HorizonKind.NEXT_SESSION], by_kind[HorizonKind.W1]
    assert eod.actual_return == pytest.approx(0.01)  # 101 close vs entry 100
    assert eod.max_adverse_excursion == 0.0 and eod.measures["path"]["excursions_from"] == "closes"
    assert nxt.actual_return == pytest.approx(call_wall * 1.03 / 100 - 1)
    assert nxt.measures["wall_hold_or_break"]["call"] == "broke"
    assert w1.measures["wall_hold_or_break"]["call"] == "broke"
    assert call_wall == 100.0 and eod.measures["wall_hold_or_break"]["call"] == "broke"  # the 101 close is above the wall at 100
    assert eod.measures["wall_details"]["call"]["max_penetration"] == pytest.approx(0.01)
    assert nxt.measures["implied_move"]["amplified"] is True  # |move| >> the 30-day 20% IV expected 1-day move
    assert nxt.measures["amplification_realized"]["predicted_direction"] == payload["direction"]
    assert nxt.measures["gamma_regime_outcome"]["regime"] == payload["regime"]
    rows = options_event_rows(store.resolved_pairs(PredictionType.OPTIONS_EVENT))
    assert [r.call_wall for r in rows] == ["broke"] * 3


@DB
def test_missing_data_unobserved_target_and_unresolvable_marking(store):
    from app.mrip.outcomes.service import OutcomeService

    ghost = store.log(NewPrediction(PredictionType.FORECAST, "NOPE", datetime(2026, 6, 10, 23, tzinfo=timezone.utc),
                                    HorizonKind.W1, "m", {"quantiles": {"0.5": 0.0}}))
    gateway = FakeGateway({"SPY": make_prices("SPY", spy_closes(40))})
    report = OutcomeService(store, gateway).resolve_pending()
    assert report.unavailable == [(ghost.id, "no price data for NOPE")] and store.pending()[0].id == ghost.id
    out = store.mark_unresolvable(ghost.id, resolver_version="outcome-v0", reason="symbol delisted")
    assert out.status is OutcomeStatus.UNRESOLVABLE and out.note == "symbol delisted" and store.pending() == []
    assert store.mark_unresolvable(ghost.id, resolver_version="x", reason="again").id == out.id  # immutable
    with pytest.raises(OutcomeError, match="reason"):
        store.mark_unresolvable(ghost.id, resolver_version="x", reason=" ")
    with pytest.raises(OutcomeError, match="no prediction"):
        store.mark_unresolvable(999, resolver_version="x", reason="r")
    assert store.resolved_pairs(PredictionType.FORECAST) == []  # unresolvable never becomes a label
