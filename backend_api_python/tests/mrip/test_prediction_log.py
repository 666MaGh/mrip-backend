"""Automatic prediction logging, outcome resolution, reliability, wiring and jobs (work 023).

Fakes only: no network, no database, except the DB-gated idempotency test at the end.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import socket
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

import app.routes.mrip as mrip_routes
import app.utils.auth as auth
from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import Latency, PriceBar, PriceSeries, Provenance
from app.mrip.forecast.types import QUANTILES, ForecastRequest, ForecastResult
from app.mrip.outcomes.autolog import (
    ATTENTION_RULE, MIN_OBS, PredictionLogger, attention_threshold, inputs_hash,
)
from app.mrip.outcomes.autoresolve import OutcomeAutoResolver, StoredPriceGateway
from app.mrip.outcomes.jobs import LOG_JOB, RESOLVE_JOB, run_outcome_resolve, run_prediction_log
from app.mrip.outcomes.labels import reliability_table
from app.mrip.outcomes.reliability import MIN_N, alert_rows, build_reliability, forecast_rows, model_health
from app.mrip.outcomes.types import (
    HorizonKind, NewPrediction, Outcome, OutcomeStatus, Prediction, PredictionType,
)
from app.mrip.research.card import ResearchCardService
from app.mrip.relationships.types import NodeType

AS_OF = date(2026, 10, 1)
NOW = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)
AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise AssertionError("prediction log tests must not touch the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)


# -- fakes -------------------------------------------------------------------

def _bars(days: list[date], start: float = 100.0, step: float = 1.0005) -> tuple[PriceBar, ...]:
    closes = [start * step ** i for i in range(len(days))]
    return tuple(PriceBar(ts=d, open=c, high=c, low=c, close=c, volume=1e6) for d, c in zip(days, closes))


def _bdays(end: date, n: int) -> list[date]:
    return [ts.date() for ts in pd.bdate_range(end=end, periods=n)]


def _bdays_from(start: date, n: int) -> list[date]:
    return [ts.date() for ts in pd.bdate_range(start=start, periods=n)]


class Prices:
    """Stores bars per symbol; ``series`` cuts at ``end`` like the real store."""

    def __init__(self, bars_by_symbol: dict[str, tuple[PriceBar, ...]]) -> None:
        self.bars_by_symbol = dict(bars_by_symbol)

    def series(self, provider: str, symbol: str, start=None, end=None) -> PriceSeries | None:
        bars = self.bars_by_symbol.get(symbol)
        if not bars:
            return None
        if end is not None:
            bars = tuple(b for b in bars if b.ts <= end)
        if not bars:
            return None
        prov = Provenance(provider=provider, gateway="fake", endpoint="fake.bars", fetched_at=NOW, latency=Latency.UNKNOWN)
        return PriceSeries(symbol=symbol, interval="1d", bars=bars, provenance=prov)


class MemStore:
    """In-memory PredictionStore with the same unique key and one-outcome-per-prediction rules."""

    def __init__(self) -> None:
        self.rows: dict[tuple, Prediction] = {}
        self.outcomes: dict[int, Outcome] = {}
        self._next = 1

    def log_with_status(self, p: NewPrediction) -> tuple[Prediction, bool]:
        key = (p.prediction_type, p.subject, p.made_at, p.horizon_kind, p.model_version)
        if key in self.rows:
            return self.rows[key], False
        row = Prediction(
            id=self._next, prediction_type=p.prediction_type, subject=p.subject, made_at=p.made_at,
            horizon_kind=p.horizon_kind, model_version=p.model_version, payload=dict(p.payload),
            benchmark=p.benchmark, entry_price=p.entry_price, expiry_date=p.expiry_date, lineage=dict(p.lineage),
        )
        self._next += 1
        self.rows[key] = row
        return row, True

    def log(self, p: NewPrediction) -> Prediction:
        return self.log_with_status(p)[0]

    def pending(self, *, subject: str | None = None) -> list[Prediction]:
        rows = [p for p in self.rows.values() if p.id not in self.outcomes and (subject is None or p.subject == subject)]
        return sorted(rows, key=lambda p: (p.made_at, p.id))

    def record_outcome(self, prediction_id: int, **fields: Any) -> Outcome:
        if prediction_id in self.outcomes:  # outcomes never change
            return self.outcomes[prediction_id]
        out = Outcome(id=len(self.outcomes) + 1, prediction_id=prediction_id, status=OutcomeStatus.RESOLVED,
                      resolved_at=NOW, **fields)
        self.outcomes[prediction_id] = out
        return out

    def counts_by_type(self) -> dict:
        return {}

    def resolved_pairs(self, prediction_type: PredictionType, *, subject: str | None = None):
        pairs = []
        for p in sorted(self.rows.values(), key=lambda p: (p.made_at, p.id)):
            o = self.outcomes.get(p.id)
            if p.prediction_type is prediction_type and o is not None and o.status is OutcomeStatus.RESOLVED:
                if subject is None or p.subject == subject:
                    pairs.append((p, o))
        return pairs


class RecordingForecast:
    def __init__(self, fail_for: set[str] = frozenset()) -> None:
        self.requests: list[ForecastRequest] = []
        self.fail_for = set(fail_for)

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        self.requests.append(request)
        if request.symbol in self.fail_for:
            raise ValueError("no data")
        q = {k: (k - 0.5) * 0.2 for k in QUANTILES}
        return ForecastResult("fake", "fake-ensemble-v1", request.symbol, request.as_of, request.horizon_days,
                              request.last_price, q, len(request.prices), ())


def _fc_pred(pid_quantiles: dict[float, float], horizon: HorizonKind = HorizonKind.M12, pid: int = 1) -> Prediction:
    return Prediction(id=pid, prediction_type=PredictionType.FORECAST, subject="NVDA", made_at=NOW,
                      horizon_kind=horizon, model_version="fake", payload={"quantiles": {str(k): v for k, v in pid_quantiles.items()}})


def _outcome(pid: int, actual: float, measures: dict | None = None) -> Outcome:
    return Outcome(id=pid, prediction_id=pid, status=OutcomeStatus.RESOLVED, resolved_at=NOW, resolver_version="test",
                   window_start=AS_OF, window_end=AS_OF, actual_return=actual, benchmark_return=0.0,
                   measures=measures or {}, max_adverse_excursion=0.0, max_favorable_excursion=0.0)


# -- logging -----------------------------------------------------------------

def _logger(prices: Prices, **kwargs: Any) -> tuple[PredictionLogger, MemStore, RecordingForecast]:
    store = MemStore()
    forecast = kwargs.pop("forecast", RecordingForecast())
    return PredictionLogger(store, price_store=prices, forecast_provider=forecast, **kwargs), store, forecast


def test_logging_is_idempotent_per_data_date_and_kind():
    days = _bdays(AS_OF, 400)
    prices = Prices({"NVDA": _bars(days), "SPY": _bars(days)})
    logger, store, _ = _logger(prices)
    first = logger.run(AS_OF, symbols=["NVDA", "SPY"], backfill=False)
    assert first.logged == {"FORECAST_SCENARIO": 10} and not first.already_logged
    second = logger.run(AS_OF, symbols=["NVDA", "SPY"], backfill=False)
    assert second.logged == {} and second.already_logged == {"FORECAST_SCENARIO": 10}
    assert len(store.rows) == 10


def test_point_in_time_cut_ignores_bars_after_as_of_and_marks_backfill():
    days = _bdays(AS_OF + timedelta(days=40), 460)  # bars continue well past as_of
    prices = Prices({"NVDA": _bars(days), "SPY": _bars(days)})
    logger, store, forecast = _logger(prices)
    as_of = AS_OF - timedelta(days=30)
    report = logger.run(as_of, symbols=["NVDA"], backfill=True)
    assert report.backfill and report.logged == {"FORECAST_SCENARIO": 5}
    assert all(r.prices.index[-1].date() <= as_of for r in forecast.requests)
    for pred in store.rows.values():
        assert pred.made_at.date() <= as_of and pred.payload["backfill"] is True
        assert pred.lineage["backfill"] is True and pred.lineage["data_as_of"] == as_of.isoformat()
        assert pred.lineage["inputs_hash"] and pred.lineage["forecast_version"] == "fake-ensemble-v1"


def test_backfill_never_logs_discover_or_related_rows():
    days = _bdays(AS_OF, 400)
    prices = Prices({"NVDA": _bars(days)})

    class ExplodingDiscover:
        def feed(self, *a, **k):
            raise AssertionError("discover must not be read for a backfill")

    logger, store, _ = _logger(prices, discover_store=ExplodingDiscover(), related_service=object())
    report = logger.run(AS_OF, symbols=["NVDA"], backfill=True)
    assert {p.prediction_type for p in store.rows.values()} == {PredictionType.FORECAST}
    assert report.logged == {"FORECAST_SCENARIO": 5}


def test_short_history_missing_and_stale_prices_are_skipped_and_counted():
    days = _bdays(AS_OF, MIN_OBS - 1)
    stale = _bdays(AS_OF - timedelta(days=10), 400)
    prices = Prices({"SHORT": _bars(days), "OLD": _bars(stale)})
    logger, store, _ = _logger(prices)
    report = logger.run(AS_OF, symbols=["SHORT", "OLD", "NONE"], backfill=True)
    assert report.skipped == {"insufficient_history": 1, "stale_prices": 1, "no_prices": 1}
    assert not store.rows


def test_forecast_failure_is_counted_not_guessed():
    days = _bdays(AS_OF, 400)
    prices = Prices({"NVDA": _bars(days)})
    logger, store, _ = _logger(prices, forecast=RecordingForecast(fail_for={"NVDA"}))
    report = logger.run(AS_OF, symbols=["NVDA"], backfill=True)
    assert report.skipped == {"forecast_unavailable": 5} and not store.rows


def test_discover_items_are_logged_with_attention_threshold_and_direction_only_when_defined():
    from app.mrip.discover.types import Kind

    days = _bdays(AS_OF, 400)
    prices = Prices({"NVDA": _bars(days), "SPY": _bars(days)})

    @dataclass(frozen=True)
    class Item:
        id: int
        subject: str
        kind: Kind
        score: float
        magnitude: float
        headline: str
        details: dict
        policy_version: str

    class Discover:
        def feed(self, as_of=None, *, limit=50, **_):
            return [
                Item(1, "NVDA", Kind.REGIME_SHIFT, 0.8, 0.5, "h", {"direction": "down"}, "rank-v1"),
                Item(2, "SPY", Kind.COT_CROWDING, 0.4, 0.3, "h", {}, "rank-v1"),
                Item(3, "A->B", Kind.RELATIONSHIP_DIVERGENCE, 0.2, 0.1, "h", {}, "rank-v1"),
            ]

    logger, store, _ = _logger(prices, discover_store=Discover())
    report = logger.run(AS_OF, symbols=["NVDA", "SPY"], backfill=False)
    assert report.skipped.get("pair_subject") == 1
    rows = [p for p in store.rows.values() if p.prediction_type is PredictionType.DISCOVER_ITEM]
    assert {(p.subject, p.horizon_kind) for p in rows} == {("NVDA", HorizonKind.W1), ("NVDA", HorizonKind.M1),
                                                           ("SPY", HorizonKind.W1), ("SPY", HorizonKind.M1)}
    nvda = next(p for p in rows if p.subject == "NVDA" and p.horizon_kind is HorizonKind.W1)
    assert nvda.payload["direction"] == "down" and nvda.payload["attention_rule"] == ATTENTION_RULE
    spy = next(p for p in rows if p.subject == "SPY" and p.horizon_kind is HorizonKind.W1)
    assert spy.payload["direction"] is None


def test_dry_run_counts_without_writing():
    days = _bdays(AS_OF, 400)
    prices = Prices({"NVDA": _bars(days)})
    logger, store, _ = _logger(prices)
    report = logger.run(AS_OF, symbols=["NVDA"], dry_run=True, backfill=True)
    assert report.would_log == {"FORECAST_SCENARIO": 5} and not store.rows


# -- attention rule -----------------------------------------------------------

def test_attention_threshold_is_trailing_median_of_absolute_h_day_returns():
    closes = pd.Series([100.0 * 1.01 ** i for i in range(300)], index=pd.bdate_range(end=AS_OF, periods=300))
    assert attention_threshold(closes, 1) == pytest.approx(0.01, rel=1e-9)
    alternating = pd.Series([100.0 * (1.02 if i % 2 else 0.99) ** (i // 2) for i in range(300)],
                            index=pd.bdate_range(end=AS_OF, periods=300))
    expected = float(np.median(np.abs(alternating.to_numpy()[1:] / alternating.to_numpy()[:-1] - 1.0)[-252:]))
    assert attention_threshold(alternating, 1) == pytest.approx(expected)
    with pytest.raises(ValueError):
        attention_threshold(closes.iloc[:100], 21)


def test_inputs_hash_changes_with_prices():
    idx = pd.bdate_range(end=AS_OF, periods=300)
    a = pd.Series(np.linspace(100, 110, 300), index=idx)
    b = a.copy()
    b.iloc[-1] += 0.01
    assert inputs_hash("NVDA", "cboe", a) == inputs_hash("NVDA", "cboe", a)
    assert inputs_hash("NVDA", "cboe", a) != inputs_hash("NVDA", "cboe", b)


# -- resolution ---------------------------------------------------------------

def _resolver(prices: Prices, store: MemStore) -> OutcomeAutoResolver:
    return OutcomeAutoResolver(store, StoredPriceGateway(prices))


def _log_forecast(store: MemStore, subject: str, made_on: date, horizon: HorizonKind = HorizonKind.W1) -> Prediction:
    quantiles = {str(q): (q - 0.5) * 0.2 for q in QUANTILES}
    return store.log(NewPrediction(
        prediction_type=PredictionType.FORECAST, subject=subject, benchmark="SPY",
        made_at=datetime.combine(made_on, datetime.min.time(), tzinfo=timezone.utc).replace(hour=23),
        horizon_kind=horizon, model_version="fake", payload={"quantiles": quantiles},
    ))


def test_resolves_only_when_horizon_is_observed_and_is_idempotent():
    days = _bdays_from(date(2026, 6, 1), 80)
    made_on = days[30]
    prices = Prices({"NVDA": _bars(days[:35]), "SPY": _bars(days[:35])})  # W1 end is days[35]
    store = MemStore()
    pred = _log_forecast(store, "NVDA", made_on)
    early = _resolver(prices, store).run(as_of=None)
    assert early.still_pending == 1 and not store.outcomes

    prices.bars_by_symbol["NVDA"] = _bars(days[:36])
    prices.bars_by_symbol["SPY"] = _bars(days[:36])
    done = _resolver(prices, store).run()
    assert done.resolved == {"forecast": 1}
    outcome = store.outcomes[pred.id]
    assert outcome.window_end == days[35]
    assert outcome.actual_return == pytest.approx(_bars(days)[35].close / _bars(days)[30].close - 1.0)
    again = _resolver(prices, store).run()
    assert again.examined == 0 and again.resolved == {}
    assert store.outcomes[pred.id] is outcome


def test_missing_prices_leave_prediction_unresolved_with_reason():
    days = _bdays_from(date(2026, 6, 1), 80)
    store = MemStore()
    _log_forecast(store, "ZZZZ", days[30])
    run = _resolver(Prices({}), store).run()
    assert run.unavailable == {"no price data for ZZZZ": 1}
    assert not store.outcomes and len(store.pending()) == 1


def test_benchmark_judged_prediction_needs_benchmark_prices():
    days = _bdays_from(date(2026, 6, 1), 80)
    store = MemStore()
    store.log(NewPrediction(prediction_type=PredictionType.DISCOVER_ITEM, subject="NVDA", benchmark="SPY",
                            made_at=datetime(2026, 6, 2, 23, tzinfo=timezone.utc), horizon_kind=HorizonKind.W1,
                            model_version="rank-v1|REGIME_SHIFT", payload={"direction": "up", "attention_threshold": 0.01}))
    prices = Prices({"NVDA": _bars(days)})
    run = _resolver(prices, store).run()
    assert run.unavailable == {"benchmark prices missing for SPY": 1} and not store.outcomes


def test_attention_hit_and_direction_hit_rules_are_exact():
    days = _bdays_from(date(2026, 6, 1), 80)
    made_on = days[30]
    store = MemStore()
    up = store.log(NewPrediction(prediction_type=PredictionType.DISCOVER_ITEM, subject="NVDA", benchmark="SPY",
                                 made_at=datetime.combine(made_on, datetime.min.time(), tzinfo=timezone.utc).replace(hour=23),
                                 horizon_kind=HorizonKind.W1, model_version="v|K",
                                 payload={"direction": "up", "attention_threshold": 0.001}))
    down = store.log(NewPrediction(prediction_type=PredictionType.RELATED_SIGNAL, subject="NVDA", benchmark="SPY",
                                   made_at=datetime.combine(made_on, datetime.min.time(), tzinfo=timezone.utc).replace(hour=23),
                                   horizon_kind=HorizonKind.W1, model_version="v|edge-1",
                                   payload={"direction": "down"}))
    prices = Prices({"NVDA": _bars(days, step=1.01), "SPY": _bars(days, step=1.0)})
    _resolver(prices, store).run()
    m_up = store.outcomes[up.id].measures
    assert m_up["attention"]["hit"] is True and m_up["direction"]["hit"] is True  # rising stock, flat benchmark
    m_down = store.outcomes[down.id].measures
    assert m_down["direction"]["hit"] is False and "attention" not in m_down


# -- reliability ----------------------------------------------------------------

def test_quantile_coverage_matches_hand_count():
    preds = [_fc_pred({q: (q - 0.5) * 0.2 for q in QUANTILES}, pid=i) for i in range(1, 41)]
    # 4 of 40 outcomes fall below P10 (-0.08 level) and every outcome is <= P50 (0.0 level)
    pairs = [(p, _outcome(p.id, -0.2 if p.id <= 4 else 0.0)) for p in preds]
    rows = forecast_rows(pairs, min_n=30)
    assert len(rows) == 1 and rows[0]["status"] == "available" and rows[0]["n"] == 40
    cov = rows[0]["coverage"]
    assert cov["P10"]["observed"] == pytest.approx(0.1) and cov["P10"]["nominal"] == 0.1
    assert cov["P25"]["observed"] == pytest.approx(0.1)
    assert cov["P50"]["observed"] == pytest.approx(1.0) and cov["P50"]["gap"] == pytest.approx(0.5)
    assert set(cov) == {"P10", "P25", "P50", "P75", "P90"}
    table = reliability_table([])
    assert table == {}


def test_min_n_rule_hides_values_below_thirty():
    preds = [_fc_pred({q: 0.0 for q in QUANTILES}, pid=i) for i in range(1, 30)]
    small = forecast_rows([(p, _outcome(p.id, 0.0)) for p in preds])
    assert small[0]["n"] == 29 and small[0]["status"] == "otillräckligt underlag"
    assert small[0]["reason"] == "n=29 < 30" and small[0]["coverage"] is None

    def alert(pid, excess, attention_hit):
        pred = Prediction(id=pid, prediction_type=PredictionType.RELATED_SIGNAL, subject="X", made_at=NOW,
                          horizon_kind=HorizonKind.W1, model_version="v", payload={"direction": "up"})
        return pred, _outcome(pid, excess, {"excess_return": excess, "direction": {"hit": excess > 0}})

    thin = alert_rows([alert(i, 0.01, True) for i in range(1, 11)], kind="RELATED_SIGNAL", attention=False)
    metric = thin[0]["metrics"]["direction"]
    assert metric["status"] == "otillräckligt underlag" and metric["hit_rate"] is None and metric["naive_baseline"] is None
    enough = alert_rows([alert(i, 0.01 if i % 3 else -0.01, True) for i in range(1, 31)], kind="RELATED_SIGNAL", attention=False)
    m = enough[0]["metrics"]["direction"]
    assert m["status"] == "available" and m["n"] == 30 and m["naive_baseline"] == pytest.approx(20 / 30)
    assert m["hit_rate"] == pytest.approx(20 / 30)


def test_attention_baseline_is_half_and_build_reliability_reads_three_kinds():
    pred = Prediction(id=1, prediction_type=PredictionType.DISCOVER_ITEM, subject="X", made_at=NOW,
                      horizon_kind=HorizonKind.M1, model_version="v", payload={})
    store = MemStore()
    store.rows[("d",)] = pred
    store.outcomes[1] = _outcome(1, 0.1, {"excess_return": 0.1, "attention": {"hit": True}})
    table = build_reliability(store)
    assert table["version"] and table["min_n"] == MIN_N
    row = next(r for r in table["rows"] if r["kind"] == "DISCOVER_ITEM")
    assert row["metrics"]["attention"]["status"] == "otillräckligt underlag"
    assert [r["kind"] for r in table["rows"]].count("FORECAST_SCENARIO") == 0


# -- research card wiring -----------------------------------------------------------

class OutcomeStore:
    def __init__(self, pairs):
        self.pairs = pairs

    def resolved_pairs(self, prediction_type, *, subject=None):
        return [(p, o) for p, o in self.pairs if p.prediction_type is prediction_type and (subject is None or p.subject == subject)]


def _card_pairs(n: int) -> list[tuple[Prediction, Outcome]]:
    out = []
    for i in range(1, n + 1):
        pred = _fc_pred({q: (q - 0.5) * 0.2 for q in QUANTILES}, horizon=HorizonKind.M12, pid=i)
        out.append((pred, _outcome(i, 0.0)))
    return out


def test_card_reliability_and_confidence_use_read_model_above_min_n():
    service = ResearchCardService(None, None, None, outcome_store=OutcomeStore(_card_pairs(30)))
    rel = service._reliability("NVDA")
    assert rel.body["status"] == "available" and rel.body["n"] == 30 and rel.body["subject_n"] == 30
    assert set(rel.body["coverage"]) == {"P10", "P25", "P50", "P75", "P90"}
    conf = service._forecast_confidence()
    assert conf.body["status"] == "available" and conf.body["horizon"] == "M12"


def test_card_stays_unavailable_below_min_n_with_reason():
    service = ResearchCardService(None, None, None, outcome_store=OutcomeStore(_card_pairs(29)))
    rel = service._reliability("NVDA")
    assert rel.body["status"] == "unavailable" and "n=29 < 30" in rel.body["reason"]
    assert rel.body["n_resolved_outcomes"] == 29
    conf = service._forecast_confidence()
    assert conf.body["status"] == "unavailable" and "n=29 < 30" in conf.body["reason"]


# -- HTTP route ---------------------------------------------------------------------

@pytest.fixture
def authed(monkeypatch):
    payload = {"sub": "tester", "user_id": 1, "_verified_user_role": "user", "_verified_username": "tester"}
    monkeypatch.setattr(auth, "verify_token", lambda token: payload if token == "test-token" else None)


class FakeRunStore:
    def __init__(self):
        self.started, self.finished, self.skipped = [], [], []

    def start(self, job, universe):
        self.started.append((job, universe))
        return 7

    def finish(self, run_id, status, report, error=None):
        self.finished.append((run_id, status, report, error))

    def record_skipped(self, job, universe, reason):
        self.skipped.append((job, universe, reason))

    def last_run(self, job, universe):
        return None


def test_model_health_route_requires_login_and_returns_shape(client, monkeypatch, authed):
    store = MemStore()
    store.rows[("k",)] = Prediction(id=1, prediction_type=PredictionType.FORECAST, subject="NVDA", made_at=NOW,
                                    horizon_kind=HorizonKind.M12, model_version="v", payload={})
    monkeypatch.setattr(mrip_routes, "_prediction_store", lambda: store)
    monkeypatch.setattr(mrip_routes, "_job_run_store", lambda: FakeRunStore())
    monkeypatch.setattr(store, "counts_by_type", lambda: {"forecast": {"logged": 1, "resolved": 0, "unresolvable": 0,
                                                                       "unresolved": 1, "backfilled": 0}}, raising=False)
    resp = client.get("/api/mrip/model-health", headers=AUTH)
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert set(data) == {"reliability", "counts", "last_job_runs"}
    assert set(data["last_job_runs"]) == {"prediction_log", "outcome_resolve"}
    assert data["counts"]["forecast"]["unresolved"] == 1
    assert data["reliability"]["min_n"] == MIN_N
    assert client.get("/api/mrip/model-health").status_code == 401


def test_model_health_helper_reports_last_runs():
    class Run:
        job, status, started_at, finished_at, report, error = "mrip-prediction-log", "completed", NOW, NOW, {"x": 1}, None

    class Runs(FakeRunStore):
        def last_run(self, job, universe):
            return Run() if job == LOG_JOB else None

    out = model_health(MemStore(), Runs(), job_names={"prediction_log": LOG_JOB, "outcome_resolve": RESOLVE_JOB})
    assert out["last_job_runs"]["prediction_log"]["status"] == "completed"
    assert out["last_job_runs"]["outcome_resolve"] is None


# -- jobs and gating ----------------------------------------------------------------

@contextlib.contextmanager
def _lock(acquired: bool):
    yield acquired


class FakeService:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def run(self, *args, **kwargs):
        if self.error:
            raise self.error
        return type("R", (), {"to_dict": lambda self: {"ok": True, "args": [str(a) for a in args]}})()


def test_log_job_lock_held_is_skipped_and_recorded():
    runs = FakeRunStore()
    result = run_prediction_log(AS_OF, service_factory=lambda: FakeService(), lock_factory=lambda k: _lock(False), run_store=runs)
    assert result.status == "skipped" and runs.skipped == [(LOG_JOB, "all", "another mrip-prediction-log run is running")]
    assert not runs.started


def test_resolve_job_failure_is_recorded_not_raised():
    runs = FakeRunStore()
    result = run_outcome_resolve(AS_OF, service_factory=lambda: FakeService(RuntimeError("boom")),
                                 lock_factory=lambda k: _lock(True), run_store=runs)
    assert result.status == "failed" and result.reason == "RuntimeError: boom"
    assert runs.finished[0][1:] == ("failed", {}, "RuntimeError: boom")


def test_dry_run_job_skips_lock_and_run_record():
    runs = FakeRunStore()
    result = run_prediction_log(AS_OF, dry_run=True, service_factory=lambda: FakeService(),
                                lock_factory=lambda k: _lock(False), run_store=runs)
    assert result.status == "completed" and result.report["ok"] is True
    assert not runs.started and not runs.skipped


def test_celery_tasks_registered_and_gated_by_env(monkeypatch):
    from app.celery_app import celery_app
    import app.tasks.mrip_sync as tasks

    for name in ("quantdinger.tasks.mrip_prediction_log", "quantdinger.tasks.mrip_outcome_resolve"):
        assert name in celery_app.tasks
        assert celery_app.conf.task_routes[name] == {"queue": "maintenance"}
    assert celery_app.conf.beat_schedule["mrip-prediction-log"]["schedule"] >= 3600
    assert celery_app.conf.beat_schedule["mrip-outcome-resolve"]["schedule"] >= 3600
    monkeypatch.setenv("ENABLE_MRIP_PREDICTION_LOG", "false")
    monkeypatch.setenv("ENABLE_MRIP_OUTCOME_RESOLVE", "false")
    assert tasks.mrip_prediction_log.run() == {"skipped": True}
    assert tasks.mrip_outcome_resolve.run() == {"skipped": True}


def test_env_keys_are_in_both_templates():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    keys = ["ENABLE_MRIP_PREDICTION_LOG", "MRIP_PREDICTION_LOG_INTERVAL_SEC", "MRIP_PREDICTION_LOG_SYMBOL_CAP",
            "ENABLE_MRIP_OUTCOME_RESOLVE", "MRIP_OUTCOME_RESOLVE_INTERVAL_SEC"]
    for template in (root / "env.example", root.parent / ".env.example"):
        text = template.read_text(encoding="utf-8")
        for key in keys:
            assert f"\n{key}=" in text, (template, key)


# -- DB-gated -----------------------------------------------------------------------

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")


@DB
def test_db_logging_is_idempotent_and_migration_extends_types():
    from app.mrip.outcomes.store import PredictionStore
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE mrip_outcomes, mrip_predictions RESTART IDENTITY CASCADE")
        conn.commit()
        cur.close()
    store = PredictionStore(db.get_db_connection)
    p = NewPrediction(prediction_type=PredictionType.RELATED_SIGNAL, subject="NVDA", benchmark="SPY",
                      made_at=NOW, horizon_kind=HorizonKind.W1, model_version="related-v1|edge-1",
                      payload={"direction": "up"})
    first, created_a = store.log_with_status(p)
    second, created_b = store.log_with_status(p)
    assert created_a and not created_b and first.id == second.id
    counts = store.counts_by_type()
    assert counts["related_signal"]["logged"] == 1 and counts["related_signal"]["unresolved"] == 1
