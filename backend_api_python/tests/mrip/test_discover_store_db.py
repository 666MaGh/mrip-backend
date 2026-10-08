from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from app.mrip.discover.types import Kind



DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")


@DB
def test_discover_store_persistence_contract():
    from app.utils.db import get_db_connection
    from app.mrip.discover.store import DiscoverStore
    from app.mrip.discover.types import Observation, RankedItem

    migration = Path(__file__).parents[2] / "migrations/mrip_20261001_discover.sql"
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(migration.read_text())
        cur.execute("TRUNCATE mrip_discover_items RESTART IDENTITY")
        conn.commit()
        cur.close()
    store = DiscoverStore(get_db_connection)
    today = date(2026, 10, 1)
    def item(kind, subject, score):
        obs = Observation(kind, subject, today, "headline", .5, details={"iteration": score})
        return RankedItem(obs, score, {"magnitude": .5}, ())
    low = item(Kind.REGIME_SHIFT, "MARKET", 30)
    high = item(Kind.PEER_DIVERGENCE, "XYZ", 90)
    assert store.save([low, high], "rank-v1") == 2
    assert store.save([low, high], "rank-v1") == 2
    feed = store.feed(today)
    assert [row.score for row in feed] == [90, 30]
    assert len(store.feed(today, limit=1)) == 1
    assert [row.kind for row in store.feed(today, kinds=[Kind.PEER_DIVERGENCE])] == [Kind.PEER_DIVERGENCE]
    high_id = next(row.id for row in feed if row.kind is Kind.PEER_DIVERGENCE)
    store.dismiss(high_id)
    assert [row.kind for row in store.feed(today)] == [Kind.REGIME_SHIFT]
    assert len(store.feed(today, include_dismissed=True)) == 2
    changed = RankedItem(Observation(Kind.PEER_DIVERGENCE, "XYZ", today, "updated", .9), 95, {"magnitude": .9}, ())
    store.save([changed], "rank-v2")
    dismissed = next(row for row in store.feed(today, include_dismissed=True) if row.kind is Kind.PEER_DIVERGENCE)
    assert dismissed.status == "dismissed" and dismissed.score == 95
    prior = date(2026, 9, 29)
    store.save([RankedItem(Observation(Kind.PEER_DIVERGENCE, "XYZ", prior, "prior", .5), 50, {}, ())], "rank-v1")
    assert store.recent_counts(today, days=7)[(Kind.PEER_DIVERGENCE.value, "XYZ")] == 1


class _FakeCursor:
    def __init__(self, log):
        self.log = log

    def execute(self, sql, params=None):
        self.log.append(params)

    def close(self):
        pass


class _FakeConn:
    def __init__(self, log):
        self.log = log

    def cursor(self):
        return _FakeCursor(self.log)

    def commit(self):
        pass


def test_store_persists_unusual_activity_details_as_json_object():
    from contextlib import contextmanager

    from app.mrip.discover.store import DiscoverStore
    from app.mrip.discover.types import Observation, RankedItem
    from app.mrip.options.vol_surface import Anomaly

    log: list = []

    @contextmanager
    def connect():
        yield _FakeConn(log)

    anomaly = Anomaly("SPY261016C00500000", date(2026, 10, 16), 500.0, "call", 1200.0, 100.0, 12.0)
    obs = Observation(Kind.UNUSUAL_OPTIONS_ACTIVITY, "SPY", date(2026, 10, 8), "Observed unusual options activity", .5, {"anomalies": [anomaly]}, {"oi_status": "ok"}, False)
    assert DiscoverStore(connect).save([RankedItem(obs, 50, {"magnitude": .5}, ())], "rank-v1") == 1
    params = log[0]
    details = json.loads(params[7])
    assert details == {"anomalies": [{"contract_symbol": "SPY261016C00500000", "expiry": "2026-10-16", "strike": 500.0, "option_type": "call", "volume": 1200.0, "open_interest": 100.0, "volume_oi_ratio": 12.0}]}
    assert json.loads(params[8]) == {"oi_status": "ok"}
