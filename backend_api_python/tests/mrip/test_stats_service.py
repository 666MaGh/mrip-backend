"""RelationshipValidator: helper tests (no DB) and end-to-end behaviour on PostgreSQL (opt-in)."""
import os
from datetime import date, datetime, timezone
from pathlib import Path as FsPath

import numpy as np
import pandas as pd
import pytest

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import PriceBar, PriceSeries, Provenance
from app.mrip.stats.service import prices_to_series

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATIONS = FsPath(__file__).resolve().parents[2] / "migrations"
N = 700
PROV = Provenance(provider="fake", gateway="test", endpoint="fake.history", fetched_at=datetime(2026, 10, 1, tzinfo=timezone.utc))


def noise(seed, n=N, scale=1.0):
    return pd.Series(np.random.default_rng(seed).normal(0, scale, n), index=pd.bdate_range("2020-01-01", periods=n))


def to_prices(returns: pd.Series, symbol: str) -> PriceSeries:
    closes = 100 * np.exp((returns.fillna(0) * 0.01).cumsum())
    bars = tuple(PriceBar(ts=ts.date(), open=None, high=None, low=None, close=float(c), volume=None) for ts, c in closes.items())
    return PriceSeries(symbol=symbol, interval="1d", bars=bars, provenance=PROV)


class FakeGateway:
    def __init__(self, series):
        self.series, self.calls = series, []

    def price_history(self, symbol, start=None, end=None, interval="1d"):
        self.calls.append(symbol)
        if symbol not in self.series:
            raise DataUnavailable(f"no data for {symbol}")
        return self.series[symbol]


def test_prices_to_series_skips_missing_closes_and_cuts_at_as_of():
    bars = (
        PriceBar(date(2026, 1, 1), None, None, None, 10.0, None),
        PriceBar(datetime(2026, 1, 2, 21, tzinfo=timezone.utc), None, None, None, None, None),
        PriceBar(date(2026, 1, 5), None, None, None, 12.0, None),
        PriceBar(date(2026, 1, 6), None, None, None, 13.0, None),
    )
    series = prices_to_series(PriceSeries("X", "1d", bars, PROV), as_of=date(2026, 1, 5))
    assert list(series.index) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05")]
    assert list(series.values) == [10.0, 12.0]


@pytest.fixture()
def env():
    from app.mrip.evidence.store import EvidenceStore
    from app.mrip.relationships.graph import RelationshipGraph
    from app.mrip.relationships.types import NodeKey, NodeType, RelationType
    from app.mrip.evidence.types import RelationshipRef
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(db._BOOTSTRAP_LEDGER_SQL)
        for name in ("mrip_20261001_relationships.sql", "mrip_20261001_evidence.sql"):
            cur.execute((MIGRATIONS / name).read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_rel_evidence, mrip_rel_edges, mrip_rel_nodes RESTART IDENTITY CASCADE")
        cur.execute("UPDATE mrip_rel_graph_state SET version = 0 WHERE id = 1")
        conn.commit()
        cur.close()
    graph = RelationshipGraph(db.get_db_connection)
    store = EvidenceStore(db.get_db_connection)
    graph.upsert_node(NodeType.COMMODITY, "copper", "Copper", {"series": {"symbol": "CPR"}})
    graph.upsert_node(NodeType.SECURITY, "miner", "Miner", {"series": {"symbol": "MNR"}})
    graph.upsert_node(NodeType.THEME, "ai", "AI")  # no series declared
    src, dst = NodeKey(NodeType.COMMODITY, "copper"), NodeKey(NodeType.SECURITY, "miner")
    graph.add_edge(src, dst, RelationType.LEADS, source="test", attributes={"expected_sign": 1})
    return graph, store, RelationshipRef(src, dst, RelationType.LEADS)


def make_validator(env, series):
    from app.mrip.stats.service import RelationshipValidator

    graph, store, ref = env
    return RelationshipValidator(graph, store, FakeGateway(series)), graph, store, ref


def leading_series(sign=1.0):
    x = noise(1)
    y = sign * 0.5 * x.shift(2) + noise(2, scale=0.8)
    return {"CPR": to_prices(x, "CPR"), "MNR": to_prices(y, "MNR"), "SPY": to_prices(noise(3), "SPY")}


@DB
def test_validated_relationship_updates_status_and_records_provenance(env):
    from app.mrip.evidence.types import SourceType, Stance
    from app.mrip.relationships.types import EdgeStatus

    validator, graph, store, ref = make_validator(env, leading_series())
    version_before = graph.current_version()
    outcome = validator.validate(ref)

    assert outcome.result.verdict.value == "validated" and outcome.status_changed
    assert outcome.edge.status is EdgeStatus.VALIDATED and graph.current_version() == version_before + 1
    (ev,) = store.list_evidence(ref)
    assert (ev.stance, ev.source_type, ev.assessed_by) == (Stance.SUPPORT, SourceType.STATISTICAL_TEST, "stats:v0-uncalibrated")
    assert ev.source_uri == "mrip://validation/v0-uncalibrated/copper->miner/LEADS"
    roles = {p["role"]: p["symbol"] for p in ev.attributes["series"]}
    assert roles == {"src": "CPR", "dst": "MNR", "market": "SPY"}
    assert '"best_lag": 2' in ev.excerpt


@DB
def test_rerun_is_idempotent_and_does_not_bump_the_graph_again(env):
    validator, graph, store, ref = make_validator(env, leading_series())
    first = validator.validate(ref)
    version = graph.current_version()
    second = validator.validate(ref)
    assert second.evidence.id == first.evidence.id and not second.status_changed
    assert graph.current_version() == version and len(store.list_evidence(ref)) == 1


@DB
def test_wrong_sign_rejects_the_edge(env):
    from app.mrip.evidence.types import Stance
    from app.mrip.relationships.types import EdgeStatus

    validator, _, store, ref = make_validator(env, leading_series(sign=-1.0))
    outcome = validator.validate(ref)
    assert outcome.edge.status is EdgeStatus.REJECTED and outcome.status_changed
    assert store.list_evidence(ref)[0].stance is Stance.CONTRADICT


@DB
def test_changed_data_supersedes_earlier_statistical_evidence(env):
    validator, graph, store, ref = make_validator(env, leading_series())
    first = validator.validate(ref)
    validator._gateway.series = leading_series()  # same relationship...
    validator._gateway.series["MNR"] = to_prices(0.5 * noise(1).shift(2) + noise(4, scale=0.8), "MNR")  # ...new data
    second = validator.validate(ref)
    active = store.list_evidence(ref)
    assert [e.id for e in active] == [second.evidence.id] and second.evidence.id != first.evidence.id
    assert any(e.id == first.evidence.id and e.retracted_at for e in store.list_evidence(ref, include_retracted=True))


@DB
def test_inconclusive_records_nothing_and_leaves_status(env):
    from app.mrip.relationships.types import EdgeStatus

    series = {"CPR": to_prices(noise(10), "CPR"), "MNR": to_prices(noise(11), "MNR"), "SPY": to_prices(noise(12), "SPY")}
    validator, graph, store, ref = make_validator(env, series)
    version = graph.current_version()
    outcome = validator.validate(ref)
    assert outcome.result.verdict.value == "inconclusive" and outcome.evidence is None
    assert outcome.edge.status is EdgeStatus.HYPOTHESIS and graph.current_version() == version
    assert store.list_evidence(ref) == []


@DB
def test_missing_series_or_data_is_reported_without_side_effects(env):
    from app.mrip.evidence.types import RelationshipRef
    from app.mrip.relationships.types import NodeKey, NodeType, RelationType

    validator, graph, store, ref = make_validator(env, {"CPR": leading_series()["CPR"]})  # MNR unavailable
    outcome = validator.validate(ref)
    assert outcome.result is None and "data unavailable" in outcome.reasons[0] and store.list_evidence(ref) == []

    graph.add_edge(NodeKey(NodeType.THEME, "ai"), ref.dst, RelationType.BENEFITS_FROM, source="test")
    no_series = validator.validate(RelationshipRef(NodeKey(NodeType.THEME, "ai"), ref.dst, RelationType.BENEFITS_FROM))
    assert no_series.result is None and "no price series declared" in no_series.reasons[0]


@DB
def test_as_of_limits_the_data_used(env):
    validator, _, _, ref = make_validator(env, leading_series())
    outcome = validator.validate(ref, as_of=date(2020, 6, 30))  # ~130 observations: too few
    assert outcome.result.verdict.value == "inconclusive" and "insufficient observations" in outcome.reasons[0]
