"""Evidence Engine against a real PostgreSQL (opt-in: MRIP_TEST_DB=1 and DATABASE_URL)."""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path as FsPath

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL"),
]

MIGRATIONS = FsPath(__file__).resolve().parents[2] / "migrations"
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


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
    graph.upsert_node(NodeType.COMPANY, "acme", "Acme")
    graph.upsert_node(NodeType.COMPANY, "gridco", "GridCo")
    src, dst = NodeKey(NodeType.COMPANY, "acme"), NodeKey(NodeType.COMPANY, "gridco")
    edge = graph.add_edge(src, dst, RelationType.SUPPLIES, source="test")
    ref = RelationshipRef(src, dst, RelationType.SUPPLIES)
    return graph, EvidenceStore(db.get_db_connection), ref, edge


def add(store, ref, stance, uri="https://sec.gov/a", **kw):
    from app.mrip.evidence.types import SourceType

    kw.setdefault("assessed_by", "manual")
    kw.setdefault("excerpt", "Acme supplies GridCo.")
    return store.add(ref, stance, kw.pop("source_type", SourceType.REGULATORY_FILING), uri, kw.pop("available_at", T0), **kw)


def test_provenance_round_trips(env):
    from app.mrip.evidence.types import SourceType, Stance

    _, store, ref, _ = env
    ev = add(
        store, ref, Stance.SUPPORT, source_title="10-K 2025", publisher="SEC", model_version="laya:x@1",
        assessor_confidence=0.82, attributes={"page": 14},
    )
    (got,) = store.list_evidence(ref)
    assert got == ev
    assert (got.source_type, got.source_uri, got.publisher, got.model_version) == (
        SourceType.REGULATORY_FILING, "https://sec.gov/a", "SEC", "laya:x@1")
    assert got.assessor_confidence == 0.82 and got.attributes == {"page": 14}
    assert got.available_at == T0 and got.ingested_at.tzinfo is not None


def test_identical_active_item_is_idempotent_but_different_excerpt_is_new(env):
    from app.mrip.evidence.types import Stance

    _, store, ref, _ = env
    a = add(store, ref, Stance.SUPPORT)
    b = add(store, ref, Stance.SUPPORT)
    c = add(store, ref, Stance.SUPPORT, excerpt="Another passage.")
    assert a.id == b.id and c.id != a.id
    assert len(store.list_evidence(ref)) == 2


def test_requires_existing_relationship_and_nodes(env):
    from app.mrip.evidence.types import EvidenceError, RelationshipRef, Stance
    from app.mrip.relationships.types import NodeKey, NodeType, RelationType

    _, store, ref, _ = env
    other_type = RelationshipRef(ref.src, ref.dst, RelationType.COMPETES_WITH)
    with pytest.raises(EvidenceError, match="no such relationship"):
        add(store, other_type, Stance.SUPPORT)
    ghost = RelationshipRef(ref.src, NodeKey(NodeType.COMPANY, "ghost"), RelationType.SUPPLIES)
    with pytest.raises(EvidenceError, match="unknown node"):
        add(store, ghost, Stance.SUPPORT)


def test_summary_flags_conflict_and_filters_by_stance(env):
    from app.mrip.evidence.types import SourceType, Stance

    _, store, ref, _ = env
    add(store, ref, Stance.SUPPORT, uri="u1")
    add(store, ref, Stance.SUPPORT, uri="u2", source_type=SourceType.COT)
    add(store, ref, Stance.CONTRADICT, uri="u3", source_type=SourceType.NEWS)
    s = store.summary(ref)
    assert s.total == 3 and s.conflicting
    assert s.by_stance[Stance.SUPPORT] == 2 and s.by_stance[Stance.CONTRADICT] == 1
    assert [e.source_uri for e in store.list_evidence(ref, stances=[Stance.CONTRADICT])] == ["u3"]


def test_retraction_hides_item_but_keeps_record_and_allows_readd(env):
    from app.mrip.evidence.types import EvidenceError, Stance

    _, store, ref, _ = env
    ev = add(store, ref, Stance.SUPPORT)
    gone = store.retract(ev.id, "misread passage")
    assert gone.retracted_at is not None and gone.retract_reason == "misread passage"
    assert store.list_evidence(ref) == []
    assert [e.id for e in store.list_evidence(ref, include_retracted=True)] == [ev.id]
    with pytest.raises(EvidenceError):
        store.retract(ev.id, "again")
    again = add(store, ref, Stance.NEUTRAL)  # same item, corrected assessment, new record
    assert again.id != ev.id and again.stance is Stance.NEUTRAL


def test_as_of_reproduces_what_was_known(env):
    from app.mrip.evidence.types import Stance

    _, store, ref, _ = env
    before = datetime.now(timezone.utc)
    early = add(store, ref, Stance.SUPPORT, uri="early", available_at=T0)
    later_public = add(store, ref, Stance.CONTRADICT, uri="future", available_at=datetime.now(timezone.utc) + timedelta(days=30))
    after_ingest = datetime.now(timezone.utc)
    store.retract(early.id, "wrong")

    assert store.list_evidence(ref, as_of=before) == []                       # not yet ingested
    assert [e.id for e in store.list_evidence(ref, as_of=after_ingest)] == [early.id]   # future-dated item excluded
    assert [e.id for e in store.list_evidence(ref, as_of=datetime.now(timezone.utc))] == []  # early retracted by now
    assert later_public.id not in [e.id for e in store.list_evidence(ref, as_of=after_ingest)]


def test_evidence_survives_edge_status_changes(env):
    from app.mrip.evidence.types import Stance
    from app.mrip.relationships.types import EdgeStatus

    graph, store, ref, edge = env
    add(store, ref, Stance.SUPPORT)
    graph.set_edge_status(edge.id, EdgeStatus.VALIDATED)  # retires the row, inserts a new one
    assert len(store.list_evidence(ref)) == 1
    assert store.summary(ref).total == 1
