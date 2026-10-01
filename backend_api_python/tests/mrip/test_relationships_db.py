"""Relationship Engine against a real PostgreSQL (opt-in: MRIP_TEST_DB=1 and DATABASE_URL).

Runs through the application's own connection layer (app.utils.db) so the
wrapper's placeholder and RETURNING handling is exercised too.
"""
import os
from pathlib import Path as FsPath

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL"),
]

MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_relationships.sql"


@pytest.fixture(scope="module")
def connect():
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(db._BOOTSTRAP_LEDGER_SQL)
        conn.commit()
        cur.close()
    return db.get_db_connection


@pytest.fixture()
def graph(connect):
    from app.mrip.relationships.graph import RelationshipGraph

    with connect() as conn:
        cur = conn.cursor()
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_rel_edges, mrip_rel_nodes RESTART IDENTITY CASCADE")
        cur.execute("UPDATE mrip_rel_graph_state SET version = 0 WHERE id = 1")
        conn.commit()
        cur.close()
    return RelationshipGraph(connect)


def keys(graph):
    """AI -> datacenter -> electricity -> grid capex -> transformers -> copper (+ a cycle edge)."""
    from app.mrip.relationships.types import NodeKey, NodeType

    spec = [
        (NodeType.THEME, "ai", "AI spending"),
        (NodeType.INDUSTRY, "datacenters", "Datacenters"),
        (NodeType.ECONOMIC_DRIVER, "electricity", "Electricity demand"),
        (NodeType.ECONOMIC_DRIVER, "grid-capex", "Grid investment"),
        (NodeType.PRODUCT, "transformers", "Transformers"),
        (NodeType.COMMODITY, "copper", "Copper"),
        (NodeType.SECURITY, "NVDA", "NVIDIA"),
    ]
    out = {}
    for node_type, key, name in spec:
        graph.upsert_node(node_type, key, name)
        out[key] = NodeKey(node_type, key)
    return out


def chain(graph, k):
    from app.mrip.relationships.types import EdgeStatus, RelationType as R

    rows = [
        (k["ai"], k["datacenters"], R.BENEFITS_FROM),
        (k["datacenters"], k["electricity"], R.CONSUMES),
        (k["electricity"], k["grid-capex"], R.DEPENDS_ON),
        (k["grid-capex"], k["transformers"], R.CONSUMES),
        (k["transformers"], k["copper"], R.CONSUMES),
    ]
    return graph.add_edges([(s, d, r, "test", EdgeStatus.HYPOTHESIS, None) for s, d, r in rows])


def test_migration_component_is_idempotent_and_checksummed(connect):
    from app.utils import db
    import logging

    log = logging.getLogger("t")
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM qd_bootstrap_migrations WHERE name = 'mrip-relationships-test'")
        conn.commit()
        cur.close()
        first = db._apply_migration_component(conn, log, name="mrip-relationships-test", path=MIGRATION)
        second = db._apply_migration_component(conn, log, name="mrip-relationships-test", path=MIGRATION)
    assert (first, second) == ("applied", "skipped")


def test_upsert_node_is_idempotent_and_updates(graph):
    from app.mrip.relationships.types import NodeType

    a = graph.upsert_node(NodeType.THEME, "ai", "AI", {"x": 1})
    b = graph.upsert_node(NodeType.THEME, "ai", "AI spending", {"x": 2})
    assert a.id == b.id and b.name == "AI spending" and b.attributes == {"x": 2}


def test_edges_bump_version_once_per_batch(graph):
    k = keys(graph)
    assert graph.current_version() == 0
    edges = chain(graph, k)
    assert graph.current_version() == 1 and {e.created_version for e in edges} == {1}


def test_third_order_traversal_finds_the_ai_to_transformer_chain(graph):
    from app.mrip.relationships.types import NodeType

    k = keys(graph)
    chain(graph, k)
    paths = graph.traverse(k["ai"], max_depth=3)
    by_depth = {}
    for p in paths:
        by_depth.setdefault(p.depth, []).append([n.key for n in p.nodes])
    assert by_depth[1] == [["ai", "datacenters"]]
    assert by_depth[2] == [["ai", "datacenters", "electricity"]]
    assert by_depth[3] == [["ai", "datacenters", "electricity", "grid-capex"]]
    deeper = graph.traverse(k["ai"], max_depth=5)
    assert max(p.depth for p in deeper) == 5 and deeper[-1].end.key == "copper"
    assert deeper[-1].end.node_type is NodeType.COMMODITY


def test_direction_in_and_both(graph):
    from app.mrip.relationships.types import Direction

    k = keys(graph)
    chain(graph, k)
    incoming = graph.traverse(k["copper"], direction=Direction.IN, max_depth=2)
    assert [[n.key for n in p.nodes] for p in incoming] == [
        ["copper", "transformers"],
        ["copper", "transformers", "grid-capex"],
    ]
    both = graph.traverse(k["electricity"], direction=Direction.BOTH, max_depth=1)
    assert {p.end.key for p in both} == {"datacenters", "grid-capex"}


def test_cycles_do_not_loop_and_rejected_edges_are_hidden(graph):
    from app.mrip.relationships.types import EdgeStatus, RelationType as R

    k = keys(graph)
    edges = chain(graph, k)
    graph.add_edge(k["copper"], k["ai"], R.CORRELATED_WITH, source="test")  # closes a cycle
    paths = graph.traverse(k["ai"], max_depth=6)
    assert all(len({n.id for n in p.nodes}) == len(p.nodes) for p in paths)  # no repeated node

    graph.set_edge_status(edges[0].id, EdgeStatus.REJECTED)
    assert graph.traverse(k["ai"], max_depth=3) == []
    rejected = graph.traverse(k["ai"], max_depth=1, statuses=[EdgeStatus.REJECTED])
    assert [p.end.key for p in rejected] == ["datacenters"]


def test_relation_type_filter(graph):
    from app.mrip.relationships.types import RelationType as R

    k = keys(graph)
    chain(graph, k)
    only_consumes = graph.traverse(k["datacenters"], relation_types=[R.CONSUMES], max_depth=3)
    assert [p.end.key for p in only_consumes] == ["electricity"]


def test_as_of_version_reproduces_older_graph(graph):
    from app.mrip.relationships.types import EdgeStatus

    k = keys(graph)
    edges = chain(graph, k)          # version 1
    v_before = graph.current_version()
    graph.set_edge_status(edges[0].id, EdgeStatus.VALIDATED)   # version 2: retire + reinsert
    graph.retire_edge(edges[1].id)                              # version 3
    now = {p.end.key for p in graph.traverse(k["ai"], max_depth=3)}
    then = {p.end.key for p in graph.traverse(k["ai"], max_depth=3, as_of_version=v_before)}
    assert now == {"datacenters"}                               # chain cut at datacenters -> electricity
    assert then == {"datacenters", "electricity", "grid-capex"}
    statuses_then = {p.edges[0].status for p in graph.traverse(k["ai"], max_depth=1, as_of_version=v_before)}
    assert statuses_then == {EdgeStatus.HYPOTHESIS}


def test_rejects_duplicate_self_loop_and_unknown_nodes(graph):
    from app.mrip.relationships.types import GraphError, NodeKey, NodeType, RelationType as R

    k = keys(graph)
    graph.add_edge(k["ai"], k["datacenters"], R.BENEFITS_FROM, source="test")
    v = graph.current_version()
    with pytest.raises(GraphError, match="already exists"):
        graph.add_edge(k["ai"], k["datacenters"], R.BENEFITS_FROM, source="test")
    with pytest.raises(GraphError, match="itself"):
        graph.add_edge(k["ai"], k["ai"], R.CORRELATED_WITH, source="test")
    with pytest.raises(GraphError, match="unknown node"):
        graph.add_edge(k["ai"], NodeKey(NodeType.THEME, "nope"), R.CORRELATED_WITH, source="test")
    assert graph.current_version() == v  # failed writes do not bump the version


def test_traversal_guard_on_path_explosion(graph):
    from app.mrip.relationships.types import GraphError, EdgeStatus, RelationType as R

    k = keys(graph)
    chain(graph, k)
    with pytest.raises(GraphError, match="max_paths"):
        graph.traverse(k["ai"], max_depth=5, max_paths=2)
