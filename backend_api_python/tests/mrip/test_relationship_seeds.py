"""Relationship seed loader: file contract and idempotence.

The unit tests use an in-memory graph (no database). The DB test is opt-in
(MRIP_TEST_DB=1 with a throwaway DATABASE_URL), like test_relationships_db.py.
"""
import json
import os
from pathlib import Path as FsPath

import pytest

from app.mrip.relationships.graph import RelationshipGraph
from app.mrip.relationships.seeds.loader import SEEDS_DIR, load_builtin_seeds, load_seed, parse_seed
from app.mrip.relationships.types import Edge, EdgeStatus, GraphError, Node, NodeKey, NodeType, RelationType

SEED_FILE = SEEDS_DIR / "ai_infrastructure.json"
MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_relationships.sql"


class _NoDb:
    def __call__(self):
        raise AssertionError("unit test must not touch the database")


class FakeGraph(RelationshipGraph):
    """In-memory stand-in with the same method contract as RelationshipGraph."""

    def __init__(self) -> None:
        super().__init__(_NoDb())
        self.nodes: dict[NodeKey, Node] = {}
        self.edges: dict[tuple[NodeKey, NodeKey, RelationType], Edge] = {}
        self.add_calls = 0

    def upsert_node(self, node_type, key, name, attributes=None):
        node = Node(id=len(self.nodes) + 1, node_type=node_type, key=key, name=name, attributes=dict(attributes or {}))
        self.nodes[NodeKey(node_type, key)] = node
        return node

    def find_active_edge(self, src, dst, relation_type):
        return self.edges.get((src, dst, relation_type))

    def add_edges(self, edges):
        self.add_calls += 1
        created = []
        for src, dst, relation_type, source, status, attributes in edges:
            assert src in self.nodes and dst in self.nodes, "loader must upsert nodes first"
            edge = Edge(
                id=len(self.edges) + 1,
                src_id=self.nodes[src].id,
                dst_id=self.nodes[dst].id,
                relation_type=relation_type,
                status=status,
                source=source,
                created_version=1,
                attributes=dict(attributes or {}),
            )
            self.edges[(src, dst, relation_type)] = edge
            created.append(edge)
        return created


def _write(tmp_path: FsPath, document: dict) -> FsPath:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _minimal(edge_relation: str = "BENEFITS_FROM") -> dict:
    return {
        "nodes": [
            {"type": "THEME", "key": "t", "name": "Theme"},
            {"type": "COMPANY", "key": "A", "name": "A Corp"},
        ],
        "edges": [
            {
                "src": {"type": "COMPANY", "key": "A"},
                "dst": {"type": "THEME", "key": "t"},
                "relation": edge_relation,
                "status": "hypothesis",
                "source": "test-seed",
                "attributes": {"expected_sign": 1},
            }
        ],
    }


def test_builtin_seed_file_parses_and_every_edge_references_a_defined_node():
    raw = json.loads(SEED_FILE.read_text(encoding="utf-8"))
    declared = {(n["type"], n["key"]) for n in raw["nodes"]}
    for edge in raw["edges"]:
        assert (edge["src"]["type"], edge["src"]["key"]) in declared
        assert (edge["dst"]["type"], edge["dst"]["key"]) in declared

    nodes, edges = parse_seed(SEED_FILE)
    assert len(nodes) == 14 and len(edges) == 13
    assert {n.key.key for n in nodes if n.key.node_type is NodeType.COMPANY} >= {"NVDA", "MSFT", "FCX"}


def test_builtin_edges_are_hypotheses_with_expected_sign():
    _, edges = parse_seed(SEED_FILE)
    assert all(e.status is EdgeStatus.HYPOTHESIS for e in edges)
    assert all(e.source == "mrip-seed-ai-infrastructure-v1" for e in edges)
    assert all(e.attributes == {"expected_sign": 1} for e in edges)


def test_company_nodes_declare_series_symbol():
    nodes, _ = parse_seed(SEED_FILE)
    for node in nodes:
        if node.key.node_type is NodeType.COMPANY:
            assert node.attributes["series"] == {"symbol": node.key.key}


def test_theme_node_declares_series_symbol_and_proxy():
    nodes, _ = parse_seed(SEED_FILE)
    theme = next(n for n in nodes if n.key == NodeKey(NodeType.THEME, "ai-infrastructure"))
    assert theme.attributes["series"] == {"symbol": "SMH"}
    assert theme.attributes["proxy"]


def test_reseed_overwrites_theme_attributes():
    graph = FakeGraph()
    load_builtin_seeds(graph)
    load_builtin_seeds(graph)
    theme = graph.nodes[NodeKey(NodeType.THEME, "ai-infrastructure")]
    assert theme.attributes["series"] == {"symbol": "SMH"}


def test_unknown_relation_type_raises_graph_error(tmp_path):
    path = _write(tmp_path, _minimal(edge_relation="NOT_A_RELATION"))
    with pytest.raises(GraphError):
        parse_seed(path)
    with pytest.raises(GraphError):
        load_seed(FakeGraph(), path)


def test_unknown_node_key_in_edge_raises_graph_error(tmp_path):
    document = _minimal()
    document["edges"][0]["dst"] = {"type": "THEME", "key": "missing"}
    with pytest.raises(GraphError):
        parse_seed(_write(tmp_path, document))


def test_unknown_node_type_raises_graph_error(tmp_path):
    document = _minimal()
    document["nodes"][1]["type"] = "PLANET"
    with pytest.raises(GraphError):
        parse_seed(_write(tmp_path, document))


def test_malformed_json_raises_graph_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(GraphError):
        parse_seed(path)


def test_second_load_adds_no_edges():
    graph = FakeGraph()
    first = load_builtin_seeds(graph)
    second = load_builtin_seeds(graph)
    assert (first.nodes, first.edges_added, first.edges_skipped) == (14, 13, 0)
    assert (second.nodes, second.edges_added, second.edges_skipped) == (14, 0, 13)
    assert len(graph.edges) == 13


def test_loader_skips_add_call_when_nothing_new(tmp_path):
    graph = FakeGraph()
    path = _write(tmp_path, _minimal())
    load_seed(graph, path)
    calls_after_first = graph.add_calls
    result = load_seed(graph, path)
    assert result.edges_added == 0 and graph.add_calls == calls_after_first


def test_seed_loads_expected_edges_into_graph():
    graph = FakeGraph()
    load_builtin_seeds(graph)
    nvda = NodeKey(NodeType.COMPANY, "NVDA")
    msft = NodeKey(NodeType.COMPANY, "MSFT")
    theme = NodeKey(NodeType.THEME, "ai-infrastructure")
    customer = graph.find_active_edge(msft, nvda, RelationType.CUSTOMER_OF)
    assert customer is not None and customer.status is EdgeStatus.HYPOTHESIS
    assert graph.find_active_edge(nvda, theme, RelationType.BENEFITS_FROM) is not None
    assert graph.find_active_edge(nvda, msft, RelationType.CUSTOMER_OF) is None


@pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
@pytest.mark.integration
def test_seed_load_against_database_is_idempotent():
    from app.utils import db

    connect = db.get_db_connection
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(db._BOOTSTRAP_LEDGER_SQL)
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_rel_edges, mrip_rel_nodes RESTART IDENTITY CASCADE")
        cur.execute("UPDATE mrip_rel_graph_state SET version = 0 WHERE id = 1")
        conn.commit()
        cur.close()

    graph = RelationshipGraph(connect)
    first = load_builtin_seeds(graph)
    second = load_builtin_seeds(graph)
    assert first.edges_added == 13
    assert second.edges_added == 0 and second.edges_skipped == 13
