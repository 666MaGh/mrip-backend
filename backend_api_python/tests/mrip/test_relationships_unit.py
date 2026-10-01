"""Relationship Engine: contract checks that need no database."""
import re
from pathlib import Path as FsPath

import pytest

from app.mrip.relationships.graph import RelationshipGraph
from app.mrip.relationships.types import (
    Direction,
    Edge,
    EdgeStatus,
    GraphError,
    Node,
    NodeKey,
    NodeType,
    Path,
    RelationType,
)

MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_relationships.sql"


def _check_values(sql: str, column: str) -> set[str]:
    match = re.search(rf"{column}\s+VARCHAR\(\d+\)\s+NOT NULL\s+(?:DEFAULT\s+'[^']*'\s+)?CHECK\s*\(\s*{column}\s+IN\s*\((.*?)\)\s*\)", sql, re.S)
    assert match, f"CHECK list for {column} not found"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def test_sql_check_constraints_match_python_enums():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert _check_values(sql, "node_type") == {n.value for n in NodeType}
    assert _check_values(sql, "relation_type") == {r.value for r in RelationType}
    assert _check_values(sql, "status") == {s.value for s in EdgeStatus}


def test_spec_vocabulary_is_complete():
    assert len(NodeType) == 14 and len(RelationType) == 14
    assert {"SUPPLIES", "HURT_BY", "FINANCED_BY", "LAGS"} <= {r.value for r in RelationType}


def _node(i):
    return Node(id=i, node_type=NodeType.COMPANY, key=f"c{i}", name=f"C{i}")


def _edge(i, src, dst):
    return Edge(i, src, dst, RelationType.SUPPLIES, EdgeStatus.HYPOTHESIS, "test", 1)


def test_path_depth_and_end():
    p = Path(nodes=(_node(1), _node(2), _node(3)), edges=(_edge(1, 1, 2), _edge(2, 2, 3)))
    assert p.depth == 2 and p.end.id == 3


@pytest.mark.parametrize("nodes,edges", [((_node(1),), ()), ((_node(1), _node(2), _node(3)), (_edge(1, 1, 2),))])
def test_path_rejects_inconsistent_shape(nodes, edges):
    with pytest.raises(ValueError):
        Path(nodes=nodes, edges=edges)


class _NoDb:
    def __call__(self):
        raise AssertionError("must not touch the database for invalid arguments")


@pytest.mark.parametrize(
    "kwargs",
    [{"max_depth": 0}, {"max_depth": 7}, {"max_paths": 0}, {"statuses": []}],
)
def test_traverse_validates_arguments_before_any_query(kwargs):
    graph = RelationshipGraph(_NoDb())
    with pytest.raises(GraphError):
        graph.traverse(NodeKey(NodeType.THEME, "ai"), **kwargs)


def test_add_edges_rejects_empty_batch_and_blank_node():
    graph = RelationshipGraph(_NoDb())
    with pytest.raises(GraphError):
        graph.add_edges([])
    with pytest.raises(GraphError):
        graph.upsert_node(NodeType.THEME, " ", "AI")
    assert Direction("both") is Direction.BOTH
