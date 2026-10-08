"""validate_relationships command: only hypothesis edges are validated, one edge's failure does not stop the rest."""
from dataclasses import dataclass, field
from types import SimpleNamespace

from app.commands.validate_relationships import validate_hypothesis_edges
from app.mrip.relationships.types import Edge, EdgeStatus, Node, NodeType, RelationType


def _node(node_id: int, key: str) -> Node:
    return Node(id=node_id, node_type=NodeType.COMPANY, key=key, name=key)


def _edge(edge_id: int, src: int, dst: int, status: EdgeStatus) -> Edge:
    return Edge(id=edge_id, src_id=src, dst_id=dst, relation_type=RelationType.SUPPLIES, status=status, source="seed", created_version=1)


@dataclass
class FakeGraph:
    edges: list
    nodes: dict = field(default_factory=dict)

    def list_active_edges(self, limit: int = 1000):
        return list(self.edges)

    def get_node_by_id(self, node_id: int):
        return self.nodes.get(node_id)


class FakeValidator:
    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.seen: list[str] = []

    def validate(self, ref):
        self.seen.append(ref.src.key)
        if ref.src.key in self.fail_on:
            raise RuntimeError("boom")
        verdict = SimpleNamespace(value="validated")
        return SimpleNamespace(result=SimpleNamespace(verdict=verdict), reasons=("ok",), status_changed=True)


def _graph() -> FakeGraph:
    nodes = {1: _node(1, "A"), 2: _node(2, "B"), 3: _node(3, "C"), 4: _node(4, "D")}
    edges = [
        _edge(10, 1, 2, EdgeStatus.HYPOTHESIS),
        _edge(11, 3, 4, EdgeStatus.VALIDATED),
        _edge(12, 1, 4, EdgeStatus.HYPOTHESIS),
        _edge(13, 3, 2, EdgeStatus.REJECTED),
    ]
    return FakeGraph(edges, nodes)


def test_only_hypothesis_edges_are_validated():
    validator = FakeValidator()
    rows, summary = validate_hypothesis_edges(_graph(), validator)
    assert validator.seen == ["A", "A"]
    assert [row["edge_id"] for row in rows] == [10, 12]
    assert summary["edges"] == 2
    assert summary["validated"] == 2
    assert summary["errors"] == 0


def test_exception_on_one_edge_does_not_stop_others():
    graph = _graph()
    graph.edges[0] = _edge(10, 3, 2, EdgeStatus.HYPOTHESIS)  # src C, will fail
    graph.edges[2] = _edge(12, 1, 4, EdgeStatus.HYPOTHESIS)  # src A, succeeds
    validator = FakeValidator(fail_on={"C"})
    rows, summary = validate_hypothesis_edges(graph, validator)
    assert [row["verdict"] for row in rows] == ["error", "validated"]
    assert "RuntimeError: boom" in rows[0]["reasons"][0]
    assert summary == {"summary": True, "edges": 2, "validated": 1, "rejected": 0, "inconclusive": 0, "errors": 1, "status_changed": 1}
