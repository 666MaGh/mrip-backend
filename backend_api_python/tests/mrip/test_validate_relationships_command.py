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


# --- --revalidate (work 022): v1 re-run, validated -> hypothesis downgrade on INCONCLUSIVE ----------------

from app.commands.validate_relationships import revalidate_edges  # noqa: E402


class RevalidateValidator:
    """Scripted v1 outcomes keyed by source key; mimics RelationshipValidator.validate's status effects."""

    def __init__(self, verdicts: dict[str, str], fail_on: set[str] | None = None) -> None:
        self.verdicts, self.fail_on = verdicts, fail_on or set()
        self.flags: list[bool] = []

    def validate(self, ref, *, downgrade_unsupported: bool = False):
        self.flags.append(downgrade_unsupported)
        if ref.src.key in self.fail_on:
            raise RuntimeError("boom")
        verdict = self.verdicts[ref.src.key]
        edge = self.current[ref.src.key]
        status = edge.status
        if verdict == "validated":
            status = EdgeStatus.VALIDATED
        elif verdict == "inconclusive" and edge.status is EdgeStatus.VALIDATED and downgrade_unsupported:
            status = EdgeStatus.HYPOTHESIS
        result = SimpleNamespace(verdict=SimpleNamespace(value=verdict))
        return SimpleNamespace(result=result, reasons=(verdict,), edge=Edge(
            edge.id, edge.src_id, edge.dst_id, edge.relation_type, status, edge.source, edge.created_version),
            status_changed=status is not edge.status)


def _revalidate_graph():
    nodes = {1: _node(1, "A"), 2: _node(2, "B"), 3: _node(3, "C"), 4: _node(4, "D"), 5: _node(5, "E")}
    edges = [
        _edge(10, 1, 2, EdgeStatus.VALIDATED),    # A -> B: validated under v0, inconclusive under v1 -> downgrade
        _edge(11, 3, 4, EdgeStatus.HYPOTHESIS),   # C -> D: hypothesis, v1 validates -> hypothesis->validated
        _edge(12, 4, 5, EdgeStatus.VALIDATED),    # D -> E: stays validated
        _edge(13, 2, 5, EdgeStatus.HYPOTHESIS),   # B -> E: stays hypothesis
        _edge(14, 5, 1, EdgeStatus.REJECTED),     # never touched
    ]
    return FakeGraph(edges, nodes)


def test_revalidate_downgrades_only_unsupported_validated_edges_and_keeps_rejected_untouched():
    graph = _revalidate_graph()
    validator = RevalidateValidator({"A": "inconclusive", "C": "validated", "D": "validated", "B": "inconclusive"})
    validator.current = {"A": graph.edges[0], "C": graph.edges[1], "D": graph.edges[2], "B": graph.edges[3]}
    rows, summary = revalidate_edges(graph, validator)
    assert [row["edge_id"] for row in rows] == [10, 11, 12, 13]  # rejected edge 14 never validated
    assert all(validator.flags) and len(validator.flags) == 4
    assert summary["transitions"] == {
        "hypothesis->hypothesis": 1,
        "hypothesis->validated": 1,
        "validated->hypothesis": 1,
        "validated->validated": 1,
    }
    assert summary["verdicts"] == {"inconclusive": 2, "validated": 2}
    assert [row["after"] for row in rows] == ["hypothesis", "validated", "validated", "hypothesis"]


def test_revalidate_isolates_per_edge_errors():
    graph = _revalidate_graph()
    validator = RevalidateValidator({"C": "validated", "D": "validated", "B": "inconclusive"}, fail_on={"A"})
    validator.current = {"C": graph.edges[1], "D": graph.edges[2], "B": graph.edges[3]}
    rows, summary = revalidate_edges(graph, validator)
    assert rows[0]["verdict"] == "error" and rows[0]["after"] == "validated"  # failed edge keeps its status
    assert "RuntimeError: boom" in rows[0]["reasons"][0]
    assert summary["transitions"]["validated->validated"] == 2 and summary["edges"] == 4
