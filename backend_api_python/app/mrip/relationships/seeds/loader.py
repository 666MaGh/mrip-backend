"""Seed loader for the Relationship Engine.

Seed files are JSON documents declaring nodes and edges. Loading is idempotent:
nodes are upserted, and an edge is added only when no active edge with the same
(source, destination, relation) exists. Existing edges are never modified.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, TypeVar

from app.mrip.relationships.graph import RelationshipGraph
from app.mrip.relationships.types import EdgeStatus, GraphError, NodeKey, NodeType, RelationType

SEEDS_DIR = Path(__file__).parent

_E = TypeVar("_E", bound=Enum)


@dataclass(frozen=True)
class SeedResult:
    nodes: int
    edges_added: int
    edges_skipped: int


@dataclass(frozen=True)
class SeedNode:
    key: NodeKey
    name: str
    attributes: Mapping[str, object]


@dataclass(frozen=True)
class SeedEdge:
    src: NodeKey
    dst: NodeKey
    relation_type: RelationType
    status: EdgeStatus
    source: str
    attributes: Mapping[str, object]


def _require_object(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise GraphError(f"{where} must be a JSON object")
    return value


def _require_list(obj: Mapping[str, object], field: str, where: str) -> list[object]:
    value = obj.get(field)
    if not isinstance(value, list):
        raise GraphError(f"{where}.{field} must be a JSON array")
    return value


def _require_str(obj: Mapping[str, object], field: str, where: str) -> str:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GraphError(f"{where}.{field} must be a non-empty string")
    return value


def _attributes(obj: Mapping[str, object], where: str) -> Mapping[str, object]:
    value = obj.get("attributes", {})
    if not isinstance(value, dict):
        raise GraphError(f"{where}.attributes must be a JSON object")
    return value


def _enum_value(enum_type: type[_E], raw: str, where: str) -> _E:
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise GraphError(f"{where}: unknown {enum_type.__name__} {raw!r}") from exc


def _node_key(value: object, where: str) -> NodeKey:
    obj = _require_object(value, where)
    node_type = _enum_value(NodeType, _require_str(obj, "type", where), f"{where}.type")
    return NodeKey(node_type, _require_str(obj, "key", where))


def _parse_nodes(raw_nodes: list[object]) -> tuple[SeedNode, ...]:
    nodes: list[SeedNode] = []
    seen: set[NodeKey] = set()
    for index, value in enumerate(raw_nodes):
        where = f"nodes[{index}]"
        obj = _require_object(value, where)
        key = _node_key(obj, where)
        if key in seen:
            raise GraphError(f"{where}: duplicate node {key.node_type.value}:{key.key}")
        seen.add(key)
        nodes.append(SeedNode(key, _require_str(obj, "name", where), _attributes(obj, where)))
    return tuple(nodes)


def _parse_edges(raw_edges: list[object], declared: frozenset[NodeKey]) -> tuple[SeedEdge, ...]:
    edges: list[SeedEdge] = []
    seen: set[tuple[NodeKey, NodeKey, RelationType]] = set()
    for index, value in enumerate(raw_edges):
        where = f"edges[{index}]"
        obj = _require_object(value, where)
        src = _node_key(obj.get("src"), f"{where}.src")
        dst = _node_key(obj.get("dst"), f"{where}.dst")
        for ref in (src, dst):
            if ref not in declared:
                raise GraphError(f"{where}: unknown node {ref.node_type.value}:{ref.key}")
        relation_type = _enum_value(RelationType, _require_str(obj, "relation", where), f"{where}.relation")
        status = _enum_value(EdgeStatus, _require_str(obj, "status", where), f"{where}.status")
        triple = (src, dst, relation_type)
        if triple in seen:
            raise GraphError(f"{where}: duplicate {relation_type.value} edge in seed file")
        seen.add(triple)
        edges.append(
            SeedEdge(
                src=src,
                dst=dst,
                relation_type=relation_type,
                status=status,
                source=_require_str(obj, "source", where),
                attributes=_attributes(obj, where),
            )
        )
    return tuple(edges)


def parse_seed(path: Path) -> tuple[tuple[SeedNode, ...], tuple[SeedEdge, ...]]:
    """Read and validate a seed file without touching the database."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphError(f"cannot read seed file {path.name}: {exc}") from exc
    root = _require_object(document, path.name)
    nodes = _parse_nodes(_require_list(root, "nodes", path.name))
    declared = frozenset(node.key for node in nodes)
    edges = _parse_edges(_require_list(root, "edges", path.name), declared)
    return nodes, edges


def load_seed(graph: RelationshipGraph, path: Path) -> SeedResult:
    """Upsert the seed's nodes and add its edges that are not already active."""
    nodes, edges = parse_seed(path)
    for node in nodes:
        graph.upsert_node(node.key.node_type, node.key.key, node.name, node.attributes)

    new_edges = [
        edge
        for edge in edges
        if graph.find_active_edge(edge.src, edge.dst, edge.relation_type) is None
    ]
    if new_edges:
        graph.add_edges(
            [
                (edge.src, edge.dst, edge.relation_type, edge.source, edge.status, edge.attributes)
                for edge in new_edges
            ]
        )
    return SeedResult(
        nodes=len(nodes),
        edges_added=len(new_edges),
        edges_skipped=len(edges) - len(new_edges),
    )


def load_builtin_seeds(graph: RelationshipGraph) -> SeedResult:
    """Load every ``*.json`` seed file shipped in this package, in name order."""
    total = SeedResult(nodes=0, edges_added=0, edges_skipped=0)
    for path in sorted(SEEDS_DIR.glob("*.json")):
        result = load_seed(graph, path)
        total = SeedResult(
            nodes=total.nodes + result.nodes,
            edges_added=total.edges_added + result.edges_added,
            edges_skipped=total.edges_skipped + result.edges_skipped,
        )
    return total
