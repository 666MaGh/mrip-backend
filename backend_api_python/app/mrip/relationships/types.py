"""Relationship Engine domain types (work 004).

Node and relation vocabularies come from the product specification and are
mirrored by CHECK constraints in migrations/mrip_20261001_relationships.sql.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class NodeType(str, Enum):
    THEME = "THEME"
    SECTOR = "SECTOR"
    INDUSTRY = "INDUSTRY"
    COMPANY = "COMPANY"
    SECURITY = "SECURITY"
    COMMODITY = "COMMODITY"
    CURRENCY = "CURRENCY"
    RATE = "RATE"
    INDEX = "INDEX"
    MACRO_SERIES = "MACRO_SERIES"
    ECONOMIC_DRIVER = "ECONOMIC_DRIVER"
    TECHNOLOGY = "TECHNOLOGY"
    PRODUCT = "PRODUCT"
    GEOGRAPHY = "GEOGRAPHY"


class RelationType(str, Enum):
    SUPPLIES = "SUPPLIES"
    CONSUMES = "CONSUMES"
    DEPENDS_ON = "DEPENDS_ON"
    CUSTOMER_OF = "CUSTOMER_OF"
    SUPPLIER_OF = "SUPPLIER_OF"
    COMPETES_WITH = "COMPETES_WITH"
    BENEFITS_FROM = "BENEFITS_FROM"
    HURT_BY = "HURT_BY"
    EXPOSED_TO = "EXPOSED_TO"
    LEADS = "LEADS"
    LAGS = "LAGS"
    CORRELATED_WITH = "CORRELATED_WITH"
    SUBSTITUTE_FOR = "SUBSTITUTE_FOR"
    FINANCED_BY = "FINANCED_BY"


class EdgeStatus(str, Enum):
    """HYPOTHESIS until statistical validation; REJECTED edges are kept for the record."""

    HYPOTHESIS = "hypothesis"
    VALIDATED = "validated"
    REJECTED = "rejected"


class Direction(str, Enum):
    OUT = "out"
    IN = "in"
    BOTH = "both"


MAX_TRAVERSAL_DEPTH = 6
DEFAULT_TRAVERSAL_DEPTH = 3  # direct, second-order and third-order effects


@dataclass(frozen=True, slots=True)
class NodeKey:
    node_type: NodeType
    key: str


@dataclass(frozen=True, slots=True)
class Node:
    id: int
    node_type: NodeType
    key: str
    name: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Edge:
    id: int
    src_id: int
    dst_id: int
    relation_type: RelationType
    status: EdgeStatus
    source: str
    created_version: int
    retired_version: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Path:
    """A walk from the start node. ``nodes`` has one more element than ``edges``.

    ``depth`` 1 is a direct relationship, 2 a second-order and 3 a third-order effect.
    """

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if len(self.nodes) != len(self.edges) + 1 or not self.edges:
            raise ValueError("a path needs at least one edge and one more node than edges")

    @property
    def depth(self) -> int:
        return len(self.edges)

    @property
    def end(self) -> Node:
        return self.nodes[-1]


class GraphError(Exception):
    """Invalid graph operation (unknown node, duplicate active edge, bad argument)."""
