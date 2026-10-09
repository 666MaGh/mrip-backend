"""RelationshipGraph: PostgreSQL-backed relationship store and traversal.

Every mutation bumps the graph version in the same transaction; edges are
append-only (retired, never deleted), so ``traverse(as_of_version=N)``
reproduces the graph exactly as it was (data lineage, work 004).

``connect`` is any callable returning a context manager that yields a
DB-API-like connection (``cursor()``, ``commit()``): ``app.utils.db.get_db_connection``
in production. Queries use ``%s`` placeholders only (never ``?``).
"""
from __future__ import annotations

import json
from typing import Any, Callable, ContextManager, Iterable, Mapping, Sequence

from app.mrip.relationships.types import (
    DEFAULT_TRAVERSAL_DEPTH,
    MAX_TRAVERSAL_DEPTH,
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

_DEFAULT_STATUSES = frozenset({EdgeStatus.HYPOTHESIS, EdgeStatus.VALIDATED})
_DEFAULT_MAX_PATHS = 10_000

_EDGE_COLUMNS = (
    "id, src_id, dst_id, relation_type, status, source, attributes, created_version, retired_version"
)

_TRAVERSE_SQL = """
WITH RECURSIVE visible AS (
    SELECT id, src_id, dst_id
    FROM mrip_rel_edges
    WHERE created_version <= %s
      AND (retired_version IS NULL OR retired_version > %s)
      AND (cardinality(%s::text[]) = 0 OR relation_type = ANY(%s::text[]))
      AND status = ANY(%s::text[])
), steps AS (
    {steps}
), walk AS (
    SELECT 1 AS depth,
           ARRAY[s.from_id, s.to_id] AS node_path,
           ARRAY[s.edge_id] AS edge_path
    FROM steps s
    WHERE s.from_id = %s
  UNION ALL
    SELECT w.depth + 1,
           w.node_path || s.to_id,
           w.edge_path || s.edge_id
    FROM walk w
    JOIN steps s ON s.from_id = w.node_path[array_length(w.node_path, 1)]
    WHERE w.depth < %s
      AND NOT (s.to_id = ANY(w.node_path))
)
SELECT depth, node_path, edge_path FROM walk ORDER BY depth, node_path, edge_path LIMIT %s
"""

_STEPS_OUT = "SELECT id AS edge_id, src_id AS from_id, dst_id AS to_id FROM visible"
_STEPS_IN = "SELECT id AS edge_id, dst_id AS from_id, src_id AS to_id FROM visible"


def _json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(value or {}), sort_keys=True)


def _node(row: Mapping[str, Any]) -> Node:
    return Node(
        id=int(row["id"]),
        node_type=NodeType(row["node_type"]),
        key=row["node_key"],
        name=row["name"],
        attributes=dict(row["attributes"] or {}),
    )


def _edge(row: Mapping[str, Any]) -> Edge:
    retired = row["retired_version"]
    return Edge(
        id=int(row["id"]),
        src_id=int(row["src_id"]),
        dst_id=int(row["dst_id"]),
        relation_type=RelationType(row["relation_type"]),
        status=EdgeStatus(row["status"]),
        source=row["source"],
        attributes=dict(row["attributes"] or {}),
        created_version=int(row["created_version"]),
        retired_version=int(retired) if retired is not None else None,
    )


class RelationshipGraph:
    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    # -- versions ---------------------------------------------------------

    def current_version(self) -> int:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT version FROM mrip_rel_graph_state WHERE id = 1")
                row = cur.fetchone()
            finally:
                cur.close()
        if row is None:
            raise GraphError("mrip_rel_graph_state is missing; has the MRIP migration been applied?")
        return int(row["version"])

    # -- nodes ------------------------------------------------------------

    def upsert_node(
        self,
        node_type: NodeType,
        key: str,
        name: str,
        attributes: Mapping[str, Any] | None = None,
    ) -> Node:
        """Create the node or update its name/attributes. Nodes are not versioned."""
        if not key.strip() or not name.strip():
            raise GraphError("node key and name must be non-empty")
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO mrip_rel_nodes (node_type, node_key, name, attributes) "
                    "VALUES (%s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (node_type, node_key) "
                    "DO UPDATE SET name = EXCLUDED.name, attributes = EXCLUDED.attributes "
                    "RETURNING id, node_type, node_key, name, attributes",
                    (node_type.value, key, name, _json(attributes)),
                )
                row = cur.fetchone()
            finally:
                cur.close()
            conn.commit()
        return _node(row)

    def get_node(self, node: NodeKey) -> Node | None:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                return self._find_node(cur, node)
            finally:
                cur.close()

    def list_nodes_in_universe(self, universe: str) -> list[Node]:
        """Return all SECURITY nodes that belong to the given universe.

        A SECURITY node belongs to a universe if its attributes contain a
        "universes" key (a JSON list) that includes the universe name.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT id, node_type, node_key, name, attributes FROM mrip_rel_nodes "
                    "WHERE node_type = %s AND attributes -> 'universes' @> %s::jsonb",
                    (NodeType.SECURITY.value, json.dumps([universe])),
                )
                return [_node(row) for row in cur.fetchall()]
            finally:
                cur.close()

    def list_priced_nodes(self) -> list[Node]:
        """Nodes that declare a price series (``attributes.series.symbol``), of any node type.

        Sector membership for validation control is read from the ``sector`` attribute.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT id, node_type, node_key, name, attributes FROM mrip_rel_nodes "
                    "WHERE attributes -> 'series' IS NOT NULL ORDER BY id"
                )
                return [_node(row) for row in cur.fetchall()]
            finally:
                cur.close()

    def get_node_by_id(self, node_id: int) -> Node | None:
        """Return a node by its database identifier."""
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT id, node_type, node_key, name, attributes FROM mrip_rel_nodes WHERE id = %s", (node_id,))
                row = cur.fetchone()
            finally:
                cur.close()
        return _node(row) if row else None

    def list_active_edges(self, limit: int = 1000) -> list[Edge]:
        """Return active hypothesis and validated edges, excluding rejected edges."""
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT " + _EDGE_COLUMNS + " FROM mrip_rel_edges WHERE retired_version IS NULL AND status = ANY(%s::text[]) ORDER BY id LIMIT %s", ([EdgeStatus.HYPOTHESIS.value, EdgeStatus.VALIDATED.value], limit))
                rows = cur.fetchall()
            finally:
                cur.close()
        return [_edge(row) for row in rows]

    def list_edges(self, statuses: Sequence[EdgeStatus], limit: int = 200) -> list[Edge]:
        """Current (non-retired) edges with one of the given statuses, oldest first."""
        if limit < 1:
            raise ValueError("limit must be positive")
        if not statuses:
            raise ValueError("statuses must not be empty")
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT " + _EDGE_COLUMNS + " FROM mrip_rel_edges "
                    "WHERE retired_version IS NULL AND status = ANY(%s::text[]) ORDER BY id LIMIT %s",
                    ([s.value for s in statuses], limit),
                )
                rows = cur.fetchall()
            finally:
                cur.close()
        return [_edge(row) for row in rows]

    def get_edge(self, edge_id: int) -> Edge | None:
        """A current (non-retired) edge by identifier."""
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT " + _EDGE_COLUMNS + " FROM mrip_rel_edges WHERE id = %s AND retired_version IS NULL",
                    (edge_id,),
                )
                row = cur.fetchone()
            finally:
                cur.close()
        return _edge(row) if row else None

    def get_nodes_by_ids(self, node_ids: Sequence[int]) -> dict[int, Node]:
        """Batch node lookup by identifier (one query)."""
        if not node_ids:
            return {}
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                return self._nodes_by_id(cur, list(node_ids))
            finally:
                cur.close()

    # -- edges ------------------------------------------------------------

    def add_edge(
        self,
        src: NodeKey,
        dst: NodeKey,
        relation_type: RelationType,
        *,
        source: str,
        status: EdgeStatus = EdgeStatus.HYPOTHESIS,
        attributes: Mapping[str, Any] | None = None,
    ) -> Edge:
        """Add one relationship (one new graph version)."""
        return self.add_edges([(src, dst, relation_type, source, status, attributes)])[0]

    def add_edges(
        self,
        edges: Iterable[tuple[NodeKey, NodeKey, RelationType, str, EdgeStatus, Mapping[str, Any] | None]],
    ) -> list[Edge]:
        """Add several relationships atomically under a single new graph version."""
        batch = list(edges)
        if not batch:
            raise GraphError("no edges given")
        created: list[Edge] = []
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                version = self._bump(cur)
                for src, dst, relation_type, source, status, attributes in batch:
                    if not source.strip():
                        raise GraphError("edge source must be non-empty")
                    created.append(self._insert_edge(cur, src, dst, relation_type, source, status, attributes, version))
            finally:
                cur.close()
            conn.commit()
        return created

    def set_edge_status(self, edge_id: int, status: EdgeStatus) -> Edge:
        """Change an edge's status: retires the old row and inserts a new one (one new version)."""
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                old = self._active_edge(cur, edge_id)
                if old.status is status:
                    return old
                version = self._bump(cur)
                self._retire(cur, edge_id, version)
                cur.execute(
                    "INSERT INTO mrip_rel_edges "
                    "(src_id, dst_id, relation_type, status, source, attributes, created_version) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s) RETURNING " + _EDGE_COLUMNS,
                    (
                        old.src_id, old.dst_id, old.relation_type.value, status.value,
                        old.source, _json(old.attributes), version,
                    ),
                )
                new = _edge(cur.fetchone())
            finally:
                cur.close()
            conn.commit()
        return new

    def retire_edge(self, edge_id: int) -> None:
        """Remove an edge from the current graph (it remains in older versions)."""
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                self._active_edge(cur, edge_id)
                self._retire(cur, edge_id, self._bump(cur))
            finally:
                cur.close()
            conn.commit()

    def find_active_edge(self, src: NodeKey, dst: NodeKey, relation_type: RelationType) -> Edge | None:
        """The current (non-retired) edge for this relationship, if any."""
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                src_node, dst_node = self._find_node(cur, src), self._find_node(cur, dst)
                if src_node is None or dst_node is None:
                    return None
                cur.execute(
                    "SELECT " + _EDGE_COLUMNS + " FROM mrip_rel_edges "
                    "WHERE src_id = %s AND dst_id = %s AND relation_type = %s AND retired_version IS NULL",
                    (src_node.id, dst_node.id, relation_type.value),
                )
                row = cur.fetchone()
            finally:
                cur.close()
        return _edge(row) if row else None

    # -- traversal --------------------------------------------------------

    def traverse(
        self,
        start: NodeKey,
        *,
        max_depth: int = DEFAULT_TRAVERSAL_DEPTH,
        direction: Direction = Direction.OUT,
        relation_types: Sequence[RelationType] | None = None,
        statuses: Iterable[EdgeStatus] | None = None,
        as_of_version: int | None = None,
        max_paths: int = _DEFAULT_MAX_PATHS,
    ) -> list[Path]:
        """All simple paths (no repeated node) of length 1..max_depth from ``start``.

        Defaults: out-edges, hypotheses and validated edges (rejected excluded),
        current graph. Results are ordered by depth, then deterministically.
        """
        if not 1 <= max_depth <= MAX_TRAVERSAL_DEPTH:
            raise GraphError(f"max_depth must be between 1 and {MAX_TRAVERSAL_DEPTH}")
        if max_paths < 1:
            raise GraphError("max_paths must be positive")
        status_set = frozenset(statuses) if statuses is not None else _DEFAULT_STATUSES
        if not status_set:
            raise GraphError("statuses must not be empty")
        steps = {
            Direction.OUT: _STEPS_OUT,
            Direction.IN: _STEPS_IN,
            Direction.BOTH: _STEPS_OUT + " UNION ALL " + _STEPS_IN,
        }[direction]
        sql = _TRAVERSE_SQL.format(steps=steps)

        with self._connect() as conn:
            cur = conn.cursor()
            try:
                start_node = self._find_node(cur, start)
                if start_node is None:
                    raise GraphError(f"unknown node {start.node_type.value}:{start.key}")
                version = as_of_version if as_of_version is not None else self._read_version(cur)
                types = [r.value for r in (relation_types or ())]
                cur.execute(
                    sql,
                    (
                        version, version, types, types,
                        sorted(s.value for s in status_set),
                        start_node.id, max_depth, max_paths + 1,
                    ),
                )
                walks = cur.fetchall()
                if len(walks) > max_paths:
                    raise GraphError(f"traversal exceeds max_paths={max_paths}; narrow the query")
                node_ids = sorted({n for w in walks for n in w["node_path"]})
                edge_ids = sorted({e for w in walks for e in w["edge_path"]})
                nodes = self._nodes_by_id(cur, node_ids)
                edges = self._edges_by_id(cur, edge_ids)
            finally:
                cur.close()
        return [
            Path(
                nodes=tuple(nodes[i] for i in w["node_path"]),
                edges=tuple(edges[i] for i in w["edge_path"]),
            )
            for w in walks
        ]

    # -- internals --------------------------------------------------------

    def _bump(self, cur: Any) -> int:
        cur.execute(
            "UPDATE mrip_rel_graph_state SET version = version + 1, updated_at = NOW() "
            "WHERE id = 1 RETURNING version"
        )
        row = cur.fetchone()
        if row is None:
            raise GraphError("mrip_rel_graph_state is missing; has the MRIP migration been applied?")
        return int(row["version"])

    def _read_version(self, cur: Any) -> int:
        cur.execute("SELECT version FROM mrip_rel_graph_state WHERE id = 1")
        row = cur.fetchone()
        if row is None:
            raise GraphError("mrip_rel_graph_state is missing; has the MRIP migration been applied?")
        return int(row["version"])

    def _find_node(self, cur: Any, node: NodeKey) -> Node | None:
        cur.execute(
            "SELECT id, node_type, node_key, name, attributes FROM mrip_rel_nodes "
            "WHERE node_type = %s AND node_key = %s",
            (node.node_type.value, node.key),
        )
        row = cur.fetchone()
        return _node(row) if row else None

    def _insert_edge(
        self,
        cur: Any,
        src: NodeKey,
        dst: NodeKey,
        relation_type: RelationType,
        source: str,
        status: EdgeStatus,
        attributes: Mapping[str, Any] | None,
        version: int,
    ) -> Edge:
        src_node, dst_node = self._find_node(cur, src), self._find_node(cur, dst)
        for ref, found in ((src, src_node), (dst, dst_node)):
            if found is None:
                raise GraphError(f"unknown node {ref.node_type.value}:{ref.key}")
        if src_node.id == dst_node.id:
            raise GraphError("an edge cannot connect a node to itself")
        cur.execute(
            "SELECT 1 FROM mrip_rel_edges "
            "WHERE src_id = %s AND dst_id = %s AND relation_type = %s AND retired_version IS NULL",
            (src_node.id, dst_node.id, relation_type.value),
        )
        if cur.fetchone():
            raise GraphError(
                f"active {relation_type.value} edge already exists: "
                f"{src.node_type.value}:{src.key} -> {dst.node_type.value}:{dst.key}"
            )
        cur.execute(
            "INSERT INTO mrip_rel_edges "
            "(src_id, dst_id, relation_type, status, source, attributes, created_version) "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s) RETURNING " + _EDGE_COLUMNS,
            (src_node.id, dst_node.id, relation_type.value, status.value, source, _json(attributes), version),
        )
        return _edge(cur.fetchone())

    def _active_edge(self, cur: Any, edge_id: int) -> Edge:
        cur.execute(
            "SELECT " + _EDGE_COLUMNS + " FROM mrip_rel_edges WHERE id = %s AND retired_version IS NULL FOR UPDATE",
            (edge_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise GraphError(f"no active edge with id {edge_id}")
        return _edge(row)

    def _retire(self, cur: Any, edge_id: int, version: int) -> None:
        cur.execute("UPDATE mrip_rel_edges SET retired_version = %s WHERE id = %s", (version, edge_id))

    def _nodes_by_id(self, cur: Any, ids: list[int]) -> dict[int, Node]:
        if not ids:
            return {}
        cur.execute(
            "SELECT id, node_type, node_key, name, attributes FROM mrip_rel_nodes WHERE id = ANY(%s)", (ids,)
        )
        return {int(r["id"]): _node(r) for r in cur.fetchall()}

    def _edges_by_id(self, cur: Any, ids: list[int]) -> dict[int, Edge]:
        if not ids:
            return {}
        cur.execute("SELECT " + _EDGE_COLUMNS + " FROM mrip_rel_edges WHERE id = ANY(%s)", (ids,))
        return {int(r["id"]): _edge(r) for r in cur.fetchall()}
