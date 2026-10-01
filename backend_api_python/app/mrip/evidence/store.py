"""EvidenceStore: append-only, provenance-carrying evidence per relationship.

``connect`` has the same contract as in RelationshipGraph (``get_db_connection``
in production; ``%s`` placeholders only).
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from typing import Any, Callable, ContextManager, Iterable, Mapping, Sequence

from app.mrip.evidence.types import (
    Evidence,
    EvidenceError,
    EvidenceSummary,
    RelationshipRef,
    SourceType,
    Stance,
)
from app.mrip.relationships.types import RelationType

_COLUMNS = (
    "id, src_id, dst_id, relation_type, stance, source_type, source_uri, source_title, publisher, "
    "excerpt, available_at, ingested_at, assessed_by, model_version, assessor_confidence, "
    "attributes, retracted_at, retract_reason"
)


def _evidence(row: Mapping[str, Any]) -> Evidence:
    confidence = row["assessor_confidence"]
    return Evidence(
        id=int(row["id"]),
        src_id=int(row["src_id"]),
        dst_id=int(row["dst_id"]),
        relation_type=RelationType(row["relation_type"]),
        stance=Stance(row["stance"]),
        source_type=SourceType(row["source_type"]),
        source_uri=row["source_uri"],
        source_title=row["source_title"],
        publisher=row["publisher"],
        excerpt=row["excerpt"],
        available_at=row["available_at"],
        ingested_at=row["ingested_at"],
        assessed_by=row["assessed_by"],
        model_version=row["model_version"],
        assessor_confidence=float(confidence) if confidence is not None else None,
        attributes=dict(row["attributes"] or {}),
        retracted_at=row["retracted_at"],
        retract_reason=row["retract_reason"],
    )


def summarize(evidence: Iterable[Evidence]) -> EvidenceSummary:
    items = list(evidence)
    stances = Counter(e.stance for e in items)
    return EvidenceSummary(
        total=len(items),
        by_stance={s: stances.get(s, 0) for s in Stance},
        by_source_type=dict(Counter(e.source_type for e in items)),
        conflicting=stances.get(Stance.SUPPORT, 0) > 0 and stances.get(Stance.CONTRADICT, 0) > 0,
        latest_available_at=max((e.available_at for e in items), default=None),
    )


class EvidenceStore:
    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    def add(
        self,
        relationship: RelationshipRef,
        stance: Stance,
        source_type: SourceType,
        source_uri: str,
        available_at: datetime,
        *,
        assessed_by: str,
        source_title: str | None = None,
        publisher: str | None = None,
        excerpt: str | None = None,
        model_version: str | None = None,
        assessor_confidence: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Evidence:
        """Record one piece of evidence; idempotent for an identical active item.

        ``available_at`` (when the information became public) must be timezone-aware
        so point-in-time reads cannot be corrupted by naive timestamps.
        """
        if not source_uri.strip():
            raise EvidenceError("source_uri is required (full provenance)")
        if not assessed_by.strip():
            raise EvidenceError("assessed_by is required")
        if available_at.tzinfo is None:
            raise EvidenceError("available_at must be timezone-aware")
        if assessor_confidence is not None and not 0.0 <= assessor_confidence <= 1.0:
            raise EvidenceError("assessor_confidence must be within [0, 1]")
        excerpt_hash = hashlib.sha256((excerpt or "").encode("utf-8")).hexdigest()

        with self._connect() as conn:
            cur = conn.cursor()
            try:
                src_id, dst_id = self._resolve_relationship(cur, relationship)
                cur.execute(
                    "INSERT INTO mrip_rel_evidence "
                    "(src_id, dst_id, relation_type, stance, source_type, source_uri, source_title, publisher, "
                    " excerpt, excerpt_hash, available_at, assessed_by, model_version, assessor_confidence, attributes) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (src_id, dst_id, relation_type, source_type, source_uri, excerpt_hash) "
                    "WHERE retracted_at IS NULL DO NOTHING RETURNING " + _COLUMNS,
                    (
                        src_id, dst_id, relationship.relation_type.value, stance.value, source_type.value,
                        source_uri, source_title, publisher, excerpt, excerpt_hash, available_at,
                        assessed_by, model_version, assessor_confidence,
                        json.dumps(dict(attributes or {}), sort_keys=True),
                    ),
                )
                row = cur.fetchone()
                if row is None:  # identical active item already stored
                    cur.execute(
                        "SELECT " + _COLUMNS + " FROM mrip_rel_evidence "
                        "WHERE src_id = %s AND dst_id = %s AND relation_type = %s AND source_type = %s "
                        "AND source_uri = %s AND excerpt_hash = %s AND retracted_at IS NULL",
                        (src_id, dst_id, relationship.relation_type.value, source_type.value, source_uri, excerpt_hash),
                    )
                    row = cur.fetchone()
            finally:
                cur.close()
            conn.commit()
        return _evidence(row)

    def retract(self, evidence_id: int, reason: str) -> Evidence:
        """Withdraw an item (kept for the record, excluded from current reads)."""
        if not reason.strip():
            raise EvidenceError("a retraction needs a reason")
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "UPDATE mrip_rel_evidence SET retracted_at = NOW(), retract_reason = %s "
                    "WHERE id = %s AND retracted_at IS NULL RETURNING " + _COLUMNS,
                    (reason, evidence_id),
                )
                row = cur.fetchone()
            finally:
                cur.close()
            if row is None:
                raise EvidenceError(f"no active evidence with id {evidence_id}")
            conn.commit()
        return _evidence(row)

    def list_evidence(
        self,
        relationship: RelationshipRef,
        *,
        as_of: datetime | None = None,
        stances: Sequence[Stance] | None = None,
        include_retracted: bool = False,
    ) -> list[Evidence]:
        """Evidence for a relationship, oldest availability first.

        ``as_of`` reproduces what was known then: available, ingested and not yet
        retracted at that instant. Without it, every non-retracted item is returned.
        """
        if as_of is not None and as_of.tzinfo is None:
            raise EvidenceError("as_of must be timezone-aware")
        clauses = ["src_id = %s", "dst_id = %s", "relation_type = %s"]
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                src_id, dst_id = self._resolve_relationship(cur, relationship, require_edge=False)
                params: list[Any] = [src_id, dst_id, relationship.relation_type.value]
                if as_of is not None:
                    clauses += ["available_at <= %s", "ingested_at <= %s", "(retracted_at IS NULL OR retracted_at > %s)"]
                    params += [as_of, as_of, as_of]
                elif not include_retracted:
                    clauses.append("retracted_at IS NULL")
                if stances:
                    clauses.append("stance = ANY(%s)")
                    params.append([s.value for s in stances])
                cur.execute(
                    "SELECT " + _COLUMNS + " FROM mrip_rel_evidence WHERE " + " AND ".join(clauses)
                    + " ORDER BY available_at, id",
                    tuple(params),
                )
                rows = cur.fetchall()
            finally:
                cur.close()
        return [_evidence(r) for r in rows]

    def summary(self, relationship: RelationshipRef, *, as_of: datetime | None = None) -> EvidenceSummary:
        return summarize(self.list_evidence(relationship, as_of=as_of))

    # -- internals --------------------------------------------------------

    def _resolve_relationship(self, cur: Any, ref: RelationshipRef, *, require_edge: bool = True) -> tuple[int, int]:
        ids = []
        for node in (ref.src, ref.dst):
            cur.execute(
                "SELECT id FROM mrip_rel_nodes WHERE node_type = %s AND node_key = %s",
                (node.node_type.value, node.key),
            )
            row = cur.fetchone()
            if row is None:
                raise EvidenceError(f"unknown node {node.node_type.value}:{node.key}")
            ids.append(int(row["id"]))
        src_id, dst_id = ids
        if require_edge:
            cur.execute(
                "SELECT 1 FROM mrip_rel_edges WHERE src_id = %s AND dst_id = %s AND relation_type = %s LIMIT 1",
                (src_id, dst_id, ref.relation_type.value),
            )
            if cur.fetchone() is None:
                raise EvidenceError("no such relationship in the graph; add the edge before attaching evidence")
        return src_id, dst_id
