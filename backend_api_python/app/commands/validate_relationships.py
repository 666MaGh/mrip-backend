from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from app.mrip.evidence.types import RelationshipRef
from app.mrip.relationships.types import EdgeStatus, NodeKey


def _edge_ref(graph: Any, edge: Any) -> tuple[RelationshipRef, Any, Any]:
    src = graph.get_node_by_id(edge.src_id)
    dst = graph.get_node_by_id(edge.dst_id)
    if src is None or dst is None:
        raise LookupError(f"missing node for edge {edge.id}")
    ref = RelationshipRef(NodeKey(src.node_type, src.key), NodeKey(dst.node_type, dst.key), edge.relation_type)
    return ref, src, dst


def validate_hypothesis_edges(graph: Any, validator: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every active edge still in ``hypothesis`` status; one row per edge, errors do not stop the run."""
    rows: list[dict[str, Any]] = []
    for edge in graph.list_active_edges():
        if edge.status is not EdgeStatus.HYPOTHESIS:
            continue
        row: dict[str, Any] = {"edge_id": edge.id}
        try:
            ref, src, dst = _edge_ref(graph, edge)
            row.update(src=src.key, relation=edge.relation_type.value, dst=dst.key)
            outcome = validator.validate(ref)
            verdict = outcome.result.verdict.value if outcome.result is not None else "inconclusive"
            row.update(verdict=verdict, status_changed=bool(outcome.status_changed), reasons=list(outcome.reasons))
        except Exception as exc:  # per-edge isolation: record and continue
            row.setdefault("src", None)
            row.setdefault("relation", edge.relation_type.value)
            row.setdefault("dst", None)
            row.update(verdict="error", status_changed=False, reasons=[f"{type(exc).__name__}: {exc}"])
        rows.append(row)
    counts = Counter(row["verdict"] for row in rows)
    summary = {
        "summary": True,
        "edges": len(rows),
        "validated": counts.get("validated", 0),
        "rejected": counts.get("rejected", 0),
        "inconclusive": counts.get("inconclusive", 0),
        "errors": counts.get("error", 0),
        "status_changed": sum(1 for row in rows if row["status_changed"]),
    }
    return rows, summary


def revalidate_edges(graph: Any, validator: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-run v1 validation on every active (hypothesis or validated) edge; rejected edges are never touched.

    An INCONCLUSIVE verdict on a currently validated edge downgrades it to hypothesis (the
    validator records the reasons as neutral evidence). Returns one row per edge plus a
    before/after transition table keyed ``"<before>-><after>"``.
    """
    rows: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    for edge in graph.list_active_edges():
        if edge.status is EdgeStatus.REJECTED:
            continue
        before = edge.status.value
        row: dict[str, Any] = {"edge_id": edge.id, "before": before}
        after = before
        try:
            ref, src, dst = _edge_ref(graph, edge)
            row.update(src=src.key, relation=edge.relation_type.value, dst=dst.key)
            outcome = validator.validate(ref, downgrade_unsupported=True)
            verdict = outcome.result.verdict.value if outcome.result is not None else "inconclusive"
            if outcome.edge is not None:
                after = outcome.edge.status.value
            row.update(verdict=verdict, reasons=list(outcome.reasons))
        except Exception as exc:  # per-edge isolation: record and continue
            row.update(verdict="error", reasons=[f"{type(exc).__name__}: {exc}"])
        row["after"] = after
        transitions[f"{before}->{after}"] += 1
        rows.append(row)
    summary = {
        "summary": True,
        "mode": "revalidate",
        "edges": len(rows),
        "verdicts": dict(Counter(row["verdict"] for row in rows)),
        "transitions": dict(sorted(transitions.items())),
    }
    return rows, summary


def main() -> int:
    from app.mrip.data.openbb_adapter import OpenBBAdapter
    from app.mrip.evidence.store import EvidenceStore
    from app.mrip.prices.gateway import StoredDataGateway
    from app.mrip.prices.store import PriceStore
    from app.mrip.relationships.graph import RelationshipGraph
    from app.mrip.stats.null_store import ValidationNullStore
    from app.mrip.stats.service import RelationshipValidator
    from app.utils.db import get_db_connection, init_database

    parser = argparse.ArgumentParser(description="Statistically validate hypothesis relationship edges")
    parser.add_argument("--price-provider", default="cboe", help="Stored price provider to read (default: cboe)")
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help="Re-run validation on all active hypothesis and validated edges; downgrade validated edges "
             "that are INCONCLUSIVE under the current policy to hypothesis. Rejected edges are untouched.",
    )
    args = parser.parse_args()
    init_database(strict_migrations=False)
    graph = RelationshipGraph(get_db_connection)
    gateway = StoredDataGateway(PriceStore(get_db_connection), provider=args.price_provider, live=OpenBBAdapter(retries=1))
    validator = RelationshipValidator(
        graph, EvidenceStore(get_db_connection), gateway, null_store=ValidationNullStore(get_db_connection),
    )
    if args.revalidate:
        rows, summary = revalidate_edges(graph, validator)
    else:
        rows, summary = validate_hypothesis_edges(graph, validator)
    for row in rows:
        print(json.dumps(row, sort_keys=True, default=str))
    print(json.dumps(summary, sort_keys=True))
    if args.revalidate:
        print_transition_table(summary)
    return 0


def print_transition_table(summary: dict[str, Any]) -> None:
    print()
    print("before -> after | edges")
    for key, count in summary["transitions"].items():
        print(f"{key} | {count}")
    print(f"verdicts: {summary['verdicts']}")


if __name__ == "__main__":
    raise SystemExit(main())
