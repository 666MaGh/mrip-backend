from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from app.mrip.evidence.types import RelationshipRef
from app.mrip.relationships.types import EdgeStatus, NodeKey


def validate_hypothesis_edges(graph: Any, validator: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every active edge still in ``hypothesis`` status; one row per edge, errors do not stop the run."""
    rows: list[dict[str, Any]] = []
    for edge in graph.list_active_edges():
        if edge.status is not EdgeStatus.HYPOTHESIS:
            continue
        row: dict[str, Any] = {"edge_id": edge.id}
        try:
            src = graph.get_node_by_id(edge.src_id)
            dst = graph.get_node_by_id(edge.dst_id)
            if src is None or dst is None:
                raise LookupError(f"missing node for edge {edge.id}")
            ref = RelationshipRef(NodeKey(src.node_type, src.key), NodeKey(dst.node_type, dst.key), edge.relation_type)
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


def main() -> int:
    from app.mrip.data.openbb_adapter import OpenBBAdapter
    from app.mrip.evidence.store import EvidenceStore
    from app.mrip.prices.gateway import StoredDataGateway
    from app.mrip.prices.store import PriceStore
    from app.mrip.relationships.graph import RelationshipGraph
    from app.mrip.stats.service import RelationshipValidator
    from app.utils.db import get_db_connection, init_database

    parser = argparse.ArgumentParser(description="Statistically validate hypothesis relationship edges")
    parser.parse_args()
    init_database(strict_migrations=False)
    graph = RelationshipGraph(get_db_connection)
    gateway = StoredDataGateway(PriceStore(get_db_connection), live=OpenBBAdapter(retries=1))
    validator = RelationshipValidator(graph, EvidenceStore(get_db_connection), gateway)
    rows, summary = validate_hypothesis_edges(graph, validator)
    for row in rows:
        print(json.dumps(row, sort_keys=True, default=str))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
