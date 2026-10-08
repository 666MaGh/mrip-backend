"""Pure helpers that parse query input and shape MRIP domain objects for JSON.

Nothing here touches Flask or the database, so every helper is unit-testable.
Input problems raise ``ApiInputError``; the route layer maps it to HTTP 400.
Modeled options quantities are always returned under a ``modeled`` object that
carries ``MODELED / ESTIMATED``; they are never mixed into ``observed``.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

from app.mrip.data.models import PriceSeries
from app.mrip.discover.store import StoredItem
from app.mrip.evidence.types import Evidence, EvidenceSummary
from app.mrip.options.analysis import MODELED_LABEL
from app.mrip.relationships.types import (
    DEFAULT_TRAVERSAL_DEPTH,
    MAX_TRAVERSAL_DEPTH,
    Direction,
    Edge,
    EdgeStatus,
    Node,
    NodeType,
    Path,
)
from app.mrip.discover.types import Kind

MAX_LIST_LIMIT = 200
DEFAULT_DISCOVER_LIMIT = 50
DEFAULT_EDGE_LIMIT = 200
PRICE_PROVIDERS: tuple[str, ...] = ("cboe", "yahoo")
DEFAULT_PRICE_PROVIDER = "yahoo"
DEFAULT_PRICE_DAYS = 365
MAX_PRICE_DAYS = 5000


class ApiInputError(ValueError):
    """A request parameter is missing, malformed or out of range."""


def _text(raw: str | None) -> str | None:
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def parse_int_in_range(raw: str | None, name: str, *, default: int, low: int, high: int) -> int:
    text = _text(raw)
    if text is None:
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise ApiInputError(f"{name} must be an integer") from exc
    if not low <= value <= high:
        raise ApiInputError(f"{name} must be between {low} and {high}")
    return value


def parse_bool(raw: str | None, name: str, *, default: bool) -> bool:
    text = _text(raw)
    if text is None:
        return default
    lowered = text.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    raise ApiInputError(f"{name} must be true or false")


def parse_date(raw: str | None, name: str) -> date | None:
    text = _text(raw)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ApiInputError(f"{name} must be a date in YYYY-MM-DD format") from exc


def parse_kinds(raw: str | None) -> tuple[Kind, ...] | None:
    text = _text(raw)
    if text is None:
        return None
    allowed = {k.value: k for k in Kind}
    kinds: list[Kind] = []
    for part in text.split(","):
        name = part.strip().upper()
        if not name:
            continue
        if name not in allowed:
            raise ApiInputError(f"unknown kind {name!r}")
        if allowed[name] not in kinds:
            kinds.append(allowed[name])
    if not kinds:
        raise ApiInputError("kinds must name at least one kind")
    return tuple(kinds)


def parse_node_type(raw: str | None) -> NodeType:
    text = _text(raw)
    if text is None:
        raise ApiInputError("node_type is required")
    try:
        return NodeType(text.upper())
    except ValueError as exc:
        raise ApiInputError(f"unknown node_type {text!r}") from exc


def parse_key(raw: str | None) -> str:
    text = _text(raw)
    if text is None:
        raise ApiInputError("key is required")
    return text


def parse_depth(raw: str | None) -> int:
    return parse_int_in_range(raw, "depth", default=DEFAULT_TRAVERSAL_DEPTH, low=1, high=MAX_TRAVERSAL_DEPTH)


def parse_direction(raw: str | None) -> Direction:
    text = _text(raw)
    if text is None:
        return Direction.BOTH
    try:
        return Direction(text.lower())
    except ValueError as exc:
        raise ApiInputError("direction must be out, in or both") from exc


def parse_edge_status(raw: str | None) -> EdgeStatus | None:
    text = _text(raw)
    if text is None:
        return None
    try:
        return EdgeStatus(text.lower())
    except ValueError as exc:
        raise ApiInputError("status must be validated, hypothesis or rejected") from exc


def parse_price_provider(raw: str | None) -> str:
    text = _text(raw)
    if text is None:
        return DEFAULT_PRICE_PROVIDER
    provider = text.lower()
    if provider not in PRICE_PROVIDERS:
        raise ApiInputError("provider must be cboe or yahoo")
    return provider


def parse_price_days(raw: str | None) -> int:
    return parse_int_in_range(raw, "days", default=DEFAULT_PRICE_DAYS, low=1, high=MAX_PRICE_DAYS)


def _iso(value: datetime | date | None) -> str | None:
    return value.isoformat() if value is not None else None


def node_json(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "node_type": node.node_type.value,
        "key": node.key,
        "name": node.name,
        "attributes": dict(node.attributes),
    }


def node_ref_json(node: Node | None, node_id: int) -> dict[str, Any]:
    if node is None:
        return {"id": node_id, "node_type": None, "key": None, "name": None}
    return {"id": node.id, "node_type": node.node_type.value, "key": node.key, "name": node.name}


def edge_json(edge: Edge, nodes: Mapping[int, Node]) -> dict[str, Any]:
    expected_sign = edge.attributes.get("expected_sign")
    return {
        "id": edge.id,
        "src": node_ref_json(nodes.get(edge.src_id), edge.src_id),
        "dst": node_ref_json(nodes.get(edge.dst_id), edge.dst_id),
        "relation_type": edge.relation_type.value,
        "status": edge.status.value,
        "source": edge.source,
        "expected_sign": expected_sign if isinstance(expected_sign, (int, float)) else None,
        "created_version": edge.created_version,
    }


def path_json(path: Path) -> dict[str, Any]:
    return {
        "depth": path.depth,
        "node_keys": [f"{n.node_type.value}:{n.key}" for n in path.nodes],
        "edge_ids": [e.id for e in path.edges],
    }


def graph_json(start: Node, paths: Sequence[Path]) -> dict[str, Any]:
    nodes: dict[int, Node] = {start.id: start}
    edges: dict[int, Edge] = {}
    for path in paths:
        for node in path.nodes:
            nodes.setdefault(node.id, node)
        for edge in path.edges:
            edges.setdefault(edge.id, edge)
    return {
        "start": node_json(start),
        "nodes": [node_json(n) for n in nodes.values()],
        "edges": [edge_json(e, nodes) for e in edges.values()],
        "paths": [path_json(p) for p in paths],
    }


def discover_item_json(item: StoredItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind.value,
        "subject": item.subject,
        "score": item.score,
        "summary": item.headline,
        "details": dict(item.details),
        "as_of": item.as_of.isoformat(),
        "dismissed": item.status == "dismissed",
        "modeled": item.modeled,
    }


def evidence_json(evidence: Evidence) -> dict[str, Any]:
    return {
        "id": evidence.id,
        "stance": evidence.stance.value,
        "source_type": evidence.source_type.value,
        "source_uri": evidence.source_uri,
        "source_title": evidence.source_title,
        "publisher": evidence.publisher,
        "excerpt": evidence.excerpt,
        "available_at": _iso(evidence.available_at),
        "ingested_at": _iso(evidence.ingested_at),
        "assessed_by": evidence.assessed_by,
        "model_version": evidence.model_version,
        "assessor_confidence": evidence.assessor_confidence,
    }


def evidence_summary_json(summary: EvidenceSummary) -> dict[str, Any]:
    return {
        "total": summary.total,
        "by_stance": {stance.value: count for stance, count in summary.by_stance.items()},
        "by_source_type": {src.value: count for src, count in summary.by_source_type.items()},
        "conflicting": summary.conflicting,
        "latest_available_at": _iso(summary.latest_available_at),
    }


def price_series_json(series: PriceSeries) -> dict[str, Any]:
    bars = [
        {"date": _iso(bar.ts), "close": bar.close}
        for bar in series.bars
    ]
    return {
        "symbol": series.symbol,
        "provider": series.provenance.provider,
        "interval": series.interval,
        "fetched_at": _iso(series.provenance.fetched_at),
        "count": len(bars),
        "bars": bars,
    }


def options_analysis_json(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Shape an ``OptionsAnalysis.to_dict()`` result; modeled values live under ``modeled`` only."""
    modeled_src = analysis.get("modeled")
    if not isinstance(modeled_src, Mapping):
        raise ValueError("options analysis is missing its modeled section")
    modeled = {
        **modeled_src,
        "modeled": True,
        "label": MODELED_LABEL,
    }
    return {
        "underlying": analysis.get("underlying"),
        "data_quality": analysis.get("data_quality"),
        "observed": analysis.get("observed"),
        "modeled": modeled,
    }


def research_card_json(card: Mapping[str, Any]) -> dict[str, Any]:
    """Shape a Research Card: an available options section keeps observed and MODELED / ESTIMATED apart."""
    out: dict[str, Any] = dict(card)
    options = card.get("options")
    if isinstance(options, Mapping) and options.get("status") == "available":
        analysis = {k: v for k, v in options.items() if k != "status"}
        out["options"] = {"status": "available", **options_analysis_json(analysis)}
    return out
