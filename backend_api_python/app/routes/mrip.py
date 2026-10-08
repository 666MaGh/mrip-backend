"""MRIP read-mostly HTTP API (mounted at ``/api/mrip``).

Every endpoint requires a Bearer token. Responses use the human envelope
``{"code": 1, "data": ...}``; errors use ``{"code": 0, "msg": ...}``.
Modeled options quantities are labelled ``MODELED / ESTIMATED`` in the response.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from flask import jsonify, request
from app.openapi.blueprint import HumanBlueprint as Blueprint

from app.mrip.api.serializers import (
    DEFAULT_DISCOVER_LIMIT,
    DEFAULT_EDGE_LIMIT,
    MAX_LIST_LIMIT,
    ApiInputError,
    discover_item_json,
    edge_json,
    evidence_json,
    evidence_summary_json,
    graph_json,
    options_analysis_json,
    parse_bool,
    parse_date,
    parse_depth,
    parse_direction,
    parse_edge_status,
    parse_int_in_range,
    parse_kinds,
    parse_key,
    parse_node_type,
    parse_price_days,
    parse_price_provider,
    price_series_json,
    research_card_json,
)
from app.mrip.calibration.store import CalibrationStore
from app.mrip.cot.engine import CotEngine
from app.mrip.discover.store import DiscoverStore
from app.mrip.evidence.store import EvidenceStore, summarize
from app.mrip.evidence.types import RelationshipRef
from app.mrip.options.analysis import analyze_options
from app.mrip.options.snapshots import OptionsSnapshotStore
from app.mrip.outcomes.store import PredictionStore
from app.mrip.prices.gateway import StoredDataGateway
from app.mrip.prices.store import PriceStore
from app.mrip.regime.engine import MarketRegimeEngine
from app.mrip.research.card import ResearchCardService
from app.mrip.research.types import UnknownSecurity
from app.mrip.relationships.graph import RelationshipGraph
from app.mrip.relationships.types import EdgeStatus, GraphError, NodeKey
from app.utils.auth import login_required
from app.utils.db import get_db_connection

mrip_blp = Blueprint('mrip', __name__)


# -- store factories (monkeypatched in tests) --------------------------------

def _discover_store() -> DiscoverStore:
    return DiscoverStore(get_db_connection)


def _graph() -> RelationshipGraph:
    return RelationshipGraph(get_db_connection)


def _evidence_store() -> EvidenceStore:
    return EvidenceStore(get_db_connection)


def _price_store() -> PriceStore:
    return PriceStore(get_db_connection)


def _snapshot_store() -> OptionsSnapshotStore:
    return OptionsSnapshotStore(get_db_connection)


def _research_card_service() -> ResearchCardService:
    # Stored data only: the HTTP path makes no network calls (live providers stay in the jobs).
    prices = _price_store()
    gateway = StoredDataGateway(prices)
    return ResearchCardService(
        _graph(), _evidence_store(), prices,
        regime_engine=MarketRegimeEngine(gateway),
        cot_engine=CotEngine(gateway),
        snapshot_store=_snapshot_store(),
        outcome_store=PredictionStore(get_db_connection),
        calibration_store=CalibrationStore(get_db_connection),
    )


# -- helpers ----------------------------------------------------------------

def _ok(data: Any):
    return jsonify({"code": 1, "data": data})


def _error(msg: str, status: int):
    return jsonify({"code": 0, "msg": msg}), status


def _arg(name: str) -> str | None:
    return request.args.get(name)


# -- discover ---------------------------------------------------------------

@mrip_blp.route('/discover', methods=['GET'])
@login_required
def list_discover_feed():
    """Ranked MRIP Discover feed for a day (latest day when ``date`` is omitted)."""
    try:
        limit = parse_int_in_range(
            _arg('limit'), 'limit', default=DEFAULT_DISCOVER_LIMIT, low=1, high=MAX_LIST_LIMIT,
        )
        kinds = parse_kinds(_arg('kinds'))
        include_dismissed = parse_bool(_arg('include_dismissed'), 'include_dismissed', default=False)
        as_of = parse_date(_arg('date'), 'date')
    except ApiInputError as exc:
        return _error(str(exc), 400)
    items = _discover_store().feed(
        as_of, limit=limit, kinds=kinds, include_dismissed=include_dismissed,
    )
    return _ok([discover_item_json(item) for item in items])


@mrip_blp.route('/discover/<int:item_id>/dismiss', methods=['POST'])
@login_required
def dismiss_discover_item(item_id: int):
    """Mark a discover item as dismissed (it stays in the record)."""
    if not _discover_store().dismiss(item_id):
        return _error(f"discover item {item_id} not found", 404)
    return _ok({"id": item_id, "dismissed": True})


# -- relationships ------------------------------------------------------------

@mrip_blp.route('/relationships/graph', methods=['GET'])
@login_required
def traverse_relationship_graph():
    """Paths and nodes reachable from one node (current graph, hypotheses and validated edges)."""
    try:
        node_type = parse_node_type(_arg('node_type'))
        key = parse_key(_arg('key'))
        depth = parse_depth(_arg('depth'))
        direction = parse_direction(_arg('direction'))
    except ApiInputError as exc:
        return _error(str(exc), 400)
    graph = _graph()
    start = graph.get_node(NodeKey(node_type, key))
    if start is None:
        return _error(f"node {node_type.value}:{key} not found", 404)
    try:
        paths = graph.traverse(NodeKey(node_type, key), max_depth=depth, direction=direction)
    except GraphError as exc:
        return _error(str(exc), 400)
    return _ok(graph_json(start, paths))


@mrip_blp.route('/relationships/edges', methods=['GET'])
@login_required
def list_relationship_edges():
    """Edges by status (default: hypothesis and validated), with node names."""
    try:
        status = parse_edge_status(_arg('status'))
        limit = parse_int_in_range(
            _arg('limit'), 'limit', default=DEFAULT_EDGE_LIMIT, low=1, high=MAX_LIST_LIMIT,
        )
    except ApiInputError as exc:
        return _error(str(exc), 400)
    graph = _graph()
    statuses = [status] if status is not None else [EdgeStatus.HYPOTHESIS, EdgeStatus.VALIDATED]
    edges = graph.list_edges(statuses, limit=limit)
    nodes = graph.get_nodes_by_ids(sorted({e.src_id for e in edges} | {e.dst_id for e in edges}))
    return _ok([edge_json(edge, nodes) for edge in edges])


@mrip_blp.route('/relationships/edges/<int:edge_id>/evidence', methods=['GET'])
@login_required
def list_edge_evidence(edge_id: int):
    """Evidence and its summary for one edge (current, non-retracted items)."""
    graph = _graph()
    edge = graph.get_edge(edge_id)
    if edge is None:
        return _error(f"edge {edge_id} not found", 404)
    nodes = graph.get_nodes_by_ids([edge.src_id, edge.dst_id])
    src, dst = nodes.get(edge.src_id), nodes.get(edge.dst_id)
    if src is None or dst is None:
        return _error(f"edge {edge_id} references a missing node", 404)
    ref = RelationshipRef(
        src=NodeKey(src.node_type, src.key),
        dst=NodeKey(dst.node_type, dst.key),
        relation_type=edge.relation_type,
    )
    items = _evidence_store().list_evidence(ref)
    return _ok({
        "edge": edge_json(edge, nodes),
        "evidence": [evidence_json(e) for e in items],
        "summary": evidence_summary_json(summarize(items)),
    })


# -- prices -----------------------------------------------------------------

@mrip_blp.route('/prices/<symbol>', methods=['GET'])
@login_required
def get_price_series(symbol: str):
    """Daily closes from the stored price series for the last ``days`` days."""
    try:
        provider = parse_price_provider(_arg('provider'))
        days = parse_price_days(_arg('days'))
    except ApiInputError as exc:
        return _error(str(exc), 400)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    series = _price_store().series(provider, symbol, start, end)
    if series is None:
        return _error(f"no stored prices for {symbol} from {provider}", 404)
    return _ok(price_series_json(series))


# -- options ----------------------------------------------------------------

@mrip_blp.route('/options/<symbol>/latest', methods=['GET'])
@login_required
def get_latest_options_analysis(symbol: str):
    """Analysis of the newest stored chain; observed and modeled parts kept apart."""
    snapshot = _snapshot_store().latest_before(symbol, datetime.now(timezone.utc))
    if snapshot is None:
        return _error(f"no stored options snapshot for {symbol}", 404)
    analysis = analyze_options(snapshot).to_dict()
    return _ok(options_analysis_json(analysis))


# -- research card ----------------------------------------------------------

@mrip_blp.route('/research/<symbol>', methods=['GET'])
@login_required
def get_research_card(symbol: str):
    """Per-security Research Card as of a date (default today UTC); unavailable sections carry a reason."""
    try:
        as_of = parse_date(_arg('as_of'), 'as_of') or datetime.now(timezone.utc).date()
    except ApiInputError as exc:
        return _error(str(exc), 400)
    try:
        card = _research_card_service().build_card(symbol, as_of)
    except UnknownSecurity:
        return _error(f"unknown security {symbol}", 404)
    return _ok(research_card_json(card))
