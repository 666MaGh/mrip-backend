"""MRIP HTTP API: endpoints against in-memory fakes, DB-free, auth patched."""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

import app.routes.mrip as mrip_routes
import app.utils.auth as auth
from app.mrip.api.serializers import (
    ApiInputError,
    evidence_summary_json,
    options_analysis_json,
    parse_bool,
    parse_depth,
    parse_kinds,
    parse_price_days,
    parse_price_provider,
)
from app.mrip.data.models import Latency, OptionContract, OptionsChainSnapshot, PriceBar, PriceSeries, Provenance
from app.mrip.discover.store import StoredItem
from app.mrip.discover.types import Kind
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType, Stance
from app.mrip.relationships.types import Direction, Edge, EdgeStatus, Node, NodeKey, NodeType, Path, RelationType
from app.mrip.options.analysis import MODELED_LABEL, analyze_options
from app.mrip.related.service import RelatedService
from app.mrip.related.types import UnknownSymbol
from app.mrip.research.types import UnknownSecurity

AUTH = {"Authorization": "Bearer test-token"}
NOW = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)


# -- auth + app ---------------------------------------------------------------

@pytest.fixture
def authed(monkeypatch):
    """Accept the test token as a valid, active user (same seam as other auth tests)."""
    payload = {"sub": "tester", "user_id": 1, "_verified_user_role": "user", "_verified_username": "tester"}
    monkeypatch.setattr(auth, "verify_token", lambda token: payload if token == "test-token" else None)


# -- fakes ------------------------------------------------------------------

def _item(item_id: int, kind: Kind = Kind.COT_CROWDING, status: str = "open", modeled: bool = False) -> StoredItem:
    return StoredItem(
        id=item_id, as_of=date(2026, 10, 1), kind=kind, subject="ES", headline="Crowded long",
        magnitude=0.7, score=0.81, components={"a": 0.5}, details={"net": 12}, data_quality={},
        modeled=modeled, policy_version="v1", status=status,
    )


class FakeDiscover:
    def __init__(self, items: list[StoredItem]) -> None:
        self.items = {i.id: i for i in items}
        self.calls: list[dict] = []

    def feed(self, as_of=None, *, limit=50, kinds=None, include_dismissed=False):
        self.calls.append({"as_of": as_of, "limit": limit, "kinds": kinds, "include_dismissed": include_dismissed})
        return list(self.items.values())[:limit]

    def dismiss(self, item_id: int) -> bool:
        if item_id not in self.items:
            return False
        self.items[item_id] = _item(item_id, status="dismissed")
        return True


NODES = {
    1: Node(1, NodeType.THEME, "ai-infrastructure", "AI infrastructure"),
    2: Node(2, NodeType.COMPANY, "NVDA", "NVIDIA"),
    3: Node(3, NodeType.COMPANY, "SMCI", "Super Micro"),
}


def _edge(edge_id: int, src: int, dst: int, status: EdgeStatus = EdgeStatus.HYPOTHESIS,
          attributes: dict | None = None) -> Edge:
    return Edge(
        id=edge_id, src_id=src, dst_id=dst, relation_type=RelationType.SUPPLIES, status=status,
        source="seed", created_version=1, attributes=attributes or {"expected_sign": 1},
    )


EDGES = {
    10: _edge(10, 1, 2),
    11: _edge(11, 2, 3, EdgeStatus.VALIDATED),
}


class FakeGraph:
    def __init__(self) -> None:
        self.traverse_calls: list[dict] = []

    def get_node(self, node: NodeKey) -> Node | None:
        for n in NODES.values():
            if n.node_type is node.node_type and n.key == node.key:
                return n
        return None

    def get_node_by_id(self, node_id: int) -> Node | None:
        return NODES.get(node_id)

    def get_nodes_by_ids(self, ids):
        return {i: NODES[i] for i in ids if i in NODES}

    def traverse(self, start, *, max_depth=3, direction=Direction.OUT, **_):
        self.traverse_calls.append({"start": start, "max_depth": max_depth, "direction": direction})
        return [Path(nodes=(NODES[1], NODES[2]), edges=(EDGES[10],)),
                Path(nodes=(NODES[1], NODES[2], NODES[3]), edges=(EDGES[10], EDGES[11]))]

    def list_edges(self, statuses, limit=200):
        return [e for e in EDGES.values() if e.status in statuses][:limit]

    def get_edge(self, edge_id: int):
        return EDGES.get(edge_id)


def _evidence(stance: Stance) -> Evidence:
    ts = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    return Evidence(
        id=1, src_id=2, dst_id=3, relation_type=RelationType.SUPPLIES, stance=stance,
        source_type=SourceType.EARNINGS_TRANSCRIPT, source_uri="https://example.test/t", available_at=ts,
        ingested_at=ts, assessed_by="laya-test", assessor_confidence=0.6,
    )


class FakeEvidence:
    def __init__(self, items: list[Evidence]) -> None:
        self.items = items
        self.refs: list[RelationshipRef] = []

    def list_evidence(self, ref: RelationshipRef, **_):
        self.refs.append(ref)
        return list(self.items)


def _series(provider: str) -> PriceSeries:
    bars = (
        PriceBar(ts=date(2026, 9, 29), open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0),
        PriceBar(ts=date(2026, 9, 30), open=1.5, high=2.5, low=1.0, close=2.25, volume=12.0),
    )
    prov = Provenance(provider=provider, gateway="mrip-store", endpoint="mrip_price_bars",
                      fetched_at=NOW, latency=Latency.UNKNOWN)
    return PriceSeries(symbol="SPY", interval="1d", bars=bars, provenance=prov)


class FakePrices:
    def __init__(self, known: set[str]) -> None:
        self.known = known
        self.calls: list[tuple] = []

    def series(self, provider, symbol, start=None, end=None):
        self.calls.append((provider, symbol, start, end))
        if symbol not in self.known:
            return None
        return _series(provider)


def _chain() -> OptionsChainSnapshot:
    exp = date(2026, 10, 16)
    prov = Provenance("cboe", "openbb", "cboe.options.chains", NOW, Latency.DELAYED)
    contracts = (
        OptionContract("C100", exp, 100.0, "call", open_interest=1000, volume=300,
                       implied_volatility=0.2, gamma=0.02),
        OptionContract("P100", exp, 100.0, "put", open_interest=800, volume=150,
                       implied_volatility=0.22, gamma=0.02),
        OptionContract("C105", exp, 105.0, "call", open_interest=500, volume=50,
                       implied_volatility=0.19, gamma=0.01),
    )
    return OptionsChainSnapshot(
        underlying="SPY", underlying_price=100.0, underlying_timestamp=NOW, snapshot_timestamp=NOW,
        oi_effective_date=date(2026, 9, 30), contracts=contracts, provenance=prov,
    )


class FakeSnapshots:
    def __init__(self, chain: OptionsChainSnapshot | None) -> None:
        self.chain = chain

    def latest_before(self, underlying: str, as_of: datetime):
        if self.chain is None or underlying != self.chain.underlying:
            return None
        return self.chain


@pytest.fixture
def fakes(monkeypatch, authed):
    fx = {
        "discover": FakeDiscover([_item(1), _item(2, kind=Kind.REGIME_SHIFT, modeled=True)]),
        "graph": FakeGraph(),
        "evidence": FakeEvidence([_evidence(Stance.SUPPORT), _evidence(Stance.CONTRADICT)]),
        "prices": FakePrices({"SPY"}),
        "snapshots": FakeSnapshots(_chain()),
    }
    monkeypatch.setattr(mrip_routes, "_discover_store", lambda: fx["discover"])
    monkeypatch.setattr(mrip_routes, "_graph", lambda: fx["graph"])
    monkeypatch.setattr(mrip_routes, "_evidence_store", lambda: fx["evidence"])
    monkeypatch.setattr(mrip_routes, "_price_store", lambda: fx["prices"])
    monkeypatch.setattr(mrip_routes, "_snapshot_store", lambda: fx["snapshots"])
    return fx


# -- discover ----------------------------------------------------------------

def test_discover_feed_happy_path(client, fakes):
    resp = client.get("/api/mrip/discover?limit=10&kinds=COT_CROWDING,REGIME_SHIFT&date=2026-10-01", headers=AUTH)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["code"] == 1
    first = body["data"][0]
    assert first == {
        "id": 1, "kind": "COT_CROWDING", "subject": "ES", "score": 0.81, "summary": "Crowded long",
        "details": {"net": 12}, "as_of": "2026-10-01", "dismissed": False, "modeled": False,
    }
    call = fakes["discover"].calls[-1]
    assert call["limit"] == 10
    assert call["kinds"] == (Kind.COT_CROWDING, Kind.REGIME_SHIFT)
    assert call["as_of"] == date(2026, 10, 1)
    assert call["include_dismissed"] is False


def test_discover_feed_rejects_bad_input(client, fakes):
    for query in ("limit=0", "limit=201", "limit=abc", "kinds=NOPE", "date=2026-13-01", "include_dismissed=maybe"):
        resp = client.get(f"/api/mrip/discover?{query}", headers=AUTH)
        assert resp.status_code == 400, query
        assert resp.get_json()["code"] == 0


def test_discover_dismiss_happy_path_and_404(client, fakes):
    resp = client.post("/api/mrip/discover/1/dismiss", headers=AUTH)
    assert resp.status_code == 200
    assert resp.get_json() == {"code": 1, "data": {"id": 1, "dismissed": True}}
    missing = client.post("/api/mrip/discover/999/dismiss", headers=AUTH)
    assert missing.status_code == 404
    assert missing.get_json()["code"] == 0


# -- relationships -------------------------------------------------------------

def test_graph_traversal_happy_path(client, fakes):
    resp = client.get("/api/mrip/relationships/graph?node_type=THEME&key=ai-infrastructure&depth=2&direction=both",
                      headers=AUTH)
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["start"]["key"] == "ai-infrastructure"
    assert {n["key"] for n in data["nodes"]} == {"ai-infrastructure", "NVDA", "SMCI"}
    edge = next(e for e in data["edges"] if e["id"] == 10)
    assert edge["relation_type"] == "SUPPLIES"
    assert edge["status"] == "hypothesis"
    assert edge["source"] == "seed"
    assert edge["expected_sign"] == 1
    assert edge["src"]["name"] == "AI infrastructure"
    assert data["paths"][1] == {
        "depth": 2, "node_keys": ["THEME:ai-infrastructure", "COMPANY:NVDA", "COMPANY:SMCI"], "edge_ids": [10, 11],
    }
    call = fakes["graph"].traverse_calls[-1]
    assert call["max_depth"] == 2
    assert call["direction"] is Direction.BOTH


def test_graph_traversal_errors(client, fakes):
    assert client.get("/api/mrip/relationships/graph?node_type=THEME&key=x&depth=7", headers=AUTH).status_code == 400
    assert client.get("/api/mrip/relationships/graph?node_type=THEME&key=x&depth=0", headers=AUTH).status_code == 400
    assert client.get("/api/mrip/relationships/graph?node_type=NOPE&key=x", headers=AUTH).status_code == 400
    assert client.get("/api/mrip/relationships/graph?node_type=THEME", headers=AUTH).status_code == 400
    assert client.get("/api/mrip/relationships/graph?node_type=THEME&key=x&direction=sideways",
                      headers=AUTH).status_code == 400
    unknown = client.get("/api/mrip/relationships/graph?node_type=THEME&key=missing", headers=AUTH)
    assert unknown.status_code == 404


def test_edges_happy_path_with_node_names(client, fakes):
    resp = client.get("/api/mrip/relationships/edges?limit=5", headers=AUTH)
    assert resp.status_code == 200
    edges = resp.get_json()["data"]
    assert [e["id"] for e in edges] == [10, 11]
    assert edges[0]["src"] == {"id": 1, "node_type": "THEME", "key": "ai-infrastructure", "name": "AI infrastructure"}
    assert edges[1]["status"] == "validated"


def test_edges_status_filter_and_bad_status(client, fakes):
    resp = client.get("/api/mrip/relationships/edges?status=validated", headers=AUTH)
    assert [e["id"] for e in resp.get_json()["data"]] == [11]
    assert client.get("/api/mrip/relationships/edges?status=bogus", headers=AUTH).status_code == 400
    assert client.get("/api/mrip/relationships/edges?limit=500", headers=AUTH).status_code == 400


def test_edge_evidence_happy_path(client, fakes):
    resp = client.get("/api/mrip/relationships/edges/10/evidence", headers=AUTH)
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["edge"]["id"] == 10
    assert len(data["evidence"]) == 2
    assert data["evidence"][0]["stance"] == "support"
    assert data["summary"]["total"] == 2
    assert data["summary"]["conflicting"] is True
    assert data["summary"]["by_stance"] == {"support": 1, "contradict": 1, "neutral": 0}
    ref = fakes["evidence"].refs[-1]
    assert ref.src == NodeKey(NodeType.THEME, "ai-infrastructure")
    assert ref.dst == NodeKey(NodeType.COMPANY, "NVDA")
    assert ref.relation_type is RelationType.SUPPLIES


def test_edge_evidence_unknown_edge_is_404(client, fakes):
    resp = client.get("/api/mrip/relationships/edges/999/evidence", headers=AUTH)
    assert resp.status_code == 404
    assert resp.get_json()["code"] == 0


# -- prices -----------------------------------------------------------------

def test_prices_happy_path(client, fakes):
    resp = client.get("/api/mrip/prices/SPY?provider=cboe&days=30", headers=AUTH)
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["provider"] == "cboe"
    assert data["bars"] == [{"date": "2026-09-29", "close": 1.5}, {"date": "2026-09-30", "close": 2.25}]
    assert data["count"] == 2
    provider, symbol, start, end = fakes["prices"].calls[-1]
    assert (provider, symbol) == ("cboe", "SPY")
    assert (end - start).days == 30


def test_prices_bad_input_and_unknown_symbol(client, fakes):
    assert client.get("/api/mrip/prices/SPY?provider=ibkr", headers=AUTH).status_code == 400
    assert client.get("/api/mrip/prices/SPY?days=5001", headers=AUTH).status_code == 400
    assert client.get("/api/mrip/prices/SPY?days=0", headers=AUTH).status_code == 400
    missing = client.get("/api/mrip/prices/NOPE", headers=AUTH)
    assert missing.status_code == 404
    assert missing.get_json()["code"] == 0


# -- options ----------------------------------------------------------------

def test_options_latest_marks_modeled_values(client, fakes):
    resp = client.get("/api/mrip/options/SPY/latest", headers=AUTH)
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["underlying"] == "SPY"
    assert "observed" in data and "total_open_interest" in data["observed"]
    assert "gex" not in data["observed"]
    modeled = data["modeled"]
    assert modeled["label"] == MODELED_LABEL
    assert modeled["modeled"] is True
    assert "gex" in modeled and "regime" in modeled and "amplification" in modeled
    assert "gex" not in data["observed"]


def test_options_latest_404_without_snapshot(client, monkeypatch, authed):
    monkeypatch.setattr(mrip_routes, "_snapshot_store", lambda: FakeSnapshots(None))
    resp = client.get("/api/mrip/options/SPY/latest", headers=AUTH)
    assert resp.status_code == 404
    assert resp.get_json()["code"] == 0


def test_options_shaping_rejects_missing_modeled_section():
    with pytest.raises(ValueError):
        options_analysis_json({"underlying": "SPY", "observed": {}})


# -- auth -------------------------------------------------------------------

@pytest.mark.parametrize("method,path", [
    ("GET", "/api/mrip/discover"),
    ("POST", "/api/mrip/discover/1/dismiss"),
    ("GET", "/api/mrip/relationships/graph?node_type=THEME&key=ai-infrastructure"),
    ("GET", "/api/mrip/relationships/edges"),
    ("GET", "/api/mrip/relationships/edges/10/evidence"),
    ("GET", "/api/mrip/prices/SPY"),
    ("GET", "/api/mrip/options/SPY/latest"),
])
def test_unauthenticated_calls_are_rejected(client, fakes, method, path):
    resp = client.open(path, method=method)
    assert resp.status_code == 401
    assert resp.get_json()["code"] == 401


# -- pure parsing helpers ---------------------------------------------------

def test_parsers_accept_valid_values_and_reject_invalid():
    assert parse_kinds(" cot_crowding , regime_shift ") == (Kind.COT_CROWDING, Kind.REGIME_SHIFT)
    assert parse_bool("TRUE", "x", default=False) is True
    assert parse_depth(None) == 3
    assert parse_price_provider(None) == "yahoo"
    assert parse_price_days(None) == 365
    with pytest.raises(ApiInputError):
        parse_kinds(" , ")
    with pytest.raises(ApiInputError):
        parse_depth("9")
    with pytest.raises(ApiInputError):
        parse_price_provider("alpaca")


def test_evidence_summary_shaping_uses_plain_values():
    from app.mrip.evidence.store import summarize

    summary = evidence_summary_json(summarize([_evidence(Stance.SUPPORT)]))
    assert summary["by_source_type"] == {"EARNINGS_TRANSCRIPT": 1}
    assert summary["latest_available_at"] == "2026-09-30T12:00:00+00:00"


# -- research card ------------------------------------------------------------

class FakeResearchCards:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, date]] = []

    def build_card(self, symbol: str, as_of: date) -> dict:
        self.calls.append((symbol, as_of))
        if self.error is not None:
            raise self.error
        analysis = analyze_options(_chain()).to_dict()
        return {
            "symbol": symbol, "as_of": as_of.isoformat(), "card_version": "research-card-v0-uncalibrated",
            "options": {"status": "available", **analysis},
            "cot": {"status": "unavailable", "reason": "no COT market mapped"},
            "why": [],
        }


@pytest.fixture
def research(monkeypatch, authed):
    fake = FakeResearchCards()
    monkeypatch.setattr(mrip_routes, "_research_card_service", lambda: fake)
    return fake


def test_research_card_happy_path_and_as_of(client, research):
    resp = client.get("/api/mrip/research/NVDA?as_of=2026-10-01", headers=AUTH)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["code"] == 1
    assert body["data"]["as_of"] == "2026-10-01"
    assert body["data"]["cot"]["status"] == "unavailable"
    assert research.calls == [("NVDA", date(2026, 10, 1))]


def test_research_card_options_keep_modeled_separate(client, research):
    data = client.get("/api/mrip/research/NVDA?as_of=2026-10-01", headers=AUTH).get_json()["data"]
    options = data["options"]
    assert options["status"] == "available"
    assert options["modeled"]["label"] == MODELED_LABEL
    assert options["modeled"]["modeled"] is True
    assert "gex" in options["modeled"]
    assert "gex" not in options["observed"]


def test_research_card_defaults_to_today_utc(client, research):
    resp = client.get("/api/mrip/research/NVDA", headers=AUTH)
    assert resp.status_code == 200
    assert research.calls[-1][1] == datetime.now(timezone.utc).date()


def test_research_card_rejects_bad_as_of(client, research):
    resp = client.get("/api/mrip/research/NVDA?as_of=2026-13-01", headers=AUTH)
    assert resp.status_code == 400
    assert resp.get_json()["code"] == 0
    assert research.calls == []


def test_research_card_unknown_security_is_404(client, monkeypatch, authed):
    fake = FakeResearchCards(error=UnknownSecurity("ZZZZ"))
    monkeypatch.setattr(mrip_routes, "_research_card_service", lambda: fake)
    resp = client.get("/api/mrip/research/ZZZZ?as_of=2026-10-01", headers=AUTH)
    assert resp.status_code == 404
    assert resp.get_json() == {"code": 0, "msg": "unknown security ZZZZ"}


def test_research_card_requires_auth(client, research):
    resp = client.get("/api/mrip/research/NVDA")
    assert resp.status_code == 401
    assert research.calls == []


# -- related neighbours -------------------------------------------------------

class FakeRelated:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, date, bool]] = []

    def build_related(self, symbol: str, as_of: date, include_hypothesis: bool = False) -> dict:
        self.calls.append((symbol, as_of, include_hypothesis))
        if self.error is not None:
            raise self.error
        return {"symbol": symbol, "as_of": as_of.isoformat(), "include_hypothesis": include_hypothesis,
                "queried": None, "rows": [{"edge_id": 1, "role": "src"}], "meta": {"disclaimer": "x"}}


@pytest.fixture
def related(monkeypatch, authed):
    stub = FakeRelated()
    monkeypatch.setattr(mrip_routes, "_related_service", lambda: stub)
    return stub


def test_related_happy_path_passes_normalised_inputs(client, related):
    resp = client.get("/api/mrip/symbols/nvda/related?as_of=2026-10-01&include_hypothesis=true", headers=AUTH)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["code"] == 1
    assert body["data"]["rows"] == [{"edge_id": 1, "role": "src"}]
    assert related.calls == [("nvda", date(2026, 10, 1), True)]


def test_related_defaults_exclude_hypothesis(client, related):
    assert client.get("/api/mrip/symbols/NVDA/related", headers=AUTH).status_code == 200
    assert related.calls[0][2] is False


def test_related_rejects_bad_input(client, related):
    assert client.get("/api/mrip/symbols/NVDA/related?as_of=2026-13-01", headers=AUTH).status_code == 400
    assert client.get("/api/mrip/symbols/NVDA/related?include_hypothesis=maybe", headers=AUTH).status_code == 400
    assert related.calls == []


def test_related_unknown_symbol_is_404(monkeypatch, client, authed):
    monkeypatch.setattr(mrip_routes, "_related_service", lambda: FakeRelated(UnknownSymbol("ZZZZ")))
    resp = client.get("/api/mrip/symbols/ZZZZ/related", headers=AUTH)
    assert resp.status_code == 404
    assert resp.get_json()["code"] == 0


def test_related_requires_auth(client, related):
    assert client.get("/api/mrip/symbols/NVDA/related").status_code == 401
    assert related.calls == []


def test_related_service_factory_wires_the_three_stores(monkeypatch):
    monkeypatch.setattr(mrip_routes, "_graph", lambda: "graph")
    monkeypatch.setattr(mrip_routes, "_evidence_store", lambda: "evidence")
    monkeypatch.setattr(mrip_routes, "_price_store", lambda: "prices")
    service = mrip_routes._related_service()
    assert isinstance(service, RelatedService)
    assert (service._graph, service._evidence, service._prices) == ("graph", "evidence", "prices")
