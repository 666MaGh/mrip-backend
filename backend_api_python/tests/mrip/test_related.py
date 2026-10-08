"""Related neighbours (work 019): every collaborator is a fake; no network, no database."""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.mrip.data.models import Latency, PriceBar, PriceSeries, Provenance
from app.mrip.discover.detectors import DiscoverPolicy
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType, Stance
from app.mrip.related.service import RelatedService, relation_sign, signal_for, translate_direction
from app.mrip.related.types import DISCLAIMER, SIGNAL_LABEL, UnknownSymbol
from app.mrip.relationships.types import Direction, Edge, EdgeStatus, Node, NodeKey, NodeType, Path, RelationType

AS_OF = date(2026, 10, 1)
NOW = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)
N_BARS = 400
SHOCK = 0.05  # daily log-return shock that makes the neighbour's recent residual extreme
POLICY = DiscoverPolicy()
THRESHOLD = POLICY.divergence_z

NVDA = Node(1, NodeType.COMPANY, "NVDA", "NVIDIA", {"series": {"symbol": "NVDA"}})
MSFT = Node(2, NodeType.COMPANY, "MSFT", "Microsoft", {"series": {"symbol": "MSFT"}})
THEME = Node(3, NodeType.THEME, "ai-infrastructure", "AI infrastructure", {})
NODES = {n.id: n for n in (NVDA, MSFT, THEME)}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise AssertionError("related tests must not touch the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)


# -- fakes ------------------------------------------------------------------

def _edge(edge_id: int, src: int, dst: int, status: EdgeStatus = EdgeStatus.VALIDATED,
          expected_sign: int | None = 1, relation: RelationType = RelationType.CORRELATED_WITH) -> Edge:
    attributes = {} if expected_sign is None else {"expected_sign": expected_sign}
    return Edge(id=edge_id, src_id=src, dst_id=dst, relation_type=relation, status=status,
                source="seed", created_version=1, attributes=attributes)


def _validation(src: int, dst: int, *, best_lag: int = 0, direction: str = "contemporaneous",
                available: date = date(2026, 9, 30), relation: RelationType = RelationType.CORRELATED_WITH,
                retracted: bool = False, source: SourceType = SourceType.STATISTICAL_TEST) -> Evidence:
    excerpt = json.dumps({"verdict": "validated", "reasons": [], "metrics": {
        "best_lag": best_lag, "direction": direction, "r": 0.61, "p_corrected": 0.001,
        "end": "2026-09-30", "expected_sign": 1,
    }}, sort_keys=True)
    ts = datetime.combine(available, datetime.min.time(), tzinfo=timezone.utc)
    return Evidence(
        id=abs(hash((src, dst, best_lag, direction, available.isoformat()))) % 10_000_000 + 1,
        src_id=src, dst_id=dst, relation_type=relation, stance=Stance.SUPPORT, source_type=source,
        source_uri=f"mrip://validation/stats-v1/{src}->{dst}/{relation.value}", available_at=ts, ingested_at=ts,
        assessed_by="stats:stats-v1", excerpt=excerpt, retracted_at=ts if retracted else None,
    )


class FakeGraph:
    def __init__(self, edges: list[Edge], nodes: dict[int, Node] = NODES) -> None:
        self.edges = edges
        self.nodes = nodes
        self.traverse_calls: list[dict] = []

    def get_node(self, node: NodeKey) -> Node | None:
        for n in self.nodes.values():
            if n.node_type is node.node_type and n.key == node.key:
                return n
        return None

    def traverse(self, start: NodeKey, *, max_depth: int, direction: Direction, statuses) -> list[Path]:
        self.traverse_calls.append({"max_depth": max_depth, "direction": direction, "statuses": frozenset(statuses)})
        node = self.get_node(start)
        paths: list[Path] = []
        for edge in self.edges:
            if edge.status not in statuses:
                continue
            if direction in (Direction.OUT, Direction.BOTH) and edge.src_id == node.id:
                paths.append(Path(nodes=(node, self.nodes[edge.dst_id]), edges=(edge,)))
            elif direction in (Direction.IN, Direction.BOTH) and edge.dst_id == node.id:
                paths.append(Path(nodes=(node, self.nodes[edge.src_id]), edges=(edge,)))
        return paths


class FakeEvidence:
    def __init__(self, items: list[Evidence], error_for: set[int] | None = None) -> None:
        self.items = items
        self.error_for = error_for or set()

    def list_evidence(self, ref: RelationshipRef, *, as_of: datetime | None = None, **_) -> list[Evidence]:
        src = next(n.id for n in NODES.values() if n.node_type is ref.src.node_type and n.key == ref.src.key)
        dst = next(n.id for n in NODES.values() if n.node_type is ref.dst.node_type and n.key == ref.dst.key)
        if dst in self.error_for or src in self.error_for:
            raise RuntimeError("evidence store down for this edge")
        return [
            e for e in self.items
            if e.src_id == src and e.dst_id == dst and e.relation_type is ref.relation_type
            and (as_of is None or e.available_at <= as_of)
        ]


def _bars(returns: np.ndarray) -> tuple[PriceBar, ...]:
    days = [ts.date() for ts in pd.bdate_range(end=AS_OF, periods=len(returns))]
    closes = 100.0 * np.exp(np.cumsum(returns))
    return tuple(PriceBar(ts=d, open=float(c), high=float(c), low=float(c), close=float(c), volume=1e6)
                 for d, c in zip(days, closes))


class FakePrices:
    def __init__(self, bars_by_symbol: dict[str, tuple[PriceBar, ...]], fail: set[str] | None = None) -> None:
        self.bars_by_symbol = bars_by_symbol
        self.fail = fail or set()
        self.calls: list[tuple[str, str, date | None]] = []

    def series(self, provider: str, symbol: str, start=None, end=None) -> PriceSeries | None:
        self.calls.append((provider, symbol, end))
        if symbol in self.fail:
            raise RuntimeError("price store down")
        bars = self.bars_by_symbol.get(symbol, ())
        if end is not None:
            bars = tuple(b for b in bars if b.ts <= end)
        if not bars:
            return None
        prov = Provenance(provider=provider, gateway="mrip-store", endpoint="mrip_price_bars",
                          fetched_at=NOW, latency=Latency.UNKNOWN)
        return PriceSeries(symbol=symbol, interval="1d", bars=bars, provenance=prov)


def _returns(seed: int, vol: float = 0.01) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, vol, N_BARS)


def _prices(*, msft_shock: float = SHOCK, shock_days: int = 5) -> dict[str, tuple[PriceBar, ...]]:
    """NVDA random walk; MSFT tracks NVDA with noise, plus a shock on its last ``shock_days`` returns."""
    nvda = _returns(1)
    msft = nvda + _returns(2, vol=0.002)
    if shock_days:
        msft[-shock_days:] += msft_shock
    return {"NVDA": _bars(nvda), "MSFT": _bars(msft)}


def _service(edges, items=(), prices=None, graph_nodes=NODES, evidence_error=None, price_fail=None) -> RelatedService:
    return RelatedService(
        FakeGraph(list(edges), graph_nodes),
        FakeEvidence(list(items), evidence_error),
        FakePrices(prices if prices is not None else _prices(), price_fail),
    )


def _row(payload: dict, edge_id: int) -> dict:
    return next(r for r in payload["rows"] if r["edge_id"] == edge_id)


# -- direction and signal units -----------------------------------------------

@pytest.mark.parametrize(("stored", "queried_is_src", "expected"), [
    ("x_leads", True, "leading"),      # src leads, queried is src
    ("x_leads", False, "lagging"),     # src leads, queried is dst
    ("y_leads", True, "lagging"),      # dst leads, queried is src
    ("y_leads", False, "leading"),     # dst leads, queried is dst
    ("contemporaneous", True, "contemporaneous"),
    ("contemporaneous", False, "contemporaneous"),
])
def test_lag_direction_translates_to_queried_symbol(stored, queried_is_src, expected):
    assert translate_direction(stored, queried_is_src) == expected


def test_lag_direction_rejects_unknown_stored_value():
    with pytest.raises(ValueError):
        translate_direction("sideways", True)


@pytest.mark.parametrize(("sigma", "sign", "expected"), [
    (3.0, 1, "up"),     # neighbour above its implied level, positive relation: queried should move up
    (-3.0, 1, "down"),  # neighbour below its implied level, positive relation: queried should move down
    (3.0, -1, "down"),  # negative relation flips the direction
    (-3.0, -1, "up"),
])
def test_signal_sign_table(sigma, sign, expected):
    signal = signal_for(validated=True, queried_direction="contemporaneous", sigma=sigma,
                        divergence_reason=None, sign=sign, threshold=THRESHOLD)
    assert signal["status"] == "available"
    assert signal["direction"] == expected
    assert signal["basis"] == "divergence"
    assert signal["label"] == SIGNAL_LABEL


def test_signal_abstains_for_hypothesis_leader_missing_and_small_divergence():
    base = dict(queried_direction=None, sigma=3.0, divergence_reason=None, sign=1, threshold=THRESHOLD)
    hyp = signal_for(**{**base, "validated": False})
    assert (hyp["status"], hyp["reason"]) == ("abstain", "relationen är inte validerad")
    leader = signal_for(**{**base, "validated": True, "queried_direction": "leading"})
    assert (leader["status"], leader["reason"], leader["direction"]) == ("abstain", "ledande aktie", "none")
    missing = signal_for(**{**base, "validated": True, "sigma": None, "divergence_reason": "saknar historik"})
    assert missing["reason"] == "saknar historik"
    small = signal_for(**{**base, "validated": True, "sigma": THRESHOLD - 0.5})
    assert (small["status"], small["direction"]) == ("abstain", "none")


def test_relation_sign_prefers_expected_sign_then_beta():
    assert relation_sign(-1, 0.9) == -1
    assert relation_sign(None, -0.4) == -1
    assert relation_sign(None, None) == 1


def test_discover_threshold_is_imported_not_hardcoded():
    service = _service([_edge(1, 1, 2)])
    assert service._policy.divergence_z == POLICY.divergence_z


# -- service behaviour --------------------------------------------------------

def test_validated_edge_src_role_with_divergence_and_signal():
    edge = _edge(10, 1, 2, expected_sign=1)  # NVDA -> MSFT
    service = _service([edge], [_validation(1, 2)])
    payload = service.build_related("NVDA", AS_OF, include_hypothesis=False)
    row = _row(payload, 10)
    assert row["role"] == "src" and row["neighbour"]["symbol"] == "MSFT"
    assert row["lag"]["status"] == "available" and row["lag"]["direction"] == "contemporaneous"
    assert row["lag"]["validated_through"] == "2026-09-30"
    assert row["divergence"]["status"] == "available" and row["divergence"]["subject"] == "neighbour"
    assert row["divergence"]["sigma"] > THRESHOLD
    assert row["signal"]["status"] == "available" and row["signal"]["direction"] == "up"


def test_validated_edge_dst_role_same_neighbour_sigma_and_signal():
    edge = _edge(11, 2, 1, expected_sign=1)  # MSFT -> NVDA, queried NVDA is dst
    service = _service([edge], [_validation(2, 1)])
    row = _row(service.build_related("NVDA", AS_OF, False), 11)
    assert row["role"] == "dst" and row["neighbour"]["symbol"] == "MSFT"
    assert row["divergence"]["subject"] == "neighbour"
    assert row["divergence"]["reference"] == "queried"
    assert row["divergence"]["sigma"] > THRESHOLD
    assert row["signal"]["direction"] == "up"


@pytest.mark.parametrize(("role_src", "expected_sign", "direction"), [
    (True, 1, "up"), (True, -1, "down"), (False, 1, "up"), (False, -1, "down"),
])
def test_sign_table_over_role_and_expected_sign(role_src, expected_sign, direction):
    edge = _edge(20, 1, 2, expected_sign=expected_sign) if role_src else _edge(20, 2, 1, expected_sign=expected_sign)
    service = _service([edge], [_validation(edge.src_id, edge.dst_id)])
    row = _row(service.build_related("NVDA", AS_OF, False), 20)
    assert row["role"] == ("src" if role_src else "dst")
    assert row["signal"]["direction"] == direction


def test_hypothesis_edge_abstains_and_is_hidden_unless_requested():
    edge = _edge(30, 1, 2, status=EdgeStatus.HYPOTHESIS)
    service = _service([edge], [_validation(1, 2)])
    assert service.build_related("NVDA", AS_OF, include_hypothesis=False)["rows"] == []
    row = _row(service.build_related("NVDA", AS_OF, include_hypothesis=True), 30)
    assert row["signal"] == {"status": "abstain", "direction": "none", "basis": None,
                             "reason": "relationen är inte validerad", "label": SIGNAL_LABEL}
    assert row["lag"] == {"status": "unavailable", "reason": "ej validerad"}


def test_rejected_edges_are_never_listed_even_if_the_graph_returns_them():
    rejected = _edge(40, 1, 2, status=EdgeStatus.REJECTED)
    service = _service([rejected], [_validation(1, 2)])
    payload = service.build_related("NVDA", AS_OF, include_hypothesis=True)
    assert payload["rows"] == []


def test_node_without_series_appears_with_unavailable_stats_and_reason():
    edge = _edge(50, 1, 3)  # NVDA -> THEME (no series symbol)
    service = _service([edge], [_validation(1, 3, relation=RelationType.CORRELATED_WITH)])
    row = _row(service.build_related("NVDA", AS_OF, False), 50)
    assert row["neighbour"] == {"node_type": "THEME", "key": "ai-infrastructure", "name": "AI infrastructure", "symbol": None}
    assert row["divergence"] == {"status": "unavailable", "reason": "grannen saknar prisserie", "subject": "neighbour"}
    assert row["signal"]["status"] == "abstain" and row["signal"]["reason"] == "grannen saknar prisserie"


def test_missing_neighbour_prices_degrade_only_that_row():
    prices = {"NVDA": _prices()["NVDA"]}  # MSFT has no stored bars
    edges = [_edge(60, 1, 2), _edge(61, 1, 3)]
    service = _service(edges, [_validation(1, 2)], prices=prices)
    payload = service.build_related("NVDA", AS_OF, False)
    row = _row(payload, 60)
    assert row["divergence"]["status"] == "unavailable"
    assert row["divergence"]["reason"] == "ingen lagrad kurs för MSFT"
    assert _row(payload, 61)["neighbour"]["key"] == "ai-infrastructure"


def test_two_edges_between_same_pair_give_two_rows():
    edges = [_edge(70, 1, 2, expected_sign=1), _edge(71, 2, 1, expected_sign=-1, relation=RelationType.SUPPLIES)]
    items = [_validation(1, 2), _validation(2, 1, relation=RelationType.SUPPLIES)]
    payload = _service(edges, items).build_related("NVDA", AS_OF, False)
    assert [r["edge_id"] for r in payload["rows"]] == [70, 71]
    assert {r["neighbour"]["key"] for r in payload["rows"]} == {"MSFT"}
    assert [r["role"] for r in payload["rows"]] == ["src", "dst"]


def test_lag_direction_is_translated_per_role_and_leader_abstains():
    # MSFT leads NVDA: stored x_leads (src=MSFT leads dst=NVDA) for the edge MSFT -> NVDA.
    as_dst = _edge(80, 2, 1, relation=RelationType.LEADS)
    items = [_validation(2, 1, best_lag=2, direction="x_leads", relation=RelationType.LEADS)]
    service = _service([as_dst], items)
    row = _row(service.build_related("NVDA", AS_OF, False), 80)
    assert row["lag"]["direction"] == "lagging" and row["lag"]["best_lag_days"] == 2
    assert row["signal"]["status"] == "available"  # queried lags, so it is not the leader

    # NVDA leads MSFT (src=NVDA, x_leads): queried NVDA is the leader -> abstain, follower divergence still shown.
    as_src = _edge(81, 1, 2, relation=RelationType.LEADS)
    items = [_validation(1, 2, best_lag=3, direction="x_leads", relation=RelationType.LEADS)]
    row = _row(_service([as_src], items).build_related("NVDA", AS_OF, False), 81)
    assert row["lag"]["direction"] == "leading"
    assert row["signal"] == {"status": "abstain", "direction": "none", "basis": None,
                             "reason": "ledande aktie", "label": SIGNAL_LABEL}
    assert row["divergence"]["status"] == "available"


def test_point_in_time_cut_ignores_later_prices_and_evidence():
    prices = _prices(shock_days=0)  # no shock inside the history...
    # ...but a shock in the days after AS_OF, which must not be visible.
    future_nvda = _returns(1)
    msft_future = future_nvda + _returns(2, vol=0.002)
    msft_future[-5:] += SHOCK
    prices = {"NVDA": _bars(future_nvda), "MSFT": _bars(msft_future)}
    cut = prices["MSFT"]
    as_of = cut[-6].ts  # the last five returns are after this day
    edge = _edge(90, 1, 2)
    late_evidence = _validation(1, 2, available=date(2026, 10, 5))  # not yet available at as_of
    service = _service([edge], [late_evidence], prices=prices)
    row = _row(service.build_related("NVDA", as_of, False), 90)
    assert row["divergence"]["sigma"] is not None and abs(row["divergence"]["sigma"]) < THRESHOLD
    assert row["signal"]["direction"] == "none"
    assert row["lag"] == {"status": "unavailable", "reason": "ej validerad"}


def test_unknown_symbol_raises_for_both_graph_and_prices():
    service = RelatedService(FakeGraph([]), FakeEvidence([]), FakePrices({}))
    with pytest.raises(UnknownSymbol):
        service.build_related("ZZZZ", AS_OF, False)


def test_known_only_to_price_store_returns_no_rows_without_error():
    service = RelatedService(FakeGraph([], {}), FakeEvidence([]), FakePrices({"NVDA": _prices()["NVDA"]}))
    payload = service.build_related("nvda", AS_OF, True)
    assert payload["rows"] == [] and payload["queried"] is None and payload["symbol"] == "NVDA"


def test_failing_row_degrades_to_unavailable_instead_of_failing_the_request():
    edges = [_edge(100, 1, 2), _edge(101, 1, 3)]
    service = _service(edges, [_validation(1, 2)], evidence_error={1})  # evidence lookup raises
    payload = service.build_related("NVDA", AS_OF, False)
    assert {r["edge_id"] for r in payload["rows"]} == {100, 101}
    failed = _row(payload, 100)
    assert failed["lag"]["status"] == "unavailable"
    assert failed["signal"]["status"] == "abstain"


def test_meta_carries_versions_counts_and_disclaimer():
    edges = [_edge(110, 1, 2), _edge(111, 1, 3, status=EdgeStatus.HYPOTHESIS)]
    payload = _service(edges, [_validation(1, 2)]).build_related("NVDA", AS_OF, True)
    meta = payload["meta"]
    assert meta["disclaimer"] == DISCLAIMER
    assert meta["as_of"] == "2026-10-01"
    assert meta["policy_version"]["divergence"] == POLICY.version
    assert meta["counts"]["edges"] == 2 and meta["counts"]["validated"] == 1 and meta["counts"]["hypothesis"] == 1
    assert meta["counts"]["signals_available"] == 1
