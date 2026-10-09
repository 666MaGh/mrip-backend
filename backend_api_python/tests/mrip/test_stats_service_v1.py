"""RelationshipValidator under validation-v1 with in-memory fakes (no database).

Covers: sector control and its fallback label, null cache hit/miss, the downgrade path
used by --revalidate, and the content of the recorded evidence.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import PriceBar, PriceSeries, Provenance
from app.mrip.evidence.types import NodeKey, RelationshipRef, SourceType, Stance
from app.mrip.relationships.types import Edge, EdgeStatus, Node, NodeType, RelationType
from app.mrip.stats.null_distribution import GLOBAL_SECTOR, NULL_SAMPLE_SIZE, NULL_SEED, NullDistribution
from app.mrip.stats.service import RelationshipValidator
from app.mrip.stats.validate import ValidationPolicy, Verdict

N = 700
DATES = pd.bdate_range("2020-01-01", periods=N)
PROV = Provenance(provider="fake", gateway="test", endpoint="fake.history", fetched_at=datetime(2026, 10, 1, tzinfo=timezone.utc))
POLICY_VERSION = "validation-v1-uncalibrated"


def noise(seed: int, scale: float = 1.0) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(0, scale, N), index=DATES)


def to_prices(returns: pd.Series, symbol: str) -> PriceSeries:
    closes = 100 * np.exp((returns.fillna(0) * 0.01).cumsum())
    bars = tuple(PriceBar(ts=ts.date(), open=None, high=None, low=None, close=float(c), volume=None) for ts, c in closes.items())
    return PriceSeries(symbol=symbol, interval="1d", bars=bars, provenance=PROV)


class FakeGateway:
    def __init__(self, returns: dict[str, pd.Series]):
        self.series = {sym: to_prices(r, sym) for sym, r in returns.items()}
        self.calls = 0

    def price_history(self, symbol, start=None, end=None, interval="1d"):
        self.calls += 1
        if symbol not in self.series:
            raise DataUnavailable(f"no data for {symbol}")
        return self.series[symbol]


@dataclass
class FakeGraph:
    nodes: dict[int, Node]
    edges: list[Edge]
    version: int = 1
    status_changes: list[tuple[int, EdgeStatus]] = field(default_factory=list)

    def find_active_edge(self, src, dst, relation_type):
        by_key = {(n.node_type, n.key): n.id for n in self.nodes.values()}
        for edge in self.edges:
            if edge.src_id == by_key.get((src.node_type, src.key)) and edge.dst_id == by_key.get((dst.node_type, dst.key)) \
                    and edge.relation_type is relation_type and edge.status in (EdgeStatus.HYPOTHESIS, EdgeStatus.VALIDATED):
                return edge
        return None

    def get_node(self, key: NodeKey):
        return next((n for n in self.nodes.values() if (n.node_type, n.key) == (key.node_type, key.key)), None)

    def list_priced_nodes(self):
        return [n for n in self.nodes.values() if "series" in n.attributes]

    def set_edge_status(self, edge_id, status):
        old = next(e for e in self.edges if e.id == edge_id)
        new = Edge(old.id, old.src_id, old.dst_id, old.relation_type, status, old.source, self.version + 1,
                   attributes=old.attributes)
        self.edges = [e for e in self.edges if e.id != edge_id] + [new]
        self.version += 1
        self.status_changes.append((edge_id, status))
        return new


@dataclass
class FakeEvidence:
    rows: list = field(default_factory=list)

    def list_evidence(self, ref, *args, **kwargs):
        return [r for r in self.rows if r.source_uri.endswith(f"{ref.src.key}->{ref.dst.key}/{ref.relation_type.value}")
                and not getattr(r, "retracted", False)]

    def add(self, ref, stance, source_type, uri, end, *, assessed_by, source_title, excerpt, attributes):
        row = type("Row", (), {})()
        row.id, row.stance, row.source_type, row.source_uri = len(self.rows) + 1, stance, source_type, uri
        row.assessed_by, row.excerpt, row.attributes, row.retracted = assessed_by, excerpt, attributes, False
        self.rows.append(row)
        return row

    def retract(self, evidence_id, reason):
        for r in self.rows:
            if r.id == evidence_id:
                r.retracted = True
        return None


class FakeNullStore:
    def __init__(self, preset: list[NullDistribution] | None = None):
        self.data = {(d.sector, d.as_of, d.policy_version, d.n, d.seed): d for d in (preset or [])}
        self.gets = 0
        self.puts = 0

    def get(self, sector, as_of, policy_version, n, seed):
        self.gets += 1
        return self.data.get((sector, as_of, policy_version, n, seed))

    def put(self, dist):
        self.puts += 1
        self.data[(dist.sector, dist.as_of, dist.policy_version, dist.n, dist.seed)] = dist


def build(sector_members: int, lagged_pair: bool = False, background: int = 40):
    """Priced SECURITY nodes: S0..S{k-1} in sector 'Tech' (shared market and sector factors), N0..N2 in
    'Niche' (too small for the sector control), and ``background`` unsectored names that only feed the
    random-pair null pool. With ``lagged_pair`` S1 carries a genuine idiosyncratic lag-2 link from S0."""
    market = noise(1)
    sector_factor = noise(2)
    returns: dict[str, pd.Series] = {"SPY": market}
    specs: list[tuple[str, str | None]] = []
    for i in range(sector_members):
        returns[f"S{i}"] = market + sector_factor + noise(100 + i, 0.7)
        specs.append((f"S{i}", "Tech"))
    if lagged_pair:
        returns["S1"] = (returns["S1"] + 2.0 * returns["S0"].shift(2)).fillna(0)
    for i in range(3):
        returns[f"N{i}"] = noise(300 + i, 0.8) + market
        specs.append((f"N{i}", "Niche"))
    for i in range(background):
        returns[f"B{i}"] = noise(400 + i)
        specs.append((f"B{i}", None))
    nodes: dict[int, Node] = {}
    for node_id, (symbol, sector) in enumerate(specs, start=1):
        attrs: dict = {"series": {"symbol": symbol}, "universes": ["test"]}
        if sector:
            attrs["sector"] = sector
        nodes[node_id] = Node(node_id, NodeType.SECURITY, symbol, symbol, attrs)
    src, dst = nodes_by_symbol(nodes, "S0"), nodes_by_symbol(nodes, "S1")
    edge = Edge(900, src, dst, RelationType.LEADS, EdgeStatus.HYPOTHESIS, "test", 1, attributes={"expected_sign": 1})
    return FakeGraph(nodes=nodes, edges=[edge]), FakeGateway(returns)


def nodes_by_symbol(nodes: dict[int, Node], symbol: str) -> int:
    return next(n.id for n in nodes.values() if n.key == symbol)


def ref_for(graph: FakeGraph, src_symbol: str, dst_symbol: str) -> RelationshipRef:
    src = next(n for n in graph.nodes.values() if n.key == src_symbol)
    dst = next(n for n in graph.nodes.values() if n.key == dst_symbol)
    return RelationshipRef(NodeKey(src.node_type, src.key), NodeKey(dst.node_type, dst.key), RelationType.LEADS)


def validator(graph, gateway, evidence=None, store=None, policy=ValidationPolicy()):
    return RelationshipValidator(graph, evidence or FakeEvidence(), gateway, policy=policy, null_store=store)


def test_shared_factor_pair_is_inconclusive_under_v1_and_records_nothing():
    graph, gateway = build(12)
    evidence = FakeEvidence()
    outcome = validator(graph, gateway, evidence).validate(ref_for(graph, "S0", "S1"), as_of=None)
    assert outcome.result.verdict is Verdict.INCONCLUSIVE
    assert outcome.edge.status is EdgeStatus.HYPOTHESIS and not outcome.status_changed
    assert evidence.rows == []
    assert outcome.result.metrics["sector_control"] == "used" and outcome.result.metrics["peers_used"] == 10
    assert any(r.startswith("does not exceed same-sector null") for r in outcome.reasons)


def test_genuine_pair_is_validated_and_evidence_carries_the_v1_metrics():
    graph, gateway = build(12, lagged_pair=True)
    evidence = FakeEvidence()
    outcome = validator(graph, gateway, evidence).validate(ref_for(graph, "S0", "S1"))
    assert outcome.result.verdict is Verdict.VALIDATED and outcome.status_changed
    assert outcome.edge.status is EdgeStatus.VALIDATED
    (row,) = evidence.rows
    assert row.stance is Stance.SUPPORT and row.source_type is SourceType.STATISTICAL_TEST
    assert row.assessed_by == f"stats:{POLICY_VERSION}"
    assert row.source_uri == f"mrip://validation/{POLICY_VERSION}/S0->S1/LEADS"
    body = json.loads(row.excerpt)["metrics"]
    for key in ("partial_r", "null_p95", "null_n", "null_label", "sector", "peers_used", "controls", "sector_control", "n_obs"):
        assert key in body
    assert body["sector"] == "Tech" and body["peers_used"] == 10 and body["sector_control"] == "used"
    assert body["controls"] == ["market_t", "market_lagged", "sector_t", "sector_lagged"]
    assert body["effect_measure"] > body["null_p95"]


def test_sector_with_fewer_than_five_peers_falls_back_and_says_so():
    graph, gateway = build(12, background=40)
    edge_src, edge_dst = nodes_by_symbol(graph.nodes, "N0"), nodes_by_symbol(graph.nodes, "N1")
    graph.edges = [Edge(901, edge_src, edge_dst, RelationType.LEADS, EdgeStatus.HYPOTHESIS, "test", 1,
                        attributes={"expected_sign": 1})]
    outcome = validator(graph, gateway).validate(ref_for(graph, "N0", "N1"))
    metrics = outcome.result.metrics
    assert metrics["sector_control"] == "unavailable" and metrics["peers_used"] == 0
    assert metrics["controls"] == ["market_t"]
    assert metrics["null_label"] == "random-pair null"


def test_null_cache_miss_computes_and_stores_then_hit_reuses_without_compute():
    graph, gateway = build(12, lagged_pair=True)
    store = FakeNullStore()
    validator(graph, gateway, store=store).validate(ref_for(graph, "S0", "S1"))
    assert store.gets == 1 and store.puts == 1
    stored = next(iter(store.data.values()))
    assert stored.sector == "Tech" and stored.policy_version == POLICY_VERSION and stored.n == NULL_SAMPLE_SIZE

    graph2, gateway2 = build(12, lagged_pair=True)
    second = validator(graph2, gateway2, store=store)
    second.validate(ref_for(graph2, "S0", "S1"))
    assert store.gets == 2 and store.puts == 1  # hit: nothing recomputed or re-stored


def test_preset_cached_null_is_the_gate_threshold():
    graph, gateway = build(12, lagged_pair=True)
    month = datetime.now(timezone.utc).date().replace(day=1)
    preset = NullDistribution(sector="Tech", as_of=month, policy_version=POLICY_VERSION, n=NULL_SAMPLE_SIZE,
                              pairs_used=200, seed=NULL_SEED, p05=-0.9, p90=0.9, p95=0.99, p99=0.995, mean=0.0, std=0.3)
    outcome = validator(graph, gateway, store=FakeNullStore([preset])).validate(ref_for(graph, "S0", "S1"))
    assert outcome.result.verdict is Verdict.INCONCLUSIVE
    assert any("p95 0.990" in r and "n=200" in r for r in outcome.reasons)


def test_downgrade_turns_an_unsupported_validated_edge_into_hypothesis_with_evidence():
    graph, gateway = build(12, lagged_pair=True)
    evidence = FakeEvidence()
    first = validator(graph, gateway, evidence).validate(ref_for(graph, "S0", "S1"))
    assert first.edge.status is EdgeStatus.VALIDATED

    # Same pair, but the sector now looks like a common factor only: a fresh gateway without the lag.
    graph_plain, gateway_plain = build(12, lagged_pair=False)
    graph.nodes = graph_plain.nodes
    gateway.series = gateway_plain.series
    outcome = validator(graph, gateway, evidence).validate(ref_for(graph, "S0", "S1"), downgrade_unsupported=True)
    assert outcome.result.verdict is Verdict.INCONCLUSIVE
    assert outcome.edge.status is EdgeStatus.HYPOTHESIS and outcome.status_changed
    neutral = [r for r in evidence.rows if r.stance is Stance.NEUTRAL]
    assert len(neutral) == 1 and json.loads(neutral[0].excerpt)["verdict"] == "inconclusive"


def test_inconclusive_on_hypothesis_edge_is_untouched_even_with_downgrade_flag():
    graph, gateway = build(12)
    evidence = FakeEvidence()
    outcome = validator(graph, gateway, evidence).validate(ref_for(graph, "S0", "S1"), downgrade_unsupported=True)
    assert outcome.edge.status is EdgeStatus.HYPOTHESIS and not outcome.status_changed and evidence.rows == []


def test_sector_membership_comes_from_priced_nodes_only():
    graph, gateway = build(12)
    v = validator(graph, gateway)
    assert v.sector_of("S3") == "Tech" and v.sector_of("B1") is None and v.sector_of("SPY") is None
    assert GLOBAL_SECTOR == "*"
