from datetime import date

from app.mrip.discover.service import DiscoverService


class Graph:
    def list_nodes_in_universe(self, universe): return []
    def list_active_edges(self, limit=1000): return []


class Store:
    def recent_counts(self, as_of): return {}
    def save(self, items, policy_version): self.items = items; return len(items)


def test_missing_inputs_save_empty_feed():
    store = Store()
    service = DiscoverService(graph=Graph(), price_store=None, snapshot_store=None, discover_store=store)
    summary = service.run(date(2026, 1, 1), option_symbols=[])
    assert summary.items_saved == 0
    assert summary.errors == {}


def test_detector_failure_is_isolated():
    class BrokenSnapshots:
        def symbols(self): return ["SPY"]
        def latest_before(self, symbol, as_of): raise RuntimeError("snapshot unavailable")
    store = Store()
    service = DiscoverService(graph=Graph(), price_store=None, snapshot_store=BrokenSnapshots(), discover_store=store)
    summary = service.run(date(2026, 1, 1))
    assert "options" in summary.errors
    assert summary.items_saved == 0


def test_detector_items_are_ranked_and_saved(monkeypatch):
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from app.mrip.discover import service as service_module
    from app.mrip.discover.types import Kind, Observation

    observation = Observation(Kind.NEAR_GAMMA_FLIP, "SPY", date(2026, 1, 1), "MODELED / ESTIMATED near flip", .9, modeled=True)
    monkeypatch.setattr(service_module, "analyze_options", lambda snapshot: object())
    monkeypatch.setattr(service_module, "gamma_events", lambda current, previous, as_of, policy: [observation])
    class Snapshots:
        def symbols(self): return ["SPY"]
        def latest_before(self, symbol, as_of): return SimpleNamespace(snapshot_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))
    store = Store()
    service = DiscoverService(graph=Graph(), price_store=None, snapshot_store=Snapshots(), discover_store=store)
    summary = service.run(date(2026, 1, 1))
    assert summary.items_saved == 1 and summary.per_detector == {Kind.NEAR_GAMMA_FLIP.value: 1}
    assert len(store.items) == 1 and store.items[0].observation is observation
    assert store.items[0].score >= 50


def test_peer_items_receive_universe_liquidity_percentile():
    from datetime import timedelta
    from types import SimpleNamespace

    class UniverseGraph(Graph):
        def list_nodes_in_universe(self, universe):
            return [SimpleNamespace(attributes={"series": {"symbol": f"S{i}"}, "sector": "Tech"}) for i in range(8)] if universe == "SP500" else []
    class Prices:
        def series(self, provider, symbol, end=None):
            start = date(2026, 1, 1)
            closes = [100.] * 21
            if symbol == "S0": closes[-1] = 250.
            bars = [SimpleNamespace(ts=start + timedelta(days=i), close=close, volume=1000 + i) for i, close in enumerate(closes)]
            return SimpleNamespace(symbol=symbol, bars=bars)
    store = Store()
    service = DiscoverService(graph=UniverseGraph(), price_store=Prices(), snapshot_store=None, discover_store=store)
    summary = service.run(date(2026, 1, 21), option_symbols=[])
    assert summary.items_saved > 0
    outlier = next(item for item in store.items if item.observation.subject == "S0")
    peers = [item for item in store.items if item.observation.kind.value == "PEER_DIVERGENCE"]
    assert peers and outlier.observation.liquidity == 1.0


def test_relationship_legs_fall_back_to_yahoo_and_prefer_cboe():
    from datetime import timedelta
    from types import SimpleNamespace

    class RelationshipGraph(Graph):
        def __init__(self):
            self.nodes = {
                1: SimpleNamespace(id=1, key="NVDA", node_type="stock", attributes={"series": {"symbol": "NVDA"}}),
                2: SimpleNamespace(id=2, key="SMH", node_type="etf", attributes={"series": {"symbol": "SMH"}}),
            }
        def list_active_edges(self, limit=1000):
            return [SimpleNamespace(src_id=1, dst_id=2, relation_type="peer", status=SimpleNamespace(value="active"))]
        def get_node_by_id(self, node_id): return self.nodes[node_id]

    class Prices:
        def __init__(self, stored):
            self.stored, self.calls = stored, []
        def series(self, provider, symbol, end=None):
            self.calls.append((provider, symbol))
            bars = self.stored.get((provider, symbol))
            if bars is None: return None
            return SimpleNamespace(symbol=symbol, bars=bars)

    def bars(closes, start=date(2026, 1, 1)):
        return [SimpleNamespace(ts=start + timedelta(days=i), close=c, volume=1000) for i, c in enumerate(closes)]

    yahoo_only = {("yahoo", "NVDA"): bars([100. + i for i in range(40)]), ("yahoo", "SMH"): bars([50. + i * 0.5 for i in range(40)])}
    prices = Prices(yahoo_only)
    service = DiscoverService(graph=RelationshipGraph(), price_store=prices, snapshot_store=None, discover_store=Store())
    summary = service.run(date(2026, 2, 9), option_symbols=[])
    assert ("yahoo", "NVDA") in prices.calls and ("yahoo", "SMH") in prices.calls
    assert "relationships" not in summary.errors

    both = {**yahoo_only, ("cboe", "NVDA"): bars([100.] * 40), ("cboe", "SMH"): bars([50.] * 40)}
    prices = Prices(both)
    service = DiscoverService(graph=RelationshipGraph(), price_store=prices, snapshot_store=None, discover_store=Store())
    service.run(date(2026, 2, 9), option_symbols=[])
    assert ("yahoo", "NVDA") not in prices.calls and ("yahoo", "SMH") not in prices.calls
