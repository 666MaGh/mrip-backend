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
