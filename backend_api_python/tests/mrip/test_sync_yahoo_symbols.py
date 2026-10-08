"""sync_yahoo_symbols: explicit-symbol Yahoo sync through the shared ingestor, with fakes."""
from datetime import date

from app.commands.sync_yahoo_symbols import PROVIDER, sync_symbols
from app.mrip.data.models import PriceBar, PriceSeries


class FakeGateway:
    def __init__(self, missing: set[str] | None = None) -> None:
        self.missing = missing or set()
        self.calls: list[str] = []

    def price_history(self, symbol, start=None, end=None, interval="1d"):
        self.calls.append(symbol)
        if symbol in self.missing:
            from app.mrip.data.gateway import DataUnavailable

            raise DataUnavailable(symbol)
        bar = PriceBar(ts=date(2026, 10, 1), open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)
        return PriceSeries(symbol=symbol, interval="1d", bars=(bar,), provenance=None)


class FakeStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, str]] = []

    def sync_states(self, provider, symbols):
        return []

    def upsert_bars(self, provider, symbol, bars):
        self.upserts.append((provider, symbol))
        return len(bars)

    def record_attempt(self, *args, **kwargs):
        self.attempts = getattr(self, "attempts", 0) + 1


def test_symbols_are_synced_under_yahoo_provider_once_each():
    gateway, store = FakeGateway(), FakeStore()
    report = sync_symbols(["SPY", "SMH", "SPY"], gateway=gateway, store=store, budget_seconds=60.0, min_interval=0.0)
    assert PROVIDER == "yahoo"
    assert sorted(set(gateway.calls)) == ["SMH", "SPY"]
    assert store.upserts and all(provider == "yahoo" for provider, _ in store.upserts)
    assert report.succeeded == 2
    assert report.failed == {}
