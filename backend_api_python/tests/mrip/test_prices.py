"""Price bar storage, ingestion, and local gateway (work 013): unit and optional DB tests."""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path as FsPath
from typing import Any, Callable, ContextManager, Sequence

import pytest

from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.data.models import Latency, PriceBar, PriceSeries, Provenance
from app.mrip.prices.gateway import StoredDataGateway
from app.mrip.prices.ingest import PriceIngestor, SyncReport
from app.mrip.prices.store import PriceStore, SyncState, collapse_bars_by_date

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_prices.sql"

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def make_bar(ts: date | datetime, close: float | None, open_: float | None = None, volume: float | None = None) -> PriceBar:
    """Helper to create a PriceBar."""
    # Handle None close by using a dummy value for OHLC calculations.
    safe_close = close if close is not None else 100.0
    return PriceBar(
        ts=ts,
        open=open_ if open_ is not None else safe_close,
        high=safe_close * 1.01,
        low=safe_close * 0.99,
        close=close,
        volume=volume,
    )


class FakeStore:
    """In-memory fake store for testing without a database."""

    def __init__(self) -> None:
        self._bars: dict[tuple[str, str], list[PriceBar]] = {}  # (provider, symbol) -> bars
        self._sync: dict[tuple[str, str], SyncState] = {}  # (provider, symbol) -> SyncState

    def upsert_bars(self, provider: str, symbol: str, bars: Sequence[PriceBar]) -> int:
        """Store bars; skip invalid ones."""
        valid = [b for b in bars if b.close is not None and b.close > 0]
        key = (provider, symbol)
        if key not in self._bars:
            self._bars[key] = []
        # Simple deduplicate by date (overwrite revisions).
        existing = {(b.ts.date() if isinstance(b.ts, datetime) else b.ts): b for b in self._bars[key]}
        for bar in valid:
            bar_date = bar.ts.date() if isinstance(bar.ts, datetime) else bar.ts
            existing[bar_date] = bar
        self._bars[key] = list(existing.values())
        self._bars[key].sort(key=lambda b: b.ts.date() if isinstance(b.ts, datetime) else b.ts)
        return len(valid)

    def series(
        self,
        provider: str,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> PriceSeries | None:
        """Retrieve bars in date range."""
        key = (provider, symbol)
        bars = self._bars.get(key, [])
        if not bars:
            return None
        filtered = []
        for bar in bars:
            bar_date = bar.ts.date() if isinstance(bar.ts, datetime) else bar.ts
            if start and bar_date < start:
                continue
            if end and bar_date > end:
                continue
            filtered.append(bar)
        if not filtered:
            return None
        return PriceSeries(
            symbol=symbol,
            interval="1d",
            bars=tuple(filtered),
            provenance=Provenance(
                provider=provider,
                gateway="mrip-store",
                endpoint="mrip_price_bars",
                fetched_at=T0,
                latency=Latency.UNKNOWN,
            ),
        )

    def last_bar_date(self, provider: str, symbol: str) -> date | None:
        """Return the most recent bar date."""
        key = (provider, symbol)
        bars = self._bars.get(key, [])
        if not bars:
            return None
        return max(b.ts.date() if isinstance(b.ts, datetime) else b.ts for b in bars)

    def record_attempt(
        self,
        provider: str,
        symbol: str,
        *,
        success: bool,
        last_bar_date: date | None = None,
        error: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Record sync attempt."""
        if now is None:
            now = datetime.now(timezone.utc)
        key = (provider, symbol)
        state = self._sync.get(key)
        if state is None:
            state = SyncState(
                provider=provider,
                symbol=symbol,
                last_bar_date=None,
                last_success_at=None,
                last_attempt_at=now,
                last_error=None,
                consecutive_failures=0,
            )
        if success:
            state = SyncState(
                provider=provider,
                symbol=symbol,
                last_bar_date=last_bar_date if last_bar_date is not None else state.last_bar_date,
                last_success_at=now,
                last_attempt_at=now,
                last_error=None,
                consecutive_failures=0,
            )
        else:
            state = SyncState(
                provider=provider,
                symbol=symbol,
                last_bar_date=state.last_bar_date,
                last_success_at=state.last_success_at,
                last_attempt_at=now,
                last_error=(error[:500] if error else None),
                consecutive_failures=state.consecutive_failures + 1,
            )
        self._sync[key] = state

    def sync_states(
        self,
        provider: str,
        symbols: Sequence[str] | None = None,
    ) -> list[SyncState]:
        """Retrieve sync states."""
        result = []
        for (p, s), state in self._sync.items():
            if p == provider and (symbols is None or s in symbols):
                result.append(state)
        result.sort(key=lambda st: st.symbol)
        return result


class FakeGateway:
    """Scripted gateway for testing ingestion logic."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, date | None]] = []  # Recorded calls: (symbol, start_date)
        self.responses: dict[str, list[PriceBar]] = {}  # symbol -> bars to return
        self.failures: dict[str, str] = {}  # symbol -> error message (DataUnavailable)

    def price_history(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
        interval: str = "1d",
    ) -> PriceSeries:
        """Record call and return scripted response or raise error."""
        self.calls.append((symbol, start))
        if symbol in self.failures:
            raise DataUnavailable(self.failures[symbol])
        bars = self.responses.get(symbol, [])
        return PriceSeries(
            symbol=symbol,
            interval=interval,
            bars=tuple(bars),
            provenance=Provenance(
                provider="cboe",
                gateway="test",
                endpoint="test.price_history",
                fetched_at=T0,
                latency=Latency.UNKNOWN,
            ),
        )

    def index_history(self, symbol: str, start: date | None = None, end: date | None = None) -> PriceSeries:
        raise DataUnavailable("not implemented")

    def vix_history(self, start: date | None = None, end: date | None = None) -> PriceSeries:
        raise DataUnavailable("not implemented")

    def options_chain(self, symbol: str) -> Any:
        raise DataUnavailable("not implemented")

    def cot(self, market: str, start: date | None = None, end: date | None = None) -> Any:
        raise DataUnavailable("not implemented")

    def macro_series(self, series_id: str, start: date | None = None, end: date | None = None) -> Any:
        raise DataUnavailable("not implemented")


class FakeClock:
    """Fake monotonic clock that advances with sleep."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ============================================================================
# Unit tests (no database)
# ============================================================================


def test_collapse_bars_by_date_removes_invalid_bars():
    bars = [
        make_bar(date(2026, 1, 1), 100.0),
        make_bar(date(2026, 1, 2), 0.0),  # Invalid: close == 0
        make_bar(date(2026, 1, 3), -50.0),  # Invalid: close < 0
        make_bar(date(2026, 1, 4), None),  # Invalid: close is None
    ]
    result = collapse_bars_by_date(bars)
    assert len(result) == 1
    assert result[0].ts == date(2026, 1, 1)
    assert result[0].close == 100.0


def test_collapse_bars_by_date_keeps_last_duplicate():
    bars = [
        make_bar(date(2026, 1, 1), 100.0),
        make_bar(date(2026, 1, 1), 105.0),  # Same date, different close
        make_bar(date(2026, 1, 2), 110.0),
    ]
    result = collapse_bars_by_date(bars)
    assert len(result) == 2
    assert result[0].ts == date(2026, 1, 1)
    assert result[0].close == 105.0  # Last one wins
    assert result[1].ts == date(2026, 1, 2)
    assert result[1].close == 110.0


def test_collapse_bars_by_date_handles_datetime_and_date():
    bars = [
        make_bar(datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc), 100.0),
        make_bar(date(2026, 1, 1), 105.0),  # Same calendar date, different ts type
        make_bar(date(2026, 1, 2), 110.0),
    ]
    result = collapse_bars_by_date(bars)
    assert len(result) == 2
    assert result[0].close == 105.0  # Last one (the date) wins
    assert result[1].close == 110.0


def test_collapse_bars_by_date_sorts_by_date():
    bars = [
        make_bar(date(2026, 1, 3), 110.0),
        make_bar(date(2026, 1, 1), 100.0),
        make_bar(date(2026, 1, 2), 105.0),
    ]
    result = collapse_bars_by_date(bars)
    assert len(result) == 3
    assert result[0].ts == date(2026, 1, 1)
    assert result[1].ts == date(2026, 1, 2)
    assert result[2].ts == date(2026, 1, 3)


def test_collapse_bars_by_date_empty_input():
    result = collapse_bars_by_date([])
    assert result == []


def test_collapse_bars_by_date_all_invalid():
    bars = [
        make_bar(date(2026, 1, 1), 0.0),
        make_bar(date(2026, 1, 2), -50.0),
        make_bar(date(2026, 1, 3), None),
    ]
    result = collapse_bars_by_date(bars)
    assert result == []


def test_fake_store_upsert_skips_invalid_bars():
    store = FakeStore()
    bars = [
        make_bar(date(2026, 1, 1), 100.0),
        make_bar(date(2026, 1, 2), 0.0),  # Invalid: close == 0
        make_bar(date(2026, 1, 3), -50.0),  # Invalid: close < 0
        make_bar(date(2026, 1, 4), None),  # Invalid: close is None
    ]
    count = store.upsert_bars("cboe", "TEST", bars)
    assert count == 1
    series = store.series("cboe", "TEST")
    assert len(series.bars) == 1
    assert series.bars[0].ts == date(2026, 1, 1)


def test_fake_store_upsert_overwrites_revisions():
    store = FakeStore()
    bars1 = [make_bar(date(2026, 1, 1), 100.0)]
    bars2 = [make_bar(date(2026, 1, 1), 105.0)]  # Revised
    store.upsert_bars("cboe", "TEST", bars1)
    store.upsert_bars("cboe", "TEST", bars2)
    series = store.series("cboe", "TEST")
    assert len(series.bars) == 1
    assert series.bars[0].close == 105.0


def test_fake_store_series_filters_by_date_range():
    store = FakeStore()
    bars = [make_bar(date(2026, 1, i), 100.0 + i) for i in range(1, 6)]
    store.upsert_bars("cboe", "TEST", bars)
    series = store.series("cboe", "TEST", start=date(2026, 1, 2), end=date(2026, 1, 4))
    assert len(series.bars) == 3
    assert series.bars[0].ts == date(2026, 1, 2)
    assert series.bars[-1].ts == date(2026, 1, 4)


def test_fake_store_last_bar_date():
    store = FakeStore()
    bars = [make_bar(date(2026, 1, i), 100.0 + i) for i in range(1, 4)]
    store.upsert_bars("cboe", "TEST", bars)
    assert store.last_bar_date("cboe", "TEST") == date(2026, 1, 3)


def test_fake_store_record_attempt_success_resets_failures():
    store = FakeStore()
    store.record_attempt("cboe", "TEST", success=False, error="err1", now=T0)
    store.record_attempt("cboe", "TEST", success=False, error="err2", now=T0 + timedelta(seconds=1))
    state = store.sync_states("cboe", ["TEST"])[0]
    assert state.consecutive_failures == 2
    assert state.last_error == "err2"

    store.record_attempt("cboe", "TEST", success=True, now=T0 + timedelta(seconds=2))
    state = store.sync_states("cboe", ["TEST"])[0]
    assert state.consecutive_failures == 0
    assert state.last_success_at == T0 + timedelta(seconds=2)
    assert state.last_error is None


def test_fake_store_record_attempt_truncates_error_to_500_chars():
    store = FakeStore()
    long_error = "x" * 600
    store.record_attempt("cboe", "TEST", success=False, error=long_error, now=T0)
    state = store.sync_states("cboe", ["TEST"])[0]
    assert state.last_error == "x" * 500


def test_ingestor_orders_symbols_never_synced_first():
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.responses = {
        "A": [make_bar(date(2026, 1, 1), 100.0)],
        "B": [make_bar(date(2026, 1, 1), 100.0)],
        "C": [make_bar(date(2026, 1, 1), 100.0)],
    }
    store = FakeStore()

    # Mark A and B as synced in the past, C as never synced.
    store.record_attempt("cboe", "A", success=True, now=T0 - timedelta(days=2))
    store.record_attempt("cboe", "B", success=True, now=T0 - timedelta(days=1))

    ingestor = PriceIngestor(
        gateway,
        store,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        now=lambda: T0,
    )
    ingestor.sync(["A", "B", "C"])

    # Expect C first (never synced), then A, then B (oldest to newest).
    symbols = [call[0] for call in gateway.calls]
    assert symbols == ["C", "A", "B"]


def test_ingestor_incremental_start_date():
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.responses = {
        "A": [make_bar(date(2026, 1, 15), 100.0)],
        "B": [make_bar(date(2026, 1, 15), 100.0)],
    }
    store = FakeStore()

    # A is never synced (should use history_start).
    # B has last_bar_date on 2026-01-10 (should use that - 7 days = 2026-01-03).
    # Sync B in the past so it's not skipped as "fresh".
    past_time = T0 - timedelta(days=5)
    bar_prev = make_bar(date(2026, 1, 10), 100.0)
    store.upsert_bars("cboe", "B", [bar_prev])
    store.record_attempt("cboe", "B", success=True, last_bar_date=date(2026, 1, 10), now=past_time)

    ingestor = PriceIngestor(
        gateway,
        store,
        history_start=date(2016, 1, 1),
        overlap_days=7,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        now=lambda: T0,
    )
    ingestor.sync(["A", "B"])

    # Check start dates.
    assert gateway.calls[0] == ("A", date(2016, 1, 1))  # history_start
    assert gateway.calls[1] == ("B", date(2026, 1, 3))  # last_bar_date (2026-01-10) - 7 days


def test_ingestor_skips_fresh_symbols():
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.responses = {
        "A": [make_bar(date(2026, 1, 1), 100.0)],
        "B": [make_bar(date(2026, 1, 1), 100.0)],
    }
    store = FakeStore()

    # A was synced 6 hours ago (skip_if_fresh_hours=12).
    # B was synced 15 hours ago (not fresh).
    now = T0
    store.record_attempt("cboe", "A", success=True, now=now - timedelta(hours=6))
    store.record_attempt("cboe", "B", success=True, now=now - timedelta(hours=15))

    ingestor = PriceIngestor(
        gateway,
        store,
        skip_if_fresh_hours=12.0,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        now=lambda: now,
    )
    report = ingestor.sync(["A", "B"])

    assert report.skipped_fresh == 1
    assert report.attempted == 1
    assert len(gateway.calls) == 1
    assert gateway.calls[0][0] == "B"


def test_ingestor_respects_budget():
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.responses = {
        "A": [make_bar(date(2026, 1, 1), 100.0)],
        "B": [make_bar(date(2026, 1, 1), 100.0)],
        "C": [make_bar(date(2026, 1, 1), 100.0)],
    }
    store = FakeStore()

    def fake_sleep(s: float) -> None:
        clock.advance(s)

    ingestor = PriceIngestor(
        gateway,
        store,
        min_interval_seconds=10.0,
        clock=clock,
        sleep=fake_sleep,
        now=lambda: T0,
    )
    report = ingestor.sync(["A", "B", "C"], budget_seconds=15.0)

    # A takes ~0s, B takes 10s (pacing), C would take 10s more = 20s total. Budget exhausted.
    assert report.attempted == 2
    assert report.not_attempted == 1
    assert report.halted_reason == "budget exhausted"


def test_ingestor_paces_calls():
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.responses = {
        "A": [make_bar(date(2026, 1, 1), 100.0)],
        "B": [make_bar(date(2026, 1, 1), 100.0)],
    }
    store = FakeStore()
    sleep_calls = []

    def track_sleep(s: float) -> None:
        sleep_calls.append(s)
        clock.advance(s)

    ingestor = PriceIngestor(
        gateway,
        store,
        min_interval_seconds=2.0,
        clock=clock,
        sleep=track_sleep,
        now=lambda: T0,
    )
    ingestor.sync(["A", "B"])

    # Should sleep ~2s between A and B.
    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 1.9  # Allow small float rounding


def test_ingestor_circuit_breaker():
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.failures = {
        "A": "err1",
        "B": "err2",
        "C": "err3",
    }
    store = FakeStore()

    ingestor = PriceIngestor(
        gateway,
        store,
        max_consecutive_failures=2,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        now=lambda: T0,
    )
    report = ingestor.sync(["A", "B", "C"])

    # A fails, B fails (2 consecutive), circuit breaker triggers.
    assert report.attempted == 2
    assert report.succeeded == 0
    assert len(report.failed) == 2
    assert report.not_attempted == 1
    assert "circuit breaker" in (report.halted_reason or "").lower() or "throttled" in (report.halted_reason or "").lower()


def test_ingestor_success_resets_failure_streak():
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.failures = {"A": "err"}
    gateway.responses = {"B": [make_bar(date(2026, 1, 1), 100.0)], "C": [make_bar(date(2026, 1, 1), 100.0)]}
    store = FakeStore()

    ingestor = PriceIngestor(
        gateway,
        store,
        max_consecutive_failures=2,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        now=lambda: T0,
    )
    report = ingestor.sync(["A", "B", "C"])

    # A fails, B succeeds (resets streak), C succeeds.
    assert report.attempted == 3
    assert report.succeeded == 2
    assert len(report.failed) == 1
    assert report.halted_reason is None


def test_ingestor_collects_failed_dict():
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.failures = {"A": "symbol not found"}
    store = FakeStore()

    ingestor = PriceIngestor(
        gateway,
        store,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        now=lambda: T0,
    )
    report = ingestor.sync(["A"])

    assert "A" in report.failed
    assert report.failed["A"] == "symbol not found"


def test_ingestor_propagates_non_dataunavailable_exceptions():
    clock = FakeClock()
    store = FakeStore()

    class BadGateway:
        def price_history(self, symbol: str, start: date | None = None, end: date | None = None, interval: str = "1d") -> Any:
            raise ValueError("unexpected error")

        def index_history(self, *args: Any, **kwargs: Any) -> Any:
            raise DataUnavailable("not implemented")

        def vix_history(self, *args: Any, **kwargs: Any) -> Any:
            raise DataUnavailable("not implemented")

        def options_chain(self, *args: Any, **kwargs: Any) -> Any:
            raise DataUnavailable("not implemented")

        def cot(self, *args: Any, **kwargs: Any) -> Any:
            raise DataUnavailable("not implemented")

        def macro_series(self, *args: Any, **kwargs: Any) -> Any:
            raise DataUnavailable("not implemented")

    ingestor = PriceIngestor(BadGateway(), store, clock=clock, sleep=lambda s: clock.advance(s), now=lambda: T0)
    with pytest.raises(ValueError):
        ingestor.sync(["A"])


def test_stored_gateway_serves_price_history_from_store():
    store = FakeStore()
    bars = [
        make_bar(date(2026, 1, 1), 100.0),
        make_bar(date(2026, 1, 2), 101.0),
        make_bar(date(2026, 1, 3), 102.0),
    ]
    store.upsert_bars("cboe", "TEST", bars)

    gateway = StoredDataGateway(store)
    series = gateway.price_history("TEST")

    assert series.symbol == "TEST"
    assert len(series.bars) == 3
    assert series.bars[0].close == 100.0


def test_stored_gateway_filters_by_date_range():
    store = FakeStore()
    bars = [make_bar(date(2026, 1, i), 100.0 + i) for i in range(1, 6)]
    store.upsert_bars("cboe", "TEST", bars)

    gateway = StoredDataGateway(store)
    series = gateway.price_history("TEST", start=date(2026, 1, 2), end=date(2026, 1, 4))

    assert len(series.bars) == 3
    assert series.bars[0].ts == date(2026, 1, 2)


def test_stored_gateway_raises_on_unsupported_interval():
    store = FakeStore()
    gateway = StoredDataGateway(store)

    with pytest.raises(ValueError, match="only 1d bars"):
        gateway.price_history("TEST", interval="1m")


def test_stored_gateway_raises_on_missing_symbol():
    store = FakeStore()
    gateway = StoredDataGateway(store)

    with pytest.raises(DataUnavailable, match="no price data found"):
        gateway.price_history("MISSING")


def test_stored_gateway_delegates_with_provider_for():
    store = FakeStore()
    bars = [make_bar(date(2026, 1, 1), 100.0)]
    store.upsert_bars("other-provider", "TEST", bars)

    def custom_provider(symbol: str) -> str:
        return "other-provider"

    gateway = StoredDataGateway(store, provider_for=custom_provider)
    series = gateway.price_history("TEST")

    assert len(series.bars) == 1


def test_stored_gateway_delegates_non_price_methods():
    store = FakeStore()

    class FakeLive:
        def index_history(self, symbol: str, start: date | None = None, end: date | None = None) -> PriceSeries:
            raise DataUnavailable("index_history called")

        def vix_history(self, start: date | None = None, end: date | None = None) -> PriceSeries:
            raise DataUnavailable("vix_history called")

        def options_chain(self, symbol: str) -> Any:
            raise DataUnavailable("options_chain called")

        def cot(self, market: str, start: date | None = None, end: date | None = None) -> Any:
            raise DataUnavailable("cot called")

        def macro_series(self, series_id: str, start: date | None = None, end: date | None = None) -> Any:
            raise DataUnavailable("macro_series called")

    gateway = StoredDataGateway(store, live=FakeLive())

    with pytest.raises(DataUnavailable, match="index_history called"):
        gateway.index_history("TEST")
    with pytest.raises(DataUnavailable, match="vix_history called"):
        gateway.vix_history()
    with pytest.raises(DataUnavailable, match="options_chain called"):
        gateway.options_chain("TEST")
    with pytest.raises(DataUnavailable, match="cot called"):
        gateway.cot("TEST")
    with pytest.raises(DataUnavailable, match="macro_series called"):
        gateway.macro_series("TEST")


def test_stored_gateway_raises_on_missing_live_gateway():
    store = FakeStore()
    gateway = StoredDataGateway(store, live=None)

    with pytest.raises(DataUnavailable, match="not stored locally"):
        gateway.index_history("TEST")


# ============================================================================
# Optional database tests (MRIP_TEST_DB=1)
# ============================================================================


@pytest.fixture()
def db_store() -> PriceStore:
    """Set up a fresh database and return a PriceStore."""
    from app.utils import db

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_price_bars RESTART IDENTITY")
        cur.execute("TRUNCATE mrip_price_sync RESTART IDENTITY")
        conn.commit()
        cur.close()
    return PriceStore(db.get_db_connection)


@DB
def test_db_upsert_and_series(db_store: PriceStore) -> None:
    bars = [make_bar(date(2026, 1, i), 100.0 + i) for i in range(1, 4)]
    count = db_store.upsert_bars("cboe", "TEST", bars)
    assert count == 3

    series = db_store.series("cboe", "TEST")
    assert series is not None
    assert len(series.bars) == 3
    assert series.bars[0].close == 101.0


@DB
def test_db_upsert_idempotency(db_store: PriceStore) -> None:
    bars = [make_bar(date(2026, 1, 1), 100.0)]
    db_store.upsert_bars("cboe", "TEST", bars)
    db_store.upsert_bars("cboe", "TEST", bars)

    series = db_store.series("cboe", "TEST")
    assert len(series.bars) == 1


@DB
def test_db_upsert_overwrites_revisions(db_store: PriceStore) -> None:
    bars1 = [make_bar(date(2026, 1, 1), 100.0)]
    bars2 = [make_bar(date(2026, 1, 1), 105.0)]
    db_store.upsert_bars("cboe", "TEST", bars1)
    db_store.upsert_bars("cboe", "TEST", bars2)

    series = db_store.series("cboe", "TEST")
    assert len(series.bars) == 1
    assert series.bars[0].close == 105.0


@DB
def test_db_series_with_date_filtering(db_store: PriceStore) -> None:
    bars = [make_bar(date(2026, 1, i), 100.0 + i) for i in range(1, 6)]
    db_store.upsert_bars("cboe", "TEST", bars)

    series = db_store.series("cboe", "TEST", start=date(2026, 1, 2), end=date(2026, 1, 4))
    assert len(series.bars) == 3
    assert series.bars[0].ts == date(2026, 1, 2)
    assert series.bars[-1].ts == date(2026, 1, 4)


@DB
def test_db_last_bar_date(db_store: PriceStore) -> None:
    bars = [make_bar(date(2026, 1, i), 100.0 + i) for i in range(1, 4)]
    db_store.upsert_bars("cboe", "TEST", bars)
    assert db_store.last_bar_date("cboe", "TEST") == date(2026, 1, 3)
    assert db_store.last_bar_date("cboe", "MISSING") is None


@DB
def test_db_record_attempt(db_store: PriceStore) -> None:
    now = T0
    db_store.record_attempt("cboe", "TEST", success=False, error="err1", now=now)
    db_store.record_attempt("cboe", "TEST", success=False, error="err2", now=now + timedelta(seconds=1))
    db_store.record_attempt("cboe", "TEST", success=True, now=now + timedelta(seconds=2))

    states = db_store.sync_states("cboe", ["TEST"])
    assert len(states) == 1
    state = states[0]
    assert state.consecutive_failures == 0
    assert state.last_success_at == now + timedelta(seconds=2)
    assert state.last_error is None


@DB
def test_db_record_attempt_error_truncation(db_store: PriceStore) -> None:
    long_error = "x" * 600
    db_store.record_attempt("cboe", "TEST", success=False, error=long_error, now=T0)

    states = db_store.sync_states("cboe", ["TEST"])
    assert len(states[0].last_error) == 500


@DB
def test_db_sync_states_filtering(db_store: PriceStore) -> None:
    now = T0
    db_store.record_attempt("cboe", "A", success=True, now=now)
    db_store.record_attempt("cboe", "B", success=True, now=now)
    db_store.record_attempt("cboe", "C", success=True, now=now)

    states = db_store.sync_states("cboe", ["A", "C"])
    symbols = [s.symbol for s in states]
    assert symbols == ["A", "C"]


@DB
def test_db_upsert_collapses_duplicate_dates(db_store: PriceStore) -> None:
    """Test that duplicate bar_dates are collapsed, keeping the last occurrence."""
    bars = [
        make_bar(date(2026, 1, 1), 100.0),
        make_bar(date(2026, 1, 1), 102.0),  # Same date, different close
        make_bar(date(2026, 1, 2), 110.0),
    ]
    count = db_store.upsert_bars("cboe", "TEST", bars)
    assert count == 2  # Only 2 distinct dates

    series = db_store.series("cboe", "TEST")
    assert len(series.bars) == 2
    assert series.bars[0].ts == date(2026, 1, 1)
    assert series.bars[0].close == 102.0  # Last occurrence wins
    assert series.bars[1].ts == date(2026, 1, 2)
    assert series.bars[1].close == 110.0


@DB
def test_db_upsert_duplicate_dates_with_datetime(db_store: PriceStore) -> None:
    """Test that datetime and date on the same calendar day are treated as duplicates."""
    bars = [
        make_bar(datetime(2026, 1, 1, 10, 30, tzinfo=timezone.utc), 100.0),
        make_bar(date(2026, 1, 1), 105.0),  # Same calendar date, different ts type
        make_bar(date(2026, 1, 2), 110.0),
    ]
    count = db_store.upsert_bars("cboe", "TEST", bars)
    assert count == 2  # Only 2 distinct dates

    series = db_store.series("cboe", "TEST")
    assert len(series.bars) == 2
    assert series.bars[0].close == 105.0  # Last occurrence (the date) wins
    assert series.bars[1].close == 110.0


@DB
def test_db_upsert_second_call_overwrites_revision(db_store: PriceStore) -> None:
    """Test that a second call with a revised close overwrites the previous one."""
    bars1 = [
        make_bar(date(2026, 1, 1), 100.0),
        make_bar(date(2026, 1, 2), 110.0),
    ]
    count1 = db_store.upsert_bars("cboe", "TEST", bars1)
    assert count1 == 2

    # Second call with a revised close for 2026-01-01
    bars2 = [
        make_bar(date(2026, 1, 1), 105.0),
    ]
    count2 = db_store.upsert_bars("cboe", "TEST", bars2)
    assert count2 == 1

    series = db_store.series("cboe", "TEST")
    assert len(series.bars) == 2
    assert series.bars[0].ts == date(2026, 1, 1)
    assert series.bars[0].close == 105.0  # Updated to revised close
    assert series.bars[1].ts == date(2026, 1, 2)
    assert series.bars[1].close == 110.0  # Unchanged


@DB
def test_db_full_ingestor_run(db_store: PriceStore) -> None:
    """Full integration: ingestor with real store and fake gateway."""
    clock = FakeClock()
    gateway = FakeGateway()
    gateway.responses = {
        "A": [make_bar(date(2026, 1, 1), 100.0), make_bar(date(2026, 1, 2), 101.0)],
        "B": [make_bar(date(2026, 1, 1), 200.0)],
    }

    ingestor = PriceIngestor(
        gateway,
        db_store,
        clock=clock,
        sleep=lambda s: clock.advance(s),
        now=lambda: T0,
    )
    report = ingestor.sync(["A", "B"])

    assert report.attempted == 2
    assert report.succeeded == 2
    assert report.bars_written == 3
    assert len(report.failed) == 0

    # Verify bars are in the database.
    series_a = db_store.series("cboe", "A")
    assert len(series_a.bars) == 2
    series_b = db_store.series("cboe", "B")
    assert len(series_b.bars) == 1
