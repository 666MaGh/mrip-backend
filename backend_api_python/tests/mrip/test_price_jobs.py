"""Price sync jobs with run tracking and advisory locking (work 014): unit and optional DB tests."""
from __future__ import annotations

import contextlib
import dataclasses
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path as FsPath
from typing import Any, Callable, ContextManager, Sequence
from unittest.mock import Mock, patch

import pytest

from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.data.models import Latency, PriceBar, PriceSeries, Provenance
from app.mrip.prices.ingest import SyncReport
from app.mrip.prices.jobs import UNIVERSES, JobResult, JobRun, JobRunStore, run_price_sync
from app.mrip.prices.store import PriceStore, SyncState

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATIONS = FsPath(__file__).resolve().parents[2] / "migrations"

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def pg_connection() -> Callable[[], ContextManager[Any]]:
    """Apply MRIP migrations and clear tables for opt-in integration tests."""
    from app.utils.db import get_db_connection

    with get_db_connection() as conn:
        cur = conn.cursor()
        for migration_name in (
            "mrip_20261001_relationships.sql",
            "mrip_20261001_prices.sql",
            "mrip_20261001_jobs.sql",
        ):
            cur.execute((MIGRATIONS / migration_name).read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_job_runs, mrip_price_bars, mrip_price_sync")
        cur.execute("TRUNCATE mrip_rel_edges, mrip_rel_nodes CASCADE")
        conn.commit()
        cur.close()
    return get_db_connection


def make_bar(ts: datetime, close: float | None, volume: float | None = None) -> PriceBar:
    """Helper to create a PriceBar."""
    safe_close = close if close is not None else 100.0
    return PriceBar(
        ts=ts,
        open=safe_close,
        high=safe_close * 1.01,
        low=safe_close * 0.99,
        close=close,
        volume=volume or 1_000_000.0,
    )


class FakeJobRunStore:
    """In-memory fake job run store for testing."""

    def __init__(self) -> None:
        self._runs: dict[int, JobRun] = {}
        self._next_id = 1

    def start(self, job: str, universe: str) -> int:
        """Start a new run."""
        run_id = self._next_id
        self._next_id += 1
        self._runs[run_id] = JobRun(
            id=run_id,
            job=job,
            universe=universe,
            started_at=T0,
            finished_at=None,
            status="running",
            report={},
            error=None,
        )
        return run_id

    def finish(self, run_id: int, status: str, report: dict[str, Any], error: str | None = None) -> None:
        """Finish a run."""
        run = self._runs[run_id]
        self._runs[run_id] = JobRun(
            id=run.id,
            job=run.job,
            universe=run.universe,
            started_at=run.started_at,
            finished_at=T0 + timedelta(seconds=1),
            status=status,
            report=report,
            error=error,
        )

    def record_skipped(self, job: str, universe: str, reason: str) -> None:
        """Record a skipped run."""
        run_id = self._next_id
        self._next_id += 1
        self._runs[run_id] = JobRun(
            id=run_id,
            job=job,
            universe=universe,
            started_at=T0,
            finished_at=T0,
            status="skipped",
            report={"reason": reason},
            error=None,
        )

    def last_run(self, job: str, universe: str) -> JobRun | None:
        """Get the most recent run."""
        candidates = [r for r in self._runs.values() if r.job == job and r.universe == universe]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.id)

    def recent(self, job: str | None = None, limit: int = 20) -> list[JobRun]:
        """Get recent runs."""
        candidates = [r for r in self._runs.values() if job is None or r.job == job]
        return sorted(candidates, key=lambda r: -r.id)[:limit]


class FakePriceStore:
    """In-memory fake price store."""

    def __init__(self) -> None:
        self._bars: dict[tuple[str, str], list[PriceBar]] = {}
        self._sync: dict[tuple[str, str], SyncState] = {}

    def upsert_bars(self, provider: str, symbol: str, bars: Sequence[PriceBar]) -> int:
        """Store bars."""
        key = (provider, symbol)
        self._bars[key] = list(bars)
        return len(bars)

    def series(self, provider: str, symbol: str, start=None, end=None) -> PriceSeries | None:
        """Retrieve bars."""
        key = (provider, symbol)
        bars = self._bars.get(key, [])
        if not bars:
            return None
        return PriceSeries(
            symbol=symbol,
            interval="1d",
            bars=tuple(bars),
            provenance=Provenance(provider=provider, gateway="test", endpoint="test", fetched_at=T0, latency=Latency.UNKNOWN),
        )

    def record_attempt(self, provider: str, symbol: str, *, success: bool, last_bar_date=None, error=None, now=None) -> None:
        """Record attempt."""
        pass


class FakeGateway(FinancialDataGateway):
    """Fake gateway that returns a fixed number of bars per symbol."""

    def __init__(self, bars_per_symbol: int = 5, raise_after: int | None = None) -> None:
        self._bars_per_symbol = bars_per_symbol
        self._raise_after = raise_after
        self._call_count = 0

    def price_history(self, symbol: str, start=None, end=None, interval="1d") -> PriceSeries:
        """Return fake bars."""
        self._call_count += 1
        if self._raise_after is not None and self._call_count > self._raise_after:
            raise DataUnavailable(f"throttled after {self._raise_after} calls")

        bars = [
            make_bar(T0 - timedelta(days=i), 100.0 + i)
            for i in range(self._bars_per_symbol)
        ]
        return PriceSeries(
            symbol=symbol,
            interval="1d",
            bars=tuple(bars),
            provenance=Provenance(provider="test", gateway="test", endpoint="test", fetched_at=T0, latency=Latency.UNKNOWN),
        )

    def index_history(self, *args, **kwargs):
        raise NotImplementedError()

    def vix_history(self, *args, **kwargs):
        raise NotImplementedError()

    def options_chain(self, *args, **kwargs):
        raise NotImplementedError()

    def cot(self, *args, **kwargs):
        raise NotImplementedError()

    def macro_series(self, *args, **kwargs):
        raise NotImplementedError()


class FakeGraph:
    """Fake relationship graph."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self._symbols = symbols or []

    def list_nodes_in_universe(self, universe: str) -> list[Any]:
        """Return fake nodes with symbol attributes."""
        from app.mrip.relationships.types import Node, NodeType

        return [
            Node(
                id=i,
                node_type=NodeType.SECURITY,
                key=symbol,
                name=symbol,
                attributes={"series": {"symbol": symbol}},
            )
            for i, symbol in enumerate(self._symbols)
        ]


# ============================================================================
# Unit tests (no DB)
# ============================================================================


def test_universes_mapping_exists():
    """Test that UNIVERSES contains expected entries."""
    assert "sp500" in UNIVERSES
    assert "omxslc" in UNIVERSES
    provider, factory, interval = UNIVERSES["sp500"]
    assert provider == "cboe"
    assert interval == 2.0
    provider, factory, interval = UNIVERSES["omxslc"]
    assert provider == "yahoo"
    assert interval == 1.0


def test_job_run_store_start_and_finish():
    """Test JobRunStore start/finish round trip."""
    store = FakeJobRunStore()
    run_id = store.start("test-job", "sp500")
    assert run_id > 0

    store.finish(run_id, "completed", {"bars": 100})

    run = store.last_run("test-job", "sp500")
    assert run is not None
    assert run.id == run_id
    assert run.status == "completed"
    assert run.report == {"bars": 100}


def test_job_run_store_record_skipped():
    """Test recording a skipped run."""
    store = FakeJobRunStore()
    store.record_skipped("test-job", "sp500", "lock held")

    run = store.last_run("test-job", "sp500")
    assert run is not None
    assert run.status == "skipped"
    assert run.report == {"reason": "lock held"}


def test_job_run_store_recent():
    """Test fetching recent runs."""
    store = FakeJobRunStore()
    store.record_skipped("job1", "sp500", "reason1")
    store.record_skipped("job1", "omxslc", "reason2")
    store.record_skipped("job2", "sp500", "reason3")

    all_runs = store.recent(job=None, limit=10)
    assert len(all_runs) == 3

    job1_runs = store.recent(job="job1", limit=10)
    assert len(job1_runs) == 2


def test_run_price_sync_unknown_universe():
    """Test run_price_sync with unknown universe."""
    with pytest.raises(ValueError, match="unknown universe"):
        run_price_sync(
            "unknown",
            budget_seconds=100.0,
            run_store=FakeJobRunStore(),
        )


def test_run_price_sync_lock_held():
    """Test run_price_sync skips when lock is held."""
    @contextlib.contextmanager
    def failing_lock(key: str) -> Any:
        yield False  # Lock not acquired

    store = FakeJobRunStore()
    result = run_price_sync(
        "sp500",
        budget_seconds=100.0,
        lock_factory=failing_lock,
        run_store=store,
    )

    assert result.status == "skipped"
    assert result.reason == "another sync is running"
    assert result.report is None


def test_run_price_sync_cooldown_after_halted():
    """Test cooldown skips after a 'halted' run."""
    store = FakeJobRunStore()
    # Record a halted run in the recent past.
    run_id = store.start("mrip-price-sync", "sp500")
    store.finish(run_id, "halted", {"halted_reason": "throttled"})

    # Now try to run again immediately.
    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    result = run_price_sync(
        "sp500",
        budget_seconds=100.0,
        lock_factory=ok_lock,
        run_store=store,
        cooldown_seconds=3600.0,  # 1 hour cooldown
        now=lambda: T0 + timedelta(seconds=1),  # 1 second later
    )

    assert result.status == "skipped"
    assert "cooling down" in result.reason


def test_run_price_sync_halted_run_older_than_cooldown_does_not_skip():
    store = FakeJobRunStore()
    run_id = store.start("mrip-price-sync", "sp500")
    store.finish(run_id, "halted", {"halted_reason": "throttled"})

    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    with patch("app.mrip.relationships.graph.RelationshipGraph") as graph_cls:
        with patch("app.mrip.prices.store.PriceStore"), patch("app.mrip.prices.ingest.PriceIngestor") as ingestor_cls:
            graph_cls.return_value.list_nodes_in_universe.return_value = [
                Mock(attributes={"series": {"symbol": "AAPL"}})
            ]
            ingestor_cls.return_value.sync.return_value = SyncReport(0, 0, {}, 1, 0, 0, None)
            result = run_price_sync(
                "sp500",
                budget_seconds=10,
                lock_factory=ok_lock,
                run_store=store,
                now=lambda: T0 + timedelta(hours=1),
                cooldown_seconds=1800,
                load_if_empty=False,
            )
    assert result.status == "completed"


def test_run_price_sync_no_cooldown_after_completed():
    """Test cooldown does not skip after 'completed' run."""
    store = FakeJobRunStore()
    # Record a completed run.
    run_id = store.start("mrip-price-sync", "sp500")
    store.finish(run_id, "completed", {"attempted": 10})

    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    # Try to run again; should not skip for cooldown.
    with patch("app.mrip.relationships.graph.RelationshipGraph") as MockGraph:
        with patch("app.mrip.prices.store.PriceStore") as MockStore:
            with patch("app.mrip.prices.ingest.PriceIngestor") as MockIngestor:
                mock_graph = Mock()
                mock_graph.list_nodes_in_universe.return_value = [
                    Mock(attributes={"series": {"symbol": "AAPL"}})
                ]
                MockGraph.return_value = mock_graph

                mock_store = Mock()
                MockStore.return_value = mock_store

                mock_ingestor = Mock()
                mock_ingestor.sync.return_value = SyncReport(
                    attempted=1,
                    succeeded=1,
                    failed={},
                    skipped_fresh=0,
                    not_attempted=0,
                    bars_written=5,
                    halted_reason=None,
                )
                MockIngestor.return_value = mock_ingestor

                result = run_price_sync(
                    "sp500",
                    budget_seconds=100.0,
                    lock_factory=ok_lock,
                    run_store=store,
                    now=lambda: T0 + timedelta(seconds=1),
                )

                # Should not be skipped; should proceed.
                assert result.status != "skipped"


def test_run_price_sync_no_symbols():
    """Test run_price_sync fails when no symbols found and load_if_empty=False."""
    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    store = FakeJobRunStore()

    with patch("app.mrip.relationships.graph.RelationshipGraph") as MockGraph:
        mock_graph = Mock()
        mock_graph.list_nodes_in_universe.return_value = []
        MockGraph.return_value = mock_graph

        result = run_price_sync(
            "sp500",
            budget_seconds=100.0,
            lock_factory=ok_lock,
            run_store=store,
            load_if_empty=False,
        )

        assert result.status == "failed"
        assert result.reason == "universe has no symbols"


def test_run_price_sync_loads_empty_universe_before_sync():
    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    store = FakeJobRunStore()
    graph = Mock()
    graph.list_nodes_in_universe.side_effect = [
        [],
        [Mock(attributes={"series": {"symbol": "AAPL"}})],
    ]
    with patch("app.mrip.relationships.graph.RelationshipGraph", return_value=graph):
        with patch("app.mrip.universe.sources.fetch_sp500", return_value=[Mock()]) as fetch_sp500:
            with patch("app.mrip.universe.loader.UniverseLoader") as loader_cls:
                with patch("app.mrip.prices.store.PriceStore"):
                    with patch("app.mrip.prices.ingest.PriceIngestor") as ingestor_cls:
                        ingestor_cls.return_value.sync.return_value = SyncReport(
                            1, 1, {}, 0, 0, 5, None
                        )
                        result = run_price_sync(
                            "sp500",
                            budget_seconds=10,
                            lock_factory=ok_lock,
                            run_store=store,
                        )
    fetch_sp500.assert_called_once_with()
    loader_cls.return_value.load.assert_called_once()
    assert result.status == "completed"


def test_run_price_sync_halted_vs_completed():
    """Test that 'halted' status maps to throttled halted_reason."""
    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    @contextlib.contextmanager
    def mock_connect() -> Any:
        yield Mock()

    store = FakeJobRunStore()

    with patch("app.mrip.relationships.graph.RelationshipGraph") as MockGraph:
        with patch("app.mrip.prices.store.PriceStore") as MockStore:
            with patch("app.mrip.prices.ingest.PriceIngestor") as MockIngestor:
                mock_graph = Mock()
                mock_graph.list_nodes_in_universe.return_value = [
                    Mock(attributes={"series": {"symbol": "AAPL"}})
                ]
                MockGraph.return_value = mock_graph
                MockStore.return_value = Mock()

                # Test halted case (throttled).
                mock_ingestor = Mock()
                mock_ingestor.sync.return_value = SyncReport(
                    attempted=5,
                    succeeded=2,
                    failed={"A": "err", "B": "err"},
                    skipped_fresh=0,
                    not_attempted=1,
                    bars_written=0,
                    halted_reason="provider appears throttled: 8 consecutive failures",
                )
                MockIngestor.return_value = mock_ingestor

                result = run_price_sync(
                    "sp500",
                    budget_seconds=100.0,
                    lock_factory=ok_lock,
                    connect=mock_connect,
                    run_store=store,
                    load_if_empty=False,
                )

                assert result.status == "halted", f"Expected halted, got {result.status}: {result.reason}"

                # Test completed case (budget exhausted, no throttle mention).
                mock_ingestor.sync.return_value = SyncReport(
                    attempted=10,
                    succeeded=10,
                    failed={},
                    skipped_fresh=0,
                    not_attempted=490,
                    bars_written=100,
                    halted_reason="budget exhausted",
                )

                result = run_price_sync(
                    "sp500",
                    budget_seconds=100.0,
                    lock_factory=ok_lock,
                    connect=mock_connect,
                    run_store=FakeJobRunStore(),  # Fresh store
                    load_if_empty=False,
                )

                assert result.status == "completed", f"Expected completed, got {result.status}: {result.reason}"


def test_run_price_sync_report_failure_truncation():
    """Test that failures are truncated to 20 items in the report."""
    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    @contextlib.contextmanager
    def mock_connect() -> Any:
        yield Mock()

    store = FakeJobRunStore()

    with patch("app.mrip.relationships.graph.RelationshipGraph") as MockGraph:
        with patch("app.mrip.prices.store.PriceStore") as MockStore:
            with patch("app.mrip.prices.ingest.PriceIngestor") as MockIngestor:
                mock_graph = Mock()
                mock_graph.list_nodes_in_universe.return_value = [
                    Mock(attributes={"series": {"symbol": f"SYM{i}"}})
                    for i in range(50)
                ]
                MockGraph.return_value = mock_graph
                MockStore.return_value = Mock()

                # Create a report with 50 failures.
                failures = {f"SYM{i}": "error message" for i in range(50)}
                mock_ingestor = Mock()
                mock_ingestor.sync.return_value = SyncReport(
                    attempted=50,
                    succeeded=0,
                    failed=failures,
                    skipped_fresh=0,
                    not_attempted=0,
                    bars_written=0,
                    halted_reason=None,
                )
                MockIngestor.return_value = mock_ingestor

                result = run_price_sync(
                    "sp500",
                    budget_seconds=100.0,
                    lock_factory=ok_lock,
                    connect=mock_connect,
                    run_store=store,
                    load_if_empty=False,
                )

                assert result.status == "completed", f"Expected completed, got {result.status}: {result.reason}"
                # Verify report is truncated to 20.
                assert len(result.report.failed) == 50
                saved = store.last_run("mrip-price-sync", "sp500")
                assert saved is not None
                assert len(saved.report["failed"]) == 20


def test_run_price_sync_exception_captured():
    """Test that unexpected exceptions are captured as failed runs."""
    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    store = FakeJobRunStore()

    with patch("app.mrip.relationships.graph.RelationshipGraph") as MockGraph:
        mock_graph = Mock()
        mock_graph.list_nodes_in_universe.side_effect = RuntimeError("connection failed")
        MockGraph.return_value = mock_graph

        result = run_price_sync(
            "sp500",
            budget_seconds=100.0,
            lock_factory=ok_lock,
            run_store=store,
            load_if_empty=False,
        )

    assert result.status == "failed"
    assert "RuntimeError" in result.reason


def test_run_price_sync_releases_lock_when_ingestor_raises():
    entered: list[bool] = []

    @contextlib.contextmanager
    def tracking_lock(key: str) -> Any:
        entered.append(True)
        try:
            yield True
        finally:
            entered.pop()

    store = FakeJobRunStore()
    with patch("app.mrip.relationships.graph.RelationshipGraph") as graph_cls:
        with patch("app.mrip.prices.store.PriceStore"), patch("app.mrip.prices.ingest.PriceIngestor") as ingestor_cls:
            graph_cls.return_value.list_nodes_in_universe.return_value = [
                Mock(attributes={"series": {"symbol": "AAPL"}})
            ]
            ingestor_cls.return_value.sync.side_effect = RuntimeError("boom")
            result = run_price_sync(
                "sp500",
                budget_seconds=10,
                lock_factory=tracking_lock,
                run_store=store,
                load_if_empty=False,
            )
    assert result.status == "failed"
    assert not entered


def test_celery_tasks_import():
    """Test that Celery tasks can be imported and registered."""
    from app.celery_app import celery_app
    from app.tasks import mrip_sync

    # Verify tasks are registered.
    assert "quantdinger.tasks.mrip_price_sync" in celery_app.tasks
    assert "quantdinger.tasks.mrip_price_sync_all" in celery_app.tasks


def test_celery_beat_schedule():
    """Test that beat schedule contains the mrip-price-sync entry."""
    from app.celery_app import celery_app

    assert "mrip-price-sync" in celery_app.conf.beat_schedule
    entry = celery_app.conf.beat_schedule["mrip-price-sync"]
    assert entry["task"] == "quantdinger.tasks.mrip_price_sync_all"
    assert entry["schedule"] >= 600  # Minimum 10 minutes


def test_celery_task_routes():
    """Test that tasks are routed to maintenance queue."""
    from app.celery_app import celery_app

    routes = celery_app.conf.task_routes
    assert routes.get("quantdinger.tasks.mrip_price_sync", {}).get("queue") == "maintenance"
    assert routes.get("quantdinger.tasks.mrip_price_sync_all", {}).get("queue") == "maintenance"


def test_mrip_price_sync_task_disabled():
    """Test that mrip_price_sync returns skipped when disabled."""
    from app.tasks.mrip_sync import mrip_price_sync

    with patch.dict(os.environ, {"ENABLE_MRIP_PRICE_SYNC": "false"}):
        result = mrip_price_sync("sp500")
        assert result == {"skipped": True}


def test_mrip_price_sync_all_task_disabled():
    from app.tasks.mrip_sync import mrip_price_sync_all

    with patch.dict(os.environ, {"ENABLE_MRIP_PRICE_SYNC": "false"}):
        assert mrip_price_sync_all() == {"skipped": True}


def test_mrip_price_sync_all_task():
    """Test mrip_price_sync_all returns results for both universes."""
    from app.tasks.mrip_sync import mrip_price_sync_all

    with patch("app.tasks.mrip_sync.mrip_price_sync") as mock_sync:
        mock_sync.return_value = {
            "status": "completed",
            "report": None,
            "reason": None,
        }

        with patch.dict(os.environ, {"ENABLE_MRIP_PRICE_SYNC": "true"}):
            result = mrip_price_sync_all()

            assert "omxslc" in result
            assert "sp500" in result
            assert mock_sync.call_count == 2


# ============================================================================
# Optional DB tests (run with MRIP_TEST_DB=1 and a throwaway DATABASE_URL)
# ============================================================================


@DB
def test_job_run_store_db_round_trip(pg_connection):
    """Test JobRunStore with real database."""
    # Truncate the table first.
    with pg_connection() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE mrip_job_runs")
        conn.commit()
        cur.close()

    store = JobRunStore(pg_connection)

    # Start a run.
    run_id = store.start("test-job", "sp500")
    assert run_id > 0

    # Finish it.
    store.finish(run_id, "completed", {"bars": 50})

    # Retrieve it.
    run = store.last_run("test-job", "sp500")
    assert run is not None
    assert run.status == "completed"
    assert run.report == {"bars": 50}

    # Test recent.
    recents = store.recent(job="test-job", limit=10)
    assert len(recents) >= 1


@DB
def test_advisory_lock_real(pg_connection):
    """Test Postgres advisory lock acquisition and release."""
    from app.mrip.prices.jobs import run_price_sync

    store = FakeJobRunStore()

    # Hold a lock on a separate connection.
    import contextlib

    @contextlib.contextmanager
    def hold_lock_separately() -> Any:
        with pg_connection() as lock_conn:
            cur = lock_conn.cursor()
            cur.execute("SELECT pg_advisory_lock(hashtext(%s))", ("mrip_price_sync:sp500",))
            try:
                yield
            finally:
                cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", ("mrip_price_sync:sp500",))
                lock_conn.commit()
            cur.close()

    with hold_lock_separately():
        # Now try to run_price_sync; it should fail to acquire lock.
        result = run_price_sync(
            "sp500",
            budget_seconds=100.0,
            connect=pg_connection,
            run_store=store,
            load_if_empty=False,
        )

        assert result.status == "skipped"
        assert "another sync is running" in result.reason


@DB
def test_full_sync_run_omxslc(pg_connection):
    """Full integration test: sync omxslc with a fake gateway."""
    from app.mrip.relationships.graph import RelationshipGraph

    # Truncate tables.
    with pg_connection() as conn:
        cur = conn.cursor()
        cur.execute("TRUNCATE mrip_job_runs")
        cur.execute("TRUNCATE mrip_price_bars")
        cur.execute("TRUNCATE mrip_price_sync")
        cur.execute("TRUNCATE mrip_rel_edges, mrip_rel_nodes CASCADE")
        conn.commit()
        cur.close()

    graph = RelationshipGraph(pg_connection)
    store = JobRunStore(pg_connection)
    gateway = FakeGateway(bars_per_symbol=5)

    @contextlib.contextmanager
    def ok_lock(key: str) -> Any:
        yield True

    result = run_price_sync(
        "omxslc",
        budget_seconds=9999.0,  # Large budget so all complete.
        min_interval=0.0,
        connect=pg_connection,
        gateway_factory=lambda: gateway,
        lock_factory=ok_lock,
        run_store=store,
        load_if_empty=True,
    )

    # Verify result.
    assert result.status == "completed"
    assert result.report is not None
    assert result.report.attempted > 0
    assert result.report.succeeded > 0

    # Verify job run was recorded.
    last_run = store.last_run("mrip-price-sync", "omxslc")
    assert last_run is not None
    assert last_run.status == "completed"
    assert last_run.report["bars_written"] > 0
    assert len(graph.list_nodes_in_universe("OMXSLC")) > 0
    second = run_price_sync(
        "omxslc",
        budget_seconds=9999.0,
        min_interval=0.0,
        connect=pg_connection,
        gateway_factory=lambda: gateway,
        lock_factory=ok_lock,
        run_store=store,
        load_if_empty=True,
    )
    assert second.status == "completed"
    assert second.report is not None
    assert second.report.skipped_fresh == len(graph.list_nodes_in_universe("OMXSLC"))
