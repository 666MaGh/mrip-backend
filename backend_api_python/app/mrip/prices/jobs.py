"""Scheduled price sync jobs with run tracking and advisory locking (work 014)."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from typing import Any, Callable, ContextManager

from app.mrip.prices.ingest import SyncReport

# universe -> (provider name stored with the bars, gateway factory, default pacing seconds)
UNIVERSES = {
    "sp500": ("cboe", lambda: __import__("app.mrip.data.openbb_adapter", fromlist=["OpenBBAdapter"]).OpenBBAdapter(retries=1), 2.0),
    "omxslc": ("yahoo", lambda: __import__("app.mrip.data.yahoo_adapter", fromlist=["YahooChartAdapter"]).YahooChartAdapter(retries=2), 1.0),
}


@dataclasses.dataclass(frozen=True, slots=True)
class JobRun:
    """Record of a single job execution."""

    id: int
    job: str
    universe: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    report: dict[str, Any]
    error: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class JobResult:
    """Result of running a price sync job."""

    status: str  # "completed", "halted", "skipped", or "failed"
    report: SyncReport | None
    reason: str | None  # explanation for "skipped" or "failed"


class JobRunStore:
    """PostgreSQL-backed store for scheduled job run tracking."""

    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    def start(self, job: str, universe: str) -> int:
        """Start a new job run; return the run ID.

        Args:
            job: Name of the job (e.g. "mrip-price-sync").
            universe: Universe name (e.g. "sp500", "omxslc").

        Returns:
            The new run ID.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO mrip_job_runs (job, universe, status)
                    VALUES (%s, %s, 'running')
                    RETURNING id
                    """,
                    (job, universe),
                )
                run_id = cur.fetchone()["id"]
                conn.commit()
                return run_id
            finally:
                cur.close()

    def finish(
        self,
        run_id: int,
        status: str,
        report: dict[str, Any],
        error: str | None = None,
    ) -> None:
        """Mark a job run as finished.

        Args:
            run_id: The run ID from start().
            status: Status ("completed", "halted", "skipped", or "failed").
            report: Report data as a dict (JSON-serializable).
            error: Error message if status is "failed".
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    UPDATE mrip_job_runs
                    SET status = %s, finished_at = NOW(), report = %s, error = %s
                    WHERE id = %s
                    """,
                    (status, json.dumps(report), error, run_id),
                )
                conn.commit()
            finally:
                cur.close()

    def record_skipped(self, job: str, universe: str, reason: str) -> None:
        """Record a skipped job run immediately.

        Args:
            job: Name of the job.
            universe: Universe name.
            reason: Reason for skipping.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO mrip_job_runs (job, universe, status, finished_at, report)
                    VALUES (%s, %s, 'skipped', NOW(), %s)
                    """,
                    (job, universe, json.dumps({"reason": reason})),
                )
                conn.commit()
            finally:
                cur.close()

    def last_run(self, job: str, universe: str) -> JobRun | None:
        """Fetch the most recent run for a job/universe pair.

        Args:
            job: Name of the job.
            universe: Universe name.

        Returns:
            JobRun or None if no run exists.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT id, job, universe, started_at, finished_at, status, report, error
                    FROM mrip_job_runs
                    WHERE job = %s AND universe = %s
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (job, universe),
                )
                row = cur.fetchone()
            finally:
                cur.close()

        if row is None:
            return None

        return JobRun(
            id=row["id"],
            job=row["job"],
            universe=row["universe"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            status=row["status"],
            report=dict(row["report"] or {}),
            error=row["error"],
        )

    def recent(self, job: str | None = None, limit: int = 20) -> list[JobRun]:
        """Fetch recent job runs.

        Args:
            job: Optional job name filter; if None, all jobs.
            limit: Maximum number of runs to return.

        Returns:
            List of JobRun objects, most recent first.
        """
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                if job is None:
                    cur.execute(
                        """
                        SELECT id, job, universe, started_at, finished_at, status, report, error
                        FROM mrip_job_runs
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, job, universe, started_at, finished_at, status, report, error
                        FROM mrip_job_runs
                        WHERE job = %s
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (job, limit),
                    )
                rows = cur.fetchall()
            finally:
                cur.close()

        return [
            JobRun(
                id=row["id"],
                job=row["job"],
                universe=row["universe"],
                started_at=row["started_at"],
                finished_at=row["finished_at"],
                status=row["status"],
                report=dict(row["report"] or {}),
                error=row["error"],
            )
            for row in rows
        ]


def run_price_sync(
    universe: str,
    *,
    budget_seconds: float,
    min_interval: float | None = None,
    connect: Callable[[], ContextManager[Any]] | None = None,
    gateway_factory: Callable[[], Any] | None = None,
    cooldown_seconds: float = 1800.0,
    now: Callable[[], datetime] | None = None,
    load_if_empty: bool = True,
    lock_factory: Callable[[str], ContextManager[bool]] | None = None,
    run_store: JobRunStore | None = None,
) -> JobResult:
    """Run price sync for a single universe with Postgres advisory locking.

    Steps:
    1. Acquire an exclusive advisory lock for the universe; if held, record skipped.
    2. Check cooldown: if last run was 'halted' and within cooldown_seconds, skip.
    3. Start a job run record.
    4. Load symbols from the relationship graph; if empty, load the universe first.
    5. Run the PriceIngestor with the given budget.
    6. Finish the job run: status 'halted' if throttled, else 'completed'.
    7. On any unexpected exception, finish as 'failed' and return the error.

    Args:
        universe: Universe name ("sp500" or "omxslc").
        budget_seconds: Maximum seconds for ingestion.
        min_interval: Minimum seconds between provider calls; defaults to UNIVERSES[universe] value.
        connect: Connection factory; defaults to app.utils.db.get_db_connection.
        gateway_factory: Gateway factory; defaults to UNIVERSES[universe] factory.
        cooldown_seconds: Seconds to wait after a 'halted' run before retrying.
        now: Current time function; defaults to datetime.now(timezone.utc).
        load_if_empty: If True, auto-load universe if no symbols found.
        lock_factory: Lock factory for testing; signature: (key: str) -> context manager yielding bool.
        run_store: JobRunStore for testing; defaults to a new one with connect.

    Returns:
        JobResult with status, report, and optional reason.

    Raises:
        ValueError: If universe is unknown.
    """
    if universe not in UNIVERSES:
        raise ValueError(f"unknown universe: {universe}")

    if connect is None:
        from app.utils.db import get_db_connection
        connect = get_db_connection

    if now is None:
        now = lambda: datetime.now(timezone.utc)

    if run_store is None:
        run_store = JobRunStore(connect)

    if gateway_factory is None:
        _, gateway_factory, _ = UNIVERSES[universe]

    if min_interval is None:
        _, _, min_interval = UNIVERSES[universe]

    provider, _, _ = UNIVERSES[universe]

    job_name = "mrip-price-sync"

    # Step 1: Acquire advisory lock.
    lock_key = f"mrip_price_sync:{universe}"

    # Use the provided lock_factory or create a default one.
    actual_lock_factory = lock_factory
    if actual_lock_factory is None:
        # Create default lock factory with lazy database access.
        import contextlib

        @contextlib.contextmanager
        def _default_lock_factory(key: str) -> Any:
            """Acquire Postgres advisory lock."""
            with connect() as lock_conn:
                cur = lock_conn.cursor()
                acquired = False
                try:
                    cur.execute(
                        "SELECT pg_try_advisory_lock(hashtext(%s)) AS acquired",
                        (key,),
                    )
                    row = cur.fetchone()
                    acquired = bool(row["acquired"] if isinstance(row, dict) else row[0])
                    yield acquired
                finally:
                    if acquired:
                        cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (key,))
                    cur.close()

        actual_lock_factory = _default_lock_factory

    try:
        # Keep this context open for the complete run. The database connection
        # owns the session-level advisory lock, so it must remain dedicated.
        with actual_lock_factory(lock_key) as lock_acquired:
            if not lock_acquired:
                run_store.record_skipped(job_name, universe, "another sync is running")
                return JobResult("skipped", None, "another sync is running")

            # Step 2: Check cooldown while the lock prevents a concurrent run.
            current_time = now()
            last_run = run_store.last_run(job_name, universe)
            if (
                last_run is not None
                and last_run.status == "halted"
                and last_run.finished_at is not None
            ):
                time_since_halt = (current_time - last_run.finished_at).total_seconds()
                if time_since_halt < cooldown_seconds:
                    reason = "cooling down after a throttled run"
                    run_store.record_skipped(job_name, universe, reason)
                    return JobResult("skipped", None, reason)

            # Step 3: Start job run.
            run_id = run_store.start(job_name, universe)

            try:
                # Step 4: Load symbols from graph.
                from app.mrip.prices.ingest import PriceIngestor
                from app.mrip.prices.store import PriceStore
                from app.mrip.relationships.graph import RelationshipGraph
                from app.mrip.universe.loader import UniverseLoader

                graph = RelationshipGraph(connect)
                nodes = graph.list_nodes_in_universe(universe.upper())
                symbols = _symbols_from_nodes(nodes)

                # If no symbols and load_if_empty, load the universe.
                if not symbols and load_if_empty:
                    from app.mrip.universe.sources import fetch_sp500, load_omxslc

                    members = fetch_sp500() if universe == "sp500" else load_omxslc()

                    loader = UniverseLoader(graph)
                    loader.load(members)

                    # Re-read symbols.
                    nodes = graph.list_nodes_in_universe(universe.upper())
                    symbols = _symbols_from_nodes(nodes)

                # If still no symbols, fail.
                if not symbols:
                    error_msg = "universe has no symbols"
                    run_store.finish(run_id, "failed", {}, error=error_msg)
                    return JobResult("failed", None, error_msg)

                # Step 5: Run ingestion.
                store = PriceStore(connect)
                gateway = gateway_factory()
                ingestor = PriceIngestor(
                    gateway,
                    store,
                    provider=provider,
                    min_interval_seconds=min_interval,
                )
                report = ingestor.sync(symbols, budget_seconds=budget_seconds)

                # Step 6: Finish job run.
                # Truncate failures to the first 20 entries in the stored report.
                report_data = dataclasses.asdict(report)
                report_data["failed"] = dict(list(report.failed.items())[:20])

                status = (
                    "halted"
                    if report.halted_reason and "throttled" in report.halted_reason.lower()
                    else "completed"
                )
                run_store.finish(run_id, status, report_data)

                return JobResult(status, report, None)

            except Exception as exc:
                # Step 7: On exception, finish as failed.
                error_text = f"{type(exc).__name__}: {str(exc)}"
                run_store.finish(run_id, "failed", {}, error=error_text)
                return JobResult("failed", None, error_text)
    except Exception:
        # Lock acquisition errors cannot safely start a run.
        try:
            run_store.record_skipped(job_name, universe, "another sync is running")
        except Exception:
            pass
        return JobResult("skipped", None, "another sync is running")


def _symbols_from_nodes(nodes: list[Any]) -> list[str]:
    """Extract price symbols from universe graph nodes."""
    symbols: list[str] = []
    for node in nodes:
        series_info = node.attributes.get("series")
        if isinstance(series_info, dict):
            symbol = series_info.get("symbol")
            if isinstance(symbol, str) and symbol:
                symbols.append(symbol)
    return symbols
