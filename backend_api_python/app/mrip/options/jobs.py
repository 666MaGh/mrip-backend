"""Scheduled collection of point-in-time options chains."""
from __future__ import annotations

import contextlib
import dataclasses
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, ContextManager, Sequence
from zoneinfo import ZoneInfo

from app.mrip.data.gateway import DataUnavailable
from app.mrip.options.snapshots import OptionsSnapshotStore, SnapshotError
from app.mrip.prices.jobs import JobResult, JobRunStore

DEFAULT_OPTION_SYMBOLS = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "UNH", "V", "HD", "AVGO", "LLY", "COST", "NFLX", "AMD")
JOB_NAME = "options_snapshots"
_NY = ZoneInfo("America/New_York")


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotReport:
    """Outcome counts for one scheduled snapshot run."""

    attempted: int
    saved: int
    skipped_fresh: int
    failed: dict[str, str]
    not_attempted: int
    halted_reason: str | None


def is_snapshot_window(now: datetime) -> bool:
    """Return whether the instant is in the weekday US post-close window."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(_NY)
    return local.weekday() < 5 and (local.hour, local.minute) >= (16, 15) and (local.hour, local.minute) <= (21, 0)


def _configured_symbols() -> tuple[str, ...]:
    raw = os.getenv("MRIP_OPTIONS_SYMBOLS", "")
    symbols = tuple(part.strip().upper() for part in raw.split(",") if part.strip())
    return symbols or DEFAULT_OPTION_SYMBOLS


def run_options_snapshots(
    symbols: Sequence[str] | None = None,
    *,
    budget_seconds: float = 900.0,
    min_age_hours: float = 20.0,
    min_interval_seconds: float = 3.0,
    max_consecutive_failures: int = 4,
    cooldown_seconds: float = 1800.0,
    connect: Callable[[], ContextManager[Any]] | None = None,
    gateway: Any | None = None,
    store: Any | None = None,
    now: datetime | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    require_window: bool = True,
    lock_factory: Callable[[str], ContextManager[bool]] | None = None,
    run_store: JobRunStore | None = None,
) -> JobResult:
    """Collect recent chains with pacing, dedupe, lock, and run tracking."""
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if require_window and not is_snapshot_window(current_time):
        if run_store is None:
            run_store = _make_run_store(connect)
        run_store.record_skipped(JOB_NAME, "options", "outside the US post-close window")
        return JobResult("skipped", None, "outside the US post-close window")
    if connect is None:
        from app.utils.db import get_db_connection
        connect = get_db_connection
    if run_store is None:
        run_store = JobRunStore(connect)
    if store is None:
        store = OptionsSnapshotStore(connect)
    if gateway is None:
        from app.mrip.data.openbb_adapter import OpenBBAdapter
        gateway = OpenBBAdapter(retries=1)
    selected = tuple(s.strip().upper() for s in symbols if s.strip()) if symbols is not None else _configured_symbols()

    if lock_factory is None:
        @contextlib.contextmanager
        def default_lock(key: str) -> Any:
            with connect() as lock_conn:
                cur = lock_conn.cursor()
                acquired = False
                try:
                    cur.execute("SELECT pg_try_advisory_lock(hashtext(%s)) AS acquired", (key,))
                    row = cur.fetchone()
                    acquired = bool(row["acquired"] if isinstance(row, dict) else row[0])
                    yield acquired
                finally:
                    if acquired:
                        cur.execute("SELECT pg_advisory_unlock(hashtext(%s))", (key,))
                    cur.close()
        lock_factory = default_lock

    try:
        with lock_factory("mrip_options_snapshots") as acquired:
            if not acquired:
                run_store.record_skipped(JOB_NAME, "options", "another snapshot run is running")
                return JobResult("skipped", None, "another snapshot run is running")
            last_run = run_store.last_run(JOB_NAME, "options")
            if last_run is not None and last_run.status == "halted" and last_run.finished_at is not None:
                if (current_time - last_run.finished_at).total_seconds() < cooldown_seconds:
                    reason = "cooling down after a throttled run"
                    run_store.record_skipped(JOB_NAME, "options", reason)
                    return JobResult("skipped", None, reason)
            run_id = run_store.start(JOB_NAME, "options")
            try:
                started = clock()
                attempted = saved = skipped_fresh = 0
                failed: dict[str, str] = {}
                halted_reason: str | None = None
                previous_call: float | None = None
                streak = 0
                not_attempted = 0
                for index, symbol in enumerate(selected):
                    if clock() - started >= budget_seconds:
                        halted_reason = "budget exhausted"
                        not_attempted = len(selected) - index
                        break
                    cutoff = current_time.timestamp() - min_age_hours * 3600
                    if any(ts.timestamp() > cutoff for ts in store.timestamps(symbol)):
                        skipped_fresh += 1  # no provider call, so no pacing either
                        continue
                    if previous_call is not None:
                        delay = min_interval_seconds - (clock() - previous_call)
                        if delay > 0:
                            sleep(delay)
                    if clock() - started >= budget_seconds:
                        halted_reason = "budget exhausted"
                        not_attempted = len(selected) - index
                        break
                    previous_call = clock()
                    attempted += 1
                    try:
                        snapshot = gateway.options_chain(symbol)
                        store.save(snapshot)
                        saved += 1
                        streak = 0
                    except (DataUnavailable, SnapshotError) as exc:
                        failed[symbol] = str(exc)
                        streak += 1
                        if isinstance(exc, DataUnavailable) and streak >= max_consecutive_failures:
                            halted_reason = f"provider appears throttled: {streak} consecutive failures"
                            not_attempted = len(selected) - index - 1
                            break
                report = SnapshotReport(attempted, saved, skipped_fresh, failed, not_attempted, halted_reason)
                status = "halted" if halted_reason and "throttled" in halted_reason.lower() else "completed"
                data = dataclasses.asdict(report)
                data["failed"] = dict(list(failed.items())[:20])
                run_store.finish(run_id, status, data)
                return JobResult(status, report, None)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                run_store.finish(run_id, "failed", {}, error=error)
                return JobResult("failed", None, error)
    except Exception:
        try:
            run_store.record_skipped(JOB_NAME, "options", "another snapshot run is running")
        except Exception:
            pass
        return JobResult("skipped", None, "another snapshot run is running")


def _make_run_store(connect: Callable[[], ContextManager[Any]] | None) -> JobRunStore:
    if connect is None:
        from app.utils.db import get_db_connection
        connect = get_db_connection
    return JobRunStore(connect)
