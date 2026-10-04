"""Scheduled options snapshot job tests."""
from __future__ import annotations

import contextlib
import dataclasses
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import pytest

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import Latency, OptionContract, OptionsChainSnapshot, Provenance
from app.mrip.options.jobs import DEFAULT_OPTION_SYMBOLS, JOB_NAME, is_snapshot_window, run_options_snapshots
from app.mrip.options.snapshots import SnapshotError
from app.mrip.prices.jobs import JobResult, JobRun

T0 = datetime(2026, 10, 5, 21, 0, tzinfo=timezone.utc)


def chain(symbol: str, ts: datetime = T0) -> OptionsChainSnapshot:
    return OptionsChainSnapshot(
        symbol, 100.0, None, ts, None,
        (OptionContract(f"{symbol}C100", date(2026, 10, 16), 100.0, "call", open_interest=10),),
        Provenance("cboe", "openbb", "cboe.options.chains", ts, Latency.DELAYED, "test"),
    )


class FakeStore:
    def __init__(self, timestamps: dict[str, list[datetime]] | None = None, save_error: Exception | None = None) -> None:
        self.times = timestamps or {}
        self.saved: list[OptionsChainSnapshot] = []
        self.save_error = save_error

    def timestamps(self, symbol: str) -> list[datetime]:
        return self.times.get(symbol, [])

    def save(self, snapshot: OptionsChainSnapshot) -> int:
        if self.save_error:
            raise self.save_error
        self.saved.append(snapshot)
        self.times.setdefault(snapshot.underlying, []).append(snapshot.snapshot_timestamp)
        return len(self.saved)


class FakeGateway:
    def __init__(self, outcomes: list[Any] | None = None) -> None:
        self.outcomes = outcomes or []
        self.calls: list[str] = []

    def options_chain(self, symbol: str) -> OptionsChainSnapshot:
        self.calls.append(symbol)
        outcome = self.outcomes.pop(0) if self.outcomes else chain(symbol)
        if isinstance(outcome, Exception):
            raise outcome
        return dataclasses.replace(outcome, underlying=symbol)


class FakeRuns:
    def __init__(self, last: JobRun | None = None) -> None:
        self.runs: list[JobRun] = []
        self.last = last
        self.next_id = 1

    def last_run(self, job: str, universe: str) -> JobRun | None:
        return self.runs[-1] if self.runs else self.last

    def start(self, job: str, universe: str) -> int:
        run_id = self.next_id
        self.next_id += 1
        self.runs.append(JobRun(run_id, job, universe, T0, None, "running", {}, None))
        return run_id

    def finish(self, run_id: int, status: str, report: dict[str, Any], error: str | None = None) -> None:
        old = self.runs[-1]
        self.runs[-1] = dataclasses.replace(old, status=status, finished_at=T0, report=report, error=error)

    def record_skipped(self, job: str, universe: str, reason: str) -> None:
        self.runs.append(JobRun(self.next_id, job, universe, T0, T0, "skipped", {"reason": reason}, None))
        self.next_id += 1


@contextlib.contextmanager
def lock(acquired: bool = True) -> Iterator[bool]:
    yield acquired


def run(**kwargs: Any) -> JobResult:
    kwargs.setdefault("require_window", False)
    kwargs.setdefault("now", T0)
    kwargs.setdefault("run_store", FakeRuns())
    kwargs.setdefault("lock_factory", lambda _key: lock())
    kwargs.setdefault("store", FakeStore())
    kwargs.setdefault("gateway", FakeGateway())
    kwargs.setdefault("sleep", lambda _s: None)  # never sleep for real in tests
    return run_options_snapshots(**kwargs)


def test_window_boundaries_weekends_naive_and_dst() -> None:
    ny = ZoneInfo("America/New_York")
    assert not is_snapshot_window(datetime(2026, 10, 5, 16, 14, tzinfo=ny))
    assert is_snapshot_window(datetime(2026, 10, 5, 16, 15, tzinfo=ny))
    assert is_snapshot_window(datetime(2026, 10, 5, 21, 0, tzinfo=ny))
    assert not is_snapshot_window(datetime(2026, 10, 5, 21, 1, tzinfo=ny))
    assert not is_snapshot_window(datetime(2026, 10, 3, 18, 0, tzinfo=ny))
    assert is_snapshot_window(datetime(2026, 3, 9, 16, 15, tzinfo=ny))
    with pytest.raises(ValueError):
        is_snapshot_window(datetime(2026, 10, 5, 16, 15))


def test_outside_window_and_disabled_window() -> None:
    runs = FakeRuns()
    skipped = run_options_snapshots(now=datetime(2026, 10, 5, 12, tzinfo=timezone.utc), run_store=runs)
    assert skipped.status == "skipped" and runs.runs[-1].report["reason"] == "outside the US post-close window"
    assert run(now=datetime(2026, 10, 5, 12, tzinfo=timezone.utc)).status == "completed"


def test_lock_and_cooldown_skip() -> None:
    runs = FakeRuns()
    result = run(run_store=runs, lock_factory=lambda _key: lock(False))
    assert result.status == "skipped"
    prior = JobRun(7, JOB_NAME, "options", T0 - timedelta(hours=1), T0 - timedelta(minutes=5), "halted", {}, None)
    cooled = run(run_store=FakeRuns(prior))
    assert cooled.status == "skipped" and "cooling down" in (cooled.reason or "")


def test_freshness_pacing_and_budget() -> None:
    store = FakeStore({"A": [T0 - timedelta(hours=1)], "B": [T0 - timedelta(hours=30)]})
    gateway = FakeGateway()
    stamps = iter([0.0] * 20)
    sleeps: list[float] = []
    result = run(symbols=["A", "B"], store=store, gateway=gateway, clock=lambda: next(stamps), sleep=sleeps.append)
    assert result.report and (result.report.skipped_fresh, result.report.saved) == (1, 1)
    assert sleeps == []  # only one real call happened: the fresh symbol is skipped without pacing
    stamps2 = iter([0.0] * 20)
    paced: list[float] = []
    run(symbols=['A', 'B'], store=FakeStore(), gateway=FakeGateway(), clock=lambda: next(stamps2), sleep=paced.append)
    assert paced == [3.0]  # two real calls at the same instant: the second waits the full interval
    ticks = iter([0.0, 0.0, 2.0, 2.0])
    budget = run(symbols=["X", "Y"], store=FakeStore(), gateway=FakeGateway(), clock=lambda: next(ticks), budget_seconds=1)
    assert budget.status == "completed" and budget.report and budget.report.not_attempted == 2


def test_failures_circuit_breaker_and_reset() -> None:
    results = run(
        symbols=["A", "B", "C"], gateway=FakeGateway([DataUnavailable("no"), DataUnavailable("no"), chain("C")]),
        store=FakeStore(), max_consecutive_failures=2,
    )
    assert results.status == "halted" and results.report and results.report.saved == 0
    breaker = run(
        symbols=["A", "B", "C"], gateway=FakeGateway([DataUnavailable("x")] * 3), store=FakeStore(), max_consecutive_failures=3,
    )
    assert breaker.status == "halted" and breaker.report and breaker.report.not_attempted == 0
    assert "3 consecutive failures" in (breaker.report.halted_reason or "")
    reset = run(
        symbols=["A", "B", "C", "D"],
        gateway=FakeGateway([DataUnavailable("x"), chain("B"), DataUnavailable("x"), chain("D")]),
        store=FakeStore(), max_consecutive_failures=2,
    )
    assert reset.status == "completed" and reset.report and (reset.report.saved, len(reset.report.failed)) == (2, 2)


def test_snapshot_error_counts_and_unexpected_exception_returns_failed() -> None:
    bad_save = run(symbols=["A"], store=FakeStore(save_error=SnapshotError("bad")), gateway=FakeGateway())
    assert bad_save.report and bad_save.report.failed == {"A": "bad"}
    broken = run(symbols=["A"], store=object(), gateway=FakeGateway())
    assert broken.status == "failed" and broken.report is None


def test_env_symbol_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.mrip.options.jobs import _configured_symbols

    monkeypatch.setenv("MRIP_OPTIONS_SYMBOLS", " spy, ,qqq ")
    assert _configured_symbols() == ("SPY", "QQQ")
    monkeypatch.setenv("MRIP_OPTIONS_SYMBOLS", " , ")
    assert _configured_symbols() == DEFAULT_OPTION_SYMBOLS


def test_celery_task_registration_route_schedule_and_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.celery_app import celery_app
    from app.tasks.mrip_sync import mrip_options_snapshots

    assert "mrip-options-snapshots" in celery_app.conf.beat_schedule
    assert celery_app.conf.task_routes["quantdinger.tasks.mrip_options_snapshots"]["queue"] == "maintenance"
    monkeypatch.setenv("ENABLE_MRIP_OPTIONS_SNAPSHOTS", "false")
    assert mrip_options_snapshots() == {"skipped": True}


DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


@DB
def test_real_stores_save_and_dedupe_snapshots() -> None:
    from app.mrip.options.jobs import run_options_snapshots
    from app.utils.db import get_db_connection

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute((MIGRATIONS / "mrip_20261001_options_snapshots.sql").read_text(encoding="utf-8"))
        cur.execute((MIGRATIONS / "mrip_20261001_jobs.sql").read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_options_snapshots, mrip_job_runs RESTART IDENTITY")
        conn.commit()
        cur.close()

    class FixedGateway:
        def options_chain(self, symbol: str) -> OptionsChainSnapshot:
            return chain(symbol)

    first = run_options_snapshots(["A", "B", "C"], connect=get_db_connection, gateway=FixedGateway(), now=T0, require_window=False, min_interval_seconds=0)
    second = run_options_snapshots(["A", "B", "C"], connect=get_db_connection, gateway=FixedGateway(), now=T0, require_window=False, min_interval_seconds=0)
    assert first.report and first.report.saved == 3
    assert second.report and second.report.skipped_fresh == 3
