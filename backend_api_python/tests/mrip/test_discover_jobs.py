from __future__ import annotations

import contextlib
import os
from datetime import date
from pathlib import Path

import pytest

from app.mrip.discover.jobs import run_discover
from app.mrip.discover.types import Kind


def test_migration_kind_check_matches_enum():
    sql = Path(__file__).parents[2].joinpath("migrations/mrip_20261001_discover.sql").read_text()
    check = sql.split("kind VARCHAR(40) NOT NULL CHECK (kind IN (")[1].split("))", 1)[0]
    kinds = {part.strip().strip("'") for part in check.split(",")}
    assert kinds == {kind.value for kind in Kind}


class FakeRunStore:
    def __init__(self): self.started = []; self.finished = []; self.skipped = []
    def start(self, job, universe): self.started.append((job, universe)); return 12
    def finish(self, run_id, status, report, error=None): self.finished.append((run_id, status, report, error))
    def record_skipped(self, job, universe, reason): self.skipped.append((job, universe, reason))


@contextlib.contextmanager
def _lock(acquired):
    yield acquired


def test_lock_held_returns_skipped_and_records_run():
    runs = FakeRunStore()
    result = run_discover(date(2026, 1, 1), service_factory=lambda: None, lock_factory=lambda key: _lock(False), run_store=runs)
    assert result.status == "skipped" and runs.skipped == [("discover", "all", "another discover run is running")]
    assert not runs.started


def test_service_exception_is_captured_as_failed_without_raising():
    runs = FakeRunStore()
    class Broken:
        def run(self, as_of): raise RuntimeError("detector failed")
    result = run_discover(date(2026, 1, 1), service_factory=lambda: Broken(), lock_factory=lambda key: _lock(True), run_store=runs)
    assert result.status == "failed" and "RuntimeError: detector failed" == result.reason
    assert runs.finished[0][1:] == ("failed", {}, "RuntimeError: detector failed")


def test_celery_task_registration_route_beat_and_disabled_switch(monkeypatch):
    from app.celery_app import celery_app
    import app.tasks.mrip_sync as tasks

    task_name = "quantdinger.tasks.mrip_discover"
    assert task_name in celery_app.tasks
    assert celery_app.conf.task_routes[task_name] == {"queue": "maintenance"}
    beat = celery_app.conf.beat_schedule["mrip-discover"]
    assert beat["task"] == task_name and beat["schedule"] >= 1800
    monkeypatch.setenv("ENABLE_MRIP_DISCOVER", "false")
    assert tasks.mrip_discover.run() == {"skipped": True}


def test_cli_initializes_database_before_running_job(monkeypatch, capsys):
    from types import SimpleNamespace
    import app.commands.run_discover as command

    calls = []
    monkeypatch.setattr(command, "init_database", lambda **kwargs: calls.append(("init", kwargs)))
    monkeypatch.setattr(command, "run_discover", lambda as_of: calls.append(("run", as_of)) or SimpleNamespace(status="completed", reason=None, report=None))
    monkeypatch.setattr("sys.argv", ["run_discover", "--as-of", "2026-10-01"])
    assert command.main() == 0
    assert calls == [("init", {"strict_migrations": False}), ("run", date(2026, 10, 1))]
    assert '"status": "completed"' in capsys.readouterr().out
