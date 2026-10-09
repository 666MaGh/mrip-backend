"""Scheduled prediction logging and outcome resolution (work 023).

Both jobs use the same pattern as the other MRIP jobs: a Postgres advisory lock per job, a row in
mrip_job_runs (universe "all") and a JSON-safe report. ``dry_run`` computes everything without taking
the lock or writing a run record or any prediction/outcome.
"""
from __future__ import annotations

import contextlib
import dataclasses
from datetime import date
from typing import Any, Callable, ContextManager

from app.mrip.prices.jobs import JobRunStore

LOG_JOB = "mrip-prediction-log"
RESOLVE_JOB = "mrip-outcome-resolve"
UNIVERSE = "all"


@dataclasses.dataclass(frozen=True, slots=True)
class AutologResult:
    status: str  # "completed", "skipped", or "failed"
    report: dict[str, Any] | None
    reason: str | None


def _default_connect() -> Callable[[], ContextManager[Any]]:
    from app.utils.db import get_db_connection

    return get_db_connection


@contextlib.contextmanager
def _default_lock(connect: Callable[[], ContextManager[Any]], key: str) -> Any:
    with connect() as conn:
        cur = conn.cursor()
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


def _log_service(connect: Callable[[], ContextManager[Any]]) -> Any:
    from app.mrip.discover.store import DiscoverStore
    from app.mrip.options.snapshots import OptionsSnapshotStore
    from app.mrip.outcomes.autolog import PredictionLogger
    from app.mrip.outcomes.store import PredictionStore
    from app.mrip.evidence.store import EvidenceStore
    from app.mrip.prices.gateway import StoredDataGateway
    from app.mrip.prices.store import PriceStore
    from app.mrip.regime.engine import MarketRegimeEngine
    from app.mrip.related.service import RelatedService
    from app.mrip.relationships.graph import RelationshipGraph

    prices = PriceStore(connect)
    graph = RelationshipGraph(connect)
    return PredictionLogger(
        PredictionStore(connect), price_store=prices, graph=graph, discover_store=DiscoverStore(connect),
        related_service=RelatedService(graph, EvidenceStore(connect), prices, snapshot_store=OptionsSnapshotStore(connect)),
        regime_engine=MarketRegimeEngine(StoredDataGateway(prices)),
    )


def _resolve_service(connect: Callable[[], ContextManager[Any]]) -> Any:
    from app.mrip.outcomes.autoresolve import OutcomeAutoResolver, StoredPriceGateway
    from app.mrip.outcomes.store import PredictionStore
    from app.mrip.prices.store import PriceStore

    return OutcomeAutoResolver(PredictionStore(connect), StoredPriceGateway(PriceStore(connect)))


def _run(
    job: str, lock_key: str, *, action: Callable[[], Any], dry_run: bool,
    connect: Callable[[], ContextManager[Any]] | None, lock_factory: Callable[[str], ContextManager[bool]] | None,
    run_store: JobRunStore | None,
) -> AutologResult:
    if dry_run:
        try:
            return AutologResult("completed", action().to_dict(), None)
        except Exception as exc:
            return AutologResult("failed", None, f"{type(exc).__name__}: {exc}")
    connect = connect or _default_connect()
    run_store = run_store or JobRunStore(connect)
    lock = lock_factory or (lambda key: _default_lock(connect, key))
    try:
        with lock(lock_key) as acquired:
            if not acquired:
                reason = f"another {job} run is running"
                run_store.record_skipped(job, UNIVERSE, reason)
                return AutologResult("skipped", None, reason)
            run_id = run_store.start(job, UNIVERSE)
            try:
                report = action().to_dict()
                run_store.finish(run_id, "completed", report)
                return AutologResult("completed", report, None)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                run_store.finish(run_id, "failed", {}, error=error)
                return AutologResult("failed", None, error)
    except Exception as exc:
        return AutologResult("failed", None, f"{type(exc).__name__}: {exc}")


def run_prediction_log(
    as_of: date | None = None, *, symbols_limit: int | None = None, dry_run: bool = False,
    connect: Callable[[], ContextManager[Any]] | None = None, service_factory: Callable[[], Any] | None = None,
    lock_factory: Callable[[str], ContextManager[bool]] | None = None, run_store: JobRunStore | None = None,
) -> AutologResult:
    as_of = as_of or date.today()
    conn_factory = connect or _default_connect()
    factory = service_factory or (lambda: _log_service(conn_factory))
    return _run(
        LOG_JOB, "mrip_prediction_log", dry_run=dry_run, connect=connect, lock_factory=lock_factory, run_store=run_store,
        action=lambda: factory().run(as_of, limit=symbols_limit, dry_run=dry_run),
    )


def run_outcome_resolve(
    as_of: date | None = None, *, limit: int | None = None, dry_run: bool = False,
    connect: Callable[[], ContextManager[Any]] | None = None, service_factory: Callable[[], Any] | None = None,
    lock_factory: Callable[[str], ContextManager[bool]] | None = None, run_store: JobRunStore | None = None,
) -> AutologResult:
    conn_factory = connect or _default_connect()
    factory = service_factory or (lambda: _resolve_service(conn_factory))
    return _run(
        RESOLVE_JOB, "mrip_outcome_resolve", dry_run=dry_run, connect=connect, lock_factory=lock_factory, run_store=run_store,
        action=lambda: factory().run(as_of=as_of, limit=limit, dry_run=dry_run),
    )
