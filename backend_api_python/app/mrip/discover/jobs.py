from __future__ import annotations

import contextlib
import dataclasses
import json
from datetime import date
from typing import Any, Callable, ContextManager

from app.mrip.prices.jobs import JobResult, JobRunStore

JOB_NAME = "discover"


def _service(connect: Callable[[], ContextManager[Any]]) -> Any:
    from app.mrip.discover.service import DiscoverService
    from app.mrip.discover.store import DiscoverStore
    from app.mrip.evidence.store import EvidenceStore
    from app.mrip.options.snapshots import OptionsSnapshotStore
    from app.mrip.prices.gateway import StoredDataGateway
    from app.mrip.prices.store import PriceStore
    from app.mrip.relationships.graph import RelationshipGraph
    from app.mrip.regime.engine import MarketRegimeEngine
    from app.mrip.cot.engine import CotEngine
    from app.mrip.data.openbb_adapter import OpenBBAdapter
    prices = PriceStore(connect)
    live = OpenBBAdapter(retries=1)
    return DiscoverService(graph=RelationshipGraph(connect), price_store=prices, snapshot_store=OptionsSnapshotStore(connect), discover_store=DiscoverStore(connect), evidence_store=EvidenceStore(connect), regime_engine=MarketRegimeEngine(StoredDataGateway(prices, live=live)), cot_engine=CotEngine(live))


def run_discover(as_of: date | None = None, *, connect: Callable[[], ContextManager[Any]] | None = None, service_factory: Callable[[], Any] | None = None, cooldown_seconds: float = 0.0, lock_factory: Callable[[str], ContextManager[bool]] | None = None, run_store: JobRunStore | None = None) -> JobResult:
    if connect is None:
        from app.utils.db import get_db_connection
        connect = get_db_connection
    if run_store is None: run_store = JobRunStore(connect)
    if service_factory is None: service_factory = lambda: _service(connect)
    if lock_factory is None:
        @contextlib.contextmanager
        def lock_factory(key: str) -> Any:
            with connect() as conn:
                cur=conn.cursor(); acquired=False
                try:
                    cur.execute("SELECT pg_try_advisory_lock(hashtext(%s)) AS acquired",(key,)); row=cur.fetchone(); acquired=bool(row["acquired"] if isinstance(row,dict) else row[0]); yield acquired
                finally:
                    if acquired: cur.execute("SELECT pg_advisory_unlock(hashtext(%s))",(key,))
                    cur.close()
    try:
        with lock_factory("mrip_discover") as acquired:
            if not acquired:
                run_store.record_skipped(JOB_NAME,"all","another discover run is running")
                return JobResult("skipped",None,"another discover run is running")
            run_id=run_store.start(JOB_NAME,"all")
            try:
                summary=service_factory().run(as_of or date.today())
                report=dataclasses.asdict(summary)
                report["as_of"]=summary.as_of.isoformat()
                run_store.finish(run_id,"completed",report)
                return JobResult("completed",summary,None)
            except Exception as exc:
                error=f"{type(exc).__name__}: {exc}"; run_store.finish(run_id,"failed",{},error=error)
                return JobResult("failed",None,error)
    except Exception as exc:
        return JobResult("failed",None,f"{type(exc).__name__}: {exc}")
