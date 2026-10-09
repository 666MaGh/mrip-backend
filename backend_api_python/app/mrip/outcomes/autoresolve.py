"""Automatic outcome resolution (work 023).

Finds pending predictions whose horizon end is observed in the stored prices and resolves them with the
existing ``OutcomeService`` (deterministic, observed prices only). Idempotent: resolved predictions are no
longer pending and outcomes are never rewritten. Missing prices, or missing benchmark prices for
benchmark-judged types, leave a prediction unresolved and are counted by reason. Nothing is guessed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from app.mrip.data.gateway import DataUnavailable
from app.mrip.outcomes.service import OutcomeService
from app.mrip.outcomes.store import PredictionStore
from app.mrip.prices.select import select_series


class StoredPriceGateway:
    """``price_history`` over stored bars, using the same per-symbol provider rule as the logger."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def price_history(self, symbol: str, start: date | None = None, end: date | None = None, interval: str = "1d") -> Any:
        if interval != "1d":
            raise ValueError(f"only 1d bars are supported, got {interval!r}")
        found = select_series(self._store, symbol, end)
        if found is None:
            raise DataUnavailable(f"no price data found for {symbol}")
        return found[1]


@dataclass(slots=True)
class ResolveRun:
    as_of: date | None
    dry_run: bool
    examined: int = 0
    resolved: Counter = field(default_factory=Counter)
    would_resolve: Counter = field(default_factory=Counter)
    still_pending: int = 0
    unavailable: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat() if self.as_of else None, "dry_run": self.dry_run,
            "examined": self.examined, "resolved": dict(self.resolved), "would_resolve": dict(self.would_resolve),
            "still_pending": self.still_pending, "unavailable": dict(self.unavailable),
        }


class OutcomeAutoResolver:
    def __init__(self, store: PredictionStore, gateway: Any) -> None:
        self._service = OutcomeService(store, gateway)

    def run(self, *, as_of: date | None = None, limit: int | None = None, dry_run: bool = False) -> ResolveRun:
        report = self._service.resolve_pending(as_of=as_of, limit=limit, dry_run=dry_run)
        run = ResolveRun(as_of=as_of, dry_run=dry_run, examined=report.examined, still_pending=len(report.still_pending))
        run.resolved.update(report.resolved_by_type)
        if dry_run:
            run.would_resolve.update({"resolvable": len(report.would_resolve)})
        for _, reason in report.unavailable:
            run.unavailable[reason] += 1
        return run
