"""CalibrationStore: immutable calibration sets and their per-segment entries."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, ContextManager, Mapping, Sequence


class CalibrationKind(str, Enum):
    FORECAST_QUANTILE_SHIFT = "forecast_quantile_shift"
    DECISION_TEMPERATURE = "decision_temperature"


class CalibrationError(Exception):
    """Invalid calibration operation."""


@dataclass(frozen=True, slots=True)
class StoredEntry:
    segment_key: str
    level: int
    n: int
    params: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StoredSet:
    id: int
    kind: CalibrationKind
    fit_as_of: date
    dims: tuple[str, ...]
    min_samples: int
    n_total: int
    policy_version: str
    metrics: Mapping[str, Any]
    created_at: datetime
    entries: tuple[StoredEntry, ...] = field(default_factory=tuple)

    @property
    def version(self) -> str:
        """Lineage identifier (``calibration_version``)."""
        return f"{self.kind.value}#{self.id}"


class CalibrationStore:
    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    def save(
        self,
        kind: CalibrationKind,
        *,
        fit_as_of: date,
        dims: Sequence[str],
        min_samples: int,
        n_total: int,
        policy_version: str,
        entries: Sequence[StoredEntry],
        metrics: Mapping[str, Any] | None = None,
    ) -> StoredSet:
        if not entries:
            raise CalibrationError("refusing to store an empty calibration set")
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO mrip_calibration_sets (kind, fit_as_of, dims, min_samples, n_total, policy_version, metrics) "
                    "VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s::jsonb) "
                    "RETURNING id, kind, fit_as_of, dims, min_samples, n_total, policy_version, metrics, created_at",
                    (kind.value, fit_as_of, json.dumps(list(dims)), min_samples, n_total, policy_version,
                     json.dumps(dict(metrics or {}), sort_keys=True)),
                )
                head = cur.fetchone()
                for e in entries:
                    cur.execute(
                        "INSERT INTO mrip_calibration_entries (set_id, segment_key, level, n, params) "
                        "VALUES (%s, %s, %s, %s, %s::jsonb)",
                        (head["id"], e.segment_key, e.level, e.n, json.dumps(dict(e.params), sort_keys=True)),
                    )
            finally:
                cur.close()
            conn.commit()
        return self._set(head, tuple(entries))

    def latest(self, kind: CalibrationKind) -> StoredSet | None:
        """The newest set of a kind, or None."""
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT id, kind, fit_as_of, dims, min_samples, n_total, policy_version, metrics, created_at "
                    "FROM mrip_calibration_sets WHERE kind = %s ORDER BY id DESC LIMIT 1",
                    (kind.value,),
                )
                head = cur.fetchone()
                if head is None:
                    return None
                cur.execute(
                    "SELECT segment_key, level, n, params FROM mrip_calibration_entries WHERE set_id = %s ORDER BY level, segment_key",
                    (head["id"],),
                )
                rows = cur.fetchall()
            finally:
                cur.close()
        return self._set(head, tuple(StoredEntry(r["segment_key"], r["level"], r["n"], dict(r["params"])) for r in rows))

    @staticmethod
    def _set(head: Mapping[str, Any], entries: tuple[StoredEntry, ...]) -> StoredSet:
        return StoredSet(
            id=int(head["id"]), kind=CalibrationKind(head["kind"]), fit_as_of=head["fit_as_of"],
            dims=tuple(head["dims"]), min_samples=head["min_samples"], n_total=head["n_total"],
            policy_version=head["policy_version"], metrics=dict(head["metrics"] or {}),
            created_at=head["created_at"], entries=entries,
        )
