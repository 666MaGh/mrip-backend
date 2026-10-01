"""PredictionStore: immutable prediction log and one outcome per prediction."""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Callable, ContextManager, Mapping

from app.mrip.outcomes.types import (
    HorizonKind, NewPrediction, Outcome, OutcomeError, OutcomeStatus, Prediction, PredictionType,
)

_P_COLUMNS = (
    "id, prediction_type, subject, benchmark, made_at, entry_price, horizon_kind, expiry_date, "
    "model_version, lineage, payload"
)
_O_COLUMNS = (
    "id, prediction_id, status, resolved_at, window_start, window_end, actual_return, benchmark_return, "
    "realized_volatility, max_adverse_excursion, max_favorable_excursion, measures, data_provenance, "
    "resolver_version, note"
)


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, default=str)


def _prediction(row: Mapping[str, Any]) -> Prediction:
    return Prediction(
        id=int(row["id"]),
        prediction_type=PredictionType(row["prediction_type"]),
        subject=row["subject"],
        benchmark=row["benchmark"],
        made_at=row["made_at"],
        entry_price=row["entry_price"],
        horizon_kind=HorizonKind(row["horizon_kind"]),
        expiry_date=row["expiry_date"],
        model_version=row["model_version"],
        lineage=dict(row["lineage"] or {}),
        payload=dict(row["payload"] or {}),
    )


def _outcome(row: Mapping[str, Any]) -> Outcome:
    return Outcome(
        id=int(row["id"]),
        prediction_id=int(row["prediction_id"]),
        status=OutcomeStatus(row["status"]),
        resolved_at=row["resolved_at"],
        resolver_version=row["resolver_version"],
        window_start=row["window_start"],
        window_end=row["window_end"],
        actual_return=row["actual_return"],
        benchmark_return=row["benchmark_return"],
        realized_volatility=row["realized_volatility"],
        max_adverse_excursion=row["max_adverse_excursion"],
        max_favorable_excursion=row["max_favorable_excursion"],
        measures=dict(row["measures"] or {}),
        data_provenance=dict(row["data_provenance"] or {}),
        note=row["note"],
    )


class PredictionStore:
    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    def log(self, p: NewPrediction) -> Prediction:
        """Log a prediction; logging the same (type, subject, made_at, horizon, model) again returns the first."""
        if p.made_at.tzinfo is None:
            raise OutcomeError("made_at must be timezone-aware")
        if not p.subject.strip() or not p.model_version.strip():
            raise OutcomeError("subject and model_version are required")
        if p.horizon_kind is HorizonKind.EXPIRY and p.expiry_date is None:
            raise OutcomeError("an EXPIRY horizon needs expiry_date")
        if p.horizon_kind in (HorizonKind.EOD, HorizonKind.NEXT_SESSION) and p.entry_price is None:
            raise OutcomeError(f"a {p.horizon_kind.value} horizon needs entry_price (the prediction was made intraday)")
        if p.entry_price is not None and p.entry_price <= 0:
            raise OutcomeError("entry_price must be positive")
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO mrip_predictions "
                    "(prediction_type, subject, benchmark, made_at, entry_price, horizon_kind, expiry_date, "
                    " model_version, lineage, payload) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb) "
                    "ON CONFLICT (prediction_type, subject, made_at, horizon_kind, model_version) DO NOTHING "
                    "RETURNING " + _P_COLUMNS,
                    (
                        p.prediction_type.value, p.subject, p.benchmark, p.made_at, p.entry_price,
                        p.horizon_kind.value, p.expiry_date, p.model_version, _json(p.lineage), _json(p.payload),
                    ),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        "SELECT " + _P_COLUMNS + " FROM mrip_predictions WHERE prediction_type = %s AND subject = %s "
                        "AND made_at = %s AND horizon_kind = %s AND model_version = %s",
                        (p.prediction_type.value, p.subject, p.made_at, p.horizon_kind.value, p.model_version),
                    )
                    row = cur.fetchone()
            finally:
                cur.close()
            conn.commit()
        return _prediction(row)

    def get(self, prediction_id: int) -> Prediction:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT " + _P_COLUMNS + " FROM mrip_predictions WHERE id = %s", (prediction_id,))
                row = cur.fetchone()
            finally:
                cur.close()
        if row is None:
            raise OutcomeError(f"no prediction with id {prediction_id}")
        return _prediction(row)

    def pending(self, *, subject: str | None = None) -> list[Prediction]:
        """Predictions without an outcome, oldest first."""
        sql = (
            "SELECT " + ", ".join("p." + c.strip() for c in _P_COLUMNS.split(",")) + " FROM mrip_predictions p "
            "LEFT JOIN mrip_outcomes o ON o.prediction_id = p.id WHERE o.id IS NULL"
        )
        params: tuple[Any, ...] = ()
        if subject is not None:
            sql += " AND p.subject = %s"
            params = (subject,)
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql + " ORDER BY p.made_at, p.id", params)
                rows = cur.fetchall()
            finally:
                cur.close()
        return [_prediction(r) for r in rows]

    def record_outcome(
        self,
        prediction_id: int,
        *,
        resolver_version: str,
        window_start: date,
        window_end: date,
        actual_return: float,
        benchmark_return: float | None,
        realized_volatility: float | None,
        max_adverse_excursion: float,
        max_favorable_excursion: float,
        measures: Mapping[str, Any],
        data_provenance: Mapping[str, Any],
    ) -> Outcome:
        """Store the resolved outcome; an existing outcome is returned unchanged (outcomes never change)."""
        return self._insert_outcome(
            prediction_id, OutcomeStatus.RESOLVED, resolver_version, None,
            dict(
                window_start=window_start, window_end=window_end, actual_return=actual_return,
                benchmark_return=benchmark_return, realized_volatility=realized_volatility,
                max_adverse_excursion=max_adverse_excursion, max_favorable_excursion=max_favorable_excursion,
            ),
            measures, data_provenance,
        )

    def mark_unresolvable(self, prediction_id: int, *, resolver_version: str, reason: str) -> Outcome:
        if not reason.strip():
            raise OutcomeError("a reason is required")
        return self._insert_outcome(prediction_id, OutcomeStatus.UNRESOLVABLE, resolver_version, reason, {}, {}, {})

    def _insert_outcome(
        self, prediction_id: int, status: OutcomeStatus, resolver_version: str, note: str | None,
        fields: Mapping[str, Any], measures: Mapping[str, Any], provenance: Mapping[str, Any],
    ) -> Outcome:
        columns = ["window_start", "window_end", "actual_return", "benchmark_return", "realized_volatility",
                   "max_adverse_excursion", "max_favorable_excursion"]
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT 1 FROM mrip_predictions WHERE id = %s", (prediction_id,))
                if cur.fetchone() is None:
                    raise OutcomeError(f"no prediction with id {prediction_id}")
                cur.execute(
                    "INSERT INTO mrip_outcomes (prediction_id, status, resolver_version, note, "
                    + ", ".join(columns) + ", measures, data_provenance) VALUES (%s, %s, %s, %s, "
                    + ", ".join("%s" for _ in columns) + ", %s::jsonb, %s::jsonb) "
                    "ON CONFLICT (prediction_id) DO NOTHING RETURNING " + _O_COLUMNS,
                    (prediction_id, status.value, resolver_version, note,
                     *[fields.get(c) for c in columns], _json(measures), _json(provenance)),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute("SELECT " + _O_COLUMNS + " FROM mrip_outcomes WHERE prediction_id = %s", (prediction_id,))
                    row = cur.fetchone()
            finally:
                cur.close()
            conn.commit()
        return _outcome(row)

    def outcome_for(self, prediction_id: int) -> Outcome | None:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT " + _O_COLUMNS + " FROM mrip_outcomes WHERE prediction_id = %s", (prediction_id,))
                row = cur.fetchone()
            finally:
                cur.close()
        return _outcome(row) if row else None

    def resolved_pairs(
        self, prediction_type: PredictionType, *, subject: str | None = None
    ) -> list[tuple[Prediction, Outcome]]:
        """(prediction, outcome) for every RESOLVED outcome of a type, oldest first."""
        sql = (
            "SELECT " + ", ".join("p." + c.strip() + " AS p_" + c.strip() for c in _P_COLUMNS.split(","))
            + ", " + ", ".join("o." + c.strip() + " AS o_" + c.strip() for c in _O_COLUMNS.split(","))
            + " FROM mrip_predictions p JOIN mrip_outcomes o ON o.prediction_id = p.id "
            "WHERE o.status = 'resolved' AND p.prediction_type = %s"
        )
        params: list[Any] = [prediction_type.value]
        if subject is not None:
            sql += " AND p.subject = %s"
            params.append(subject)
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(sql + " ORDER BY p.made_at, p.id", tuple(params))
                rows = cur.fetchall()
            finally:
                cur.close()
        pairs = []
        for r in rows:
            p = {k[2:]: v for k, v in r.items() if k.startswith("p_")}
            o = {k[2:]: v for k, v in r.items() if k.startswith("o_")}
            pairs.append((_prediction(p), _outcome(o)))
        return pairs
