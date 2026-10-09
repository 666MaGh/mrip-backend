"""Reliability read-model over resolved predictions (work 023).

Built only from resolved outcomes (labels from ``app.mrip.outcomes.labels``) and never from a
prediction's own claim of confidence.

- FORECAST_SCENARIO per horizon: observed share of outcomes at or below each predicted quantile
  P10/P25/P50/P75/P90 against the nominal level (coverage). A calibrated forecast has observed = nominal.
- DISCOVER_ITEM per horizon: attention hit rate (|move| > logged trailing median) against 0.5, the
  naive baseline by construction; direction hit rate for items that carry a direction against the
  naive "always up" baseline (share of outcomes with positive excess return).
- RELATED_SIGNAL per horizon: direction hit rate (sign of excess return vs benchmark) against the
  same "always up" baseline.

Minimum-n rule: any metric with n < MIN_N reports status "otillräckligt underlag" and no value.
Backfilled rows are counted separately (``backfilled_n``) and are included unless excluded by the caller.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.mrip.outcomes.labels import CalibrationRow, forecast_calibration_rows, reliability_table
from app.mrip.outcomes.types import Outcome, Prediction, PredictionType

RELIABILITY_VERSION = "reliability-v1"
MIN_N = 30
UNDERSIZED = "otillräckligt underlag"
QUANTILE_LABELS = {0.1: "P10", 0.25: "P25", 0.5: "P50", 0.75: "P75", 0.9: "P90"}
HORIZON_ORDER = ("W1", "M1", "M3", "M6", "M12")
ATTENTION_BASELINE = 0.5  # the logged threshold is a trailing median, so |move| > median happens half the time

Pair = tuple[Prediction, Outcome]


def insufficient(n: int, min_n: int = MIN_N) -> str:
    return f"n={n} < {min_n}"


def _horizon_key(kind: str) -> tuple[int, str]:
    return (HORIZON_ORDER.index(kind) if kind in HORIZON_ORDER else len(HORIZON_ORDER), kind)


def _is_backfill(prediction: Prediction) -> bool:
    return bool(prediction.payload.get("backfill"))


def forecast_rows(pairs: Sequence[Pair], *, min_n: int = MIN_N) -> list[dict[str, Any]]:
    by_horizon: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_horizon.setdefault(pair[0].horizon_kind.value, []).append(pair)
    rows = []
    for horizon in sorted(by_horizon, key=_horizon_key):
        group = by_horizon[horizon]
        labels: list[CalibrationRow] = forecast_calibration_rows(group)
        n = len(labels)
        row: dict[str, Any] = {
            "kind": "FORECAST_SCENARIO", "horizon": horizon, "n": n,
            "backfilled_n": sum(1 for p, _ in group if _is_backfill(p)),
            "status": "available" if n >= min_n else UNDERSIZED,
            "reason": None if n >= min_n else insufficient(n, min_n),
            "coverage": None,
        }
        if n >= min_n:
            table = reliability_table(labels)
            row["coverage"] = {
                QUANTILE_LABELS[q]: {"nominal": q, "observed": table[q].observed, "n": table[q].n,
                                     "gap": table[q].observed - q}
                for q in QUANTILE_LABELS if q in table
            }
        rows.append(row)
    return rows


def _metric(hits: list[bool], baseline: float | None, min_n: int) -> dict[str, Any]:
    n = len(hits)
    if n < min_n:
        return {"n": n, "status": UNDERSIZED, "reason": insufficient(n, min_n), "hit_rate": None, "naive_baseline": None}
    return {
        "n": n, "status": "available", "reason": None,
        "hit_rate": sum(hits) / n, "naive_baseline": baseline,
    }


def alert_rows(pairs: Sequence[Pair], *, kind: str, attention: bool, min_n: int = MIN_N) -> list[dict[str, Any]]:
    by_horizon: dict[str, list[Pair]] = {}
    for pair in pairs:
        by_horizon.setdefault(pair[0].horizon_kind.value, []).append(pair)
    rows = []
    for horizon in sorted(by_horizon, key=_horizon_key):
        group = by_horizon[horizon]
        excess = [o.measures["excess_return"] for _, o in group if "excess_return" in o.measures]
        always_up = (sum(1 for x in excess if x > 0) / len(excess)) if excess else None
        metrics: dict[str, Any] = {}
        if attention:
            att = [bool(o.measures["attention"]["hit"]) for _, o in group if "attention" in o.measures]
            metrics["attention"] = _metric(att, ATTENTION_BASELINE, min_n)
        directional = [bool(o.measures["direction"]["hit"]) for _, o in group if "direction" in o.measures]
        metrics["direction"] = _metric(directional, always_up, min_n)
        rows.append({
            "kind": kind, "horizon": horizon, "n": len(group),
            "backfilled_n": sum(1 for p, _ in group if _is_backfill(p)),
            "metrics": metrics,
        })
    return rows


def build_reliability(store: Any, *, min_n: int = MIN_N) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rows += forecast_rows(store.resolved_pairs(PredictionType.FORECAST), min_n=min_n)
    rows += alert_rows(store.resolved_pairs(PredictionType.DISCOVER_ITEM), kind="DISCOVER_ITEM", attention=True, min_n=min_n)
    rows += alert_rows(store.resolved_pairs(PredictionType.RELATED_SIGNAL), kind="RELATED_SIGNAL", attention=False, min_n=min_n)
    return {"version": RELIABILITY_VERSION, "min_n": min_n, "rows": rows}


def model_health(store: Any, run_store: Any, *, job_names: Mapping[str, str]) -> dict[str, Any]:
    """Reliability table, logged/resolved/unresolved counts and the last run of each automatic job."""
    last: dict[str, Any] = {}
    for label, job in job_names.items():
        run = run_store.last_run(job, "all")
        last[label] = None if run is None else {
            "job": run.job, "status": run.status, "started_at": run.started_at.isoformat(),
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "report": run.report, "error": run.error,
        }
    return {"reliability": build_reliability(store), "counts": store.counts_by_type(), "last_job_runs": last}
