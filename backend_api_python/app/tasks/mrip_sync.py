"""MRIP scheduled price sync tasks managed by Celery Beat (work 014)."""
from __future__ import annotations

import os

from app.celery_app import celery_app


def _enabled(name: str, default: str = "true") -> bool:
    """Check if a feature is enabled via environment variable."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@celery_app.task(name="quantdinger.tasks.mrip_price_sync")
def mrip_price_sync(universe: str) -> dict[str, object]:
    """Sync prices for a single universe with time and resource budgets.

    Args:
        universe: Universe name ("sp500" or "omxslc").

    Returns:
        JSON-serializable dict with status, report, and optional reason.
        Returns {"skipped": True} if ENABLE_MRIP_PRICE_SYNC is false.

    Raises:
        ValueError: If universe is unknown (unhandled, returns task error).
    """
    if not _enabled("ENABLE_MRIP_PRICE_SYNC"):
        return {"skipped": True}

    from app.mrip.prices.jobs import run_price_sync

    budget_seconds = int(os.getenv("MRIP_PRICE_SYNC_BUDGET_SEC", "1500"))
    cooldown_seconds = int(os.getenv("MRIP_PRICE_SYNC_COOLDOWN_SEC", "1800"))

    result = run_price_sync(
        universe,
        budget_seconds=float(budget_seconds),
        cooldown_seconds=float(cooldown_seconds),
    )

    return {
        "status": result.status,
        "report": __serialize_report(result.report),
        "reason": result.reason,
    }


@celery_app.task(name="quantdinger.tasks.mrip_price_sync_all")
def mrip_price_sync_all() -> dict[str, object]:
    """Sync prices for all universes sequentially.

    Runs "omxslc" first (fast), then "sp500" (paced to avoid throttling).
    Returns {"skipped": True} if ENABLE_MRIP_PRICE_SYNC is false.

    Returns:
        Dict mapping universe name to result dict.
    """
    if not _enabled("ENABLE_MRIP_PRICE_SYNC"):
        return {"skipped": True}

    results: dict[str, object] = {}
    for universe in ["omxslc", "sp500"]:
        try:
            result_dict = mrip_price_sync(universe)
            results[universe] = result_dict
        except Exception as exc:
            results[universe] = {
                "status": "failed",
                "report": None,
                "reason": f"{type(exc).__name__}: {str(exc)}",
            }

    return results


def __serialize_report(report: object) -> dict[str, object] | None:
    """Serialize SyncReport to a JSON-safe dict."""
    if report is None:
        return None

    # SyncReport is a dataclass; convert to dict.
    import dataclasses
    if dataclasses.is_dataclass(report):
        return dict(dataclasses.asdict(report))

    return None
