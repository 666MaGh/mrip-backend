"""Log today's (or a past as-of date's) automatic predictions.

Usage:
    python -m app.commands.log_predictions [--as-of YYYY-MM-DD] [--limit N] [--dry-run]

Logs FORECAST_SCENARIO, DISCOVER_ITEM and RELATED_SIGNAL predictions at decision time with their
provenance. An --as-of before today is a backfill: only FORECAST_SCENARIO rows are written, marked
backfill=true. --limit caps the number of symbols. --dry-run computes counts and writes nothing.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

from app.mrip.outcomes.jobs import run_prediction_log
from app.utils.db import init_database


def main() -> int:
    parser = argparse.ArgumentParser(description="Log automatic MRIP predictions")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--limit", type=int, default=None, help="maximum number of symbols")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    init_database(strict_migrations=False)
    result = run_prediction_log(args.as_of, symbols_limit=args.limit, dry_run=args.dry_run)
    print(json.dumps({"status": result.status, "reason": result.reason, "report": result.report},
                     default=str, sort_keys=True))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
