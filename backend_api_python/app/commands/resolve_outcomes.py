"""Resolve pending predictions whose horizon has been observed in the stored prices.

Usage:
    python -m app.commands.resolve_outcomes [--as-of YYYY-MM-DD] [--limit N] [--dry-run]

Resolution is idempotent: resolved predictions are never resolved again. Predictions with missing
prices are left unresolved and counted by reason. --limit caps the pending predictions examined.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

from app.mrip.outcomes.jobs import run_outcome_resolve
from app.utils.db import init_database


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve automatic MRIP predictions")
    parser.add_argument("--as-of", type=date.fromisoformat, default=None,
                        help="only use prices up to this date (default: all stored prices)")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of pending predictions to examine")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    init_database(strict_migrations=False)
    result = run_outcome_resolve(args.as_of, limit=args.limit, dry_run=args.dry_run)
    print(json.dumps({"status": result.status, "reason": result.reason, "report": result.report},
                     default=str, sort_keys=True))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
