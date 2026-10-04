"""Synchronize price history for a trading universe.

Usage:
    python -m app.commands.sync_prices sp500|omxslc [--budget-seconds 1500] [--min-interval N]

Fetches daily OHLCV data incrementally and stores it in the local database. sp500 uses
CBOE (via OpenBB, 2 s pacing); omxslc uses Yahoo's chart endpoint (private research only,
1 s pacing). Tracks sync state and failure streaks per symbol. Paced to respect
provider rate limits and can be interrupted and resumed.
"""
from __future__ import annotations

import argparse
import sys

from app.mrip.prices.jobs import UNIVERSES, run_price_sync
from app.utils.db import init_database


def main() -> None:
    """Entry point for syncing prices."""
    parser = argparse.ArgumentParser(description="Synchronize price history for a trading universe")
    parser.add_argument(
        "universe",
        choices=sorted(UNIVERSES),
        help="Universe to sync",
    )
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=1500.0,
        help="Maximum seconds to spend on ingestion (default: 1500)",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=None,
        help="Minimum seconds between provider calls (default: per universe)",
    )
    args = parser.parse_args()

    # Initialize the database and migrations.
    init_database(strict_migrations=False)

    # Run the price sync job.
    result = run_price_sync(
        args.universe,
        budget_seconds=args.budget_seconds,
        min_interval=args.min_interval,
        load_if_empty=True,
    )

    # Print the result.
    if result.report is not None:
        print(
            f"SyncReport(attempted={result.report.attempted}, succeeded={result.report.succeeded}, "
            f"failed={len(result.report.failed)}, skipped_fresh={result.report.skipped_fresh}, "
            f"not_attempted={result.report.not_attempted}, bars_written={result.report.bars_written}, "
            f"halted_reason={result.report.halted_reason!r})"
        )
    elif result.reason:
        print(f"{result.status}: {result.reason}")
    else:
        print(f"{result.status}")

    sys.exit(0)


if __name__ == "__main__":
    main()
