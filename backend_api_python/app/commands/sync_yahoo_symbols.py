"""Synchronize daily price history from Yahoo for an explicit list of symbols.

Usage:
    python -m app.commands.sync_yahoo_symbols NVDA AVGO SPY [--budget-seconds 600] [--min-interval 1.0]

Bypasses the universe job and stores bars under provider "yahoo" (private research only),
so that relationship-seed series such as SMH or SPY can be priced without a universe.
Uses the same incremental PriceIngestor as sync_prices.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from app.mrip.prices.ingest import PriceIngestor, SyncReport

PROVIDER = "yahoo"


def sync_symbols(
    symbols: Sequence[str],
    *,
    gateway: Any,
    store: Any,
    budget_seconds: float,
    min_interval: float,
) -> SyncReport:
    """Ingest the given symbols through the shared ingestor under the Yahoo provider name."""
    ingestor = PriceIngestor(gateway, store, provider=PROVIDER, min_interval_seconds=min_interval)
    return ingestor.sync(list(dict.fromkeys(symbols)), budget_seconds=budget_seconds)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Synchronize Yahoo price history for explicit symbols")
    parser.add_argument("symbols", nargs="+", help="Ticker symbols, e.g. NVDA SMH SPY")
    parser.add_argument("--budget-seconds", type=float, default=600.0, help="Maximum seconds to spend (default: 600)")
    parser.add_argument("--min-interval", type=float, default=1.0, help="Seconds between Yahoo calls (default: 1.0)")
    args = parser.parse_args()

    from app.mrip.data.yahoo_adapter import YahooChartAdapter
    from app.mrip.prices.store import PriceStore
    from app.utils.db import get_db_connection, init_database

    init_database(strict_migrations=False)
    report = sync_symbols(
        args.symbols,
        gateway=YahooChartAdapter(retries=2),
        store=PriceStore(get_db_connection),
        budget_seconds=args.budget_seconds,
        min_interval=args.min_interval,
    )
    print(
        f"SyncReport(attempted={report.attempted}, succeeded={report.succeeded}, "
        f"failed={report.failed}, skipped_fresh={report.skipped_fresh}, "
        f"not_attempted={report.not_attempted}, bars_written={report.bars_written}, "
        f"halted_reason={report.halted_reason!r})"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
