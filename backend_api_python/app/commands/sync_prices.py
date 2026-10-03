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

from app.mrip.data.openbb_adapter import OpenBBAdapter
from app.mrip.data.yahoo_adapter import YahooChartAdapter
from app.mrip.prices.ingest import PriceIngestor
from app.mrip.prices.store import PriceStore
from app.mrip.relationships.graph import RelationshipGraph
from app.utils.db import get_db_connection, init_database


# universe -> (provider name stored with the bars, gateway factory, default pacing seconds)
UNIVERSES = {
    "sp500": ("cboe", lambda: OpenBBAdapter(retries=1), 2.0),
    "omxslc": ("yahoo", lambda: YahooChartAdapter(retries=2), 1.0),
}


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
        default=None,
        help="Maximum seconds to spend on ingestion (default: no limit)",
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

    # Create the graph and fetch universe symbols.
    graph = RelationshipGraph(get_db_connection)
    nodes = graph.list_nodes_in_universe(args.universe.upper())
    symbols = []
    for node in nodes:
        series_info = node.attributes.get("series")
        if series_info and isinstance(series_info, dict):
            symbol = series_info.get("symbol")
            if symbol:
                symbols.append(symbol)

    if not symbols:
        print(f"No symbols found in universe {args.universe}")
        sys.exit(0)

    # Build the gateway and store.
    provider, make_gateway, default_interval = UNIVERSES[args.universe]
    gateway = make_gateway()
    store = PriceStore(get_db_connection)

    # Run the ingestor.
    ingestor = PriceIngestor(
        gateway,
        store,
        provider=provider,
        min_interval_seconds=args.min_interval if args.min_interval is not None else default_interval,
    )
    report = ingestor.sync(symbols, budget_seconds=args.budget_seconds)

    # Print the report.
    print(
        f"SyncReport(attempted={report.attempted}, succeeded={report.succeeded}, "
        f"failed={len(report.failed)}, skipped_fresh={report.skipped_fresh}, "
        f"not_attempted={report.not_attempted}, bars_written={report.bars_written}, "
        f"halted_reason={report.halted_reason!r})"
    )

    sys.exit(0)


if __name__ == "__main__":
    main()
