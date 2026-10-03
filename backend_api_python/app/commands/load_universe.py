"""Load universe constituents into the relationship graph.

Usage:
    python -m app.commands.load_universe sp500
    python -m app.commands.load_universe omxslc

Fetches the constituent list from the data source and loads it into the
relationship graph (SECURITY nodes), merging with any existing nodes.
"""
from __future__ import annotations

import argparse
import sys

from app.mrip.universe.loader import UniverseLoader
from app.mrip.universe.sources import fetch_sp500, load_omxslc
from app.mrip.relationships.graph import RelationshipGraph
from app.utils.db import get_db_connection, init_database


def main() -> None:
    """Entry point for loading a universe."""
    parser = argparse.ArgumentParser(description="Load trading universe constituents")
    parser.add_argument(
        "universe",
        choices=["sp500", "omxslc"],
        help="Universe to load (sp500 or omxslc)",
    )
    args = parser.parse_args()

    # Initialize the database and migrations.
    init_database(strict_migrations=False)

    # Create the graph.
    graph = RelationshipGraph(get_db_connection)

    # Fetch and load.
    if args.universe == "sp500":
        members = fetch_sp500()
    elif args.universe == "omxslc":
        members = load_omxslc()
    else:
        raise ValueError(f"unknown universe: {args.universe}")

    loader = UniverseLoader(graph)
    report = loader.load(members)

    # Print the report.
    print(f"LoadReport(created={report.created}, updated={report.updated}, "
          f"unchanged={report.unchanged}, dropped={report.dropped})")


if __name__ == "__main__":
    main()
