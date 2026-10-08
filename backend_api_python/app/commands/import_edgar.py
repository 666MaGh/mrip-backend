"""Import candidate customer/supplier relationships from SEC EDGAR 10-K filings.

Examples:
    python -m app.commands.import_edgar --tickers NVDA,MSFT --dry-run
    python -m app.commands.import_edgar --universe SP500 --limit 20 --year 2026

Requires EDGAR_USER_AGENT (descriptive, with contact). Optional EDGAR_CACHE_DIR.
Output: HYPOTHESIS edges with SUPPORT evidence only. Nothing is validated here.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone

from app.mrip.edgar.client import EdgarClient, cache_dir_from_env, user_agent_from_env
from app.mrip.edgar.types import EdgarError
from app.mrip.edgar.service import EdgarImportService, normalize_ticker
from app.mrip.evidence.store import EvidenceStore
from app.mrip.relationships.graph import RelationshipGraph
from app.mrip.relationships.types import GraphError
from app.utils.db import get_db_connection, init_database

SP500_UNIVERSE = "SP500"


def _parse_tickers(raw: str) -> list[str]:
    tickers = [normalize_ticker(t) for t in raw.split(",") if t.strip()]
    if not tickers:
        raise argparse.ArgumentTypeError("--tickers needs at least one ticker")
    return list(dict.fromkeys(tickers))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import SEC 10-K customer/supplier hypotheses (work 021)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", type=_parse_tickers, help="Comma-separated filer tickers (also the universe)")
    group.add_argument("--universe", choices=[SP500_UNIVERSE], help="Filers are the loaded universe members")
    parser.add_argument("--limit", type=int, default=None, help="With --universe: process only the first N members")
    parser.add_argument("--year", type=int, default=None, help="Only 10-Ks filed in this calendar year")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse, write nothing")
    args = parser.parse_args()

    user_agent = user_agent_from_env()
    if not user_agent:
        print("EDGAR_USER_AGENT is empty. Set a descriptive User-Agent (e.g. 'MRIP research you@example.com').", file=sys.stderr)
        return 2
    if args.limit is not None and args.limit < 1:
        print("--limit must be positive", file=sys.stderr)
        return 2

    try:
        client = EdgarClient(user_agent, cache_dir_from_env())
    except EdgarError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    init_database(strict_migrations=False)
    graph = RelationshipGraph(get_db_connection)
    if args.tickers:
        filers = args.tickers
        universe = args.tickers
    else:
        members = graph.list_nodes_in_universe(SP500_UNIVERSE)
        if not members:
            print("SP500 universe is not loaded; run app.commands.load_universe sp500 first.", file=sys.stderr)
            return 2
        universe = sorted({normalize_ticker(m.key.rsplit(":", 1)[-1]) for m in members})
        filers = universe[: args.limit] if args.limit else universe

    service = EdgarImportService(graph, EvidenceStore(get_db_connection), client, clock=lambda: datetime.now(timezone.utc))
    try:
        summary = service.run(filers=filers, universe=universe, year=args.year, dry_run=args.dry_run)
    except (EdgarError, GraphError) as exc:
        print(f"import failed: {exc}", file=sys.stderr)
        return 1

    data = dataclasses.asdict(summary)
    data["names_unmatched_top"] = summary.top_unmatched()
    data.pop("names_unmatched", None)
    print(json.dumps({"import_edgar": data}, default=str, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
