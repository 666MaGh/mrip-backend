"""Collect point-in-time options chain snapshots."""
from __future__ import annotations

import argparse
import sys

from app.mrip.options.jobs import run_options_snapshots
from app.utils.db import init_database


def main() -> None:
    """Run the scheduled options snapshot job from the command line."""
    parser = argparse.ArgumentParser(description="Collect point-in-time options chains")
    parser.add_argument("symbols", nargs="*", help="Symbols to snapshot; defaults to configured symbols")
    parser.add_argument("--budget-seconds", type=float, default=900.0)
    parser.add_argument("--ignore-window", action="store_true", help="Run outside the normal post-close window")
    args = parser.parse_args()
    init_database(strict_migrations=False)
    result = run_options_snapshots(
        args.symbols or None,
        budget_seconds=args.budget_seconds,
        require_window=not args.ignore_window,
    )
    print(result)
    sys.exit(0)


if __name__ == "__main__":
    main()
