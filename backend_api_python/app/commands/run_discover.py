from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import date

from app.mrip.discover.jobs import run_discover
from app.utils.db import init_database


def main() -> int:
    parser=argparse.ArgumentParser(description="Run the daily MRIP Discover feed")
    parser.add_argument("--as-of", type=date.fromisoformat)
    args=parser.parse_args()
    init_database(strict_migrations=False)
    result=run_discover(args.as_of)
    print(json.dumps({"status":result.status,"reason":result.reason,"summary":dataclasses.asdict(result.report) if result.report else None},default=str,sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
