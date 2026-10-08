from __future__ import annotations

import argparse
import dataclasses
import json

from app.mrip.relationships.graph import RelationshipGraph
from app.mrip.relationships.seeds.loader import load_builtin_seeds
from app.utils.db import get_db_connection, init_database


def main() -> int:
    parser=argparse.ArgumentParser(description="Load the built-in MRIP relationship seeds (idempotent)")
    parser.parse_args()
    init_database(strict_migrations=False)
    result=load_builtin_seeds(RelationshipGraph(get_db_connection))
    print(json.dumps({"seed_relationships":dataclasses.asdict(result)},sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
