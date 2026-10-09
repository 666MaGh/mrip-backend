"""mrip_validation_null cache: hit/miss and idempotent insert (PostgreSQL, opt-in like the other DB tests)."""
import os
from datetime import date
from pathlib import Path

import pytest

from app.mrip.stats.null_distribution import NullDistribution

DB = pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
MIGRATION = Path(__file__).parents[2] / "migrations" / "mrip_20261009_validation_null.sql"


def _dist(p95: float = 0.1234567890123) -> NullDistribution:
    return NullDistribution(
        sector="Information Technology", as_of=date(2026, 10, 1), policy_version="validation-v1-uncalibrated",
        n=200, pairs_used=187, seed=20261009, p05=-0.07, p90=0.05, p95=p95, p99=0.2, mean=0.001, std=0.04,
    )


@DB
def test_null_cache_miss_then_hit_and_insert_is_idempotent():
    from app.mrip.stats.null_store import ValidationNullStore
    from app.utils.db import get_db_connection

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_validation_null RESTART IDENTITY")
        conn.commit()
        cur.close()
    store = ValidationNullStore(get_db_connection)
    dist = _dist()
    assert store.get(dist.sector, dist.as_of, dist.policy_version, 200, dist.seed) is None  # miss
    store.put(dist)
    store.put(_dist(p95=0.9))  # same key: ignored, the first computation stays
    hit = store.get(dist.sector, dist.as_of, dist.policy_version, 200, dist.seed)
    assert hit == dist and hit.label == "same-sector null"
    assert store.get(dist.sector, dist.as_of, dist.policy_version, 300, dist.seed) is None  # other n is another key
    assert store.get(dist.sector, date(2026, 11, 1), dist.policy_version, 200, dist.seed) is None  # other month
