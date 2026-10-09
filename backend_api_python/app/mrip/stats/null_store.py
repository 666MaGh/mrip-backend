"""Database cache for same-sector / random-pair nulls (table ``mrip_validation_null``).

The key is (sector, as_of month, policy_version, n, seed); a hit is reused, a miss is
computed once by the caller and stored. SQL uses ``%s`` placeholders only.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, ContextManager

from app.mrip.stats.null_distribution import NullDistribution


class ValidationNullStore:
    def __init__(self, connect: Callable[[], ContextManager[Any]]) -> None:
        self._connect = connect

    def get(self, sector: str, as_of: date, policy_version: str, n: int, seed: int) -> NullDistribution | None:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT sector, as_of, policy_version, n, pairs_used, seed, p05, p90, p95, p99, mean, std "
                    "FROM mrip_validation_null WHERE sector = %s AND as_of = %s AND policy_version = %s "
                    "AND n = %s AND seed = %s",
                    (sector, as_of, policy_version, n, seed),
                )
                row = cur.fetchone()
            finally:
                cur.close()
        if row is None:
            return None
        return _from_row(row)

    def put(self, dist: NullDistribution) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO mrip_validation_null "
                    "(sector, as_of, policy_version, n, pairs_used, seed, p05, p90, p95, p99, mean, std) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (sector, as_of, policy_version, n, seed) DO NOTHING",
                    (
                        dist.sector, dist.as_of, dist.policy_version, dist.n, dist.pairs_used, dist.seed,
                        dist.p05, dist.p90, dist.p95, dist.p99, dist.mean, dist.std,
                    ),
                )
            finally:
                cur.close()
            conn.commit()


def _from_row(row: Any) -> NullDistribution:
    if isinstance(row, dict):
        values = row
    else:
        keys = ("sector", "as_of", "policy_version", "n", "pairs_used", "seed", "p05", "p90", "p95", "p99", "mean", "std")
        values = dict(zip(keys, row, strict=True))
    return NullDistribution(
        sector=str(values["sector"]),
        as_of=values["as_of"],
        policy_version=str(values["policy_version"]),
        n=int(values["n"]),
        pairs_used=int(values["pairs_used"]),
        seed=int(values["seed"]),
        p05=float(values["p05"]),
        p90=float(values["p90"]),
        p95=float(values["p95"]),
        p99=float(values["p99"]),
        mean=float(values["mean"]),
        std=float(values["std"]),
    )
