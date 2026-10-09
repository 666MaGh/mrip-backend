"""Null baseline for the relationship validator (read-only).

Runs the same validation call as RelationshipValidator (stored yahoo closes via
StoredDataGateway, SPY as market control, expected_sign=+1, default ValidationPolicy,
as_of = today) on:
  (a) random ordered pairs of universe tickers, excluding any pair that already has an
      edge in mrip_rel_edges (either direction),
  (b) the sec-edgar-10k candidate edges with expected_sign +1,
  (c) random same-sector ordered pairs (sector from SECURITY node attributes), if available.

Writes nothing to the database. Prints a summary table and the most common reasons per group.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.mrip.data.gateway import DataUnavailable  # noqa: E402
from app.mrip.prices.gateway import StoredDataGateway  # noqa: E402
from app.mrip.prices.store import PriceStore  # noqa: E402
from app.mrip.relationships.types import RelationType  # noqa: E402
from app.mrip.stats.service import prices_to_series  # noqa: E402
from app.mrip.stats.transforms import log_returns  # noqa: E402
from app.mrip.stats.validate import ValidationPolicy, Verdict, validate_relationship  # noqa: E402
from app.utils.db import get_db_connection  # noqa: E402

SYMS_FILE = Path("/tmp/syms.txt")
MARKET = "SPY"
PROVIDER = "yahoo"
SEED = 20261009
N_RANDOM = 300
EXPECTED_SIGN = 1
SOURCE = "sec-edgar-10k"


def _rows(cur) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in cur.fetchall():
        if not isinstance(row, Mapping):
            raise TypeError("expected mapping rows from the database cursor")
        out.append(dict(row))
    return out


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        import json

        value = json.loads(value)
    return dict(value or {})


def _normalize_reason(reason: str) -> str:
    return re.sub(r"[-+]?\d+(\.\d+)?", "N", reason)


def main() -> int:
    symbols = [s.strip() for s in SYMS_FILE.read_text().split(",") if s.strip()]
    universe = [s for s in symbols if s != MARKET]
    policy = ValidationPolicy()
    gateway = StoredDataGateway(PriceStore(get_db_connection), provider=PROVIDER)

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, node_type, node_key, attributes FROM mrip_rel_nodes")
        nodes = {r["id"]: r for r in _rows(cur)}
        cur.execute("SELECT src_id, dst_id, relation_type, source, status, attributes FROM mrip_rel_edges")
        all_edges = _rows(cur)
        cur.execute(
            "SELECT e.src_id, e.dst_id, e.relation_type, e.attributes, "
            "s.attributes AS src_attrs, d.attributes AS dst_attrs "
            "FROM mrip_rel_edges e "
            "JOIN mrip_rel_nodes s ON s.id = e.src_id JOIN mrip_rel_nodes d ON d.id = e.dst_id "
            "WHERE e.source = %s AND e.retired_version IS NULL",
            (SOURCE,),
        )
        sec_rows = _rows(cur)

    def symbol_of(attrs: Any) -> str | None:
        series = _as_dict(attrs).get("series")
        sym = _as_dict(series).get("symbol") if series is not None else None
        return str(sym) if sym else None

    sym_by_node = {nid: symbol_of(n["attributes"]) for nid, n in nodes.items()}
    sector_by_symbol: dict[str, str] = {}
    for n in nodes.values():
        sym = symbol_of(n["attributes"])
        sector = _as_dict(n["attributes"]).get("sector")
        if sym and sector:
            sector_by_symbol[sym] = str(sector)

    existing: set[tuple[str, str]] = set()
    for e in all_edges:
        a, b = sym_by_node.get(e["src_id"]), sym_by_node.get(e["dst_id"])
        if a and b:
            existing.add((a, b))
            existing.add((b, a))

    sec_edges = []
    for r in sec_rows:
        if _as_dict(r["attributes"]).get("expected_sign") != EXPECTED_SIGN:
            continue
        a, b = symbol_of(r["src_attrs"]), symbol_of(r["dst_attrs"])
        if a and b:
            sec_edges.append((a, b, RelationType(r["relation_type"])))
    print(f"[baseline] sec-edgar-10k edges with +1 and symbols: {len(sec_edges)}; "
          f"universe={len(universe)} sector_known={sum(s in sector_by_symbol for s in universe)}")

    rel_counts = Counter(rt for _, _, rt in sec_edges)
    null_rel = rel_counts.most_common(1)[0][0] if rel_counts else None
    print(f"[baseline] relation_type for random pairs (most common among sec edges): "
          f"{null_rel.value if null_rel else None}; sec relation mix: "
          f"{ {k.value: v for k, v in rel_counts.items()} }")

    # Preload returns once (same transform as RelationshipValidator.validate).
    cache: dict[str, pd.Series | None] = {}
    for sym in [*universe, MARKET]:
        try:
            cache[sym] = log_returns(prices_to_series(gateway.price_history(sym), date.today()))
        except DataUnavailable:
            cache[sym] = None
    market = cache[MARKET]

    def run(a: str, b: str, rel: RelationType) -> tuple[str, float | None, str]:
        ra, rb = cache.get(a), cache.get(b)
        if ra is None or rb is None or market is None:
            return "data unavailable", None, "data unavailable"
        result = validate_relationship(
            ra, rb, market=market, expected_sign=EXPECTED_SIGN, relation_type=rel, policy=policy,
        )
        r = result.metrics.get("r")
        reason = "validated" if result.verdict is Verdict.VALIDATED else (
            _normalize_reason(result.reasons[0]) if result.reasons else "none")
        return result.verdict.value, (abs(r) if r is not None else None), reason

    rng = random.Random(SEED)
    all_pairs = [(a, b) for a in universe for b in universe if a != b and (a, b) not in existing]
    random_pairs = rng.sample(all_pairs, min(N_RANDOM, len(all_pairs)))
    same_sector = [(a, b) for a, b in all_pairs
                   if a in sector_by_symbol and sector_by_symbol.get(a) == sector_by_symbol.get(b)]
    sector_pairs = rng.sample(same_sector, min(N_RANDOM, len(same_sector)))

    groups: dict[str, list[tuple[str, float | None, str]]] = {}
    groups["(a) random pairs"] = [run(a, b, null_rel) for a, b in random_pairs]
    groups["(b) sec-edgar-10k edges"] = [run(a, b, rel) for a, b, rel in sec_edges]
    if sector_pairs:
        groups["(c) same-sector random"] = [run(a, b, null_rel) for a, b in sector_pairs]
    else:
        print("[baseline] (c) skipped: no sector info on universe nodes")

    print()
    print("group | n | validated | rejected | inconclusive | share validated | median abs r")
    for name, rows in groups.items():
        n = len(rows)
        v = sum(1 for x in rows if x[0] == "validated")
        j = sum(1 for x in rows if x[0] == "rejected")
        i = sum(1 for x in rows if x[0] not in ("validated", "rejected"))
        rs = [x[1] for x in rows if x[1] is not None]
        med = float(np.median(rs)) if rs else float("nan")
        share = v / n if n else float("nan")
        print(f"{name} | {n} | {v} | {j} | {i} | {share:.1%} | {med:.3f}")

    print()
    for name, rows in groups.items():
        top = Counter(x[2] for x in rows).most_common(3)
        print(f"{name} top reasons: " + "; ".join(f"{k} ({c})" for k, c in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
