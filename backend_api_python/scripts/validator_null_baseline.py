"""Null baseline for the relationship validator (read-only; writes nothing).

Runs the production validation pipeline (RelationshipValidator.assess_symbols: stored
closes via StoredDataGateway, SPY market control, sector peer control and the same-sector
null gate under validation-v1-uncalibrated, expected_sign=+1, as_of = today) on:
  (a) random ordered pairs of stored-price universe symbols, excluding any pair that already
      has an edge in mrip_rel_edges (either direction),
  (b) the sec-edgar-10k candidate edges with expected_sign +1,
  (c) random same-sector ordered pairs (sector from the ``sector`` node attribute),
  (d) random cross-sector ordered pairs (both sectors known, different).
Null distributions are computed in memory (no cache table is written).

Prints the share VALIDATED per group. For (c) the share is the false-positive rate of the
v1 rules on same-sector noise; the gate targets <= 5% by construction at the p95 level.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.mrip.data.gateway import DataUnavailable  # noqa: E402
from app.mrip.evidence.store import EvidenceStore  # noqa: E402
from app.mrip.prices.gateway import StoredDataGateway  # noqa: E402
from app.mrip.prices.store import PriceStore  # noqa: E402
from app.mrip.relationships.graph import RelationshipGraph  # noqa: E402
from app.mrip.relationships.types import RelationType  # noqa: E402
from app.mrip.stats.service import RelationshipValidator  # noqa: E402
from app.mrip.stats.validate import ValidationPolicy, Verdict  # noqa: E402
from app.utils.db import get_db_connection  # noqa: E402

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
        value = json.loads(value)
    return dict(value or {})


def _normalize_reason(reason: str) -> str:
    text = re.sub(r"\(partial_r [-+]?\d+(\.\d+)?", "(partial_r X", reason)
    return re.sub(r"[-+]?\d+(\.\d+)?", "N", text)


def _symbol_of(attrs: Any) -> str | None:
    series = _as_dict(attrs).get("series")
    sym = _as_dict(series).get("symbol") if series is not None else None
    return str(sym) if sym else None


def main() -> int:
    graph = RelationshipGraph(get_db_connection)
    gateway = StoredDataGateway(PriceStore(get_db_connection), provider=PROVIDER)
    validator = RelationshipValidator(graph, EvidenceStore(get_db_connection), gateway, policy=ValidationPolicy())
    sector_by_symbol = {s: sec for s, sec in validator.priced_sectors().items()}
    universe = sorted(sector_by_symbol)

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, attributes FROM mrip_rel_nodes")
        sym_by_node = {r["id"]: _symbol_of(r["attributes"]) for r in _rows(cur)}
        cur.execute("SELECT src_id, dst_id FROM mrip_rel_edges WHERE retired_version IS NULL")
        all_edges = _rows(cur)
        cur.execute(
            "SELECT e.relation_type, e.attributes, s.attributes AS src_attrs, d.attributes AS dst_attrs "
            "FROM mrip_rel_edges e "
            "JOIN mrip_rel_nodes s ON s.id = e.src_id JOIN mrip_rel_nodes d ON d.id = e.dst_id "
            "WHERE e.source = %s AND e.retired_version IS NULL",
            (SOURCE,),
        )
        sec_rows = _rows(cur)
        cur.close()

    existing: set[tuple[str, str]] = set()
    for e in all_edges:
        a, b = sym_by_node.get(e["src_id"]), sym_by_node.get(e["dst_id"])
        if a and b:
            existing.add((a, b))
            existing.add((b, a))

    sec_edges: list[tuple[str, str, RelationType]] = []
    for r in sec_rows:
        if _as_dict(r["attributes"]).get("expected_sign") != EXPECTED_SIGN:
            continue
        a, b = _symbol_of(r["src_attrs"]), _symbol_of(r["dst_attrs"])
        if a and b:
            sec_edges.append((a, b, RelationType(r["relation_type"])))
    rel_counts = Counter(rel for _, _, rel in sec_edges)
    null_rel = rel_counts.most_common(1)[0][0] if rel_counts else RelationType.SUPPLIES
    print(f"[baseline] universe with stored prices={len(universe)} sector_known={sum(1 for s in universe if sector_by_symbol[s])}; "
          f"sec-edgar-10k +1 edges={len(sec_edges)}; relation for random pairs={null_rel.value}")

    all_pairs = [(a, b) for a in universe for b in universe if a != b and (a, b) not in existing]
    same = [(a, b) for a, b in all_pairs if sector_by_symbol[a] and sector_by_symbol[a] == sector_by_symbol[b]]
    cross = [(a, b) for a, b in all_pairs
             if sector_by_symbol[a] and sector_by_symbol[b] and sector_by_symbol[a] != sector_by_symbol[b]]
    rng = random.Random(SEED)
    random_pairs = rng.sample(all_pairs, min(N_RANDOM, len(all_pairs)))
    same_pairs = rng.sample(same, min(N_RANDOM, len(same)))
    cross_pairs = rng.sample(cross, min(N_RANDOM, len(cross)))
    print(f"[baseline] pools: random={len(all_pairs)} same-sector={len(same)} cross-sector={len(cross)}")

    def run(a: str, b: str, rel: RelationType) -> tuple[str, float | None, str]:
        try:
            result = validator.assess_symbols(a, b, relation_type=rel, expected_sign=EXPECTED_SIGN)
        except DataUnavailable:
            return "data unavailable", None, "data unavailable"
        r = result.metrics.get("partial_r")
        reason = "validated" if result.verdict is Verdict.VALIDATED else (
            _normalize_reason(result.reasons[0]) if result.reasons else "none")
        return result.verdict.value, (abs(r) if r is not None else None), reason

    groups: dict[str, list[tuple[str, float | None, str]]] = {}
    groups["(a) random pairs"] = [run(a, b, null_rel) for a, b in random_pairs]
    groups["(b) sec-edgar-10k edges"] = [run(a, b, rel) for a, b, rel in sec_edges]
    groups["(c) same-sector random"] = [run(a, b, null_rel) for a, b in same_pairs]
    groups["(d) cross-sector random"] = [run(a, b, null_rel) for a, b in cross_pairs]

    print()
    print("group | n | validated | rejected | inconclusive | share validated | median abs partial r")
    for name, rows in groups.items():
        n = len(rows)
        v = sum(1 for x in rows if x[0] == "validated")
        j = sum(1 for x in rows if x[0] == "rejected")
        i = n - v - j
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
