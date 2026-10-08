"""Feed a real MSFT->NVDA CUSTOMER_OF series with in-memory shocks to Discover's relationship detector.

Read-only: loads prices and the graph edge from the DB, perturbs the return
arrays in memory only, and prints the observations emitted for:
  baseline - real returns, no shock
  A        - NVDA (dst) last-day return shifted by -6 sigma, MSFT (src) unchanged
  B        - MSFT (src) T-2 return shifted by +6 sigma, NVDA (dst) flat on T-1 and T
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.mrip.discover.detectors import DiscoverPolicy, relationship_events  # noqa: E402
from app.mrip.prices.store import PriceStore  # noqa: E402
from app.mrip.stats.correlation import lead_lag  # noqa: E402
from app.mrip.relationships.graph import RelationshipGraph  # noqa: E402
from app.mrip.stats.service import prices_to_series  # noqa: E402
from app.mrip.stats.transforms import align, log_returns  # noqa: E402
from app.utils.db import get_db_connection, init_database  # noqa: E402

SRC_SYMBOL, DST_SYMBOL, RELATION = "MSFT", "NVDA", "CUSTOMER_OF"
SHOCK_SIGMA = 6.0


def _symbol(node) -> str | None:
    val = node.attributes.get("series", {})
    return val.get("symbol") if isinstance(val, dict) else None


def _find_validated_edge(graph: RelationshipGraph):
    for edge in graph.list_active_edges(limit=10000):
        if edge.relation_type.value != RELATION or edge.status.value != "validated":
            continue
        src, dst = graph.get_node_by_id(edge.src_id), graph.get_node_by_id(edge.dst_id)
        if src and dst and _symbol(src) == SRC_SYMBOL and _symbol(dst) == DST_SYMBOL:
            return edge
    return None


def _load_returns(price_store: PriceStore, symbol: str, as_of: date) -> tuple[pd.Series, str]:
    # Same provider order as DiscoverService._series: cboe first, then yahoo.
    for provider in ("cboe", "yahoo"):
        series = price_store.series(provider, symbol, end=as_of)
        if series is not None and series.bars:
            return log_returns(prices_to_series(series, as_of)), provider
    raise SystemExit(f"no price bars for {symbol}")


def _show(label: str, observations) -> None:
    print(f"\n== {label}: {len(observations)} observation(s)")
    for obs in observations:
        print(f"  kind={obs.kind.value} magnitude={obs.magnitude:.3f} details={dict(obs.details)}")


def main() -> int:
    init_database(strict_migrations=False)
    connect = get_db_connection
    graph = RelationshipGraph(connect)
    edge = _find_validated_edge(graph)
    if edge is None:
        print(f"no validated {SRC_SYMBOL}->{DST_SYMBOL} {RELATION} edge in mrip_rel_edges")
        return 1

    prices = PriceStore(connect)
    src_raw, src_provider = _load_returns(prices, SRC_SYMBOL, date.today())
    dst_raw, dst_provider = _load_returns(prices, DST_SYMBOL, date.today())
    src, dst = align(src_raw, dst_raw)
    as_of = src.index[-1].date()
    policy = DiscoverPolicy()
    print(f"edge id={edge.id} status={edge.status.value} pairs={len(src)} as_of={as_of} "
          f"providers={src_provider}/{dst_provider} fit_gate={policy.relationship_fit_days}+{policy.relationship_recent_days}")

    subject = f"{SRC_SYMBOL}->{DST_SYMBOL}"
    sigma_src, sigma_dst = float(src.std(ddof=1)), float(dst.std(ddof=1))
    print(f"sigma_src={sigma_src:.5f} sigma_dst={sigma_dst:.5f} shock={SHOCK_SIGMA}sigma")
    lag = lead_lag(src.iloc[:-policy.relationship_recent_days], dst.iloc[:-policy.relationship_recent_days], 5)
    print(f"lead_lag best_lag={lag.best_lag} corrected_p={lag.corrected_p_value:.4g} (DELAYED_REACTION needs best_lag>0 and p<0.05)")

    def detect(s: pd.Series, d: pd.Series):
        return relationship_events(subject, s, d, as_of, policy, edge.status.value)

    _show("baseline", detect(src, dst))

    dst_a = dst.copy()
    dst_a.iloc[-1] += -SHOCK_SIGMA * sigma_dst
    _show("A (NVDA last-day -6 sigma, MSFT unchanged)", detect(src, dst_a))

    src_b, dst_b = src.copy(), dst.copy()
    src_b.iloc[-3] += SHOCK_SIGMA * sigma_src
    dst_b.iloc[-2:] = 0.0
    _show("B (MSFT T-2 +6 sigma, NVDA flat T-1..T)", detect(src_b, dst_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
