"""Shared stored-price provider selection (one provider per symbol, one common provider per pair).

Every collaborator is a fake; no network, no database.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from app.mrip.data.models import Latency, PriceBar, PriceSeries, Provenance
from app.mrip.discover.detectors import DiscoverPolicy
from app.mrip.outcomes import autolog
from app.mrip.prices.select import PRICE_PROVIDERS, select_pair, select_series
from app.mrip.related.service import RelatedService
from app.mrip.relationships.types import Edge, EdgeStatus, Node, NodeType, Path, RelationType
from app.mrip.research.card import ResearchCardService
from app.mrip.security_regime.service import load_close_prices

NOW = datetime(2026, 10, 9, 19, 0, tzinfo=timezone.utc)
STALE = date(2026, 10, 1)  # cboe stops here in the stored data
FRESH = date(2026, 10, 9)  # yahoo is current
N_BARS = 300  # enough history for the relationship fit gate (250 + 5 recent returns)


class Store:
    """Fake price store keyed by (provider, symbol); ``series`` cuts bars at ``end`` like the real store."""

    def __init__(self, data: dict[tuple[str, str], tuple[PriceBar, ...]]) -> None:
        self.data = data
        self.calls: list[tuple[str, str, date | None]] = []

    def series(self, provider: str, symbol: str, start=None, end=None) -> PriceSeries | None:
        self.calls.append((provider, symbol, end))
        bars = self.data.get((provider, symbol), ())
        if end is not None:
            bars = tuple(b for b in bars if b.ts <= end)
        if not bars:
            return None
        prov = Provenance(provider=provider, gateway="fake", endpoint="fake.bars", fetched_at=NOW, latency=Latency.UNKNOWN)
        return PriceSeries(symbol=symbol, interval="1d", bars=bars, provenance=prov)


def bars(end: date, seed: int, n: int = N_BARS) -> tuple[PriceBar, ...]:
    rng = np.random.default_rng(seed)
    days = [ts.date() for ts in pd.bdate_range(end=end, periods=n)]
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    return tuple(PriceBar(ts=d, open=float(c), high=float(c), low=float(c), close=float(c), volume=1e6)
                 for d, c in zip(days, closes))


def _node(node_id: int, symbol: str) -> Node:
    return Node(node_id, NodeType.COMPANY, symbol, symbol, {"series": {"symbol": symbol}})


def test_stale_cboe_loses_to_fresh_yahoo():
    store = Store({("cboe", "NVDA"): bars(STALE, 1), ("yahoo", "NVDA"): bars(FRESH, 2)})
    provider, series = select_series(store, "NVDA", FRESH)
    assert provider == "yahoo"
    assert series.bars[-1].ts == FRESH


def test_tie_on_latest_bar_goes_to_yahoo():
    store = Store({("cboe", "NVDA"): bars(FRESH, 1), ("yahoo", "NVDA"): bars(FRESH, 2)})
    assert select_series(store, "NVDA", FRESH)[0] == "yahoo"
    assert PRICE_PROVIDERS[0] == "yahoo"


def test_symbol_only_in_cboe_uses_cboe():
    store = Store({("cboe", "^VIX"): bars(STALE, 1)})
    assert select_series(store, "^VIX", FRESH)[0] == "cboe"


def test_nothing_stored_returns_none():
    assert select_series(Store({}), "NVDA", FRESH) is None


def test_autolog_uses_the_shared_helper():
    assert autolog.select_series is select_series


def test_pair_uses_one_provider_for_both_legs_even_when_cboe_has_both():
    store = Store({
        ("cboe", "NVDA"): bars(STALE, 1), ("cboe", "SMH"): bars(STALE, 2),
        ("yahoo", "NVDA"): bars(FRESH, 3), ("yahoo", "SMH"): bars(FRESH, 4),
    })
    pair = select_pair(store, "NVDA", "SMH", FRESH, min_overlap=100)
    assert pair is not None and pair.provider == "yahoo"
    assert pair.left.provenance.provider == pair.right.provenance.provider == "yahoo"


def test_pair_with_no_common_provider_is_none_though_each_leg_has_data():
    store = Store({("cboe", "NVDA"): bars(FRESH, 1), ("yahoo", "SMH"): bars(FRESH, 2)})
    assert select_series(store, "NVDA", FRESH) is not None and select_series(store, "SMH", FRESH) is not None
    assert select_pair(store, "NVDA", "SMH", FRESH, min_overlap=100) is None


def test_pair_below_min_overlap_is_none():
    store = Store({("cboe", "NVDA"): bars(FRESH, 1, n=20), ("cboe", "SMH"): bars(FRESH, 2, n=20)})
    assert select_pair(store, "NVDA", "SMH", FRESH, min_overlap=50) is None
    assert select_pair(store, "NVDA", "SMH", FRESH, min_overlap=10) is not None


def test_related_divergence_mixed_providers_is_unavailable_with_reason():
    store = Store({("cboe", "NVDA"): bars(FRESH, 1), ("yahoo", "SMH"): bars(FRESH, 2)})
    service = RelatedService(graph=None, evidence_store=None, price_store=store)
    divergence, beta = service._divergence("NVDA", "SMH", FRESH)
    assert (divergence["status"], divergence["reason"], beta) == ("unavailable", "olika kurskällor", None)


def test_related_divergence_uses_fresh_yahoo_over_stale_cboe_for_both_legs():
    store = Store({
        ("cboe", "NVDA"): bars(STALE, 1), ("cboe", "SMH"): bars(STALE, 2),
        ("yahoo", "NVDA"): bars(FRESH, 3), ("yahoo", "SMH"): bars(FRESH, 4),
    })
    service = RelatedService(graph=None, evidence_store=None, price_store=store)
    divergence, beta = service._divergence("NVDA", "SMH", FRESH)
    assert divergence["status"] == "available"
    assert divergence["price_provider"] == "yahoo" and beta is not None


def test_security_regime_loader_prefers_fresh_yahoo():
    store = Store({("cboe", "NVDA"): bars(STALE, 1), ("yahoo", "NVDA"): bars(FRESH, 2)})
    closes = load_close_prices(store, "NVDA", FRESH)
    assert closes is not None and closes.index[-1].date() == FRESH


def test_research_card_series_uses_shared_rule_and_pair_reports_mixed_providers():
    store = Store({("cboe", "NVDA"): bars(FRESH, 1), ("yahoo", "SMH"): bars(FRESH, 2)})
    card = object.__new__(ResearchCardService)  # only the lookups under test; no graph or engines needed
    card._prices = store
    card._policy = DiscoverPolicy()
    assert card._series("NVDA", FRESH)[0] == "cboe"
    nvda, smh = _node(1, "NVDA"), _node(2, "SMH")
    edge = Edge(id=7, src_id=1, dst_id=2, relation_type=RelationType.CORRELATED_WITH, status=EdgeStatus.VALIDATED,
                source="seed", created_version=1, attributes={})
    outcome = card._divergence(Path(nodes=(nvda, smh), edges=(edge,)), FRESH)
    assert outcome.body["status"] == "unavailable"
    assert "olika kurskällor" in outcome.body["reason"]
