"""Shared stored-price provider selection for every consumer that reads prices.

Rules
- One provider per symbol: the stored series whose most recent bar on or before ``end`` is the most
  recent one (ties: yahoo first). yahoo closes are dividend-adjusted and cboe closes are not, and cboe
  stops earlier, so a single series must never mix them.
- Pairs (relationship legs, related neighbours, divergence) use ONE common provider for both legs. The
  provider that has both legs with enough overlap and the most recent common bar wins. If no provider
  has both legs with enough overlap, there is no pair selection (``None``) and the caller reports the
  pair as unavailable. Nothing is mixed across providers.
- Stored data only. No network.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.mrip.data.models import PriceSeries

PRICE_PROVIDERS: tuple[str, ...] = ("yahoo", "cboe")  # ties and tie-breaks: earlier wins


@dataclass(frozen=True, slots=True)
class PairSelection:
    provider: str
    left: PriceSeries
    right: PriceSeries


def _day(ts: datetime | date) -> date:
    return ts.date() if isinstance(ts, datetime) else ts


def _last_day(series: PriceSeries) -> date:
    return _day(series.bars[-1].ts)


def select_series(store: Any, symbol: str, end: date | None = None) -> tuple[str, PriceSeries] | None:
    """The stored series with the most recent bar on or before ``end``; ``None`` when nothing is stored."""
    best: tuple[str, PriceSeries, date] | None = None
    for provider in PRICE_PROVIDERS:
        series = store.series(provider, symbol, end=end)
        if series is None or not series.bars:
            continue
        last = _last_day(series)
        if best is None or last > best[2]:
            best = (provider, series, last)
    return None if best is None else (best[0], best[1])


def select_pair(store: Any, left: str, right: str, end: date | None = None, *, min_overlap: int = 0) -> PairSelection | None:
    """Both legs from ONE provider: the one with the most recent common bar and at least ``min_overlap``
    common closing days (ties: yahoo first). ``None`` when no provider has both legs with enough overlap."""
    best: tuple[PairSelection, date] | None = None
    for provider in PRICE_PROVIDERS:
        a = store.series(provider, left, end=end)
        b = store.series(provider, right, end=end)
        if a is None or b is None or not a.bars or not b.bars:
            continue
        common = {_day(bar.ts) for bar in a.bars if bar.close is not None} & {
            _day(bar.ts) for bar in b.bars if bar.close is not None
        }
        if len(common) < max(min_overlap, 1):
            continue
        latest = max(common)
        if best is None or latest > best[1]:
            best = (PairSelection(provider, a, b), latest)
    return None if best is None else best[0]
