from datetime import date

import numpy as np
import pandas as pd

from app.mrip.discover.detectors import DiscoverPolicy, relationship_events
from app.mrip.discover.types import Kind

AS_OF = date(2026, 10, 1)
SHOCK_SIGMA = 6.0


def _series(seed: int = 7, days: int = 400) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-06-03", periods=days)
    src = rng.normal(0.0, 0.02, days)
    dst = 0.6 * src + rng.normal(0.0, 0.015, days)
    return pd.Series(src, index=index), pd.Series(dst, index=index)


def _kinds(events) -> set[Kind]:
    return {event.kind for event in events}


def test_baseline_emits_no_relationship_observation():
    src, dst = _series()
    events = relationship_events("MSFT->NVDA", src, dst, AS_OF, DiscoverPolicy())
    assert Kind.RELATIONSHIP_DIVERGENCE not in _kinds(events)
    assert Kind.DELAYED_REACTION not in _kinds(events)


def test_dst_last_day_shock_emits_relationship_divergence():
    src, dst = _series()
    sigma_dst = float(dst.std(ddof=1))
    shocked = dst.copy()
    shocked.iloc[-1] += -SHOCK_SIGMA * sigma_dst
    events = relationship_events("MSFT->NVDA", src, shocked, AS_OF, DiscoverPolicy())
    divergence = [event for event in events if event.kind is Kind.RELATIONSHIP_DIVERGENCE]
    assert divergence and divergence[0].details["z"] < 0
    assert 0 <= divergence[0].magnitude <= 1


def test_src_shock_on_contemporaneous_pair_does_not_emit_delayed_reaction():
    # Finding: DELAYED_REACTION requires a significant positive lead-lag (best_lag > 0).
    # A contemporaneously correlated pair (best_lag == 0) cannot emit it, even with scenario B's shock.
    src, dst = _series()
    shocked_src, shocked_dst = src.copy(), dst.copy()
    shocked_src.iloc[-3] += SHOCK_SIGMA * float(src.std(ddof=1))
    shocked_dst.iloc[-2:] = 0.0
    events = relationship_events("MSFT->NVDA", shocked_src, shocked_dst, AS_OF, DiscoverPolicy())
    assert Kind.DELAYED_REACTION not in _kinds(events)


def test_src_shock_without_dst_response_emits_delayed_reaction_on_lagged_pair():
    # Positive control (same shape as the existing fixture in test_discover_detectors.py):
    # dst follows src with a 2-day lag and is flat across the whole 5-day recent window,
    # so the lead in src is unanswered and the expected-response threshold is exceeded.
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2024-06-03", periods=400)
    src = pd.Series(rng.normal(0.0, 0.02, 400), index=index)
    dst = src.shift(2).fillna(0.0)
    shocked_src, shocked_dst = src.copy(), dst.copy()
    shocked_src.iloc[-3] += SHOCK_SIGMA * float(src.std(ddof=1))
    shocked_dst.iloc[-5:] = 0.0
    events = relationship_events("MSFT->NVDA", shocked_src, shocked_dst, AS_OF, DiscoverPolicy())
    delayed = [event for event in events if event.kind is Kind.DELAYED_REACTION]
    assert delayed and delayed[0].details["lag"] == 2
