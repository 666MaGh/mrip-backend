"""Live Market Regime Engine on real CBOE data. Opt in with MRIP_LIVE_OPENBB=1."""
import os
from datetime import date

import pytest

from app.mrip.data.openbb_adapter import OpenBBAdapter
from app.mrip.regime.engine import MarketRegimeEngine

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_LIVE_OPENBB") != "1", reason="set MRIP_LIVE_OPENBB=1 to run live"),
]


def test_live_regime_now_and_at_known_past_dates():
    engine = MarketRegimeEngine(OpenBBAdapter())
    now = engine.at()
    print(f"\nnow {now.as_of}: {now.regime.value}  VIX {now.vix:.2f} (pct {now.vix_percentile:.2f}, chg {now.vix_change:+.2f})"
          f"  rv {now.realized_vol:.3f}  iv/rv {now.implied_to_realized:.2f}  VIX/VIX3M {now.vix_term_ratio:.2f}"
          f"  trend {now.index_trend.value if now.index_trend else None}  dd {now.index_drawdown:.3f}"
          f"  10y {now.rates_yield:.2f}% {now.rates_direction.value if now.rates_direction else None}"
          f"  USD {now.usd_direction.value if now.usd_direction else None}  unavailable {now.unavailable}")
    assert now.unavailable == () and now.vix > 0
    covid = engine.at(date(2020, 3, 16))
    print(f"2020-03-16: {covid.regime.value} VIX {covid.vix:.1f} trend {covid.index_trend} dd {covid.index_drawdown:.2f}")
    assert covid.regime.value == "CRISIS" and covid.vix > 60
    calm = engine.at(date(2017, 11, 3))
    print(f"2017-11-03: {calm.regime.value} VIX {calm.vix:.1f}")
    assert calm.regime.value == "LOW_VOL"
