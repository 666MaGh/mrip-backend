"""Live COT engine against the CFTC feed. Opt in with MRIP_LIVE_OPENBB=1."""
import os
from datetime import date

import pytest

from app.mrip.cot.engine import DEFAULT_MARKETS, CotEngine
from app.mrip.data.openbb_adapter import OpenBBAdapter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_LIVE_OPENBB") != "1", reason="set MRIP_LIVE_OPENBB=1 to run live"),
]


def test_every_registered_market_resolves_and_is_point_in_time():
    engine = CotEngine(OpenBBAdapter())
    print()
    for m in DEFAULT_MARKETS:
        s = engine.snapshot(m)
        p = lambda v: "  n/a" if v is None else f"{v:5.2f}"  # noqa: E731
        print(f"{m.name:16} report {s.report_date} (avail {s.available_date}) {s.group:14} net {s.net_position:>10,.0f}"
              f"  pct1y {p(s.percentile_1y)} pct3y {p(s.percentile_3y)} pct5y {p(s.percentile_5y)} z {p(s.zscore)}"
              f"  crowd {p(s.crowding_score)} {s.crowding_side} div {p(s.price_position_divergence)} {s.warnings}")
        assert s.available_date <= date.today() and s.net_position == s.net_position
    past = engine.snapshot("Gold", as_of=date(2022, 6, 1))
    assert past.available_date <= date(2022, 6, 1) and past.report_date < date(2022, 6, 1)
