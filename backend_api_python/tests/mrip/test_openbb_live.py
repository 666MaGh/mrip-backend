"""Live OpenBB smoke tests (network). Opt in with MRIP_LIVE_OPENBB=1."""
import os
from datetime import date, timedelta

import pytest

from app.mrip.data.models import Latency
from app.mrip.data.openbb_adapter import OpenBBAdapter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_LIVE_OPENBB") != "1", reason="set MRIP_LIVE_OPENBB=1 to run live"),
]

RECENT = date.today() - timedelta(days=21)


@pytest.fixture(scope="module")
def adapter() -> OpenBBAdapter:
    return OpenBBAdapter()


def test_live_options_chain_has_greeks_and_delayed_provenance(adapter):
    snap = adapter.options_chain("SPY")
    assert snap.underlying_price and snap.underlying_price > 0
    assert snap.oi_effective_date is None
    assert snap.provenance.latency is Latency.DELAYED
    cov = snap.coverage
    assert cov.contracts > 1000
    assert cov.with_greeks > 0 and cov.with_open_interest > 0
    assert {c.option_type for c in snap.contracts} == {"call", "put"}


def test_live_vix_and_equity_history(adapter):
    vix = adapter.vix_history(start=RECENT)
    assert vix.bars and vix.bars[-1].close and vix.bars[-1].close > 0
    spy = adapter.price_history("SPY", start=RECENT)
    assert spy.bars and spy.bars[-1].volume


def test_live_cot_copper(adapter):
    series = adapter.cot("085692", start=date.today() - timedelta(days=120))
    rec = series.records[-1]
    assert rec.open_interest and rec.long_positions.get("commercial") is not None
