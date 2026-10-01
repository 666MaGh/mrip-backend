"""Live options analysis smoke test on a real chain. Opt in with MRIP_LIVE_OPENBB=1."""
import json
import os

import pytest

from app.mrip.data.openbb_adapter import OpenBBAdapter
from app.mrip.options.analysis import analyze_options

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_LIVE_OPENBB") != "1", reason="set MRIP_LIVE_OPENBB=1 to run live"),
]


def test_live_spy_chain_produces_a_coherent_analysis():
    snapshot = OpenBBAdapter().options_chain("SPY")
    result = analyze_options(snapshot)
    gex = result.modeled.gex
    spot = snapshot.underlying_price
    assert gex.contracts_used > 1000 and gex.gross_gex > 0 and -1.0 <= gex.tilt <= 1.0
    assert all(0.5 * spot < k < 1.5 * spot for k in (gex.walls.call_wall, gex.walls.put_wall) if k)
    assert result.modeled.label == "MODELED / ESTIMATED"
    assert result.data_quality.oi_status == "unknown" and result.data_quality.latency.value == "delayed"
    assert result.observed.term_structure and result.observed.put_call.volume is not None
    print(json.dumps({
        "spot": spot, "net_gex": round(gex.net_gex), "gross_gex": round(gex.gross_gex), "tilt": round(gex.tilt, 3),
        "regime": result.modeled.regime.value, "flip": gex.flip, "walls": gex.walls,
        "zero_dte_share": gex.zero_dte_share, "amplification": result.modeled.amplification.level.value,
        "excluded": gex.excluded, "source": gex.gamma_source,
    }, default=str, indent=1))
    json.dumps(result.to_dict())
