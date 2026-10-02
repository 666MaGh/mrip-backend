"""COT engine: point-in-time snapshots, positioning assessment and evidence (fake gateway, no network)."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from app.mrip.cot.engine import (
    CotEngine, CotMarket, CotSnapshot, PositioningStance, assess_positioning, record_cot_evidence,
)
from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import CotRecord, CotSeries, Latency, PriceBar, PriceSeries, Provenance
from app.mrip.evidence.types import RelationshipRef, SourceType, Stance
from app.mrip.relationships.types import NodeKey, NodeType, RelationType

PROV = Provenance("cftc", "openbb", "cftc.cot", datetime(2026, 10, 1, tzinfo=timezone.utc), Latency.UNKNOWN)
MARKET = CotMarket("999999", "Test Metal", "TMX")
N = 330  # weekly reports, ~6.3 years
FIRST = date(2020, 1, 7)  # a Tuesday


def tuesday(i):
    return FIRST + timedelta(weeks=i)


def cot_series(nets):
    records = tuple(
        CotRecord(tuesday(i), "Test Metal", 100_000.0, {"managed_money": 50_000.0 + n / 2}, {"managed_money": 50_000.0 - n / 2})
        for i, n in enumerate(nets)
    )
    return CotSeries(MARKET.code, records, PROV)


def price_series(values):
    days = pd.bdate_range("2019-12-02", periods=len(values))
    bars = tuple(PriceBar(d.date(), None, None, None, float(v), None) for d, v in zip(days, values))
    return PriceSeries("TMX", "1d", bars, PROV)


class Gateway:
    def __init__(self, cot, prices=None):
        self._cot, self._prices, self.cot_calls = cot, prices, []

    def cot(self, market, start=None, end=None):
        self.cot_calls.append((market, start, end))
        return self._cot

    def price_history(self, symbol, start=None, end=None, interval="1d"):
        if self._prices is None:
            raise DataUnavailable(symbol)
        return self._prices


def rising_nets(n=N):
    rng = np.random.default_rng(1)
    return list(np.linspace(-20_000, 40_000, n) + rng.normal(0, 500, n))


def test_snapshot_reports_positioning_features_for_the_latest_published_report():
    gw = Gateway(cot_series(rising_nets()), price_series(np.linspace(100, 150, 1700)))
    snap = CotEngine(gw, [MARKET]).snapshot("999999", as_of=date(2026, 5, 15))
    assert snap.group == "managed_money" and snap.market is MARKET
    assert snap.report_date == tuesday(N - 1) and snap.available_date == snap.report_date + timedelta(days=3)
    assert snap.available_date <= snap.as_of
    assert snap.percentile_3y > 0.95 and snap.percentile_1y > 0.95 and snap.percentile_5y > 0.95   # at the highs
    assert snap.crowded and snap.crowding_side == "LONG" and snap.crowding_score > 0.9
    assert snap.net_position == pytest.approx(rising_nets()[-1], abs=1e-6) and snap.open_interest == 100_000.0
    assert snap.net_pct_oi == pytest.approx(snap.net_position / 100_000.0)
    assert snap.price_position_divergence is not None and snap.warnings == ()
    assert snap.provenance["cot"]["code"] == "999999" and snap.provenance["price"]["symbol"] == "TMX"
    market, start, end = gw.cot_calls[0]
    assert market == "999999" and end == date(2026, 5, 15) and (date(2026, 5, 15) - start).days > 365 * 5


def test_a_report_is_invisible_until_its_friday_publication():
    gw = Gateway(cot_series(rising_nets()))
    engine = CotEngine(gw, [MARKET])
    last_tuesday = tuesday(N - 1)
    for offset, expected in ((0, N - 2), (1, N - 2), (2, N - 2), (3, N - 1)):  # Tue, Wed, Thu, Fri
        snap = engine.snapshot(MARKET, as_of=last_tuesday + timedelta(days=offset))
        assert snap.report_date == tuesday(expected), offset
    with pytest.raises(DataUnavailable, match="no COT report published"):
        engine.snapshot(MARKET, as_of=date(2019, 1, 1))


def test_short_history_and_missing_price_proxy_are_reported_not_hidden():
    short = Gateway(cot_series(rising_nets(80)))
    snap = CotEngine(short, [MARKET]).snapshot(MARKET, as_of=date(2022, 1, 1))
    assert snap.percentile_3y is None and snap.crowding_score is None and not snap.crowded
    assert any("less than 3 years" in w for w in snap.warnings) and any("price proxy" in w for w in snap.warnings)
    assert snap.price_position_divergence is None
    with pytest.raises(KeyError):
        CotEngine(short, [MARKET]).market("nope")
    assert CotEngine(short, [MARKET]).market("test metal") is MARKET


def snapshot_with(percentile):
    return CotSnapshot(MARKET, "managed_money", date(2026, 3, 15), date(2026, 3, 10), date(2026, 3, 13), 1.0, None, None, None,
                       None, percentile, None, None, None, None, False, None)


@pytest.mark.parametrize(
    "pct,direction,expected",
    [
        (0.90, 1, PositioningStance.CONFIRMING), (0.60, 1, PositioningStance.CONFIRMING), (0.55, 1, PositioningStance.NEUTRAL),
        (0.40, 1, PositioningStance.CONTRADICTING), (0.10, 1, PositioningStance.CONTRADICTING),
        (0.10, -1, PositioningStance.CONFIRMING), (0.90, -1, PositioningStance.CONTRADICTING), (0.50, -1, PositioningStance.NEUTRAL),
        (0.90, None, PositioningStance.UNKNOWN), (None, 1, PositioningStance.UNKNOWN), (0.9, 0, PositioningStance.UNKNOWN),
    ],
)
def test_positioning_is_assessed_against_the_expected_direction(pct, direction, expected):
    assert assess_positioning(snapshot_with(pct), direction) is expected


class RecordingStore:
    def __init__(self):
        self.calls = []

    def add(self, relationship, stance, source_type, uri, available_at, **kw):
        self.calls.append((relationship, stance, source_type, uri, available_at, kw))
        return "stored"


REL = RelationshipRef(NodeKey(NodeType.COMMODITY, "copper"), NodeKey(NodeType.SECURITY, "miner"), RelationType.SUPPLIES)


def test_cot_evidence_maps_stance_provenance_and_availability_and_skips_unknown():
    store = RecordingStore()
    snap = CotEngine(Gateway(cot_series(rising_nets())), [MARKET]).snapshot(MARKET, as_of=date(2026, 5, 15))
    assert record_cot_evidence(store, REL, snap, +1) == "stored"
    _, stance, source_type, uri, available_at, kw = store.calls[0]
    assert (stance, source_type) is not None and stance is Stance.SUPPORT and source_type is SourceType.COT
    assert uri == f"cftc://cot/999999/{snap.report_date.isoformat()}"
    assert available_at == datetime.combine(snap.available_date, datetime.min.time(), tzinfo=timezone.utc).replace(hour=20, minute=30)
    assert kw["publisher"] == "CFTC" and kw["assessed_by"] == "cot:cot-v0-uncalibrated" and '"expected_direction": 1' in kw["excerpt"]
    record_cot_evidence(store, REL, snap, -1)
    assert store.calls[1][1] is Stance.CONTRADICT  # crowded long contradicts an expected net-short market
    assert record_cot_evidence(store, REL, snapshot_with(None), +1) is None and len(store.calls) == 2
