from datetime import date

import pandas as pd

from app.mrip.discover.detectors import DiscoverPolicy, peer_events
from app.mrip.discover.types import Kind


def test_peer_outlier_and_small_sector_ignored():
    values = {f"S{i}": pd.Series([0.0] * 19 + ([1.0] if i == 0 else [0.0])) for i in range(8)}
    values.update({f"T{i}": pd.Series([1.0] * 20) for i in range(3)})
    sectors = {s: "large" if s.startswith("S") else "small" for s in values}
    events = peer_events(values, sectors, date(2026, 1, 1), DiscoverPolicy(peer_z=1.0))
    assert all(e.kind is Kind.PEER_DIVERGENCE and e.subject.startswith("S") for e in events)
    assert events

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from app.mrip.cot.engine import CotMarket, CotSnapshot
from app.mrip.data.models import Latency, OptionContract, OptionsChainSnapshot, Provenance
from app.mrip.discover.detectors import cot_events, gamma_events, regime_events, relationship_events
from app.mrip.options.analysis import analyze_options
from app.mrip.options.regime import GammaRegime
from app.mrip.regime.engine import MarketRegime
from app.mrip.stats.regimes import Regime

AS_OF = date(2026, 10, 1)
SNAPSHOT_AT = datetime(2026, 10, 1, 19, tzinfo=timezone.utc)


def _analysis():
    contracts = tuple(OptionContract(f"X{i}", date(2026, 10, 16), float(i), "call" if i % 2 else "put", 1000, volume=10, implied_volatility=.2, gamma=.03) for i in range(95, 106))
    snapshot = OptionsChainSnapshot("TEST", 100., datetime(2026, 10, 1, 18, 58, tzinfo=timezone.utc), SNAPSHOT_AT, None, contracts, Provenance("test", "test", "test", SNAPSHOT_AT, Latency.REALTIME))
    return analyze_options(snapshot)


def _edited_analysis(*, regime=GammaRegime.POSITIVE_GAMMA, tilt=.4, flip_status="unavailable", flip_distance=None, call_distance=None, put_distance=None, zero_dte=0., anomalies=()):
    analysis = _analysis()
    gex = replace(analysis.modeled.gex, net_gex=tilt * analysis.modeled.gex.gross_gex, flip=replace(analysis.modeled.gex.flip, status=flip_status, distance_pct=flip_distance), walls=replace(analysis.modeled.gex.walls, call_wall_distance_pct=call_distance, put_wall_distance_pct=put_distance), zero_dte_share=zero_dte)
    modeled = replace(analysis.modeled, regime=regime, gex=gex)
    observed = replace(analysis.observed, anomalies=anomalies)
    return replace(analysis, modeled=modeled, observed=observed)


def test_gamma_thresholds_modes_and_modeled_label():
    policy = DiscoverPolicy(flip_near_pct=.02, wall_near_pct=.01, zero_dte_high=.3, anomaly_cap=4)
    exact = _edited_analysis(flip_status="found", flip_distance=.02, call_distance=.01, put_distance=-.01, zero_dte=.3, anomalies=("a", "b", "c", "d"))
    events = gamma_events(exact, None, AS_OF, policy)
    by_kind = {event.kind: event for event in events}
    assert {Kind.NEAR_GAMMA_FLIP, Kind.WALL_PROXIMITY, Kind.HIGH_ZERO_DTE, Kind.UNUSUAL_OPTIONS_ACTIVITY} <= set(by_kind)
    assert by_kind[Kind.NEAR_GAMMA_FLIP].magnitude == pytest.approx(0)
    assert by_kind[Kind.WALL_PROXIMITY].magnitude == pytest.approx(0)
    assert by_kind[Kind.HIGH_ZERO_DTE].magnitude == pytest.approx(.3)
    for event in events:
        assert 0 <= event.magnitude <= 1
        if event.kind is Kind.UNUSUAL_OPTIONS_ACTIVITY:
            assert not event.modeled
        else:
            assert event.modeled and "MODELED / ESTIMATED" in event.headline and event.details["label"] == "MODELED / ESTIMATED"
    below = gamma_events(_edited_analysis(flip_status="found", flip_distance=.0200001, call_distance=.0100001, put_distance=None, zero_dte=.299999, anomalies=()), None, AS_OF, policy)
    assert not {Kind.NEAR_GAMMA_FLIP, Kind.WALL_PROXIMITY, Kind.HIGH_ZERO_DTE, Kind.UNUSUAL_OPTIONS_ACTIVITY} & {event.kind for event in below}


def test_gamma_regime_change_requires_previous_known_regimes():
    current = _edited_analysis(regime=GammaRegime.POSITIVE_GAMMA, tilt=.8)
    previous = _edited_analysis(regime=GammaRegime.NEGATIVE_GAMMA, tilt=-.2)
    assert not [event for event in gamma_events(current, None, AS_OF) if event.kind is Kind.GAMMA_REGIME_CHANGE]
    changed = [event for event in gamma_events(current, previous, AS_OF) if event.kind is Kind.GAMMA_REGIME_CHANGE]
    assert len(changed) == 1 and changed[0].magnitude == pytest.approx(1)
    unknown = _edited_analysis(regime=GammaRegime.UNKNOWN)
    assert not [event for event in gamma_events(current, unknown, AS_OF) if event.kind is Kind.GAMMA_REGIME_CHANGE]


def _market(regime, ratio=None):
    return MarketRegime(AS_OF, regime, 20., None, None, None, None, ratio, None, None, None, None, None, (), "test")


def test_regime_shift_and_backwardation_transitions():
    events = regime_events(_market(Regime.CRISIS, 1.2), _market(Regime.LOW_VOL, .9), AS_OF)
    assert next(event for event in events if event.kind is Kind.REGIME_SHIFT).magnitude == 1
    assert next(event for event in events if event.kind is Kind.TERM_BACKWARDATION).magnitude == pytest.approx(1)
    assert not [event for event in regime_events(_market(Regime.CRISIS, 1.2), _market(Regime.STRESS, 1.1), AS_OF) if event.kind is Kind.TERM_BACKWARDATION]
    assert not [event for event in regime_events(_market(Regime.CRISIS, .99), _market(Regime.LOW_VOL, None), AS_OF) if event.kind is Kind.TERM_BACKWARDATION]


def test_cot_thresholds_and_magnitude_bounds():
    snapshot = CotSnapshot(
        market=CotMarket("1", "Gold"), group="Managed Money", as_of=AS_OF,
        report_date=AS_OF, available_date=AS_OF, net_position=100., net_change=1.,
        net_pct_oi=.2, open_interest=500., percentile_1y=1., percentile_3y=1.,
        percentile_5y=1., zscore=3., crowding_score=1., crowding_side="long",
        crowded=True, price_position_divergence=2.,
    )
    events = cot_events(snapshot, AS_OF)
    assert {event.kind for event in events} == {Kind.COT_CROWDING, Kind.COT_DIVERGENCE}
    assert [event.magnitude for event in events] == [1., .5]
    assert all(0 <= event.magnitude <= 1 for event in events)
    below = replace(snapshot, crowded=False, price_position_divergence=1.999999)
    assert cot_events(below, AS_OF) == []


def test_relationship_divergence_boundary_faithful_series_and_short_history():
    rng = np.random.default_rng(314)
    src = pd.Series(rng.normal(0, .01, 320), index=pd.date_range("2025-01-01", periods=320))
    noise = rng.normal(0, .0001, 320)
    follower = 1.7 * src + noise
    base_policy = DiscoverPolicy(relationship_fit_days=250, relationship_recent_days=5)
    faithful = relationship_events("A->B", src, follower, AS_OF, base_policy)
    assert not faithful
    divergent = follower.copy()
    divergent.iloc[-5:] += .01
    result = relationship_events("A->B", src, divergent, AS_OF, base_policy)
    divergence = next(event for event in result if event.kind is Kind.RELATIONSHIP_DIVERGENCE)
    assert divergence.details["edge_status"] is None and divergence.details["beta"] == pytest.approx(1.7, abs=.1)
    assert 0 <= divergence.magnitude <= 1
    z = abs(divergence.details["z"])
    assert any(event.kind is Kind.RELATIONSHIP_DIVERGENCE for event in relationship_events("A->B", src, divergent, AS_OF, replace(base_policy, divergence_z=z)))
    assert not [event for event in relationship_events("A->B", src, divergent, AS_OF, replace(base_policy, divergence_z=z + 1e-6)) if event.kind is Kind.RELATIONSHIP_DIVERGENCE]
    assert relationship_events("A->B", src.iloc[:254], follower.iloc[:254], AS_OF, base_policy) == []


def test_delayed_reaction_finds_leading_source_with_no_recent_response():
    rng = np.random.default_rng(9)
    src = pd.Series(rng.normal(0, .01, 360), index=pd.date_range("2025-01-01", periods=360))
    dst = src.shift(2).fillna(0.)
    src.iloc[-5:] += .12
    dst.iloc[-5:] = 0.
    events = relationship_events("LEAD->LAG", src, dst, AS_OF, DiscoverPolicy())
    delayed = [event for event in events if event.kind is Kind.DELAYED_REACTION]
    assert delayed and delayed[0].details["lag"] == 2
    assert 0 <= delayed[0].magnitude <= 1


def test_peer_threshold_is_inclusive_and_just_above_threshold_is_silent():
    values = {f"P{i}": pd.Series([0.] * 19 + ([.1] if i == 0 else [0.])) for i in range(8)}
    sectors = {symbol: "sector" for symbol in values}
    measured = peer_events(values, sectors, AS_OF, DiscoverPolicy(peer_z=0))
    outlier = next(event for event in measured if event.subject == "P0")
    z = abs(outlier.details["z"])
    assert peer_events(values, sectors, AS_OF, DiscoverPolicy(peer_z=z))
    assert not peer_events(values, sectors, AS_OF, DiscoverPolicy(peer_z=z + 1e-6))
    assert all(0 <= event.magnitude <= 1 for event in measured)
