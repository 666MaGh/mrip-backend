"""Modeled GEX, walls, gamma flip, regimes and amplification against hand-computed golden chains."""
import json
from datetime import date, datetime, timezone

import numpy as np
import pytest

from app.mrip.data.models import Latency, OptionContract, OptionsChainSnapshot, Provenance
from app.mrip.options.analysis import LookAheadError, MODELED_LABEL, analyze_options, assert_not_after
from app.mrip.options.bs import bs_gamma_array
from app.mrip.options.gex import GexAssumptions, compute_gex, seconds_to_expiry, years_to_expiry
from app.mrip.options.regime import (
    AmplificationLevel, Direction, ExternalContext, GammaRegime, RegimeMethod, assess_amplification,
    classify_gamma_regime,
)

# 2026-10-01 15:00 New York (EDT, UTC-4) = 19:00 UTC.
SNAP_TS = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)
EXP = date(2026, 10, 16)
PROV = Provenance("cboe", "openbb", "cboe.options.chains", SNAP_TS, Latency.DELAYED)


def contract(strike, kind, oi, gamma=None, iv=None, expiry=EXP, multiplier=100, volume=0):
    return OptionContract(
        contract_symbol=f"{kind[0].upper()}{strike}{expiry:%y%m%d}", expiry=expiry, strike=float(strike),
        option_type=kind, open_interest=oi, volume=volume, implied_volatility=iv, gamma=gamma,
        contract_multiplier=multiplier,
    )


def snapshot(contracts, spot=100.0, oi_date=None, ts=SNAP_TS):
    return OptionsChainSnapshot(
        underlying="TEST", underlying_price=spot, underlying_timestamp=datetime(2026, 10, 1, 14, 58),
        snapshot_timestamp=ts, oi_effective_date=oi_date, contracts=tuple(contracts), provenance=PROV,
    )


def golden():
    """S=100 so S^2*0.01 = 100. Contract GEX = sign * gamma * OI * 100 (multiplier) * 100."""
    return snapshot([
        contract(100, "call", 1000, gamma=0.04),  # +400_000
        contract(100, "put", 800, gamma=0.04),    # -320_000
        contract(105, "call", 2000, gamma=0.02),  # +400_000
        contract(95, "put", 1500, gamma=0.02),    # -300_000
    ])


def test_golden_chain_matches_hand_computation():
    g = compute_gex(golden())
    assert g.call_gex == pytest.approx(800_000) and g.put_gex == pytest.approx(-620_000)
    assert g.net_gex == pytest.approx(180_000) and g.gross_gex == pytest.approx(1_420_000)
    assert g.by_strike == {95.0: pytest.approx(-300_000), 100.0: pytest.approx(80_000), 105.0: pytest.approx(400_000)}
    assert g.by_expiry == {EXP: pytest.approx(180_000)}
    assert g.tilt == pytest.approx(180_000 / 1_420_000)
    # Concentration: gross by strike = {100: 720k, 105: 400k, 95: 300k}
    assert [c.strike for c in g.concentrations] == [100.0, 105.0, 95.0]
    assert g.top_strikes_share == pytest.approx(1.0)
    assert g.hhi == pytest.approx((72 / 142) ** 2 + (40 / 142) ** 2 + (30 / 142) ** 2)
    assert g.contracts_total == 4 and g.contracts_used == 4 and g.gamma_source == {"provider": 4, "black_scholes": 0}


def test_walls_use_the_right_side_of_spot_and_break_ties_toward_spot():
    g = compute_gex(golden())
    assert (g.walls.call_wall, g.walls.put_wall) == (100.0, 100.0)  # 100 ties 105 on call GEX: nearer wins
    assert g.walls.call_wall_distance_pct == pytest.approx(0.0)
    far = snapshot([contract(110, "call", 5000, gamma=0.03), contract(105, "call", 100, gamma=0.03),
                    contract(90, "put", 5000, gamma=0.03), contract(95, "put", 100, gamma=0.03)])
    w = compute_gex(far).walls
    assert (w.call_wall, w.put_wall) == (110.0, 90.0)
    assert w.call_wall_distance_pct == pytest.approx(0.10) and w.put_wall_distance_pct == pytest.approx(-0.10)


def test_contract_multiplier_scales_gex():
    base = compute_gex(snapshot([contract(100, "call", 1000, gamma=0.04)]))
    mini = compute_gex(snapshot([contract(100, "call", 1000, gamma=0.04, multiplier=10)]))
    assert mini.net_gex == pytest.approx(base.net_gex / 10)


def test_put_call_sign_assumption_is_explicit_and_reversible():
    default = compute_gex(golden())
    flipped = compute_gex(golden(), GexAssumptions(call_sign=-1, put_sign=1))
    assert flipped.net_gex == pytest.approx(-default.net_gex) and flipped.gross_gex == pytest.approx(default.gross_gex)
    with pytest.raises(ValueError):
        GexAssumptions(call_sign=0)


def test_missing_greeks_fall_back_to_black_scholes_and_unusable_contracts_are_excluded():
    chain = snapshot([
        contract(100, "call", 1000, iv=0.25),              # no provider gamma -> Black-Scholes
        contract(100, "put", 1000, gamma=0.04, iv=0.25),   # provider gamma
        contract(105, "call", 1000),                       # nothing to compute gamma from
        contract(95, "put", None, gamma=0.02),             # no open interest
        contract(90, "put", 0, gamma=0.02),                # zero open interest
    ])
    g = compute_gex(chain)
    assert g.gamma_source == {"provider": 1, "black_scholes": 1}
    assert g.excluded == {"no_open_interest": 2, "no_gamma_source": 1, "expired": 0}
    t = years_to_expiry(EXP, SNAP_TS, 0.25)
    expected_bs = float(bs_gamma_array(100.0, np.array([100.0]), np.array([t]), np.array([0.25]))[0])
    assert g.call_gex == pytest.approx(expected_bs * 1000 * 100 * 100)


def test_expiration_handling_zero_dte_and_expired_contracts():
    today = date(2026, 10, 1)
    assert seconds_to_expiry(today, SNAP_TS) == pytest.approx(3600)  # 16:00 ET is one hour after 15:00 ET
    after_close = datetime(2026, 10, 1, 20, 30, tzinfo=timezone.utc)
    assert seconds_to_expiry(today, after_close) < 0
    assert years_to_expiry(today, after_close, 0.25) == pytest.approx(0.25 * 3600 / (365 * 24 * 3600))  # floored
    with pytest.raises(ValueError):
        seconds_to_expiry(today, datetime(2026, 10, 1, 19, 0))
    chain = snapshot([
        contract(100, "call", 1000, iv=0.2, expiry=today),                 # 0DTE: finite thanks to the floor
        contract(100, "put", 1000, iv=0.2, expiry=date(2026, 9, 30)),      # already expired
        contract(100, "call", 1000, iv=0.2),
    ])
    g = compute_gex(chain)
    assert g.excluded["expired"] == 1 and g.contracts_used == 2
    assert np.isfinite(g.net_gex) and 0 < g.zero_dte_share < 1
    assert g.short_dated_share == pytest.approx(g.zero_dte_share)  # the other expiry is 15 days out


def net_gex_at(chain, spot):
    """Independent recomputation of modeled net GEX at a hypothetical spot (Black-Scholes, default assumptions)."""
    total = 0.0
    for c in chain.contracts:
        t = years_to_expiry(c.expiry, chain.snapshot_timestamp, 0.25)
        gamma = float(bs_gamma_array(spot, np.array([c.strike]), np.array([t]), np.array([c.implied_volatility]))[0])
        sign = 1 if c.option_type == "call" else -1
        total += sign * gamma * c.open_interest * c.contract_multiplier * spot * spot * 0.01
    return total


def flip_chain():
    cs = []
    for k in range(90, 101):
        cs.append(contract(k, "put", 3000, iv=0.25))
    for k in range(100, 111):
        cs.append(contract(k, "call", 3000, iv=0.25))
    return snapshot(cs)


def test_gamma_flip_is_found_and_the_modeled_net_gex_changes_sign_around_it():
    chain = flip_chain()
    g = compute_gex(chain)
    assert g.flip.status == "found" and 90 < g.flip.level < 110
    assert g.flip.distance_pct == pytest.approx((g.flip.level - 100) / 100)
    below, above = net_gex_at(chain, g.flip.level * 0.995), net_gex_at(chain, g.flip.level * 1.005)
    assert below * above < 0
    assert abs(net_gex_at(chain, g.flip.level)) < 0.05 * max(abs(below), abs(above))


def test_gamma_flip_absent_or_unavailable_cases():
    calls_only = compute_gex(snapshot([contract(k, "call", 1000, iv=0.25) for k in range(95, 106)]))
    assert calls_only.flip.status == "none_positive_in_range" and calls_only.flip.level is None
    puts_only = compute_gex(snapshot([contract(k, "put", 1000, iv=0.25) for k in range(95, 106)]))
    assert puts_only.flip.status == "none_negative_in_range"
    provider_only = compute_gex(golden())  # provider gamma but no IV: cannot re-evaluate at other spots
    assert provider_only.flip.status == "unavailable" and provider_only.flip.distance_pct is None


def test_invalid_inputs():
    with pytest.raises(ValueError):
        compute_gex(snapshot([contract(100, "call", 1, gamma=0.1)], spot=None))
    with pytest.raises(ValueError):
        GexAssumptions(flip_grid_points=2)


def test_empty_gamma_gives_no_tilt_and_unknown_regime():
    g = compute_gex(snapshot([contract(100, "call", 0, gamma=0.1)]))
    assert g.gross_gex == 0 and g.tilt is None and g.hhi is None and g.walls.call_wall is None
    assert classify_gamma_regime(g.tilt) is GammaRegime.UNKNOWN


@pytest.mark.parametrize(
    "tilt,expected",
    [(0.5, "POSITIVE_GAMMA"), (0.2, "POSITIVE_GAMMA"), (0.0, "NEUTRAL_GAMMA"), (-0.19, "NEUTRAL_GAMMA"),
     (-0.2, "NEGATIVE_GAMMA"), (-0.59, "NEGATIVE_GAMMA"), (-0.6, "EXTREME_NEGATIVE_GAMMA"), (-1.0, "EXTREME_NEGATIVE_GAMMA")],
)
def test_regime_boundaries(tilt, expected):
    assert classify_gamma_regime(tilt).value == expected


def test_regime_method_validation():
    with pytest.raises(ValueError):
        RegimeMethod(positive_tilt=-0.5)


def big_chain(sign_calls=1.0, put_scale=1.0, expiry=EXP, n=60):
    cs = []
    for i in range(n):
        k = 80 + i
        cs.append(contract(k, "call", int(1000 * sign_calls), gamma=0.03, expiry=expiry))
        cs.append(contract(k, "put", int(1000 * put_scale), gamma=0.03, expiry=expiry))
    return snapshot(cs)


def test_amplification_positive_gamma_is_stabilizing_low():
    g = compute_gex(big_chain(sign_calls=3.0))
    a = assess_amplification(g, classify_gamma_regime(g.tilt))
    assert (a.level, a.direction) == (AmplificationLevel.LOW, Direction.STABILIZING)


def test_amplification_negative_gamma_accumulates_risk_factors():
    g = compute_gex(big_chain(put_scale=5.0))  # puts dominate -> strongly negative tilt
    regime = classify_gamma_regime(g.tilt)
    assert regime is GammaRegime.EXTREME_NEGATIVE_GAMMA
    base = assess_amplification(g, regime)
    # 2 points for extreme negative gamma + 1 for a call wall at spot (strike 100 is in the chain) = HIGH.
    assert base.direction is Direction.AMPLIFYING and base.level is AmplificationLevel.HIGH
    assert any("call wall" in f for f in base.contributing) and len(base.contributing) == 2
    hot = assess_amplification(g, regime, ExternalContext(momentum=0.08, realized_vol=0.55))
    assert hot.level is AmplificationLevel.EXTREME and len(hot.contributing) == 4
    calm = assess_amplification(g, regime, ExternalContext(momentum=0.01, realized_vol=0.15))
    assert calm.level is AmplificationLevel.HIGH and len(calm.contradicting) == 2


def test_amplification_uncertain_on_thin_or_neutral_data():
    thin = compute_gex(golden())  # 4 usable contracts
    a = assess_amplification(thin, classify_gamma_regime(thin.tilt))
    assert (a.level, a.direction) == (AmplificationLevel.UNCERTAIN, Direction.UNCERTAIN)
    assert any("usable contracts" in r for r in a.uncertain_because)
    neutral = compute_gex(big_chain(put_scale=1.0))
    n = assess_amplification(neutral, classify_gamma_regime(neutral.tilt))
    assert n.level is AmplificationLevel.UNCERTAIN and "close to neutral" in n.uncertain_because[0]


def test_analysis_separates_observed_from_modeled_and_reports_what_is_unknown():
    result = analyze_options(big_chain(put_scale=2.0))
    assert result.modeled.label == MODELED_LABEL == "MODELED / ESTIMATED"
    observed_fields = set(result.observed.__dataclass_fields__)
    assert not observed_fields & {"gex", "flip", "walls", "regime", "amplification"}
    dq = result.data_quality
    assert dq.oi_effective_date is None and dq.oi_status == "unknown" and dq.latency is Latency.DELAYED
    assert any("open-interest as-of date is not stated" in w for w in dq.warnings)
    assert dq.coverage["open_interest"] == 1.0 and dq.contracts == 120
    assert result.modeled.assumptions["dealer_convention"] == "dealers long calls / short puts"


@pytest.mark.parametrize(
    "oi_date,status",
    [(date(2026, 9, 30), "current"), (date(2026, 10, 1), "current"), (date(2026, 9, 28), "stale")],
)
def test_open_interest_staleness(oi_date, status):
    result = analyze_options(_with_oi(oi_date))
    assert result.data_quality.oi_status == status
    assert (status == "stale") == any("business days old" in w for w in result.data_quality.warnings)


def _with_oi(oi_date):
    import dataclasses

    return dataclasses.replace(big_chain(), oi_effective_date=oi_date)


def test_analysis_is_deterministic_and_json_serializable():
    a, b = analyze_options(big_chain(put_scale=2.0)), analyze_options(big_chain(put_scale=2.0))
    assert a == b
    payload = json.dumps(a.to_dict(), sort_keys=True)  # raises on any leaked numpy/enum/date object
    assert '"MODELED / ESTIMATED"' in payload


def test_look_ahead_guard():
    chain = big_chain()
    assert_not_after(chain, SNAP_TS)  # equal is fine
    with pytest.raises(LookAheadError):
        assert_not_after(chain, datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc))
    with pytest.raises(LookAheadError):
        analyze_options(chain, as_of=datetime(2026, 9, 30, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        assert_not_after(chain, datetime(2026, 10, 2))
