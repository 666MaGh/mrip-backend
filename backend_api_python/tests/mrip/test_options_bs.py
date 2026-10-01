"""Black-Scholes-Merton: textbook values, identities and finite-difference checks."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.mrip.options.bs import BsGreeks, bs_gamma_array, bs_greeks, implied_vol

S, K, R, SIG, T = 42.0, 40.0, 0.10, 0.20, 0.5


def _price(opt: str, s: float = S, k: float = K, t: float = T, v: float = SIG, r: float = R, q: float = 0.0) -> float:
    return bs_greeks(opt, s, k, t, v, r, q).price  # type: ignore[arg-type]


def test_hull_textbook_values() -> None:
    call = bs_greeks("call", S, K, T, SIG, R)
    put = bs_greeks("put", S, K, T, SIG, R)
    assert isinstance(call, BsGreeks)
    assert call.price == pytest.approx(4.76, abs=0.005)
    assert put.price == pytest.approx(0.81, abs=0.005)
    assert call.delta == pytest.approx(0.7791, abs=5e-5)


def test_put_call_parity_with_dividend_yield() -> None:
    q = 0.03
    call = _price("call", q=q)
    put = _price("put", q=q)
    assert call - put == pytest.approx(S * math.exp(-q * T) - K * math.exp(-R * T), abs=1e-12)


def test_call_minus_put_delta_is_discount_factor() -> None:
    q = 0.03
    dc = bs_greeks("call", S, K, T, SIG, R, q).delta
    dp = bs_greeks("put", S, K, T, SIG, R, q).delta
    assert dc - dp == pytest.approx(math.exp(-q * T), abs=1e-12)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_gamma_is_finite_difference_of_delta(opt: str) -> None:
    q, h = 0.02, 1e-3
    up = bs_greeks(opt, S + h, K, T, SIG, R, q).delta  # type: ignore[arg-type]
    dn = bs_greeks(opt, S - h, K, T, SIG, R, q).delta  # type: ignore[arg-type]
    g = bs_greeks(opt, S, K, T, SIG, R, q).gamma  # type: ignore[arg-type]
    assert g == pytest.approx((up - dn) / (2 * h), rel=1e-6)


def test_gamma_equal_for_call_and_put() -> None:
    assert bs_greeks("call", S, K, T, SIG, R, 0.01).gamma == pytest.approx(
        bs_greeks("put", S, K, T, SIG, R, 0.01).gamma, abs=1e-14
    )


@pytest.mark.parametrize("opt", ["call", "put"])
def test_vega_is_finite_difference_per_vol_point(opt: str) -> None:
    q, h = 0.02, 1e-5
    up = _price(opt, v=SIG + h, q=q)
    dn = _price(opt, v=SIG - h, q=q)
    expected = (up - dn) / (2 * h) * 0.01
    assert bs_greeks(opt, S, K, T, SIG, R, q).vega == pytest.approx(expected, rel=1e-6)  # type: ignore[arg-type]


@pytest.mark.parametrize("opt", ["call", "put"])
def test_theta_is_minus_dprice_dT_per_calendar_day(opt: str) -> None:
    q, h = 0.02, 1e-5
    dprice_dt = (_price(opt, t=T + h, q=q) - _price(opt, t=T - h, q=q)) / (2 * h)
    assert bs_greeks(opt, S, K, T, SIG, R, q).theta == pytest.approx(-dprice_dt / 365.0, rel=1e-6)  # type: ignore[arg-type]


@pytest.mark.parametrize("opt", ["call", "put"])
def test_rho_is_per_one_percentage_point(opt: str) -> None:
    q, h = 0.02, 1e-6
    dprice_dr = (_price(opt, r=R + h, q=q) - _price(opt, r=R - h, q=q)) / (2 * h)
    assert bs_greeks(opt, S, K, T, SIG, R, q).rho == pytest.approx(dprice_dr * 0.01, rel=1e-6)  # type: ignore[arg-type]
    # A real 1pp rate move changes the price by about rho.
    moved = _price(opt, r=R + 0.01, q=q) - _price(opt, q=q)
    assert moved == pytest.approx(bs_greeks(opt, S, K, T, SIG, R, q).rho, rel=0.05)  # type: ignore[arg-type]


def test_gamma_array_matches_scalar() -> None:
    strikes = np.array([30.0, 38.0, 40.0, 42.0, 55.0])
    ts = np.array([0.02, 0.1, 0.5, 1.0, 2.0])
    vols = np.array([0.5, 0.3, 0.2, 0.25, 0.4])
    arr = bs_gamma_array(S, strikes, ts, vols, R, 0.015)
    for i in range(len(strikes)):
        scalar = bs_greeks("call", S, float(strikes[i]), float(ts[i]), float(vols[i]), R, 0.015).gamma
        assert abs(arr[i] - scalar) < 1e-12


def test_gamma_array_nan_for_invalid_entries() -> None:
    strikes = np.array([40.0, 0.0, 40.0, 40.0, np.nan, 40.0, -5.0])
    ts = np.array([0.5, 0.5, 0.0, 0.5, 0.5, np.inf, 0.5])
    vols = np.array([0.2, 0.2, 0.2, 0.0, 0.2, 0.2, 0.2])
    out = bs_gamma_array(S, strikes, ts, vols)
    assert np.isfinite(out[0]) and out[0] > 0
    assert np.isnan(out[1:]).all()


def test_gamma_array_invalid_spot_is_all_nan() -> None:
    assert np.isnan(bs_gamma_array(0.0, np.array([40.0]), np.array([0.5]), np.array([0.2]))).all()


@pytest.mark.parametrize("opt", ["call", "put"])
def test_implied_vol_round_trip_grid(opt: str) -> None:
    # Strikes within ~8% of spot: deeper ITM at short maturity is limited by float price precision.
    for strike in (92.0, 95.0, 100.0, 105.0, 108.0):
        for vol in (0.10, 0.25, 0.60, 1.20):
            for t in (0.05, 0.25, 1.0, 2.0):
                p = bs_greeks(opt, 100.0, strike, t, vol, 0.03, 0.01).price  # type: ignore[arg-type]
                iv = implied_vol(opt, p, 100.0, strike, t, 0.03, 0.01)  # type: ignore[arg-type]
                assert iv is not None, (opt, strike, vol, t)
                assert iv == pytest.approx(vol, abs=1e-8), (opt, strike, vol, t)


def test_implied_vol_hull_example() -> None:
    iv = implied_vol("call", bs_greeks("call", S, K, T, SIG, R).price, S, K, T, R)
    assert iv == pytest.approx(SIG, abs=1e-8)


def test_implied_vol_none_cases() -> None:
    # Below intrinsic (forward intrinsic for the call is 42 - 40*exp(-0.05) = 3.95).
    assert implied_vol("call", 3.0, S, K, T, R) is None
    # Put-side: below forward intrinsic (K*exp(-rT) - S) is impossible for a deep ITM put.
    assert implied_vol("put", 5.0, 30.0, 40.0, T, R) is None  # lower bound 40*exp(-0.05) - 30 = 8.05
    assert implied_vol("put", 0.0, S, K, T, R) is None
    # At or above the upper bound (call <= spot, put <= discounted strike).
    assert implied_vol("call", S, S, K, T, R) is None
    assert implied_vol("call", S + 1.0, S, K, T, R) is None
    assert implied_vol("put", K * math.exp(-R * T) + 0.01, S, K, T, R) is None
    # Non-positive maturity.
    assert implied_vol("call", 4.76, S, K, 0.0, R) is None
    assert implied_vol("call", 4.76, S, K, -1.0, R) is None
    # Price implying vol outside [1e-4, 5.0]: bracket fails.
    p_hi = bs_greeks("call", S, K, T, 8.0, R).price
    assert implied_vol("call", p_hi, S, K, T, R) is None
    # ATM, r=q=0: vol 1e-5 gives a positive price whose implied vol is below the 1e-4 floor.
    p_lo = bs_greeks("call", 100.0, 100.0, 0.5, 1e-5).price
    assert p_lo > 0
    assert implied_vol("call", p_lo, 100.0, 100.0, 0.5) is None


@pytest.mark.parametrize(
    "args",
    [
        ("call", 0.0, K, T, SIG),
        ("call", S, 0.0, T, SIG),
        ("call", S, K, 0.0, SIG),
        ("call", S, K, T, 0.0),
        ("call", S, K, T, -0.1),
        ("call", S, K, -1.0, SIG),
        ("straddle", S, K, T, SIG),
    ],
)
def test_bs_greeks_value_errors(args: tuple) -> None:
    with pytest.raises(ValueError):
        bs_greeks(*args)


def test_implied_vol_invalid_option_type() -> None:
    with pytest.raises(ValueError):
        implied_vol("straddle", 1.0, S, K, T)  # type: ignore[arg-type]
