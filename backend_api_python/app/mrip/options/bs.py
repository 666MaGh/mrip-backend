"""Black-Scholes-Merton pricing and Greeks with a continuous dividend yield.

Conventions (all outputs are per share, not per contract):
- ``t_years`` is time to expiry in years; ``vol`` is annualised, decimal (0.25 = 25%).
- ``theta`` is per CALENDAR DAY (annual theta / 365).
- ``vega`` is per 1 volatility point (d price / d vol * 0.01).
- ``rho`` is per 1 percentage point of the risk-free rate (d price / d r * 0.01).
- ``rate`` and ``dividend_yield`` are continuously compounded decimals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]

_IV_LOW = 1e-4
_IV_HIGH = 5.0


@dataclass(frozen=True, slots=True)
class BsGreeks:
    price: float
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float  # per 1 vol point
    rho: float  # per 1 percentage point of rate


def _check_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _d1_d2(
    spot: float, strike: float, t_years: float, vol: float, rate: float, dividend_yield: float
) -> tuple[float, float]:
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    return d1, d1 - vol * sqrt_t


def _price(
    option_type: OptionType,
    spot: float,
    strike: float,
    t_years: float,
    vol: float,
    rate: float,
    dividend_yield: float,
) -> float:
    d1, d2 = _d1_d2(spot, strike, t_years, vol, rate, dividend_yield)
    fwd = spot * math.exp(-dividend_yield * t_years)
    disc = strike * math.exp(-rate * t_years)
    if option_type == "call":
        return float(fwd * norm.cdf(d1) - disc * norm.cdf(d2))
    return float(disc * norm.cdf(-d2) - fwd * norm.cdf(-d1))


def bs_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    t_years: float,
    vol: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> BsGreeks:
    """Price and Greeks of a European option. Raises ValueError on invalid input."""
    _check_type(option_type)
    if spot <= 0 or strike <= 0 or t_years <= 0 or vol <= 0:
        raise ValueError("spot, strike, t_years and vol must be > 0")
    d1, d2 = _d1_d2(spot, strike, t_years, vol, rate, dividend_yield)
    sqrt_t = math.sqrt(t_years)
    disc_q = math.exp(-dividend_yield * t_years)
    disc_r = math.exp(-rate * t_years)
    pdf_d1 = float(norm.pdf(d1))
    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t * 0.01
    decay = -spot * disc_q * pdf_d1 * vol / (2.0 * sqrt_t)
    if option_type == "call":
        delta = disc_q * float(norm.cdf(d1))
        theta_year = decay - rate * strike * disc_r * float(norm.cdf(d2)) + dividend_yield * spot * disc_q * float(norm.cdf(d1))
        rho = strike * t_years * disc_r * float(norm.cdf(d2)) * 0.01
    else:
        delta = -disc_q * float(norm.cdf(-d1))
        theta_year = decay + rate * strike * disc_r * float(norm.cdf(-d2)) - dividend_yield * spot * disc_q * float(norm.cdf(-d1))
        rho = -strike * t_years * disc_r * float(norm.cdf(-d2)) * 0.01
    return BsGreeks(
        price=_price(option_type, spot, strike, t_years, vol, rate, dividend_yield),
        delta=delta,
        gamma=gamma,
        theta=theta_year / 365.0,
        vega=vega,
        rho=rho,
    )


def bs_gamma_array(
    spot: float,
    strikes: np.ndarray,
    t_years: np.ndarray,
    vols: np.ndarray,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> np.ndarray:
    """Per-share gamma (same for calls and puts) for one spot and many contracts.

    Entries with strike <= 0, t_years <= 0, vol <= 0 or non-finite input are NaN.
    """
    k = np.asarray(strikes, dtype=float)
    t = np.asarray(t_years, dtype=float)
    v = np.asarray(vols, dtype=float)
    k, t, v = np.broadcast_arrays(k, t, v)
    valid = (
        np.isfinite(k) & np.isfinite(t) & np.isfinite(v)
        & (k > 0) & (t > 0) & (v > 0)
        & bool(np.isfinite(spot) and spot > 0)
        & bool(np.isfinite(rate) and np.isfinite(dividend_yield))
    )
    out = np.full(k.shape, np.nan, dtype=float)
    if not valid.any():
        return out
    kk, tt, vv = k[valid], t[valid], v[valid]
    sqrt_t = np.sqrt(tt)
    d1 = (np.log(spot / kk) + (rate - dividend_yield + 0.5 * vv * vv) * tt) / (vv * sqrt_t)
    out[valid] = np.exp(-dividend_yield * tt) * norm.pdf(d1) / (spot * vv * sqrt_t)
    return out


def implied_vol(
    option_type: OptionType,
    price: float,
    spot: float,
    strike: float,
    t_years: float,
    rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float | None:
    """Implied volatility by Brent root-finding on [1e-4, 5.0]; None if no solution."""
    _check_type(option_type)
    values = (price, spot, strike, t_years, rate, dividend_yield)
    if not all(math.isfinite(x) for x in values):
        return None
    if spot <= 0 or strike <= 0 or t_years <= 0:
        return None
    fwd = spot * math.exp(-dividend_yield * t_years)
    disc = strike * math.exp(-rate * t_years)
    if option_type == "call":
        lower, upper = max(fwd - disc, 0.0), fwd
    else:
        lower, upper = max(disc - fwd, 0.0), disc
    if price <= lower or price >= upper:
        return None

    def objective(sigma: float) -> float:
        return _price(option_type, spot, strike, t_years, sigma, rate, dividend_yield) - price

    f_lo, f_hi = objective(_IV_LOW), objective(_IV_HIGH)
    if f_lo > 0 or f_hi < 0:
        return None
    if f_lo == 0:
        return _IV_LOW
    if f_hi == 0:
        return _IV_HIGH
    try:
        return float(brentq(objective, _IV_LOW, _IV_HIGH, xtol=1e-14, rtol=1e-13, maxiter=200))
    except (ValueError, RuntimeError):
        return None
