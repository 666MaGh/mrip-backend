"""Deterministic regime classification of one security from its daily closes.

Every value uses only closes on or before ``as_of``; the input is cut defensively so a
later bar can never leak in. Volatility and trend are independent dimensions: each is
unavailable with its own reason when its history is too short, and the whole regime is
unavailable only when neither can be computed.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.mrip.regime import features as f
from app.mrip.security_regime.types import (
    SecurityRegime,
    SecurityRegimePolicy,
    TrendLabel,
    TrendState,
    VolLabel,
    VolatilityState,
)

_TRADING_DAYS_PER_YEAR = 252


def _unavailable_volatility(reason: str) -> VolatilityState:
    return VolatilityState("unavailable", None, None, None, reason)


def _unavailable_trend(reason: str) -> TrendState:
    return TrendState("unavailable", None, None, None, None, None, None, reason)


def unavailable_regime(reason: str, policy: SecurityRegimePolicy = SecurityRegimePolicy()) -> SecurityRegime:
    """A regime that could not be computed, with the reason stated."""
    return SecurityRegime(
        status="unavailable", as_of=None, obs=0,
        volatility=_unavailable_volatility(reason), trend=_unavailable_trend(reason),
        drawdown_252d_pct=None, return_1m_pct=None, return_3m_pct=None,
        reason=reason, policy_version=policy.version,
    )


def vol_label(percentile: float, policy: SecurityRegimePolicy) -> VolLabel:
    """LOW below the low percentile, EXTREME from the extreme percentile, HIGH from the high one."""
    if percentile >= policy.vol_extreme_pct:
        return VolLabel.EXTREME
    if percentile >= policy.vol_high_pct:
        return VolLabel.HIGH
    if percentile < policy.vol_low_pct:
        return VolLabel.LOW
    return VolLabel.NORMAL


def _volatility(prices: pd.Series, policy: SecurityRegimePolicy) -> VolatilityState:
    if len(prices) < policy.vol_min_obs:
        return _unavailable_volatility(
            f"för kort historik för volatilitet: {len(prices)} av {policy.vol_min_obs} observationer"
        )
    vols = f.realized_volatility(prices, policy.vol_window).dropna()
    if vols.empty:
        return _unavailable_volatility("ingen giltig volatilitet kan beräknas")
    current = float(vols.iloc[-1])
    window = vols.iloc[-policy.vol_lookback:].to_numpy()
    percentile = float(np.mean(window <= current))
    return VolatilityState(
        status="available",
        label=vol_label(percentile, policy),
        realized_vol_20d_pct=round(current * 100.0, 4),
        percentile_3y=round(percentile, 4),
        reason=None,
    )


def _trend_labels(prices: pd.Series, policy: SecurityRegimePolicy) -> pd.Series:
    """Trend label per date: NaN until the slow SMA and the return both exist."""
    close = prices
    fast = close.rolling(policy.sma_fast, min_periods=policy.sma_fast).mean()
    slow = close.rolling(policy.sma_slow, min_periods=policy.sma_slow).mean()
    ret = f.pct_change(close, policy.trend_return_days)
    valid = (fast.notna() & slow.notna() & ret.notna()).to_numpy()
    up = ((close > fast) & (fast > slow) & (ret > 0)).to_numpy()
    down = ((close < fast) & (fast < slow) & (ret < 0)).to_numpy()
    labels = np.full(len(close), None, dtype=object)
    for i in np.flatnonzero(valid):
        labels[i] = TrendLabel.UPTREND if up[i] else TrendLabel.DOWNTREND if down[i] else TrendLabel.SIDEWAYS
    return pd.Series(labels, index=close.index, dtype=object)


def _trend(prices: pd.Series, policy: SecurityRegimePolicy) -> TrendState:
    if len(prices) < policy.trend_min_obs:
        return _unavailable_trend(
            f"för kort historik för trend: {len(prices)} av {policy.trend_min_obs} observationer"
        )
    labels = _trend_labels(prices, policy)
    label = labels.iloc[-1]
    if label is None:
        return _unavailable_trend("trend kan inte beräknas för sista datumet")
    since = 0
    for value in reversed(labels.to_numpy()[-(policy.since_cap + 1):]):
        if value != label:
            break
        since += 1
    return TrendState(
        status="available",
        label=label,
        close=round(float(prices.iloc[-1]), 4),
        sma50=round(float(prices.iloc[-policy.sma_fast:].mean()), 4),
        sma200=round(float(prices.iloc[-policy.sma_slow:].mean()), 4),
        return_3m_pct=_pct_return(prices, policy.return_3m_days),
        since_days=min(since, policy.since_cap),
        reason=None,
    )


def _pct_return(prices: pd.Series, days: int) -> float | None:
    if len(prices) <= days:
        return None
    base = float(prices.iloc[-(days + 1)])
    if base == 0:
        return None
    return round((float(prices.iloc[-1]) / base - 1.0) * 100.0, 4)


def classify_security_regime(
    prices: pd.Series,
    as_of: date,
    policy: SecurityRegimePolicy = SecurityRegimePolicy(),
) -> SecurityRegime:
    """Regime of a security as known at the close of ``as_of``.

    ``prices`` are daily closes indexed by a unique, ascending DatetimeIndex. Bars after
    ``as_of`` are ignored. Non-positive prices make the volatility unavailable.
    """
    if not isinstance(prices, pd.Series) or prices.empty:
        return unavailable_regime("ingen kurshistorik", policy)
    closes = prices.dropna().astype(float)
    closes = closes.loc[: pd.Timestamp(as_of)]
    if closes.empty:
        return unavailable_regime(f"ingen kurs på eller före {as_of.isoformat()}", policy)
    try:
        volatility = _volatility(closes, policy)
    except ValueError as exc:  # non-positive or malformed prices
        volatility = _unavailable_volatility(f"ogiltiga kurser: {exc}")
    trend = _trend(closes, policy)
    if volatility.status == "unavailable" and trend.status == "unavailable":
        reason = f"{volatility.reason}; {trend.reason}"
    else:
        reason = None
    drawdown = None
    if len(closes) >= 2 and bool((closes > 0).all()):
        drawdown = round(float(f.drawdown(closes, policy.drawdown_window).iloc[-1]) * 100.0, 4)
    return SecurityRegime(
        status="available" if reason is None else "unavailable",
        as_of=closes.index[-1].date(),
        obs=len(closes),
        volatility=volatility,
        trend=trend,
        drawdown_252d_pct=drawdown,
        return_1m_pct=_pct_return(closes, policy.return_1m_days),
        return_3m_pct=_pct_return(closes, policy.return_3m_days),
        reason=reason,
        policy_version=policy.version,
    )
