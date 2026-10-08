"""Types, labels and the versioned policy for the security regime (work 020).

The policy is uncalibrated: every threshold is an initial value and is reported with
``SecurityRegimePolicy.version`` so that a stored view can be traced to its rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

SECURITY_REGIME_VERSION = "security-regime-v0-uncalibrated"
DISCLAIMER = "Historiska mönster och beskrivande regimer, inte prognos eller investeringsråd."
DISCREPANCY_LABEL = "Avvikelse att granska, ej signal"


class VolLabel(str, Enum):
    """Volatility level from the percentile of the 20d realised vol over about 3 years."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TrendLabel(str, Enum):
    """Trend from close versus SMA50/SMA200 and the 63-day return."""

    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS = "SIDEWAYS"


# Ordering used for "levels apart" in the volatility discrepancy rule.
VOL_LEVEL: dict[VolLabel, int] = {VolLabel.LOW: 0, VolLabel.NORMAL: 1, VolLabel.HIGH: 2, VolLabel.EXTREME: 3}


@dataclass(frozen=True, slots=True)
class SecurityRegimePolicy:
    version: str = SECURITY_REGIME_VERSION
    vol_window: int = 20
    vol_min_obs: int = 300
    vol_lookback: int = 756  # about three years of trading days
    vol_low_pct: float = 0.25  # percentile strictly below this is LOW
    vol_high_pct: float = 0.75  # at or above this is HIGH
    vol_extreme_pct: float = 0.95  # at or above this is EXTREME
    trend_min_obs: int = 210
    sma_fast: int = 50
    sma_slow: int = 200
    trend_return_days: int = 63
    since_cap: int = 252
    drawdown_window: int = 252
    return_1m_days: int = 21
    return_3m_days: int = 63


@dataclass(frozen=True, slots=True)
class VolatilityState:
    status: str  # "available" | "unavailable"
    label: VolLabel | None
    realized_vol_20d_pct: float | None  # annualised, in percent
    percentile_3y: float | None  # share of the lookback window at or below today's vol, in (0, 1]
    reason: str | None


@dataclass(frozen=True, slots=True)
class TrendState:
    status: str
    label: TrendLabel | None
    close: float | None
    sma50: float | None
    sma200: float | None
    return_3m_pct: float | None
    since_days: int | None  # trading days the current label has held, capped at policy.since_cap
    reason: str | None


@dataclass(frozen=True, slots=True)
class SecurityRegime:
    status: str  # "available" when at least one dimension is available
    as_of: date | None  # last trading day on or before the requested as_of
    obs: int
    volatility: VolatilityState
    trend: TrendState
    drawdown_252d_pct: float | None
    return_1m_pct: float | None
    return_3m_pct: float | None
    reason: str | None
    policy_version: str
