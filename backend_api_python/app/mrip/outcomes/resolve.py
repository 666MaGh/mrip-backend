"""Resolve a past prediction against OBSERVED daily bars (work 009).

Everything here is deterministic and reads only observed prices; nothing uses the
prediction's own content as truth. ``bars`` is a DataFrame with a sorted, unique
DatetimeIndex (one row per trading day) and a ``close`` column; ``open``/``high``/
``low`` are optional.

Rules:
  1. The entry is the close of the last bar on or before ``made_on`` unless an
     explicit ``entry_price`` is given. Intraday kinds (EOD, NEXT_SESSION) REQUIRE
     an explicit entry and an observed ``made_on`` session.
  2. An outcome is resolvable only once its end bar is observed; otherwise
     ``locate_window`` returns None (never a partial outcome).
  3. Excursions look at bars start+1..end only. The made_on bar is excluded
     because part of it pre-dates an intraday entry; EOD has no excursion bars
     (only the end close counts). The final close is always an observed point.
  4. Sign convention is long perspective: adverse <= 0 <= favorable.
  5. Nothing after ``window.end_pos`` is ever read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

OUTCOME_VERSION = "outcome-v0-uncalibrated"
TRADING_DAYS_PER_YEAR = 252
HORIZON_BARS: dict[str, int] = {"W1": 5, "M1": 21, "M3": 63, "M6": 126, "M12": 252}
KIND_EOD = "EOD"
KIND_NEXT_SESSION = "NEXT_SESSION"
KIND_EXPIRY = "EXPIRY"
HORIZON_KINDS: tuple[str, ...] = (*HORIZON_BARS, KIND_EOD, KIND_NEXT_SESSION, KIND_EXPIRY)

_MIN_VOL_RETURNS = 3


@dataclass(frozen=True, slots=True)
class Window:
    start_pos: int
    end_pos: int
    start_date: date
    end_date: date
    entry_price: float
    kind: str


@dataclass(frozen=True, slots=True)
class PathStats:
    actual_return: float
    realized_volatility: float | None
    max_adverse_excursion: float
    max_favorable_excursion: float
    n_bars: int
    excursions_from: str  # "intraday_range" | "closes"


class WallResult(str, Enum):
    UNTESTED = "untested"
    HELD = "held"
    BROKE = "broke"


@dataclass(frozen=True, slots=True)
class WallOutcome:
    result: WallResult
    side: str
    level: float
    first_touch_date: date | None
    first_break_date: date | None
    max_penetration: float


@dataclass(frozen=True, slots=True)
class ImpliedMoveOutcome:
    expected_move: float
    abs_return_to_expected: float
    excursion_to_expected: float
    realized_to_implied_vol: float | None
    amplified: bool
    version: str


def _check_bars(bars: pd.DataFrame) -> None:
    if "close" not in bars.columns:
        raise ValueError("bars must have a 'close' column")
    idx = bars.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("bars must have a DatetimeIndex")
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("bars index must be sorted and unique")


def _bar_date(bars: pd.DataFrame, pos: int) -> date:
    return bars.index[pos].date()


def _last_pos_on_or_before(bars: pd.DataFrame, day: date) -> int:
    """Position of the last bar dated <= ``day``; -1 if there is none."""
    return int(bars.index.searchsorted(pd.Timestamp(day) + pd.Timedelta(days=1), side="left")) - 1


def locate_window(
    bars: pd.DataFrame,
    made_on: date,
    kind: str,
    expiry: date | None = None,
    entry_price: float | None = None,
) -> Window | None:
    """Map a prediction to its observed bar window, or None if not yet observable."""
    if kind not in HORIZON_KINDS:
        raise ValueError(f"unknown horizon kind: {kind}")
    _check_bars(bars)
    intraday = kind in (KIND_EOD, KIND_NEXT_SESSION)
    if intraday and entry_price is None:
        raise ValueError(f"entry_price is required for {kind}")
    if kind == KIND_EXPIRY and expiry is None:
        raise ValueError("expiry is required for EXPIRY")
    if entry_price is not None and not entry_price > 0:
        raise ValueError("entry_price must be positive")

    start_pos = _last_pos_on_or_before(bars, made_on)
    if start_pos < 0:
        raise ValueError("no data on or before made_on")
    last_pos = len(bars) - 1

    if intraday and _bar_date(bars, start_pos) != made_on:
        return None  # the made_on session is not observed

    if kind in HORIZON_BARS:
        end_pos = start_pos + HORIZON_BARS[kind]
    elif kind == KIND_EOD:
        end_pos = start_pos
    elif kind == KIND_NEXT_SESSION:
        end_pos = start_pos + 1
    else:
        assert expiry is not None
        if _bar_date(bars, last_pos) < expiry:
            return None  # data has not reached the expiry date yet
        end_pos = _last_pos_on_or_before(bars, expiry)
        if end_pos <= start_pos:
            return None

    if end_pos > last_pos:
        return None

    entry = float(bars["close"].iloc[start_pos]) if entry_price is None else float(entry_price)
    if not entry > 0:
        raise ValueError("entry price must be positive")
    return Window(
        start_pos=start_pos,
        end_pos=end_pos,
        start_date=_bar_date(bars, start_pos),
        end_date=_bar_date(bars, end_pos),
        entry_price=entry,
        kind=kind,
    )


def _excursion_range(window: Window) -> tuple[int, int]:
    """Half-open [lo, hi) bar positions that count for excursions."""
    if window.kind == KIND_EOD:
        return window.end_pos, window.end_pos
    return window.start_pos + 1, window.end_pos + 1


def path_stats(bars: pd.DataFrame, window: Window) -> PathStats:
    """Return, realized vol and long-perspective excursions over the window."""
    _check_bars(bars)
    closes = bars["close"].to_numpy(dtype=float)
    entry = window.entry_price
    end_close = float(closes[window.end_pos])
    actual_return = end_close / entry - 1.0

    vol: float | None = None
    if window.kind not in (KIND_EOD, KIND_NEXT_SESSION):
        rets = np.diff(np.log(closes[window.start_pos : window.end_pos + 1]))
        if len(rets) >= _MIN_VOL_RETURNS:
            vol = float(np.std(rets, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))

    lo, hi = _excursion_range(window)
    low_pts = [end_close]
    high_pts = [end_close]
    source = "closes"
    if hi > lo:
        has_range = "high" in bars.columns and "low" in bars.columns
        if has_range:
            highs = bars["high"].to_numpy(dtype=float)[lo:hi]
            lows = bars["low"].to_numpy(dtype=float)[lo:hi]
            if not (np.isnan(highs).any() or np.isnan(lows).any()):
                low_pts.append(float(lows.min()))
                high_pts.append(float(highs.max()))
                source = "intraday_range"
        if source == "closes":
            sl = closes[lo:hi]
            low_pts.append(float(sl.min()))
            high_pts.append(float(sl.max()))

    return PathStats(
        actual_return=actual_return,
        realized_volatility=vol,
        max_adverse_excursion=min(0.0, min(low_pts) / entry - 1.0),
        max_favorable_excursion=max(0.0, max(high_pts) / entry - 1.0),
        n_bars=window.end_pos - window.start_pos,
        excursions_from=source,
    )


def benchmark_return(bars: pd.DataFrame, window: Window) -> float | None:
    """Benchmark return over the window dates; None if its data does not cover them."""
    _check_bars(bars)
    if len(bars) == 0:
        return None
    start = _last_pos_on_or_before(bars, window.start_date)
    if start < 0 or _bar_date(bars, len(bars) - 1) < window.end_date:
        return None
    end = _last_pos_on_or_before(bars, window.end_date)
    start_close = float(bars["close"].iloc[start])
    if not start_close > 0:
        return None
    return float(bars["close"].iloc[end]) / start_close - 1.0


def wall_outcome(
    bars: pd.DataFrame,
    window: Window,
    level: float,
    side: Literal["call", "put"],
    touch_tolerance: float = 0.005,
) -> WallOutcome:
    """Did price touch / close through a modeled wall level during the window?"""
    if not level > 0:
        raise ValueError("level must be positive")
    if side not in ("call", "put"):
        raise ValueError("side must be 'call' or 'put'")
    if touch_tolerance < 0:
        raise ValueError("touch_tolerance must be >= 0")
    _check_bars(bars)

    lo, hi = _excursion_range(window)
    if hi == lo:  # EOD: only the end close is observed (high = low = close)
        lo, hi = window.end_pos, window.end_pos + 1
    closes = bars["close"].to_numpy(dtype=float)[lo:hi]
    extreme_col = "high" if side == "call" else "low"
    if extreme_col in bars.columns:
        extremes = bars[extreme_col].to_numpy(dtype=float)[lo:hi]
        if window.kind == KIND_EOD:
            extremes = closes
    else:
        extremes = closes

    if side == "call":
        touched = extremes >= level * (1.0 - touch_tolerance)
        broke = closes > level
        penetration = max(0.0, float(extremes.max()) / level - 1.0)
    else:
        touched = extremes <= level * (1.0 + touch_tolerance)
        broke = closes < level
        penetration = max(0.0, 1.0 - float(extremes.min()) / level)

    def first_date(mask: np.ndarray) -> date | None:
        hits = np.flatnonzero(mask)
        return _bar_date(bars, lo + int(hits[0])) if len(hits) else None

    touch_date = first_date(touched)
    break_date = first_date(broke)
    if break_date is not None:
        result = WallResult.BROKE
    elif touch_date is not None:
        result = WallResult.HELD
    else:
        result = WallResult.UNTESTED
    return WallOutcome(
        result=result,
        side=side,
        level=float(level),
        first_touch_date=touch_date,
        first_break_date=break_date,
        max_penetration=penetration if result is not WallResult.UNTESTED else 0.0,
    )


def implied_move_outcome(
    stats: PathStats,
    atm_iv: float,
    horizon_bars: int,
    amplification_ratio: float = 1.5,
) -> ImpliedMoveOutcome:
    """Compare the realized path with the move implied by ATM IV at the horizon."""
    if not atm_iv > 0:
        raise ValueError("atm_iv must be positive")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")
    expected = atm_iv * math.sqrt(horizon_bars / TRADING_DAYS_PER_YEAR)
    excursion = max(abs(stats.max_adverse_excursion), stats.max_favorable_excursion)
    exc_ratio = excursion / expected
    rv = stats.realized_volatility
    return ImpliedMoveOutcome(
        expected_move=expected,
        abs_return_to_expected=abs(stats.actual_return) / expected,
        excursion_to_expected=exc_ratio,
        realized_to_implied_vol=None if rv is None else rv / atm_iv,
        amplified=exc_ratio >= amplification_ratio,
        version=OUTCOME_VERSION,
    )


def horizon_bars_for(kind: str, window: Window) -> int:
    """Number of bars the horizon spans (0 for EOD, 1 for NEXT_SESSION)."""
    if kind in HORIZON_BARS:
        return HORIZON_BARS[kind]
    if kind == KIND_EOD:
        return 0
    if kind == KIND_NEXT_SESSION:
        return 1
    if kind == KIND_EXPIRY:
        return window.end_pos - window.start_pos
    raise ValueError(f"unknown horizon kind: {kind}")
