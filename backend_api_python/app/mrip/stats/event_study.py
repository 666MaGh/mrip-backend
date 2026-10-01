"""Market-model event study (OLS alpha/beta fitted before the event window)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from app.mrip.stats.transforms import align

_MIN_ESTIMATION_OBS = 30
# Cross-sectional std at or below this is treated as zero (returns are ~1e-2 scale).
_ZERO_STD = 1e-12


@dataclass(frozen=True, slots=True)
class EventStudyResult:
    n_events: int
    n_skipped: int
    relative_days: tuple[int, ...]
    aar: tuple[float, ...]
    caar: tuple[float, ...]
    car_mean: float
    car_tstat: float | None
    per_event_car: tuple[float, ...]


def event_study(
    returns: pd.Series,
    market: pd.Series,
    event_dates: Sequence[pd.Timestamp | str],
    pre: int = 5,
    post: int = 5,
    estimation_window: int = 120,
    gap: int = 5,
) -> EventStudyResult:
    """Average abnormal returns around events.

    The estimation window is the `estimation_window` positions that end `gap`
    positions before the event window starts, so it never overlaps it and no
    data from the event window or later is used for the fit.
    """
    if pre < 0 or post < 0:
        raise ValueError("pre and post must be >= 0")
    if estimation_window < _MIN_ESTIMATION_OBS:
        raise ValueError(f"estimation_window must be >= {_MIN_ESTIMATION_OBS}")
    if gap < 0:
        raise ValueError("gap must be >= 0")

    r, m = align(returns, market)
    index = pd.DatetimeIndex(r.index)
    rv = r.to_numpy(dtype=float)
    mv = m.to_numpy(dtype=float)
    n = len(rv)

    abnormal: list[np.ndarray] = []
    skipped = 0
    for raw in event_dates:
        t0 = int(index.searchsorted(pd.Timestamp(raw), side="left"))
        start = t0 - pre
        end = t0 + post  # inclusive
        est_end = start - gap  # exclusive
        est_start = est_end - estimation_window
        if t0 >= n or est_start < 0 or end >= n:
            skipped += 1
            continue
        if est_end - est_start < _MIN_ESTIMATION_OBS:
            skipped += 1
            continue
        design = np.column_stack([np.ones(est_end - est_start), mv[est_start:est_end]])
        (alpha, beta), *_ = np.linalg.lstsq(design, rv[est_start:est_end], rcond=None)
        abnormal.append(rv[start : end + 1] - (alpha + beta * mv[start : end + 1]))

    if not abnormal:
        raise ValueError("no usable events")

    matrix = np.vstack(abnormal)
    aar = matrix.mean(axis=0)
    cars = matrix.sum(axis=1)
    car_mean = float(cars.mean())
    tstat: float | None = None
    if len(cars) >= 2:
        std = float(cars.std(ddof=1))
        if std > _ZERO_STD:
            tstat = car_mean / (std / np.sqrt(len(cars)))
    return EventStudyResult(
        n_events=len(abnormal),
        n_skipped=skipped,
        relative_days=tuple(range(-pre, post + 1)),
        aar=tuple(float(x) for x in aar),
        caar=tuple(float(x) for x in np.cumsum(aar)),
        car_mean=car_mean,
        car_tstat=None if tstat is None else float(tstat),
        per_event_car=tuple(float(x) for x in cars),
    )
