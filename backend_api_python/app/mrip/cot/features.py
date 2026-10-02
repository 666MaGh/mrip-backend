"""CFTC Commitments of Traders positioning features.

COT reports are as of a Tuesday but published on the following Friday, so a report is
only usable from ``report_date + COT_PUBLICATION_LAG_DAYS``; ``point_in_time`` enforces
this. Every rolling feature is strictly trailing: the value at report t uses only
reports at or before t. Warm-up positions are NaN, never back-filled.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.mrip.data.models import CotSeries
from app.mrip.regime.features import trailing_percentile
from app.mrip.stats.transforms import zscore

# A Tuesday report is published the following Friday.
COT_PUBLICATION_LAG_DAYS = 3

_MANAGED_MONEY = "managed_money"
_NON_COMMERCIAL = "non_commercial"

# Float noise allowance when testing a crowding score against its threshold.
_SCORE_TOLERANCE = 1e-12


def _check_min(value: int, minimum: int, name: str) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _check_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Require a positioning frame: ascending unique DatetimeIndex and the given columns."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("frame must have a DatetimeIndex")
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame index must be monotonic increasing")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {', '.join(missing)}")


def _select_group(series: CotSeries, group: str | None) -> str:
    """Resolve the trader group: explicit (must exist in the data) or auto-selected."""
    records = series.records
    if not records:
        raise ValueError("COT series has no records")
    if group is not None:
        if not any(group in r.long_positions or group in r.short_positions for r in records):
            raise ValueError(f"trader group {group!r} not present in the data")
        return group
    complete = sum(
        r.long_positions.get(_MANAGED_MONEY) is not None
        and r.short_positions.get(_MANAGED_MONEY) is not None
        for r in records
    )
    # At least half of the records must carry both sides for managed money.
    return _MANAGED_MONEY if 2 * complete >= len(records) else _NON_COMMERCIAL


def positioning_frame(series: CotSeries, group: str | None = None) -> tuple[pd.DataFrame, str]:
    """Positioning frame indexed by report_date, plus the trader group actually used.

    `group=None` picks managed money when at least half of the records have both a long
    and a short value for it, otherwise non-commercial. Records where the group's long or
    short is None are dropped; duplicate report dates keep the last record.
    """
    used = _select_group(series, group)
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for record in series.records:
        long_ = record.long_positions.get(used)
        short = record.short_positions.get(used)
        if long_ is None or short is None:
            continue
        oi = record.open_interest
        net = float(long_) - float(short)
        # Re-inserting a key keeps its slot; the value is the last record seen.
        rows[pd.Timestamp(record.report_date)] = {
            "gross_long": float(long_),
            "gross_short": float(short),
            "net_position": net,
            "open_interest": float("nan") if oi is None else float(oi),
        }
    if not rows:
        raise ValueError(f"no records with both long and short positions for {used!r}")

    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame.index.name = "report_date"
    frame.insert(0, "available_date", frame.index + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS))
    open_interest = frame["open_interest"]
    frame["net_pct_oi"] = frame["net_position"] / open_interest.where(open_interest != 0)
    frame["net_change"] = frame["net_position"].diff(1)
    frame["net_change_4w"] = frame["net_position"].diff(4)
    columns = [
        "available_date", "gross_long", "gross_short", "net_position",
        "open_interest", "net_pct_oi", "net_change", "net_change_4w",
    ]
    return frame[columns], used


def add_percentiles(
    frame: pd.DataFrame,
    weeks_1y: int = 52,
    weeks_3y: int = 156,
    weeks_5y: int = 260,
    zscore_weeks: int = 156,
) -> pd.DataFrame:
    """Copy of `frame` with trailing percentile ranks and z-score of `net_position`.

    `percentile_1y/3y/5y` is the share of the trailing window (current value included)
    that is <= the current value. `zscore` uses the trailing mean and sample std (ddof=1)
    over `zscore_weeks` reports, NaN where the std is 0. All are NaN until the window is full.
    """
    for name, value in (
        ("weeks_1y", weeks_1y), ("weeks_3y", weeks_3y),
        ("weeks_5y", weeks_5y), ("zscore_weeks", zscore_weeks),
    ):
        _check_min(value, 2, name)
    _check_frame(frame, ("net_position",))
    out = frame.copy()
    net = out["net_position"].astype(float)
    out["percentile_1y"] = trailing_percentile(net, weeks_1y)
    out["percentile_3y"] = trailing_percentile(net, weeks_3y)
    out["percentile_5y"] = trailing_percentile(net, weeks_5y)
    out["zscore"] = zscore(net, zscore_weeks)
    return out


def add_crowding(frame: pd.DataFrame, extreme: float = 0.8) -> pd.DataFrame:
    """Copy of `frame` with crowding score, side and flag derived from `percentile_3y`.

    `crowding_score = |2 * percentile_3y - 1|` in [0, 1]; `crowding_side` is LONG above
    0.5, SHORT below, NONE at exactly 0.5; `crowded` is score >= `extreme`. Score and side
    are NaN where the percentile is NaN, and `crowded` is False there.
    """
    if not 0 < extreme <= 1:
        raise ValueError("extreme must satisfy 0 < extreme <= 1")
    _check_frame(frame, ("percentile_3y",))
    out = frame.copy()
    pct = out["percentile_3y"].astype(float)
    score = (2.0 * pct - 1.0).abs()
    side = np.where(pct > 0.5, "LONG", np.where(pct < 0.5, "SHORT", "NONE")).astype(object)
    side[pct.isna().to_numpy()] = np.nan
    out["crowding_score"] = score
    out["crowding_side"] = pd.Series(side, index=out.index, dtype=object)
    out["crowded"] = (score >= extreme - _SCORE_TOLERANCE).fillna(False).astype(bool)
    return out


def point_in_time(frame: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Rows whose `available_date` is on or before `as_of` (a report is hidden until published)."""
    _check_frame(frame, ("available_date",))
    visible = frame["available_date"] <= pd.Timestamp(as_of)
    return frame.loc[visible.to_numpy()].copy()


def _trailing_return(prices: pd.Series, at: pd.DatetimeIndex, weeks: int) -> pd.Series:
    """Price at the last observation <= t over the price at the last one <= t - 7*weeks days, minus 1."""
    index = prices.index
    values = prices.to_numpy(dtype=float)
    pos_now = index.searchsorted(at, side="right") - 1
    pos_then = index.searchsorted(at - pd.Timedelta(days=7 * weeks), side="right") - 1
    valid = (pos_now >= 0) & (pos_then >= 0)
    out = np.full(len(at), np.nan)
    out[valid] = values[pos_now[valid]] / values[pos_then[valid]] - 1.0
    return pd.Series(out, index=at)


def price_position_divergence(
    frame: pd.DataFrame,
    prices: pd.Series,
    price_weeks: int = 4,
    zscore_weeks: int = 156,
) -> pd.Series:
    """Trailing z-score of the price return minus that of `net_change_4w`, indexed like `frame`.

    The price return at report date t spans `price_weeks` weeks ending at the last price
    observation <= t. Positive means price rose more than positioning, negative means
    positioning rose more than price. NaN until both z-scores are defined.
    """
    _check_min(price_weeks, 1, "price_weeks")
    _check_min(zscore_weeks, 2, "zscore_weeks")
    _check_frame(frame, ("net_change_4w",))
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if prices.index.tz is not None:
        raise ValueError("prices index must be timezone-naive")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be monotonic increasing")
    clean = prices.astype(float).dropna()
    if bool((clean <= 0).any()):
        raise ValueError("prices must be strictly positive")

    price_return = _trailing_return(clean, frame.index, price_weeks)
    divergence = zscore(price_return, zscore_weeks) - zscore(
        frame["net_change_4w"].astype(float), zscore_weeks
    )
    divergence.name = "divergence"
    return divergence
