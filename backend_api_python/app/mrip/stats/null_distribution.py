"""Same-sector null distribution for the validation v1 effect-size gate (work 022).

A relationship must not only be significant; its partial correlation (controlling
for the market and the sector peer average) must exceed what random pairs produce
under the same controls. The null is the distribution of the CONTEMPORANEOUS
signed partial correlation over seeded random pairs:

  * sector null  : random unordered pairs inside sector S, each controlled for the
                   market and the equal-weighted mean return of the other sector
                   members (both pair members excluded);
  * random-pair null: random unordered universe pairs, market-only control. Used as
                   the fallback when no sector control is available.

The statistic is the SAME one validation uses: lead/lag is selected by raw correlation
(``lead_lag``), and the signed partial correlation is taken at that lag with the controls
at t and t-|lag|. Only walk-forward and the significance check are skipped, so a null
costs a few hundred small regressions. Pure functions: caching lives in
``app.mrip.stats.null_store``.
"""
from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.mrip.stats.correlation import CorrResult, lagged_pair, lead_lag, partial_correlation
from app.mrip.stats.transforms import align

NULL_SAMPLE_SIZE = 200
NULL_SEED = 20261009
MIN_SECTOR_PEERS = 5
MIN_NULL_PAIRS = 30
GLOBAL_SECTOR = "*"


@dataclass(frozen=True, slots=True)
class NullDistribution:
    sector: str  # sector name, or GLOBAL_SECTOR for the random-pair null
    as_of: date  # first day of the month the null is valid for
    policy_version: str
    n: int  # requested sample size
    pairs_used: int  # pairs that produced a finite partial correlation
    seed: int
    p05: float
    p90: float
    p95: float
    p99: float
    mean: float
    std: float

    @property
    def label(self) -> str:
        return "random-pair null" if self.sector == GLOBAL_SECTOR else "same-sector null"


def sector_peer_average(frame: pd.DataFrame, exclude: Sequence[str], min_peers: int = MIN_SECTOR_PEERS) -> pd.Series | None:
    """Equal-weighted mean return of the sector members in ``frame`` other than ``exclude``.

    Returns None when fewer than ``min_peers`` peers have data. A date is kept only
    if at least ``min_peers`` peers have a return on that date.
    """
    peers = frame.drop(columns=[c for c in exclude if c in frame.columns])
    usable = [c for c in peers.columns if peers[c].notna().any()]
    if len(usable) < min_peers:
        return None
    block = peers[usable]
    average = block.mean(axis=1, skipna=True)
    return average.where(block.notna().sum(axis=1) >= min_peers).rename("sector_peer")


def partial_at_lag(
    xs: pd.Series, ys: pd.Series, market: pd.Series, sector: pd.Series | None, lag: int
) -> CorrResult:
    """Partial correlation of x and y at ``lag``, controlling for the market and the sector peer average.

    The controls are taken at both paired dates (t and t-|lag|), as in validate_relationship.
    Inputs must already be aligned on one index.
    """
    x_at_lag, y_now = lagged_pair(xs, ys, lag)
    controls = pd.DataFrame({"market_t": market})
    if lag != 0:
        controls["market_lagged"] = market.shift(abs(lag))
    if sector is not None:
        controls["sector_t"] = sector
        if lag != 0:
            controls["sector_lagged"] = sector.shift(abs(lag))
    pairs = pd.concat([x_at_lag.rename("x"), y_now.rename("y")], axis=1).dropna()
    return partial_correlation(pairs["x"], pairs["y"], controls.loc[pairs.index].dropna())


def signed_best_lag_partial(
    x: pd.Series, y: pd.Series, market: pd.Series, peer: pd.Series | None, max_lag: int
) -> float:
    """Signed partial correlation at the lag that the lead/lag search selects for (x, y); NaN if undefined."""
    series = [x.rename("x"), y.rename("y"), market.rename("market")]
    if peer is not None:
        series.append(peer.rename("sector"))
    aligned = align(*series)
    xs, ys, m = aligned[0], aligned[1], aligned[2]
    s = aligned[3] if peer is not None else None
    try:
        best = lead_lag(xs, ys, max_lag).best_lag
        return float(partial_at_lag(xs, ys, m, s, best).r)
    except ValueError:
        return float("nan")


def compute_null(
    *,
    sector: str,
    as_of: date,
    policy_version: str,
    returns: Mapping[str, pd.Series],
    market: pd.Series,
    pool: Sequence[tuple[str, str]],
    max_lag: int,
    sector_members: Sequence[str] | None = None,
    n: int = NULL_SAMPLE_SIZE,
    seed: int = NULL_SEED,
) -> NullDistribution | None:
    """Seeded null over ``n`` pairs drawn from ``pool`` (unordered pairs of symbols with returns).

    With ``sector_members`` the sector control is applied (peer average of the other members);
    without it the null is market-only. Returns None when fewer than ``MIN_NULL_PAIRS`` pairs
    yield a finite partial correlation.
    """
    if not pool:
        return None
    frame = pd.DataFrame({s: returns[s] for s in sector_members}) if sector_members is not None else None
    ordered_pool = sorted({tuple(sorted(pair)) for pair in pool})
    sample = random.Random(seed).sample(ordered_pool, min(n, len(ordered_pool)))
    values: list[float] = []
    for a, b in sample:
        peer = None
        if frame is not None:
            peer = sector_peer_average(frame, (a, b))
            if peer is None:
                continue
        r = signed_best_lag_partial(returns[a], returns[b], market, peer, max_lag)
        if np.isfinite(r):
            values.append(r)
    if len(values) < MIN_NULL_PAIRS:
        return None
    arr = np.asarray(values, dtype=float)
    p05, p90, p95, p99 = np.percentile(arr, [5, 90, 95, 99])
    return NullDistribution(
        sector=sector,
        as_of=as_of,
        policy_version=policy_version,
        n=n,
        pairs_used=len(values),
        seed=seed,
        p05=float(p05),
        p90=float(p90),
        p95=float(p95),
        p99=float(p99),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)),
    )
