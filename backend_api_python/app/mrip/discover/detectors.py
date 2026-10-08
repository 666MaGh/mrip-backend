from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from app.mrip.discover.types import Kind, Observation
from app.mrip.options.analysis import OptionsAnalysis, MODELED_LABEL
from app.mrip.regime.engine import MarketRegime
from app.mrip.cot.engine import CotSnapshot
from app.mrip.stats.correlation import lead_lag
from app.mrip.stats.transforms import align


@dataclass(frozen=True)
class DiscoverPolicy:
    version: str = "discover-v0-uncalibrated"
    flip_near_pct: float = .02
    wall_near_pct: float = .01
    zero_dte_high: float = .30
    anomaly_cap: int = 5
    cot_divergence_abs: float = 2.0
    divergence_z: float = 2.0
    peer_z: float = 2.5
    peer_min_size: int = 5
    peer_window: int = 20
    relationship_fit_days: int = 250
    relationship_recent_days: int = 5
    delayed_response_ratio: float = .5
    regime_ordinal: Mapping[str, int] = None

    def __post_init__(self) -> None:
        if self.regime_ordinal is None:
            object.__setattr__(self, "regime_ordinal", {"LOW_VOL": 0, "NORMAL": 1, "ELEVATED": 2, "STRESS": 3, "CRISIS": 4})


def _quality(analysis: OptionsAnalysis) -> dict[str, object]:
    return {"completeness": analysis.data_quality.coverage.get("greeks", 0), "warnings": list(analysis.data_quality.warnings), "oi_status": analysis.data_quality.oi_status}


def _modeled_details(details: dict[str, object]) -> dict[str, object]:
    return {**details, "label": MODELED_LABEL}


def gamma_events(current: OptionsAnalysis, previous: OptionsAnalysis | None, as_of: date, policy: DiscoverPolicy = DiscoverPolicy()) -> list[Observation]:
    out: list[Observation] = []; q = _quality(current); m = current.modeled
    regime = m.regime.value if hasattr(m.regime, "value") else str(m.regime)
    if previous is not None:
        old = previous.modeled.regime.value if hasattr(previous.modeled.regime, "value") else str(previous.modeled.regime)
        if regime != old and "UNKNOWN" not in (regime, old):
            now_tilt, old_tilt = m.gex.tilt or 0.0, previous.modeled.gex.tilt or 0.0
            mag = min(1., abs(now_tilt - old_tilt))
            out.append(Observation(Kind.GAMMA_REGIME_CHANGE, current.underlying, as_of, f"{MODELED_LABEL}: gamma regime changed {old} to {regime}", mag, _modeled_details({"previous_regime": old, "regime": regime, "tilt_now": now_tilt, "tilt_previous": old_tilt}), q, True))
    flip = m.gex.flip
    if getattr(flip, "status", None) == "found" and flip.distance_pct is not None:
        distance = abs(float(flip.distance_pct))
        if distance <= policy.flip_near_pct:
            out.append(Observation(Kind.NEAR_GAMMA_FLIP, current.underlying, as_of, f"{MODELED_LABEL}: near modeled gamma flip", 1-distance/policy.flip_near_pct, _modeled_details({"distance_pct": flip.distance_pct}), q, True))
    for side, distance_value in (("call", m.gex.walls.call_wall_distance_pct), ("put", m.gex.walls.put_wall_distance_pct)):
        distance = abs(float(distance_value)) if distance_value is not None else None
        if distance is not None and distance <= policy.wall_near_pct:
            out.append(Observation(Kind.WALL_PROXIMITY, current.underlying, as_of, f"{MODELED_LABEL}: near modeled {side} wall", 1-distance/policy.wall_near_pct, _modeled_details({"side": side, "distance_pct": distance}), q, True))
    anomalies = current.observed.anomalies
    if anomalies:
        out.append(Observation(Kind.UNUSUAL_OPTIONS_ACTIVITY, current.underlying, as_of, "Observed unusual options activity", min(1., len(anomalies)/policy.anomaly_cap), {"anomalies": list(anomalies)}, q, False))
    share = m.gex.zero_dte_share
    if share is not None and share >= policy.zero_dte_high:
        out.append(Observation(Kind.HIGH_ZERO_DTE, current.underlying, as_of, f"{MODELED_LABEL}: high modeled 0DTE concentration", min(1., float(share)), _modeled_details({"zero_dte_share": float(share)}), q, True))
    return out


def regime_events(now: MarketRegime, earlier: MarketRegime | None, as_of: date, policy: DiscoverPolicy = DiscoverPolicy()) -> list[Observation]:
    out=[]
    if earlier and now.regime != earlier.regime:
        a, b = now.regime.value, earlier.regime.value
        out.append(Observation(Kind.REGIME_SHIFT, "MARKET", as_of, f"Market regime shifted from {b} to {a}", abs(policy.regime_ordinal.get(a,0)-policy.regime_ordinal.get(b,0))/4, {"previous": b, "regime": a}))
    if now.term_backwardation is True and (earlier is None or earlier.term_backwardation is not True):
        ratio=now.vix_term_ratio or 1
        out.append(Observation(Kind.TERM_BACKWARDATION,"MARKET",as_of,"VIX term structure entered backwardation",min(1.,max(0.,(ratio-1)*5)),{"vix_term_ratio":ratio}))
    return out


def cot_events(snapshot: CotSnapshot, as_of: date, policy: DiscoverPolicy = DiscoverPolicy()) -> list[Observation]:
    out=[]; subject=snapshot.market.name
    if snapshot.crowded and snapshot.crowding_score is not None:
        out.append(Observation(Kind.COT_CROWDING,subject,as_of,"Crowded COT positioning",min(1.,max(0.,snapshot.crowding_score)),{"side":snapshot.crowding_side,"percentile_1y":snapshot.percentile_1y,"percentile_3y":snapshot.percentile_3y}))
    div=snapshot.price_position_divergence
    if div is not None and abs(div)>=policy.cot_divergence_abs:
        out.append(Observation(Kind.COT_DIVERGENCE,subject,as_of,"Price and positioning diverged",min(1.,abs(div)/4),{"price_position_divergence":div}))
    return out


def relationship_events(subject: str, src_returns: pd.Series, dst_returns: pd.Series, as_of: date, policy: DiscoverPolicy = DiscoverPolicy(), edge_status: str | None = None) -> list[Observation]:
    src,dst=align(src_returns,dst_returns); n=policy.relationship_fit_days; recent=policy.relationship_recent_days
    if len(src)<n+recent:return []
    sx,dy=src.iloc[:-recent],dst.iloc[:-recent]; beta=float(np.dot(sx,dy)/np.dot(sx,sx)) if float(np.dot(sx,sx)) else 0.
    residual=dy-beta*sx
    historical=residual.rolling(recent).sum().dropna(); scale=float(historical.std(ddof=1))
    recent_res=float((dst.iloc[-recent:]-beta*src.iloc[-recent:]).sum()); z=recent_res/scale if scale>0 else 0.
    details={"edge_status":edge_status,"beta":beta,"z":z}
    out=[]
    if abs(z)>=policy.divergence_z:out.append(Observation(Kind.RELATIONSHIP_DIVERGENCE,subject,as_of,"Relationship residual diverged from its fitted history",min(1.,abs(z)/4),details))
    try: lag=lead_lag(sx,dy,5)
    except ValueError: lag=None
    src_hist=src.iloc[:-recent]; src_scale=float(src_hist.rolling(recent).sum().std(ddof=1)); sz=float(src.iloc[-recent:].sum())/src_scale if src_scale>0 else 0.
    dst_sum=float(dst.iloc[-recent:].sum())
    if lag and lag.best_lag>0 and lag.corrected_p_value<.05 and abs(sz)>=policy.divergence_z and abs(dst_sum)<policy.delayed_response_ratio*abs(beta*float(src.iloc[-recent:].sum())):
        out.append(Observation(Kind.DELAYED_REACTION,subject,as_of,"Leading series moved without the expected response",min(1.,abs(sz)/4),{**details,"z":sz,"lag":lag.best_lag,"corrected_p_value":lag.corrected_p_value}))
    return out


def peer_events(returns_by_symbol: Mapping[str,pd.Series], sector_by_symbol: Mapping[str,str], as_of: date, policy: DiscoverPolicy=DiscoverPolicy()) -> list[Observation]:
    sectors:dict[str,list[tuple[str,float]]]={}
    for symbol,series in returns_by_symbol.items():
        sector=sector_by_symbol.get(symbol)
        if not sector or len(series)<policy.peer_window:continue
        value=float(series.iloc[-policy.peer_window:].sum()); sectors.setdefault(sector,[]).append((symbol,value))
    out=[]
    for sector,members in sectors.items():
        if len(members)<policy.peer_min_size:continue
        vals=np.array([v for _,v in members]); median=float(np.median(vals)); sd=float(vals.std(ddof=1))
        if sd==0:continue
        for symbol,value in members:
            rel=value-median; z=rel/sd
            if abs(z)>=policy.peer_z:out.append(Observation(Kind.PEER_DIVERGENCE,symbol,as_of,"Peer return diverged from sector",min(1.,abs(z)/4),{"sector":sector,"relative_return":rel,"z":z}))
    return out
