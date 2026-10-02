"""COT / Positioning Engine: CFTC Commitments of Traders as evidence, never a standalone signal.

Reports are as of Tuesday and published Friday, so every snapshot is point-in-time:
at ``as_of`` only reports published by then are visible. The engine summarises the
speculative group's net position (percentiles, z-score, crowding, price/positioning
divergence) and can turn it into EVIDENCE for a relationship (CONFIRMING /
CONTRADICTING / NEUTRAL against an expected direction). It never emits a trade signal.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Any, Sequence

import pandas as pd

from app.mrip.cot import features as cf
from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.evidence.store import EvidenceStore
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType, Stance
from app.mrip.stats.service import prices_to_series

POLICY_VERSION = "cot-v0-uncalibrated"
_HISTORY_YEARS = 6  # enough for the 5-year percentile


@dataclass(frozen=True, slots=True)
class CotMarket:
    code: str  # CFTC contract market code
    name: str
    price_symbol: str | None = None  # optional price proxy for the divergence feature


# Codes verified against the CFTC feed (see work 012); price proxies are exchange-listed ETFs.
DEFAULT_MARKETS: tuple[CotMarket, ...] = (
    CotMarket("085692", "Copper", None),
    CotMarket("088691", "Gold", "GLD"),
    CotMarket("067651", "WTI Crude Oil", "USO"),
    CotMarket("023651", "Natural Gas", "UNG"),
    CotMarket("13874A", "S&P 500 E-mini", "SPY"),
)


class PositioningStance(str, Enum):
    CONFIRMING = "confirming"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CotSnapshot:
    market: CotMarket
    group: str
    as_of: date
    report_date: date
    available_date: date
    net_position: float
    net_change: float | None
    net_pct_oi: float | None
    open_interest: float | None
    percentile_1y: float | None
    percentile_3y: float | None
    percentile_5y: float | None
    zscore: float | None
    crowding_score: float | None
    crowding_side: str | None
    crowded: bool
    price_position_divergence: float | None
    policy_version: str = POLICY_VERSION
    warnings: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)


def _f(v: Any) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def assess_positioning(
    snapshot: CotSnapshot, expected_direction: int | None, neutral_band: tuple[float, float] = (0.4, 0.6)
) -> PositioningStance:
    """Is speculative positioning (3y percentile) aligned with the direction a relationship implies?

    ``expected_direction`` +1 means the relationship expects the market to be net long-positioned
    (rising), -1 net short. Without a direction or a percentile the answer is UNKNOWN.
    """
    if expected_direction not in (-1, 1):
        return PositioningStance.UNKNOWN
    pct = snapshot.percentile_3y
    if pct is None:
        return PositioningStance.UNKNOWN
    lo, hi = neutral_band
    aligned = pct if expected_direction == 1 else 1.0 - pct
    if aligned >= hi:
        return PositioningStance.CONFIRMING
    if aligned <= lo:
        return PositioningStance.CONTRADICTING
    return PositioningStance.NEUTRAL


class CotEngine:
    def __init__(self, gateway: FinancialDataGateway, markets: Sequence[CotMarket] = DEFAULT_MARKETS) -> None:
        self._gw = gateway
        self._markets = {m.code: m for m in markets} | {m.name.lower(): m for m in markets}

    def market(self, key: str) -> CotMarket:
        m = self._markets.get(key) or self._markets.get(key.lower())
        if m is None:
            raise KeyError(f"unknown COT market {key!r}")
        return m

    def snapshot(self, market: str | CotMarket, as_of: date | None = None, *, group: str | None = None) -> CotSnapshot:
        m = market if isinstance(market, CotMarket) else self.market(market)
        today = as_of or date.today()
        series = self._gw.cot(m.code, start=today - timedelta(days=365 * _HISTORY_YEARS), end=as_of)
        frame, used = cf.positioning_frame(series, group)
        frame = cf.add_crowding(cf.add_percentiles(frame))
        warnings: list[str] = []
        prov: dict[str, Any] = {"cot": {"provider": series.provenance.provider, "endpoint": series.provenance.endpoint,
                                         "fetched_at": series.provenance.fetched_at.isoformat(), "code": m.code}}
        if m.price_symbol:
            try:
                ps = self._gw.price_history(m.price_symbol)
                prices = prices_to_series(ps, today)
                frame["price_position_divergence"] = cf.price_position_divergence(frame, prices)
                prov["price"] = {"symbol": ps.symbol, "provider": ps.provenance.provider}
            except DataUnavailable:
                warnings.append(f"price proxy {m.price_symbol} unavailable: no divergence")
        visible = cf.point_in_time(frame, today)
        if visible.empty:
            raise DataUnavailable(f"no COT report published on or before {today} for {m.name}")
        row = visible.iloc[-1]
        if _f(row.get("percentile_3y")) is None:
            warnings.append("less than 3 years of history: percentiles/z-score unavailable")
        return CotSnapshot(
            market=m, group=used, as_of=today, report_date=visible.index[-1].date(),
            available_date=pd.Timestamp(row["available_date"]).date(),
            net_position=float(row["net_position"]), net_change=_f(row["net_change"]), net_pct_oi=_f(row["net_pct_oi"]),
            open_interest=_f(row["open_interest"]),
            percentile_1y=_f(row.get("percentile_1y")), percentile_3y=_f(row.get("percentile_3y")),
            percentile_5y=_f(row.get("percentile_5y")), zscore=_f(row.get("zscore")),
            crowding_score=_f(row.get("crowding_score")),
            crowding_side=row.get("crowding_side") if isinstance(row.get("crowding_side"), str) else None,
            crowded=bool(row.get("crowded", False)),
            price_position_divergence=_f(row.get("price_position_divergence")),
            warnings=tuple(warnings), provenance=prov,
        )


def record_cot_evidence(
    store: EvidenceStore, relationship: RelationshipRef, snapshot: CotSnapshot, expected_direction: int
) -> Evidence | None:
    """Store the positioning read as evidence for a relationship; UNKNOWN is not evidence and stores nothing."""
    stance = {
        PositioningStance.CONFIRMING: Stance.SUPPORT,
        PositioningStance.CONTRADICTING: Stance.CONTRADICT,
        PositioningStance.NEUTRAL: Stance.NEUTRAL,
    }.get(assess_positioning(snapshot, expected_direction))
    if stance is None:
        return None
    available_at = datetime.combine(snapshot.available_date, time(20, 30), tzinfo=timezone.utc)  # Friday release
    excerpt = json.dumps(
        {
            "group": snapshot.group, "report_date": snapshot.report_date.isoformat(),
            "net_position": snapshot.net_position, "net_change": snapshot.net_change,
            "percentile_3y": snapshot.percentile_3y, "zscore": snapshot.zscore,
            "crowding_side": snapshot.crowding_side, "crowded": snapshot.crowded,
            "expected_direction": expected_direction,
        },
        sort_keys=True,
    )
    return store.add(
        relationship, stance, SourceType.COT, f"cftc://cot/{snapshot.market.code}/{snapshot.report_date.isoformat()}",
        available_at, assessed_by=f"cot:{snapshot.policy_version}",
        source_title=f"CFTC COT {snapshot.market.name} ({snapshot.group})", publisher="CFTC",
        excerpt=excerpt, attributes={"provenance": snapshot.provenance},
    )
