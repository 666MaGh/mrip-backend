"""Security regime service: queried and neighbour regimes, modeled gamma, and discrepancies (work 020).

Rules
- Deterministic only. LAYA computes nothing here; no numbers come from a model except the
  options gamma block, which is kept under ``modeled`` and labelled ``MODELED / ESTIMATED``.
- Point-in-time: every price is cut at ``as_of``; the options snapshot must be taken on or
  before the end of ``as_of`` (``OptionsSnapshotStore.latest_before``).
- Stored data only: prices come from the first provider with stored bars (cboe, then yahoo).
  No network. A missing or short history is reported as unavailable with a reason.
- Neighbours never get a gamma block. Only the queried symbol does.
- Discrepancies are attention flags, not signals. They never say buy or sell.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from enum import Enum
from typing import Any, Protocol

import pandas as pd

from app.mrip.data.models import PriceSeries
from app.mrip.options.analysis import MODELED_LABEL, analyze_options
from app.mrip.related.types import UnknownSymbol
from app.mrip.relationships.types import Edge, EdgeStatus, Node, NodeKey, NodeType
from app.mrip.security_regime.features import classify_security_regime, unavailable_regime
from app.mrip.security_regime.types import (
    DISCLAIMER,
    TrendLabel,
    VOL_LEVEL,
    SecurityRegime,
    SecurityRegimePolicy,
    VolLabel,
)
from app.mrip.stats.service import prices_to_series

logger = logging.getLogger(__name__)

_PROVIDERS = ("cboe", "yahoo")
_LOOKUP_TYPES = (NodeType.COMPANY, NodeType.SECURITY)
_TREND_SV = {
    TrendLabel.UPTREND: "uppåttrend",
    TrendLabel.DOWNTREND: "nedåttrend",
    TrendLabel.SIDEWAYS: "sidledes rörelse",
}
_VOL_SV = {VolLabel.LOW: "låg", VolLabel.NORMAL: "normal", VolLabel.HIGH: "hög", VolLabel.EXTREME: "extrem"}


class PricePort(Protocol):
    def series(self, provider: str, symbol: str, start: date | None = None, end: date | None = None) -> PriceSeries | None: ...


class SnapshotPort(Protocol):
    def latest_before(self, underlying: str, as_of: datetime) -> Any: ...


class GraphPort(Protocol):
    def get_node(self, node: NodeKey) -> Node | None: ...


class DiscrepancyCode(str, Enum):
    TREND_MISMATCH = "TREND_MISMATCH"
    TREND_DIVERGING = "TREND_DIVERGING"
    VOLATILITY_MISMATCH = "VOLATILITY_MISMATCH"


@dataclass(frozen=True, slots=True)
class Discrepancy:
    code: DiscrepancyCode
    severity: str  # "attention" | "info"
    dimension: str  # "trend" | "volatility"
    text: str
    queried_label: str
    neighbour_label: str

    def to_json(self) -> dict[str, str]:
        return {
            "code": self.code.value, "severity": self.severity, "dimension": self.dimension,
            "text": self.text, "queried_label": self.queried_label, "neighbour_label": self.neighbour_label,
        }


def load_close_prices(prices: PricePort, symbol: str, as_of: date) -> pd.Series | None:
    """Closes up to ``as_of`` from the first provider with stored bars; None when there are none."""
    for provider in _PROVIDERS:
        series = prices.series(provider, symbol, end=as_of)
        if series is not None and series.bars:
            return prices_to_series(series, as_of)
    return None


def find_discrepancies(
    queried_regime: SecurityRegime,
    neighbour_regime: SecurityRegime,
    edge: Edge,
    *,
    queried_symbol: str = "Den valda aktien",
    neighbour_symbol: str = "grannen",
) -> list[Discrepancy]:
    """Flags where the two regimes disagree in a way the edge's relation makes notable.

    Trend rules depend on ``expected_sign`` (+1 co-moving, -1 opposite); without a valid sign
    the trend is not compared. Volatility rules do not depend on the sign. Hypothesis edges
    never raise "attention".
    """
    validated = edge.status is EdgeStatus.VALIDATED
    sign = edge.attributes.get("expected_sign")
    found: list[Discrepancy] = []
    if (
        queried_regime.status == "available" and neighbour_regime.status == "available"
        and sign in (1, -1)
    ):
        found.extend(_trend_discrepancies(
            queried_regime.trend.label, neighbour_regime.trend.label, int(sign), validated,
            queried_symbol, neighbour_symbol,
        ))
    found.extend(_volatility_discrepancies(
        queried_regime.volatility.label, neighbour_regime.volatility.label, validated,
        queried_symbol, neighbour_symbol,
    ))
    return found


def _trend_discrepancies(
    q: TrendLabel | None, n: TrendLabel | None, sign: int, validated: bool, qs: str, ns: str,
) -> list[Discrepancy]:
    """Trend table (sign = expected_sign of the edge):

    - +1 and opposite trends (UPTREND vs DOWNTREND): TREND_MISMATCH.
    - -1 and the same non-SIDEWAYS trend: TREND_MISMATCH.
    - SIDEWAYS vs a trending label, either sign: TREND_DIVERGING (info).
    - Everything else: no flag.
    """
    if q is None or n is None:
        return []
    if q is n:
        if sign == -1 and q is not TrendLabel.SIDEWAYS:
            return [_trend_mismatch(q, n, sign, validated, qs, ns)]
        return []
    if TrendLabel.SIDEWAYS in (q, n):
        return [Discrepancy(
            DiscrepancyCode.TREND_DIVERGING, "info", "trend",
            f"{qs} är i {_TREND_SV[q]} men {ns} i {_TREND_SV[n]}; trenderna skiljer sig åt",
            q.value, n.value,
        )]
    opposite = {q, n} == {TrendLabel.UPTREND, TrendLabel.DOWNTREND}
    if opposite and sign == 1:
        return [_trend_mismatch(q, n, sign, validated, qs, ns)]
    return []


def _trend_mismatch(
    q: TrendLabel, n: TrendLabel, sign: int, validated: bool, qs: str, ns: str,
) -> Discrepancy:
    if sign == 1:
        text = f"{qs} är i {_TREND_SV[q]} men {ns} i {_TREND_SV[n]}; sambandet brukar vara samrörande"
    else:
        text = f"{qs} och {ns} är båda i {_TREND_SV[q]}, fast sambandet brukar vara motriktat"
    return Discrepancy(
        DiscrepancyCode.TREND_MISMATCH, "attention" if validated else "info", "trend",
        text, q.value, n.value,
    )


def _volatility_discrepancies(
    q: VolLabel | None, n: VolLabel | None, validated: bool, qs: str, ns: str,
) -> list[Discrepancy]:
    if q is None or n is None or abs(VOL_LEVEL[q] - VOL_LEVEL[n]) < 2:
        return []
    extreme = VolLabel.EXTREME in (q, n)
    severity = "attention" if validated and extreme else "info"
    return [Discrepancy(
        DiscrepancyCode.VOLATILITY_MISMATCH, severity, "volatility",
        f"{qs} har {_VOL_SV[q]} volatilitet medan {ns} har {_VOL_SV[n]} volatilitet; regimerna skiljer sig tydligt",
        q.value, n.value,
    )]


def regime_json(regime: SecurityRegime) -> dict[str, Any]:
    """Plain JSON for a regime (labels as strings, no modeled block)."""
    vol, trend = regime.volatility, regime.trend
    return {
        "status": regime.status,
        "as_of": regime.as_of.isoformat() if regime.as_of else None,
        "obs": regime.obs,
        "reason": regime.reason,
        "volatility": {
            "status": vol.status,
            "label": vol.label.value if vol.label else None,
            "realized_vol_20d_pct": vol.realized_vol_20d_pct,
            "percentile_3y": vol.percentile_3y,
            "reason": vol.reason,
        },
        "trend": {
            "status": trend.status,
            "label": trend.label.value if trend.label else None,
            "close": trend.close,
            "sma50": trend.sma50,
            "sma200": trend.sma200,
            "return_3m_pct": trend.return_3m_pct,
            "since_days": trend.since_days,
            "reason": trend.reason,
        },
        "drawdown_252d_pct": regime.drawdown_252d_pct,
        "return_1m_pct": regime.return_1m_pct,
        "return_3m_pct": regime.return_3m_pct,
        "policy_version": regime.policy_version,
    }


def _modeled_unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "label": MODELED_LABEL, "reason": reason}


class SecurityRegimeService:
    def __init__(
        self,
        prices: PricePort,
        *,
        snapshots: SnapshotPort | None = None,
        graph: GraphPort | None = None,
        policy: SecurityRegimePolicy = SecurityRegimePolicy(),
    ) -> None:
        self._prices = prices
        self._snapshots = snapshots
        self._graph = graph
        self._policy = policy

    # -- public -----------------------------------------------------------------

    def classify(self, prices: pd.Series | None, as_of: date) -> SecurityRegime:
        """Regime from closes already loaded; None means no stored prices."""
        if prices is None:
            return unavailable_regime("ingen lagrad kurs", self._policy)
        return classify_security_regime(prices, as_of, self._policy)

    def queried_json(self, symbol: str, as_of: date, prices: pd.Series | None) -> dict[str, Any]:
        """Queried-symbol regime plus its optional modeled gamma block."""
        return {**regime_json(self.classify(prices, as_of)), "modeled": self.modeled(symbol, as_of)}

    def build_regime(self, symbol: str, as_of: date) -> dict[str, Any]:
        """Payload for ``GET /symbols/<symbol>/regime``. Raises UnknownSymbol if neither graph nor prices know it."""
        sym = symbol.strip().upper()
        prices = load_close_prices(self._prices, sym, as_of)
        if prices is None and not self._in_graph(sym):
            raise UnknownSymbol(sym)
        return {
            "symbol": sym,
            "as_of": as_of.isoformat(),
            "queried_regime": self.queried_json(sym, as_of, prices),
            "meta": {
                "as_of": as_of.isoformat(),
                "policy_version": self._policy.version,
                "disclaimer": DISCLAIMER,
            },
        }

    def modeled(self, symbol: str, as_of: date) -> dict[str, Any]:
        """Gamma regime from the latest stored options snapshot on or before ``as_of``; MODELED / ESTIMATED."""
        if self._snapshots is None:
            return _modeled_unavailable("ingen options-snapshot-lagring konfigurerad")
        end = datetime.combine(as_of, time.max, tzinfo=timezone.utc)
        try:
            snapshot = self._snapshots.latest_before(symbol, end)
            if snapshot is None:
                return _modeled_unavailable(f"ingen lagrad options-snapshot för {symbol} på eller före {as_of.isoformat()}")
            analysis = analyze_options(snapshot, as_of=end).to_dict()
        except Exception:  # the modeled block must never fail the regime
            logger.exception("modeled gamma failed for %s", symbol)
            return _modeled_unavailable("internt fel vid options-analys")
        modeled = analysis["modeled"]
        return {
            "status": "available",
            "label": MODELED_LABEL,
            "snapshot_timestamp": snapshot.snapshot_timestamp.isoformat(),
            "gamma_regime": modeled["regime"],
            "amplification_level": modeled["amplification"]["level"],
            "regime_method_version": modeled["regime_method_version"],
            "warnings": list(analysis["data_quality"]["warnings"]),
            "reason": None,
        }

    @property
    def policy_version(self) -> str:
        return self._policy.version

    # -- helpers ----------------------------------------------------------------

    def _in_graph(self, sym: str) -> bool:
        if self._graph is None:
            return False
        return any(self._graph.get_node(NodeKey(t, sym)) is not None for t in _LOOKUP_TYPES)
