"""RelationshipValidator: run the statistical validation for a graph relationship.

Reads price series through the data gateway, applies ``validate_relationship``,
records the outcome as STATISTICAL_TEST evidence and moves the edge status to
``validated``/``rejected`` when (and only when) the verdict is definite.
An inconclusive run records nothing and leaves the edge as it is: lack of data or
significance is not evidence. Node series are declared in node attributes:
``{"series": {"symbol": "NVDA"}}``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.data.models import PriceSeries
from app.mrip.evidence.store import EvidenceStore
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType
from app.mrip.relationships.graph import RelationshipGraph
from app.mrip.relationships.types import Edge, EdgeStatus, GraphError, Node
from app.mrip.stats.transforms import log_returns
from app.mrip.stats.validate import ValidationPolicy, ValidationResult, Verdict, validate_relationship


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    result: ValidationResult | None  # None when no validation could run (missing data)
    reasons: tuple[str, ...]
    edge: Edge
    evidence: Evidence | None = None
    status_changed: bool = False


def prices_to_series(series: PriceSeries, as_of: date | None = None) -> pd.Series:
    """Close prices indexed by date (UTC date for datetimes), optionally cut at ``as_of``."""
    points = {}
    for bar in series.bars:
        if bar.close is None:
            continue
        day = bar.ts.date() if isinstance(bar.ts, datetime) else bar.ts
        if as_of is None or day <= as_of:
            points[pd.Timestamp(day)] = bar.close
    return pd.Series(points, dtype=float).sort_index()


def _symbol(node: Node) -> str | None:
    series = node.attributes.get("series")
    symbol = series.get("symbol") if isinstance(series, dict) else None
    return str(symbol) if symbol else None


class RelationshipValidator:
    def __init__(
        self,
        graph: RelationshipGraph,
        evidence: EvidenceStore,
        gateway: FinancialDataGateway,
        policy: ValidationPolicy = ValidationPolicy(),
        market_symbol: str | None = "SPY",
    ) -> None:
        self._graph, self._evidence, self._gateway = graph, evidence, gateway
        self._policy, self._market_symbol = policy, market_symbol

    def validate(
        self, relationship: RelationshipRef, *, expected_sign: int | None = None, as_of: date | None = None
    ) -> ValidationOutcome:
        edge = self._graph.find_active_edge(relationship.src, relationship.dst, relationship.relation_type)
        if edge is None:
            raise GraphError("no active edge for this relationship")
        src, dst = self._graph.get_node(relationship.src), self._graph.get_node(relationship.dst)
        sign = expected_sign if expected_sign is not None else edge.attributes.get("expected_sign")

        symbols = {"src": _symbol(src), "dst": _symbol(dst)}
        missing = [role for role, sym in symbols.items() if sym is None]
        if missing:
            return ValidationOutcome(None, (f"no price series declared on node(s): {', '.join(missing)}",), edge)

        try:
            fetched = {role: self._gateway.price_history(sym) for role, sym in symbols.items()}
            market_prices = None
            if self._market_symbol and self._market_symbol not in symbols.values():
                market_prices = self._gateway.price_history(self._market_symbol)
        except DataUnavailable as exc:
            return ValidationOutcome(None, (f"data unavailable: {exc}",), edge)

        returns = {role: log_returns(prices_to_series(p, as_of)) for role, p in fetched.items()}
        market = log_returns(prices_to_series(market_prices, as_of)) if market_prices is not None else None
        result = validate_relationship(
            returns["src"], returns["dst"], market=market, expected_sign=sign,
            relation_type=relationship.relation_type, policy=self._policy,
        )
        if result.verdict is Verdict.INCONCLUSIVE:
            return ValidationOutcome(result, result.reasons, edge)

        evidence = self._record(relationship, result, fetched, market_prices)
        new_status = EdgeStatus.VALIDATED if result.verdict is Verdict.VALIDATED else EdgeStatus.REJECTED
        changed = edge.status is not new_status
        if changed:
            edge = self._graph.set_edge_status(edge.id, new_status)
        return ValidationOutcome(result, result.reasons, edge, evidence, changed)

    def _record(
        self,
        relationship: RelationshipRef,
        result: ValidationResult,
        fetched: dict[str, PriceSeries],
        market_prices: PriceSeries | None,
    ) -> Evidence:
        uri = (
            f"mrip://validation/{result.policy_version}/"
            f"{relationship.src.key}->{relationship.dst.key}/{relationship.relation_type.value}"
        )
        excerpt = json.dumps(
            {"verdict": result.verdict.value, "reasons": list(result.reasons), "metrics": result.metrics},
            sort_keys=True,
        )
        previous = [
            e for e in self._evidence.list_evidence(relationship)
            if e.source_type is SourceType.STATISTICAL_TEST and e.source_uri == uri
        ]
        for old in previous:
            if old.excerpt == excerpt:
                return old  # identical run already recorded
        for old in previous:
            self._evidence.retract(old.id, "superseded by a newer validation run")
        end = datetime.fromisoformat(result.metrics["end"]).replace(tzinfo=timezone.utc)
        provenance: list[dict[str, Any]] = [
            {"role": role, "provider": p.provenance.provider, "endpoint": p.provenance.endpoint,
             "fetched_at": p.provenance.fetched_at.isoformat(), "symbol": p.symbol}
            for role, p in {**fetched, **({"market": market_prices} if market_prices else {})}.items()
        ]
        return self._evidence.add(
            relationship,
            result.stance,
            SourceType.STATISTICAL_TEST,
            uri,
            end,
            assessed_by=f"stats:{result.policy_version}",
            source_title=f"Statistical validation ({result.policy_version})",
            excerpt=excerpt,
            attributes={"series": provenance},
        )
