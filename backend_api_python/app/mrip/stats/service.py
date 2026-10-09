"""RelationshipValidator: run the statistical validation for a graph relationship.

Reads price series through the data gateway, applies ``validate_relationship``,
records the outcome as STATISTICAL_TEST evidence and moves the edge status to
``validated``/``rejected`` when (and only when) the verdict is definite.
An inconclusive run records nothing and leaves the edge as it is: lack of data or
significance is not evidence. Node series are declared in node attributes:
``{"series": {"symbol": "NVDA"}}``; the sector is read from the ``sector`` attribute.

Validation v1 (default policy) adds a sector control (peer average of the other
stored-price members of the same sector) and a same-sector null gate. Nulls are
cached in ``mrip_validation_null`` through ``ValidationNullStore`` when one is given.

``downgrade_unsupported=True`` (used by ``--revalidate``) turns an INCONCLUSIVE verdict
on a currently VALIDATED edge into HYPOTHESIS, recording the reasons as neutral evidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import combinations
from typing import Any

import pandas as pd

from app.mrip.data.gateway import DataUnavailable, FinancialDataGateway
from app.mrip.data.models import PriceSeries
from app.mrip.evidence.store import EvidenceStore
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType
from app.mrip.relationships.graph import RelationshipGraph
from app.mrip.relationships.types import Edge, EdgeStatus, GraphError, Node, RelationType
from app.mrip.stats.null_distribution import (
    GLOBAL_SECTOR,
    MIN_SECTOR_PEERS,
    NULL_SAMPLE_SIZE,
    NULL_SEED,
    NullDistribution,
    compute_null,
    sector_peer_average,
)
from app.mrip.stats.null_store import ValidationNullStore
from app.mrip.stats.transforms import log_returns
from app.mrip.stats.validate import (
    SectorControl,
    ValidationPolicy,
    ValidationResult,
    Verdict,
    validate_relationship,
)


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


def _sector(node: Node) -> str | None:
    sector = node.attributes.get("sector")
    return str(sector) if sector else None


class RelationshipValidator:
    def __init__(
        self,
        graph: RelationshipGraph,
        evidence: EvidenceStore,
        gateway: FinancialDataGateway,
        policy: ValidationPolicy = ValidationPolicy(),
        market_symbol: str | None = "SPY",
        null_store: ValidationNullStore | None = None,
    ) -> None:
        self._graph, self._evidence, self._gateway = graph, evidence, gateway
        self._policy, self._market_symbol = policy, market_symbol
        self._null_store = null_store
        self._prices_cache: dict[str, PriceSeries] = {}
        self._returns_cache: dict[tuple[str, date | None], pd.Series] = {}
        self._nulls: dict[tuple[str, date, str, int, int], NullDistribution | None] = {}
        self._universe: dict[str, str | None] | None = None  # priced symbol -> sector

    # ----- data access -------------------------------------------------------------------

    def _forget_loaded(self) -> None:
        """Drop loaded series and in-memory nulls so the next validate() reads the stores afresh.

        The universe (symbol -> sector, restricted to symbols with stored prices) is kept: it changes
        only when the graph is re-seeded and is expensive to probe symbol by symbol.
        """
        self._prices_cache.clear()
        self._returns_cache.clear()
        self._nulls.clear()

    def _prices(self, symbol: str) -> PriceSeries:
        if symbol not in self._prices_cache:
            self._prices_cache[symbol] = self._gateway.price_history(symbol)
        return self._prices_cache[symbol]

    def _returns(self, symbol: str, as_of: date | None) -> pd.Series:
        key = (symbol, as_of)
        if key not in self._returns_cache:
            self._returns_cache[key] = log_returns(prices_to_series(self._prices(symbol), as_of))
        return self._returns_cache[key]

    def _priced_universe(self) -> dict[str, str | None]:
        """Symbols with stored prices (market excluded) mapped to their sector; loaded once per instance.

        A symbol can appear on several nodes (e.g. a SECURITY and a COMPANY node); the sector is taken
        from any of them that declares one. Symbols without stored prices are left out.
        """
        if self._universe is None:
            sectors: dict[str, str | None] = {}
            for node in self._graph.list_priced_nodes():
                symbol = _symbol(node)
                if not symbol or symbol == self._market_symbol:
                    continue
                if sectors.get(symbol) is None:
                    sectors[symbol] = _sector(node)
            mapping: dict[str, str | None] = {}
            for symbol, sector in sectors.items():
                try:
                    self._returns(symbol, None)
                except DataUnavailable:
                    continue
                mapping[symbol] = sector
            self._universe = mapping
        return self._universe

    def sector_of(self, symbol: str) -> str | None:
        return self._priced_universe().get(symbol)

    def priced_sectors(self) -> dict[str, str | None]:
        """Stored-price symbols (market excluded) with their sector, as seen by the validator."""
        return dict(self._priced_universe())

    def _sector_frame(self, members: list[str], as_of: date | None) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for symbol in members:
            try:
                columns[symbol] = self._returns(symbol, as_of)
            except DataUnavailable:
                continue
        return pd.DataFrame(columns)

    # ----- nulls -------------------------------------------------------------------------

    def _null(self, sector: str, as_of: date | None) -> NullDistribution | None:
        """Cached null for ``sector`` (GLOBAL_SECTOR = random-pair null).

        The null is pair-independent (the cache key does not name a pair), so the pair under test is
        not excluded from the sample: it is one member of a pool of hundreds of pairs.
        """
        month = (as_of or date.today()).replace(day=1)
        key = (sector, month, self._policy.version, NULL_SAMPLE_SIZE, NULL_SEED)
        if key in self._nulls:
            return self._nulls[key]
        if self._null_store is not None:
            cached = self._null_store.get(sector, month, self._policy.version, NULL_SAMPLE_SIZE, NULL_SEED)
            if cached is not None:
                self._nulls[key] = cached
                return cached

        universe = self._priced_universe()
        if sector == GLOBAL_SECTOR:
            members: list[str] | None = None
            symbols = sorted(universe)
        else:
            members = sorted(s for s, sec in universe.items() if sec == sector)
            if len(members) < MIN_SECTOR_PEERS + 2:
                self._nulls[key] = None
                return None
            symbols = members
        if not self._market_symbol:
            return None
        returns: dict[str, pd.Series] = {}
        for symbol in symbols:
            try:
                returns[symbol] = self._returns(symbol, as_of)
            except DataUnavailable:
                continue
        pool = list(combinations(sorted(returns), 2))
        dist = compute_null(
            sector=sector,
            as_of=month,
            policy_version=self._policy.version,
            returns=returns,
            market=self._returns(self._market_symbol, as_of),
            pool=pool,
            max_lag=self._policy.max_lag,
            sector_members=[m for m in (members or []) if m in returns] if members is not None else None,
        )
        if dist is not None and self._null_store is not None:
            self._null_store.put(dist)
        self._nulls[key] = dist
        return dist

    def _sector_context(
        self, src: str, dst: str, src_sector: str | None, dst_sector: str | None, as_of: date | None
    ) -> tuple[SectorControl | None, str, NullDistribution | None]:
        """Sector control, its status and the null to gate against (sector null, else random-pair null)."""
        if not self._policy.sector_control:
            return None, "disabled", self._null(GLOBAL_SECTOR, as_of)
        if src_sector is None or src_sector != dst_sector:
            return None, "not_applicable", self._null(GLOBAL_SECTOR, as_of)
        members = [s for s, sec in self._priced_universe().items() if sec == src_sector]
        frame = self._sector_frame(members, as_of)
        peer = sector_peer_average(frame, (src, dst))
        if peer is None:
            return None, "unavailable", self._null(GLOBAL_SECTOR, as_of)
        peers = int(sum(1 for c in frame.columns if c not in (src, dst) and frame[c].notna().any()))
        null = self._null(src_sector, as_of)
        if null is None:
            return None, "unavailable", self._null(GLOBAL_SECTOR, as_of)
        return SectorControl(series=peer, sector=src_sector, peers=peers), "used", null

    def _evaluate(
        self,
        src: str,
        dst: str,
        src_sector: str | None,
        dst_sector: str | None,
        *,
        relation_type: RelationType | None,
        expected_sign: int | None,
        as_of: date | None,
    ) -> ValidationResult:
        src_r = self._returns(src, as_of)
        dst_r = self._returns(dst, as_of)
        market_r = None
        if self._market_symbol and self._market_symbol not in (src, dst):
            market_r = self._returns(self._market_symbol, as_of)
        sector, status, null = (None, "not_applicable", None)
        if market_r is not None:
            sector, status, null = self._sector_context(src, dst, src_sector, dst_sector, as_of)
        return validate_relationship(
            src_r, dst_r, market=market_r, sector=sector, sector_status=status, null=null,
            expected_sign=expected_sign, relation_type=relation_type, policy=self._policy,
        )

    # ----- public API --------------------------------------------------------------------

    def assess_symbols(
        self,
        src_symbol: str,
        dst_symbol: str,
        *,
        relation_type: RelationType | None = None,
        expected_sign: int | None = None,
        as_of: date | None = None,
    ) -> ValidationResult:
        """Evaluate a symbol pair without touching the graph or evidence (used by the null baseline).

        Loaded series and nulls are reused across calls on the same instance (batch work).
        Raises DataUnavailable when a series cannot be loaded.
        """
        return self._evaluate(
            src_symbol, dst_symbol, self.sector_of(src_symbol), self.sector_of(dst_symbol),
            relation_type=relation_type, expected_sign=expected_sign, as_of=as_of,
        )

    def validate(
        self,
        relationship: RelationshipRef,
        *,
        expected_sign: int | None = None,
        as_of: date | None = None,
        downgrade_unsupported: bool = False,
    ) -> ValidationOutcome:
        self._forget_loaded()  # every validation run reads the stored series afresh
        edge = self._graph.find_active_edge(relationship.src, relationship.dst, relationship.relation_type)
        if edge is None:
            raise GraphError("no active edge for this relationship")
        src, dst = self._graph.get_node(relationship.src), self._graph.get_node(relationship.dst)
        if src is None or dst is None:
            raise GraphError("relationship endpoints are missing")
        sign = expected_sign if expected_sign is not None else edge.attributes.get("expected_sign")

        src_symbol, dst_symbol = _symbol(src), _symbol(dst)
        missing = [role for role, sym in (("src", src_symbol), ("dst", dst_symbol)) if sym is None]
        if missing or src_symbol is None or dst_symbol is None:
            return ValidationOutcome(None, (f"no price series declared on node(s): {', '.join(missing)}",), edge)

        try:
            fetched = {"src": self._prices(src_symbol), "dst": self._prices(dst_symbol)}
            market_prices = None
            if self._market_symbol and self._market_symbol not in (src_symbol, dst_symbol):
                market_prices = self._prices(self._market_symbol)
        except DataUnavailable as exc:
            return ValidationOutcome(None, (f"data unavailable: {exc}",), edge)

        # The sector is the symbol's universe sector (the same source the sector nulls use); the node
        # attribute is only a fallback for symbols outside the universe.
        result = self._evaluate(
            src_symbol, dst_symbol,
            self.sector_of(src_symbol) or _sector(src), self.sector_of(dst_symbol) or _sector(dst),
            relation_type=relationship.relation_type, expected_sign=sign, as_of=as_of,
        )
        if result.verdict is Verdict.INCONCLUSIVE:
            if downgrade_unsupported and edge.status is EdgeStatus.VALIDATED:
                evidence = self._record(relationship, result, fetched, market_prices)
                edge = self._graph.set_edge_status(edge.id, EdgeStatus.HYPOTHESIS)
                return ValidationOutcome(result, result.reasons, edge, evidence, True)
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
        end_text = result.metrics.get("end")
        end = (
            datetime.fromisoformat(end_text).replace(tzinfo=timezone.utc)
            if end_text else datetime.now(timezone.utc)
        )
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
