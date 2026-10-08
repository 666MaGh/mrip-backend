"""Research Card: one security, composed from the existing MRIP engines (work 017).

Sections (each independent; missing inputs give ``status: "unavailable"``):

- ``security`` / ``theme`` / ``relationship``: graph node and up to ``MAX_PATHS``
  traversal paths (BOTH directions, depth 3) from the security to a THEME node.
- ``relationship_confidence`` / ``evidence``: statistical validation and evidence
  summary for the edges on the primary path. No calibrated confidence exists yet, so
  the confidence number is always unavailable; the raw validation verdict is shown.
  Evidence level is a deterministic label (``evidence-level-v0-uncalibrated``), not a score.
- ``divergence``: current relationship residual in sigma, computed by the same fit
  Discover uses (``discover.detectors.relationship_divergence``) for the primary edge.
- ``vix_regime``: ``MarketRegimeEngine.at``.
- ``cot``: COT snapshot of a CFTC market whose price proxy is the security or its
  theme proxy; positioning stance (confirming / contradicting / neutral) uses the
  primary edge's ``expected_sign``. Unknown direction gives unavailable.
- ``options``: latest stored chain on or before ``as_of`` through ``analyze_options``.
  Observed fields and MODELED / ESTIMATED fields stay in separate objects.
- ``potential_12m``: scenario distribution of the 12-month (252 trading days) simple
  return from an equal-weight ensemble of the naive and empirical baselines on stored
  prices. Bear = P25, Base = P50, Bull = P75, P10 and P90 bound the range. This is a
  scenario distribution, NOT a price target. Prices are last close times (1 + return).
- ``forecast_confidence``: unavailable unless a fitted forecast calibration is stored.
- ``historical_reliability``: unavailable until resolved forecast outcomes exist and a
  reliability metric is defined; the count of resolved outcomes is reported.

LAYA is never used here: every number comes from deterministic code or statistics.
Trading is disabled; nothing in this module places orders.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from typing import Any, Callable, Mapping, Sequence

from app.mrip.calibration.forecast import load_forecast_calibration
from app.mrip.cot.engine import PositioningStance, assess_positioning
from app.mrip.discover.detectors import DiscoverPolicy, relationship_divergence
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType
from app.mrip.evidence.store import summarize
from app.mrip.forecast.baselines import NaiveBaseline, StatisticalBaseline
from app.mrip.forecast.ensemble import EnsembleProvider
from app.mrip.forecast.types import ForecastProvider, ForecastRequest, ForecastUnavailable, Horizon
from app.mrip.options.analysis import analyze_options
from app.mrip.outcomes.types import PredictionType
from app.mrip.relationships.types import DEFAULT_TRAVERSAL_DEPTH, Direction, Node, NodeKey, NodeType, Path
from app.mrip.stats.service import prices_to_series
from app.mrip.stats.transforms import log_returns
from app.mrip.research.types import CARD_VERSION, EVIDENCE_LEVEL_VERSION, MAX_PATHS, SectionOutcome, UnknownSecurity, WhyEntry, unavailable

POTENTIAL_DEFINITION = (
    "Scenario distribution of the 12-month (252 trading days) simple return. "
    "Bear = P25, Base = P50, Bull = P75; P10 and P90 bound the range. "
    "Not a price target."
)
DIVERGENCE_DEFINITION = (
    "Sigma of the current residual of the primary edge's pair against its fitted history "
    "(Discover relationship divergence, same fit and policy)."
)
FORECAST_VERSION_TAG = "potential-12m-v0-uncalibrated"
ENSEMBLE_WEIGHTS = (0.5, 0.5)
SECTION_ORDER = (
    "security", "theme", "relationship", "relationship_confidence", "evidence", "divergence",
    "vix_regime", "cot", "options", "potential_12m", "forecast_confidence", "historical_reliability",
)


def default_forecast_provider() -> ForecastProvider:
    """Equal-weight ensemble of the two non-parametric baselines; TimesFM is not required."""
    return EnsembleProvider([(NaiveBaseline(), ENSEMBLE_WEIGHTS[0]), (StatisticalBaseline(), ENSEMBLE_WEIGHTS[1])])


def _symbol(node: Node | None) -> str | None:
    if node is None:
        return None
    series = node.attributes.get("series")
    value = series.get("symbol") if isinstance(series, Mapping) else None
    return value if isinstance(value, str) and value else None


def _validation_policy_version(uri: str) -> str | None:
    # validation evidence URIs look like mrip://validation/<policy_version>/<src>-><dst>/<relation>
    parts = uri.split("/")
    return parts[3] if len(parts) > 3 else None


def _iso(value: datetime | date | None) -> str | None:
    return value.isoformat() if value is not None else None


class ResearchCardService:
    def __init__(
        self,
        graph: Any,
        evidence_store: Any,
        price_store: Any,
        *,
        regime_engine: Any = None,
        cot_engine: Any = None,
        snapshot_store: Any = None,
        outcome_store: Any = None,
        calibration_store: Any = None,
        forecast_provider: ForecastProvider | None = None,
        divergence_policy: DiscoverPolicy = DiscoverPolicy(),
    ) -> None:
        self._graph = graph
        self._evidence = evidence_store
        self._prices = price_store
        self._regime = regime_engine
        self._cot_engine = cot_engine
        self._snapshots = snapshot_store
        self._outcomes = outcome_store
        self._calibration = calibration_store
        self._forecast = forecast_provider or default_forecast_provider()
        self._policy = divergence_policy

    # -- lookups ----------------------------------------------------------------

    def _series(self, symbol: str, as_of: date) -> tuple[str, Any] | None:
        """First stored series with bars for cboe, then yahoo (same order as Discover)."""
        for provider in ("cboe", "yahoo"):
            series = self._prices.series(provider, symbol, end=as_of)
            if series is not None and series.bars:
                return provider, series
        return None

    def _security_node(self, symbol: str) -> Node | None:
        for node_type in (NodeType.COMPANY, NodeType.SECURITY):
            node = self._graph.get_node(NodeKey(node_type, symbol))
            if node is not None:
                return node
        return None

    def _paths(self, node: Node) -> list[Path]:
        paths = self._graph.traverse(
            NodeKey(node.node_type, node.key), max_depth=DEFAULT_TRAVERSAL_DEPTH, direction=Direction.BOTH,
        )
        to_theme = [p for p in paths if p.end.node_type is NodeType.THEME]
        to_theme.sort(key=lambda p: (p.depth, p.end.key))
        return to_theme[:MAX_PATHS]

    def _path_evidence(self, path: Path) -> list[Evidence]:
        nodes = {n.id: n for n in path.nodes}
        items: list[Evidence] = []
        for edge in path.edges:
            src, dst = nodes[edge.src_id], nodes[edge.dst_id]
            ref = RelationshipRef(NodeKey(src.node_type, src.key), NodeKey(dst.node_type, dst.key), edge.relation_type)
            items.extend(self._evidence.list_evidence(ref))
        return items

    # -- public -----------------------------------------------------------------

    def build_card(self, symbol: str, as_of: date) -> dict[str, Any]:
        sym = symbol.strip().upper()
        node = self._security_node(sym)
        price = self._series(sym, as_of)
        if node is None and price is None:
            raise UnknownSecurity(sym)

        paths: list[Path] = []
        path_error: str | None = None
        if node is None:
            path_error = "security is not in the relationship graph"
        else:
            try:
                paths = self._paths(node)
            except Exception as exc:  # section isolation: graph failure must not sink the card
                path_error = f"{type(exc).__name__}: {exc}"
            else:
                if not paths:
                    path_error = f"no path from {sym} to a THEME within depth {DEFAULT_TRAVERSAL_DEPTH}"
        primary = paths[0] if paths else None
        theme_node = primary.end if primary else None

        evidence_items: list[Evidence] | None = None
        evidence_error: str | None = None
        if primary is not None:
            try:
                evidence_items = self._path_evidence(primary)
            except Exception as exc:
                evidence_error = f"{type(exc).__name__}: {exc}"
        else:
            evidence_error = path_error

        outcomes: dict[str, SectionOutcome] = {
            "security": self._security(sym, node, price),
            "theme": self._theme(paths, path_error),
            "relationship": self._relationship(sym, paths, path_error),
            "relationship_confidence": self._confidence(primary, evidence_items, evidence_error),
            "evidence": self._evidence_section(primary, evidence_items, evidence_error),
            "divergence": self._safe(lambda: self._divergence(primary, as_of)),
            "vix_regime": self._safe(lambda: self._vix(as_of)),
            "cot": self._safe(lambda: self._cot(sym, theme_node, primary, as_of)),
            "options": self._safe(lambda: self._options(sym, as_of)),
            "potential_12m": self._safe(lambda: self._potential(sym, as_of)),
            "forecast_confidence": self._safe(self._forecast_confidence),
            "historical_reliability": self._safe(lambda: self._reliability(sym)),
        }
        card: dict[str, Any] = {"symbol": sym, "as_of": as_of.isoformat(), "card_version": CARD_VERSION}
        for name in SECTION_ORDER:
            card[name] = dict(outcomes[name].body)
        card["why"] = [
            WhyEntry(name, outcomes[name].inputs, outcomes[name].versions, outcomes[name].evidence_ids).to_dict()
            for name in SECTION_ORDER
        ]
        return card

    # -- section helpers --------------------------------------------------------

    @staticmethod
    def _safe(fn: Callable[[], SectionOutcome]) -> SectionOutcome:
        """Isolate one section: any engine failure becomes an ABSTAIN with the error as reason."""
        try:
            return fn()
        except Exception as exc:
            return unavailable(f"{type(exc).__name__}: {exc}")

    def _security(self, sym: str, node: Node | None, price: tuple[str, Any] | None) -> SectionOutcome:
        inputs = (f"relationship_graph:{sym}" if node else "relationship_graph:none", f"price_store:{sym}")
        return SectionOutcome(
            body={
                "status": "available", "symbol": sym, "name": node.name if node else None,
                "graph_node_type": node.node_type.value if node else None,
                "price_provider": price[0] if price else None,
            },
            inputs=inputs,
        )

    def _theme(self, paths: Sequence[Path], error: str | None) -> SectionOutcome:
        if not paths:
            return unavailable(error or "no theme reachable")
        end = paths[0].end
        return SectionOutcome(
            body={"status": "available", "name": end.name, "key": end.key, "depth": paths[0].depth},
            inputs=("relationship_graph:traverse",), versions=(f"traverse-max-depth-{DEFAULT_TRAVERSAL_DEPTH}",),
        )

    def _relationship(self, sym: str, paths: Sequence[Path], error: str | None) -> SectionOutcome:
        if not paths:
            return unavailable(error or "no relationship path")
        body_paths = []
        for path in paths:
            body_paths.append({
                "depth": path.depth,
                "nodes": [{"type": n.node_type.value, "key": n.key, "name": n.name} for n in path.nodes],
                "edges": [
                    {"id": e.id, "relation_type": e.relation_type.value, "status": e.status.value,
                     "source": e.source, "src_id": e.src_id, "dst_id": e.dst_id}
                    for e in path.edges
                ],
            })
        return SectionOutcome(
            body={"status": "available", "paths": body_paths},
            inputs=("relationship_graph:traverse",), versions=(f"traverse-max-depth-{DEFAULT_TRAVERSAL_DEPTH}",),
        )

    def _confidence(self, primary: Path | None, items: list[Evidence] | None, error: str | None) -> SectionOutcome:
        if primary is None or items is None:
            return unavailable(error or "no primary relationship path")
        edge_status = primary.edges[0].status.value
        validations = [e for e in items if e.source_type is SourceType.STATISTICAL_TEST]
        confidence = {"status": "unavailable", "reason": "no calibrated relationship confidence exists yet"}
        if not validations:
            return unavailable("no statistical validation evidence for the primary path", edge_status=edge_status,
                               confidence=confidence)
        latest = max(validations, key=lambda e: e.available_at)
        try:
            excerpt = json.loads(latest.excerpt or "{}")
        except json.JSONDecodeError:
            excerpt = {}
        policy_version = _validation_policy_version(latest.source_uri)
        return SectionOutcome(
            body={
                "status": "available", "edge_status": edge_status,
                "validation": {
                    "verdict": excerpt.get("verdict"), "reasons": excerpt.get("reasons", []),
                    "metrics": excerpt.get("metrics", {}), "recorded_at": _iso(latest.available_at),
                    "policy_version": policy_version,
                },
                "confidence": confidence,
            },
            inputs=("evidence_store:STATISTICAL_TEST",),
            versions=(f"validation:{policy_version}",) if policy_version else (),
            evidence_ids=(latest.id,),
        )

    def _evidence_section(self, primary: Path | None, items: list[Evidence] | None, error: str | None) -> SectionOutcome:
        if items is None:
            return unavailable(error or "no primary relationship path")
        if not items:
            return unavailable("no evidence recorded for the primary path")
        summary = summarize(items)
        by_stance = {stance.value: n for stance, n in summary.by_stance.items()}
        by_source = {source.value: n for source, n in summary.by_source_type.items()}
        if summary.conflicting:
            level = "conflicting"
        elif len(by_source) >= 2:
            level = "multi_source"
        else:
            level = "single_source"
        return SectionOutcome(
            body={
                "status": "available", "level": level, "level_version": EVIDENCE_LEVEL_VERSION,
                "total": summary.total, "by_stance": by_stance, "by_source_type": by_source,
                "conflicting": summary.conflicting, "latest_available_at": _iso(summary.latest_available_at),
            },
            inputs=("evidence_store:primary_path",), versions=(EVIDENCE_LEVEL_VERSION,),
            evidence_ids=tuple(sorted(e.id for e in items)),
        )

    def _divergence(self, primary: Path | None, as_of: date) -> SectionOutcome:
        if primary is None:
            return unavailable("no primary relationship path")
        edge = primary.edges[0]
        nodes = {n.id: n for n in primary.nodes}
        src, dst = nodes[edge.src_id], nodes[edge.dst_id]
        a, b = _symbol(src), _symbol(dst)
        if a is None or b is None:
            return unavailable("primary edge has no price series declared on its nodes")
        pa, pb = self._series(a, as_of), self._series(b, as_of)
        if pa is None or pb is None:
            return unavailable(f"no stored price series for {a} or {b}")
        fit = relationship_divergence(
            log_returns(prices_to_series(pa[1], as_of)), log_returns(prices_to_series(pb[1], as_of)), self._policy,
        )
        inputs = (f"price_store:{pa[0]}:{a}", f"price_store:{pb[0]}:{b}")
        versions = (self._policy.version,)
        if fit is None:
            return unavailable(
                f"insufficient history: need {self._policy.relationship_fit_days + self._policy.relationship_recent_days} aligned returns",
                inputs=inputs, versions=versions,
            )
        beta, z = fit
        return SectionOutcome(
            body={
                "status": "available", "sigma": round(z, 4), "beta": round(beta, 6), "pair": f"{a}->{b}",
                "edge_id": edge.id, "definition": DIVERGENCE_DEFINITION, "policy_version": self._policy.version,
            },
            inputs=inputs, versions=versions,
        )

    def _vix(self, as_of: date) -> SectionOutcome:
        if self._regime is None:
            return unavailable("no regime engine configured")
        regime = self._regime.at(as_of)
        return SectionOutcome(
            body={
                "status": "available", "regime": regime.regime.value, "vix": regime.vix,
                "vix_percentile": regime.vix_percentile, "term_backwardation": regime.term_backwardation,
                "unavailable_inputs": list(regime.unavailable), "policy_version": regime.policy_version,
            },
            inputs=("regime_engine:VIX",), versions=(regime.policy_version,),
        )

    def _cot(self, sym: str, theme_node: Node | None, primary: Path | None, as_of: date) -> SectionOutcome:
        if self._cot_engine is None:
            return unavailable("no COT engine configured")
        proxies = {sym, _symbol(theme_node) or ""}
        markets = [m for m in self._cot_engine.markets() if m.price_symbol in proxies]
        if not markets:
            return unavailable(f"no COT market has {sym} or its theme proxy as price proxy")
        snapshot = self._cot_engine.snapshot(markets[0], as_of)
        expected = primary.edges[0].attributes.get("expected_sign") if primary else None
        direction = expected if expected in (-1, 1) else None
        stance = assess_positioning(snapshot, direction)
        body: dict[str, Any] = {
            "market": snapshot.market.name, "code": snapshot.market.code, "report_date": snapshot.report_date.isoformat(),
            "percentile_3y": snapshot.percentile_3y, "crowded": snapshot.crowded, "crowding_side": snapshot.crowding_side,
            "policy_version": snapshot.policy_version, "warnings": list(snapshot.warnings),
        }
        inputs = (f"cot:{snapshot.market.code}",)
        versions = (snapshot.policy_version,)
        if stance is PositioningStance.UNKNOWN:
            return unavailable("no expected direction or percentile for positioning stance", inputs=inputs,
                               versions=versions, observed=body)
        return SectionOutcome(body={"status": "available", "stance": stance.value, **body}, inputs=inputs, versions=versions)

    def _options(self, sym: str, as_of: date) -> SectionOutcome:
        if self._snapshots is None:
            return unavailable("no options snapshot store configured")
        end = datetime.combine(as_of, time.max, tzinfo=timezone.utc)
        snapshot = self._snapshots.latest_before(sym, end)
        if snapshot is None:
            return unavailable(f"no stored options snapshot for {sym} on or before {as_of.isoformat()}")
        analysis = analyze_options(snapshot, as_of=end).to_dict()
        inputs = (f"options_snapshot:{snapshot.underlying}:{snapshot.snapshot_timestamp.isoformat()}",)
        versions = (analysis["modeled"]["regime_method_version"],)
        return SectionOutcome(body={"status": "available", **analysis}, inputs=inputs, versions=versions)

    def _potential(self, sym: str, as_of: date) -> SectionOutcome:
        found = self._series(sym, as_of)
        if found is None:
            return unavailable(f"no stored prices for {sym}")
        provider, series = found
        prices = prices_to_series(series, as_of)
        inputs = (f"price_store:{provider}:{sym}",)
        try:
            result = self._forecast.forecast(ForecastRequest(symbol=sym, prices=prices, horizon_days=Horizon.M12.value))
        except (ForecastUnavailable, ValueError) as exc:
            return unavailable(f"forecast unavailable: {exc}", inputs=inputs, versions=(FORECAST_VERSION_TAG,))
        q = result.return_quantiles
        returns = {"P10": q[0.1], "P25": q[0.25], "P50": q[0.5], "P75": q[0.75], "P90": q[0.9]}
        last = result.last_price
        def scenario(value: float) -> dict[str, float]:
            return {"return": value, "price": last * (1.0 + value)}
        return SectionOutcome(
            body={
                "status": "available", "definition": POTENTIAL_DEFINITION, "horizon_days": Horizon.M12.value,
                "as_of": result.as_of.isoformat(), "last_price": last, "n_obs": result.n_obs,
                "bear": scenario(returns["P25"]), "base": scenario(returns["P50"]), "bull": scenario(returns["P75"]),
                "percentiles": {k: {"return": v, "price": last * (1.0 + v)} for k, v in returns.items()},
                "warnings": list(result.warnings), "evaluation": "not evaluated in card (no walk-forward metadata)",
            },
            inputs=inputs,
            versions=(result.provider_version, FORECAST_VERSION_TAG),
        )

    def _forecast_confidence(self) -> SectionOutcome:
        if self._calibration is None:
            return unavailable("no calibration store configured; forecast is uncalibrated")
        fitted = load_forecast_calibration(self._calibration)
        if fitted is None:
            return unavailable("no fitted forecast calibration stored; forecast is uncalibrated")
        return SectionOutcome(
            body={
                "status": "available", "calibration_version": fitted.version, "fit_as_of": _iso(fitted.fit_as_of),
                "n_total": fitted.n_total,
                "note": "segment-level quantile shifts; no scalar confidence is published",
            },
            inputs=("calibration_store:FORECAST_QUANTILE_SHIFT",), versions=(fitted.version,),
        )

    def _reliability(self, sym: str) -> SectionOutcome:
        if self._outcomes is None:
            return unavailable("no outcome store configured")
        resolved = self._outcomes.resolved_pairs(PredictionType.FORECAST, subject=sym)
        if not resolved:
            return unavailable(f"no resolved forecast outcomes for {sym}", inputs=("outcome_store:FORECAST",),
                               n_resolved_outcomes=0)
        return unavailable("reliability metric not defined in v0", inputs=("outcome_store:FORECAST",),
                           n_resolved_outcomes=len(resolved))
