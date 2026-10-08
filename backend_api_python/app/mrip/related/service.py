"""Related neighbours of a security (work 019): direct edges only, both directions, stored data only.

Rules
- Deterministic and statistical only. LAYA computes nothing here.
- One row per active edge, so several edges may link the same pair. Rejected edges are never listed;
  hypothesis edges only when ``include_hypothesis`` is set.
- Lead/lag is read from the latest non-retracted STATISTICAL_TEST evidence and is never recomputed.
  It is only reported for validated edges; otherwise it is unavailable with the reason "ej validerad".
  Stored direction is src -> dst (``x_leads`` means src leads). It is translated to the QUERIED symbol:
  queried is src: x_leads -> "leading", y_leads -> "lagging"; queried is dst: the reverse.
- Divergence is always the NEIGHBOUR relative to the QUERIED symbol (``subject="neighbour"``):
  neighbour_returns = beta * queried_returns + residual, and sigma is the z of the neighbour's latest
  residual window. This is ``relationship_divergence(queried, neighbour)`` with the Discover policy.
  The same convention is used for both roles, so the row's sigma does not depend on role.
- Signal for the QUERIED symbol (``basis="divergence"``), in this order:
  1. Not validated -> abstain "relationen är inte validerad".
  2. Queried symbol leads the neighbour (translated lag direction "leading") -> abstain "ledande aktie".
  3. Divergence unavailable -> abstain with its reason.
  4. |sigma| below ``DiscoverPolicy.divergence_z`` -> abstain, direction "none".
  5. Otherwise direction = "up" if sigma * s > 0 else "down", where s is the edge's ``expected_sign``
     (or the sign of beta when the edge has none). Reason: a positive sigma with s = +1 means the
     neighbour sits above the level the relationship implies for the queried symbol, so the queried
     symbol is expected to move up toward it. Negative s flips the direction.
- Point-in-time: prices and evidence are cut at ``as_of``. The graph is the current graph version.
- Stored data only: no network. A failing row degrades to unavailable with a reason, never a 500.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Protocol

import pandas as pd

from app.mrip.discover.detectors import DiscoverPolicy, relationship_divergence
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType
from app.mrip.related.types import DISCLAIMER, RELATED_VERSION, SIGNAL_LABEL, UnknownSymbol
from app.mrip.relationships.types import Direction, Edge, EdgeStatus, Node, NodeKey, NodeType, Path
from app.mrip.stats.service import prices_to_series
from app.mrip.stats.transforms import log_returns

logger = logging.getLogger(__name__)

_PROVIDERS = ("cboe", "yahoo")
_LOOKUP_TYPES = (NodeType.COMPANY, NodeType.SECURITY)
_VALIDATED_ONLY = frozenset({EdgeStatus.VALIDATED})
_WITH_HYPOTHESIS = frozenset({EdgeStatus.VALIDATED, EdgeStatus.HYPOTHESIS})
_UNVALIDATED = "ej validerad"


class GraphPort(Protocol):
    def get_node(self, node: NodeKey) -> Node | None: ...
    def traverse(self, start: NodeKey, *, max_depth: int, direction: Direction, statuses: frozenset[EdgeStatus]) -> list[Path]: ...


class EvidencePort(Protocol):
    def list_evidence(self, relationship: RelationshipRef, *, as_of: datetime | None = None) -> list[Evidence]: ...


class PricePort(Protocol):
    def series(self, provider: str, symbol: str, start: date | None = None, end: date | None = None) -> Any: ...


def translate_direction(stored: str, queried_is_src: bool) -> str:
    """Stored direction (src -> dst) seen from the queried symbol: "leading", "lagging" or "contemporaneous"."""
    if stored == "contemporaneous":
        return "contemporaneous"
    if stored not in ("x_leads", "y_leads"):
        raise ValueError(f"unknown stored direction {stored!r}")
    src_leads = stored == "x_leads"
    return "leading" if src_leads == queried_is_src else "lagging"


def relation_sign(expected_sign: object, beta: float | None) -> int:
    """s used by the signal rule: the edge's expected_sign when it is +1/-1, else the sign of beta."""
    if expected_sign in (1, -1):
        return int(expected_sign)  # type: ignore[arg-type]
    if beta is not None and beta != 0:
        return 1 if beta > 0 else -1
    return 1


def _abstain(reason: str) -> dict[str, Any]:
    return {"status": "abstain", "direction": "none", "basis": None, "reason": reason, "label": SIGNAL_LABEL}


def signal_for(
    *,
    validated: bool,
    queried_direction: str | None,
    sigma: float | None,
    divergence_reason: str | None,
    sign: int,
    threshold: float,
) -> dict[str, Any]:
    """Signal for the queried symbol under the rule in the module docstring (pure, unit-tested)."""
    if not validated:
        return _abstain("relationen är inte validerad")
    if queried_direction == "leading":
        return _abstain("ledande aktie")
    if sigma is None:
        return _abstain(divergence_reason or "avvikelse saknas")
    if abs(sigma) < threshold:
        return _abstain(f"avvikelsen {sigma:.2f} sigma understiger tröskeln {threshold:g}")
    direction = "up" if sigma * sign > 0 else "down"
    return {
        "status": "available", "direction": direction, "basis": "divergence",
        "reason": f"avvikelse {sigma:.2f} sigma mot grannen", "label": SIGNAL_LABEL,
    }


def _symbol(node: Node) -> str | None:
    series = node.attributes.get("series")
    value = series.get("symbol") if isinstance(series, Mapping) else None
    return value if isinstance(value, str) and value else None


def _metrics(excerpt: str | None) -> dict[str, Any] | None:
    if not excerpt:
        return None
    try:
        data = json.loads(excerpt)
    except json.JSONDecodeError:
        return None
    metrics = data.get("metrics") if isinstance(data, dict) else None
    return metrics if isinstance(metrics, dict) else None


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


class RelatedService:
    def __init__(
        self,
        graph: GraphPort,
        evidence_store: EvidencePort,
        price_store: PricePort,
        *,
        policy: DiscoverPolicy = DiscoverPolicy(),
    ) -> None:
        self._graph = graph
        self._evidence = evidence_store
        self._prices = price_store
        self._policy = policy

    # -- public -----------------------------------------------------------------

    def build_related(self, symbol: str, as_of: date, include_hypothesis: bool = False) -> dict[str, Any]:
        sym = symbol.strip().upper()
        queried = self._security_node(sym)
        queried_prices = self._prices_until(sym, as_of)
        if queried is None and queried_prices is None:
            raise UnknownSymbol(sym)
        queried_returns = self._returns(queried_prices)

        statuses = _WITH_HYPOTHESIS if include_hypothesis else _VALIDATED_ONLY
        paths = self._graph.traverse(
            NodeKey(queried.node_type, queried.key), max_depth=1, direction=Direction.BOTH, statuses=statuses,
        ) if queried is not None else []
        rows: list[dict[str, Any]] = []
        neighbour_cache: dict[str, pd.Series | None] = {}
        seen: set[int] = set()
        for path in sorted(paths, key=lambda p: (p.end.node_type.value, p.end.key, p.edges[0].id)):
            edge = path.edges[0]
            neighbour = path.end
            if edge.id in seen or edge.status not in statuses or neighbour.id == queried.id:
                continue
            seen.add(edge.id)
            role = "src" if edge.src_id == queried.id else "dst"
            rows.append(self._safe_row(
                sym, as_of, queried, queried_returns, edge, neighbour, role, neighbour_cache,
            ))

        return {
            "symbol": sym,
            "as_of": as_of.isoformat(),
            "include_hypothesis": include_hypothesis,
            "queried": _node_json(queried),
            "rows": rows,
            "meta": {
                "as_of": as_of.isoformat(),
                "policy_version": {"related": RELATED_VERSION, "divergence": self._policy.version},
                "counts": _counts(rows),
                "disclaimer": DISCLAIMER,
            },
        }

    # -- lookups ----------------------------------------------------------------

    def _security_node(self, sym: str) -> Node | None:
        for node_type in _LOOKUP_TYPES:
            node = self._graph.get_node(NodeKey(node_type, sym))
            if node is not None:
                return node
        return None

    def _prices_until(self, symbol: str, as_of: date) -> pd.Series | None:
        """Close prices up to ``as_of`` from the first provider with stored bars; None when there are none."""
        for provider in _PROVIDERS:
            series = self._prices.series(provider, symbol, end=as_of)
            if series is not None and series.bars:
                return prices_to_series(series, as_of)
        return None

    @staticmethod
    def _returns(prices: pd.Series | None) -> pd.Series | None:
        if prices is None or len(prices) < 2:
            return None
        return log_returns(prices)

    # -- rows -------------------------------------------------------------------

    def _safe_row(
        self,
        sym: str,
        as_of: date,
        queried: Node | None,
        queried_returns: pd.Series | None,
        edge: Edge,
        neighbour: Node,
        role: str,
        cache: dict[str, pd.Series | None],
    ) -> dict[str, Any]:
        try:
            return self._row(sym, as_of, queried, queried_returns, edge, neighbour, role, cache)
        except Exception:  # one neighbour must not fail the whole response
            logger.exception("related row failed for edge %s", edge.id)
            failed = _unavailable("internt fel vid beräkning av raden")
            return _assemble(edge, neighbour, role, lag=failed, divergence=failed, signal=_abstain(failed["reason"]))

    def _row(
        self,
        sym: str,
        as_of: date,
        queried: Node | None,
        queried_returns: pd.Series | None,
        edge: Edge,
        neighbour: Node,
        role: str,
        cache: dict[str, pd.Series | None],
    ) -> dict[str, Any]:
        queried_is_src = role == "src"
        validated = edge.status is EdgeStatus.VALIDATED
        lag, queried_direction = self._lag(edge, queried, neighbour, queried_is_src, as_of)

        n_sym = _symbol(neighbour)
        beta: float | None = None
        if n_sym is None:
            divergence = {**_unavailable("grannen saknar prisserie"), "subject": "neighbour"}
        else:
            if n_sym not in cache:
                cache[n_sym] = self._returns(self._prices_until(n_sym, as_of))
            divergence, beta = self._divergence(sym, queried_returns, n_sym, cache[n_sym])

        sigma = divergence.get("sigma")
        signal = signal_for(
            validated=validated,
            queried_direction=queried_direction,
            sigma=sigma,
            divergence_reason=divergence.get("reason"),
            sign=relation_sign(edge.attributes.get("expected_sign"), beta),
            threshold=self._policy.divergence_z,
        )
        return _assemble(edge, neighbour, role, lag=lag, divergence=divergence, signal=signal)

    def _lag(
        self, edge: Edge, queried: Node | None, neighbour: Node, queried_is_src: bool, as_of: date,
    ) -> tuple[dict[str, Any], str | None]:
        if edge.status is not EdgeStatus.VALIDATED or queried is None:
            return _unavailable(_UNVALIDATED), None
        src, dst = (queried, neighbour) if queried_is_src else (neighbour, queried)
        ref = RelationshipRef(NodeKey(src.node_type, src.key), NodeKey(dst.node_type, dst.key), edge.relation_type)
        cut = datetime.combine(as_of, time.max, tzinfo=timezone.utc)
        items = [
            e for e in self._evidence.list_evidence(ref, as_of=cut)
            if e.source_type is SourceType.STATISTICAL_TEST and e.retracted_at is None
        ]
        if not items:
            return _unavailable(_UNVALIDATED), None
        latest = max(items, key=lambda e: (e.available_at, e.id))
        metrics = _metrics(latest.excerpt)
        if metrics is None or "best_lag" not in metrics or "direction" not in metrics:
            return _unavailable("valideringsutdraget saknar lag"), None
        direction = translate_direction(str(metrics["direction"]), queried_is_src)
        lag = {
            "status": "available",
            "best_lag_days": abs(int(metrics["best_lag"])),
            "direction": direction,
            "r": metrics.get("r"),
            "partial_r": metrics.get("partial_r"),
            "p_corrected": metrics.get("p_corrected"),
            "validated_through": metrics.get("end"),
        }
        return lag, direction

    def _divergence(
        self, sym: str, queried_returns: pd.Series | None, n_sym: str, neighbour_returns: pd.Series | None,
    ) -> tuple[dict[str, Any], float | None]:
        window = self._policy.relationship_recent_days
        if queried_returns is None:
            return {**_unavailable(f"saknar kurshistorik för {sym}"), "subject": "neighbour"}, None
        if neighbour_returns is None:
            return {**_unavailable(f"ingen lagrad kurs för {n_sym}"), "subject": "neighbour"}, None
        fit = relationship_divergence(queried_returns, neighbour_returns, self._policy)
        if fit is None:
            need = self._policy.relationship_fit_days + window
            return {**_unavailable(f"för kort historik: behöver {need} gemensamma avkastningar"),
                    "subject": "neighbour"}, None
        beta, z = fit
        return {
            "status": "available", "subject": "neighbour", "reference": "queried",
            "sigma": round(z, 4), "beta": round(beta, 6), "window_days": window, "reason": None,
        }, beta


def _node_json(node: Node | None) -> dict[str, Any] | None:
    if node is None:
        return None
    return {"node_type": node.node_type.value, "key": node.key, "name": node.name, "symbol": _symbol(node)}


def _assemble(
    edge: Edge, neighbour: Node, role: str, *, lag: dict[str, Any], divergence: dict[str, Any], signal: dict[str, Any],
) -> dict[str, Any]:
    return {
        "edge_id": edge.id,
        "neighbour": _node_json(neighbour),
        "role": role,
        "relation_type": edge.relation_type.value,
        "status": edge.status.value,
        "source": edge.source,
        "expected_sign": edge.attributes.get("expected_sign"),
        "lag": lag,
        "divergence": divergence,
        "signal": signal,
    }


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "edges": len(rows),
        "validated": sum(1 for r in rows if r["status"] == EdgeStatus.VALIDATED.value),
        "hypothesis": sum(1 for r in rows if r["status"] == EdgeStatus.HYPOTHESIS.value),
        "lag_available": sum(1 for r in rows if r["lag"]["status"] == "available"),
        "divergence_available": sum(1 for r in rows if r["divergence"]["status"] == "available"),
        "signals_available": sum(1 for r in rows if r["signal"]["status"] == "available"),
    }
