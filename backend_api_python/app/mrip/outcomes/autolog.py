"""Automatic, point-in-time prediction logging (work 023).

Logs predictions made by EXISTING engines; no new modelling happens here.

Prediction kinds
- FORECAST_SCENARIO (``PredictionType.FORECAST``): ensemble quantiles of the simple return for
  1w/1m/3m/6m/12m, from the same provider the Research Card uses (``default_forecast_provider``).
- DISCOVER_ITEM (``DISCOVER_ITEM``): each saved Discover item of the day, for 1w and 1m. The item's
  expected direction is logged only when the detector defines one (``details.direction`` up/down).
- RELATED_SIGNAL (``RELATED_SIGNAL``): every ``signal.status == "available"`` up/down row of the
  related service, for 1w and 1m.

Point-in-time rules
- The data date is the last stored bar on or before ``as_of``. Every engine only sees prices up to
  that bar, and the prediction is made at 23:00 UTC of that date with the close of that bar as
  entry. Bars older than ``MAX_STALE_DAYS`` before ``as_of`` skip the symbol (no guessing).
- Symbols need at least ``MIN_OBS`` stored closes; otherwise they are skipped and counted.
- Backfill (``as_of`` before today) logs FORECAST_SCENARIO only. The relationship graph, its
  validation evidence and the Discover detectors are current versions, not point-in-time, so
  DISCOVER_ITEM and RELATED_SIGNAL rows are never backfilled. Backfilled rows carry
  ``payload.backfill = true`` and ``lineage.backfill = true``; they can be excluded or weighted
  separately later.

Attention rule (DISCOVER_ITEM), fixed and documented: at logging time the threshold is the median of
|h-day simple return| over the last ``ATTENTION_WINDOW`` (252) overlapping h-day returns ending on the
data date. An outcome is an attention hit when the realised |return| over the horizon is strictly
larger than that threshold. The naive baseline for attention is 0.5 (the median by construction).

Idempotency: the unique key (type, subject, made_at, horizon, model_version) makes a second run for the
same data date a no-op. Discriminators (Discover kind, related edge) are part of model_version.
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from app.mrip.data.models import PriceSeries
from app.mrip.forecast.types import ForecastProvider, ForecastRequest, ForecastUnavailable
from app.mrip.outcomes import resolve
from app.mrip.outcomes.predictions import forecast_prediction
from app.mrip.outcomes.store import PredictionStore
from app.mrip.outcomes.types import HorizonKind, NewPrediction, PredictionType
from app.mrip.relationships.types import NodeType
from app.mrip.stats.service import prices_to_series

AUTOLOG_VERSION = "prediction-log-v1"
POLICY_VERSION = "outcome-policy-v1"
BENCHMARK = "SPY"
MIN_OBS = 300
MAX_STALE_DAYS = 4
ATTENTION_WINDOW = 252
ATTENTION_RULE = "median(|h-day return|) over last 252 overlapping returns ending on the data date"
FORECAST_HORIZON_DAYS = (5, 21, 63, 126, 252)
ALERT_HORIZONS = (HorizonKind.W1, HorizonKind.M1)
# One provider per symbol: the one whose latest bar is most recent (ties: yahoo first). yahoo closes are
# dividend-adjusted and cboe closes are not, and cboe stops earlier, so mixing them inside one prediction
# would put dividend gaps into returns. The choice is recorded in lineage.price_provider.
PRICE_PROVIDERS = ("yahoo", "cboe")


def select_series(store: Any, symbol: str, end: date | None = None) -> tuple[str, PriceSeries] | None:
    """The stored series with the most recent bar on or before ``end`` (see PRICE_PROVIDERS)."""
    best: tuple[str, PriceSeries, date] | None = None
    for provider in PRICE_PROVIDERS:
        series = store.series(provider, symbol, end=end)
        if series is None or not series.bars:
            continue
        last = series.bars[-1].ts
        last_day = last.date() if isinstance(last, datetime) else last
        if best is None or last_day > best[2]:
            best = (provider, series, last_day)
    return None if best is None else (best[0], best[1])

KIND_LABEL = {
    PredictionType.FORECAST: "FORECAST_SCENARIO",
    PredictionType.DISCOVER_ITEM: "DISCOVER_ITEM",
    PredictionType.RELATED_SIGNAL: "RELATED_SIGNAL",
}


class SkipReason:
    NO_PRICES = "no_prices"
    INSUFFICIENT_HISTORY = "insufficient_history"
    STALE_PRICES = "stale_prices"
    PAIR_SUBJECT = "pair_subject"
    NOT_IN_SCOPE = "not_in_scope"
    FORECAST_UNAVAILABLE = "forecast_unavailable"


@dataclass(slots=True)
class LogReport:
    as_of: date
    backfill: bool
    dry_run: bool
    symbols: int = 0
    logged: Counter = field(default_factory=Counter)
    already_logged: Counter = field(default_factory=Counter)
    would_log: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(), "backfill": self.backfill, "dry_run": self.dry_run,
            "symbols": self.symbols, "logged": dict(self.logged), "already_logged": dict(self.already_logged),
            "would_log": dict(self.would_log), "skipped": dict(self.skipped),
        }


@dataclass(frozen=True, slots=True)
class LoadedPrices:
    provider: str
    prices: pd.Series
    data_date: date


def attention_threshold(prices: pd.Series, horizon_days: int) -> float:
    """Median |h-day simple return| over the trailing ATTENTION_WINDOW returns ending on the last date."""
    closes = prices.to_numpy(dtype=float)
    if len(closes) < ATTENTION_WINDOW + horizon_days + 1:
        raise ValueError("not enough history for the attention threshold")
    returns = closes[horizon_days:] / closes[:-horizon_days] - 1.0
    return float(np.median(np.abs(returns[-ATTENTION_WINDOW:])))


def inputs_hash(symbol: str, provider: str, prices: pd.Series) -> str:
    digest = hashlib.sha256()
    digest.update(f"{symbol}|{provider}|{prices.index[0].date()}|{prices.index[-1].date()}|{len(prices)}".encode())
    digest.update(np.ascontiguousarray(prices.to_numpy(dtype=float)).tobytes())
    return digest.hexdigest()


def _made_at(data_date: date) -> datetime:
    return datetime.combine(data_date, time(23, 0), tzinfo=timezone.utc)


def _symbol_of(node: Any) -> str | None:
    series = node.attributes.get("series") if isinstance(node.attributes, Mapping) else None
    value = series.get("symbol") if isinstance(series, Mapping) else None
    return value.strip().upper() if isinstance(value, str) and value.strip() else None


class PredictionLogger:
    def __init__(
        self,
        store: PredictionStore,
        *,
        price_store: Any,
        graph: Any = None,
        discover_store: Any = None,
        related_service: Any = None,
        regime_engine: Any = None,
        forecast_provider: ForecastProvider | None = None,
    ) -> None:
        from app.mrip.research.card import default_forecast_provider

        self._store = store
        self._prices = price_store
        self._graph = graph
        self._discover = discover_store
        self._related = related_service
        self._regime = regime_engine
        self._forecast = forecast_provider or default_forecast_provider()

    # -- public -------------------------------------------------------------

    def default_symbols(self, as_of: date) -> list[str]:
        """Graph COMPANY/SECURITY nodes with a series symbol, Discover item subjects, and SPY."""
        found: set[str] = {BENCHMARK}
        if self._graph is not None:
            for node in self._graph.list_priced_nodes():
                if node.node_type in (NodeType.COMPANY, NodeType.SECURITY):
                    sym = _symbol_of(node)
                    if sym:
                        found.add(sym)
        if self._discover is not None:
            for item in self._discover.feed(as_of, limit=500):
                if "->" not in item.subject:
                    found.add(item.subject.strip().upper())
        return sorted(found)

    def run(
        self,
        as_of: date,
        *,
        symbols: Sequence[str] | None = None,
        limit: int | None = None,
        dry_run: bool = False,
        backfill: bool | None = None,
    ) -> LogReport:
        backfill = (as_of < date.today()) if backfill is None else backfill
        report = LogReport(as_of=as_of, backfill=backfill, dry_run=dry_run)
        universe = [s.strip().upper() for s in symbols] if symbols is not None else self.default_symbols(as_of)
        universe = list(dict.fromkeys(universe))
        if limit is not None:  # the benchmark is always kept: without it no benchmark-judged row can resolve
            rest = [s for s in universe if s != BENCHMARK]
            universe = ([BENCHMARK] if BENCHMARK in universe else []) + rest
            universe = universe[: max(0, limit)]
        report.symbols = len(universe)
        scope = set(universe)

        regime = self._regime_context(as_of)
        for sym in universe:
            try:
                self._log_symbol(sym, as_of, backfill, regime, report, dry_run)
            except Exception as exc:  # one symbol must not sink the run
                report.skipped[f"error:{type(exc).__name__}"] += 1
        if not backfill and self._discover is not None:
            self._log_discover(as_of, scope, regime, report, dry_run)
        return report

    # -- internals ----------------------------------------------------------

    def _series(self, symbol: str, as_of: date) -> tuple[str, PriceSeries] | None:
        return select_series(self._prices, symbol, as_of)

    def _load(self, symbol: str, as_of: date, report: LogReport) -> LoadedPrices | None:
        found = self._series(symbol, as_of)
        if found is None:
            report.skipped[SkipReason.NO_PRICES] += 1
            return None
        provider, series = found
        prices = prices_to_series(series, as_of)
        if len(prices) < MIN_OBS:
            report.skipped[SkipReason.INSUFFICIENT_HISTORY] += 1
            return None
        data_date = prices.index[-1].date()
        if (as_of - data_date).days > MAX_STALE_DAYS:
            report.skipped[SkipReason.STALE_PRICES] += 1
            return None
        return LoadedPrices(provider, prices, data_date)

    def _regime_context(self, as_of: date) -> dict[str, Any] | None:
        if self._regime is None:
            return None
        try:
            regime = self._regime.at(as_of)
        except Exception:  # the regime is context; a failure must not block logging
            return None
        return {"regime": regime.regime.value, "engine": type(self._regime).__name__}

    def _lineage(self, loaded: LoadedPrices, symbol: str, as_of: date, backfill: bool,
                 regime: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
        return {
            "logger": AUTOLOG_VERSION, "policy": POLICY_VERSION, "as_of": as_of.isoformat(),
            "data_as_of": loaded.data_date.isoformat(), "inputs_hash": inputs_hash(symbol, loaded.provider, loaded.prices),
            "price_provider": loaded.provider, "regime_version": regime and f"{regime['engine']}:{regime['regime']}",
            "backfill": backfill, **extra,
        }

    def _persist(self, predictions: list[NewPrediction], report: LogReport, dry_run: bool) -> None:
        for p in predictions:
            label = KIND_LABEL[p.prediction_type]
            if dry_run:
                report.would_log[label] += 1
                continue
            _, created = self._store.log_with_status(p)
            (report.logged if created else report.already_logged)[label] += 1

    def _log_symbol(self, sym: str, as_of: date, backfill: bool, regime: dict[str, Any] | None,
                    report: LogReport, dry_run: bool) -> None:
        loaded = self._load(sym, as_of, report)
        if loaded is None:
            return
        predictions: list[NewPrediction] = []
        for h in FORECAST_HORIZON_DAYS:
            try:
                result = self._forecast.forecast(ForecastRequest(symbol=sym, prices=loaded.prices, horizon_days=h))
            except (ForecastUnavailable, ValueError):
                report.skipped[SkipReason.FORECAST_UNAVAILABLE] += 1
                continue
            base = forecast_prediction(result, benchmark=BENCHMARK, market_regime=regime)
            predictions.append(dataclasses.replace(
                base,
                payload={**base.payload, "backfill": backfill},
                lineage=self._lineage(loaded, sym, as_of, backfill, regime, forecast_version=result.provider_version,
                                      horizon_days=h),
            ))
        if not backfill and self._related is not None:
            predictions.extend(self._related_predictions(sym, as_of, loaded, regime, report))
        self._persist(predictions, report, dry_run)

    def _related_predictions(self, sym: str, as_of: date, loaded: LoadedPrices,
                             regime: dict[str, Any] | None, report: LogReport) -> list[NewPrediction]:
        from app.mrip.related.types import RELATED_VERSION

        out: list[NewPrediction] = []
        built = self._related.build_related(sym, loaded.data_date)
        for row in built.get("rows", []):
            signal = row.get("signal") or {}
            if signal.get("status") != "available" or signal.get("direction") not in ("up", "down"):
                continue
            for horizon in ALERT_HORIZONS:
                out.append(NewPrediction(
                    prediction_type=PredictionType.RELATED_SIGNAL,
                    subject=sym,
                    benchmark=BENCHMARK,
                    made_at=_made_at(loaded.data_date),
                    horizon_kind=horizon,
                    model_version=f"{RELATED_VERSION}|edge-{row['edge_id']}",
                    payload={
                        "direction": signal["direction"], "basis": signal.get("basis"),
                        "reason": signal.get("reason"), "edge_id": row["edge_id"],
                        "relation_type": row.get("relation_type"), "expected_sign": row.get("expected_sign"),
                        "neighbour": row.get("neighbour"), "sigma": (row.get("divergence") or {}).get("sigma"),
                        "backfill": False,
                    },
                    lineage=self._lineage(loaded, sym, as_of, False, regime, policy_related=RELATED_VERSION),
                ))
        return out

    def _log_discover(self, as_of: date, scope: set[str], regime: dict[str, Any] | None,
                      report: LogReport, dry_run: bool) -> None:
        for item in self._discover.feed(as_of, limit=500):
            subject = item.subject.strip().upper()
            if "->" in subject:
                report.skipped[SkipReason.PAIR_SUBJECT] += 1
                continue
            if subject not in scope:
                report.skipped[SkipReason.NOT_IN_SCOPE] += 1
                continue
            loaded = self._load(subject, as_of, report)
            if loaded is None:
                continue
            direction = item.details.get("direction") if isinstance(item.details, Mapping) else None
            direction = direction if direction in ("up", "down") else None
            predictions = []
            for horizon in ALERT_HORIZONS:
                threshold = attention_threshold(loaded.prices, _horizon_days(horizon))
                predictions.append(NewPrediction(
                    prediction_type=PredictionType.DISCOVER_ITEM,
                    subject=subject,
                    benchmark=BENCHMARK,
                    made_at=_made_at(loaded.data_date),
                    horizon_kind=horizon,
                    model_version=f"{item.policy_version}|{item.kind.value}",
                    payload={
                        "kind": item.kind.value, "score": item.score, "magnitude": item.magnitude,
                        "headline": item.headline, "direction": direction, "attention_threshold": threshold,
                        "attention_rule": ATTENTION_RULE, "discover_item_id": item.id, "backfill": False,
                    },
                    lineage=self._lineage(loaded, subject, as_of, False, regime, policy_discover=item.policy_version),
                ))
            self._persist(predictions, report, dry_run)


def _horizon_days(kind: HorizonKind) -> int:
    return resolve.HORIZON_BARS[kind.value]


__all__ = [
    "AUTOLOG_VERSION", "ATTENTION_RULE", "BENCHMARK", "KIND_LABEL", "LogReport", "POLICY_VERSION",
    "PredictionLogger", "attention_threshold", "inputs_hash", "select_series",
]
