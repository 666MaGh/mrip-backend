"""Research Card service: every collaborator is a fake; no network, no database."""
from __future__ import annotations

import socket
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.mrip.api.serializers import research_card_json
from app.mrip.cot.engine import CotMarket, CotSnapshot
from app.mrip.data.models import Latency, OptionContract, OptionsChainSnapshot, PriceBar, PriceSeries, Provenance
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType, Stance
from app.mrip.forecast.types import ForecastUnavailable
from app.mrip.options.analysis import MODELED_LABEL
from app.mrip.relationships.types import Direction, Edge, EdgeStatus, Node, NodeKey, NodeType, Path, RelationType
from app.mrip.research.card import SECTION_ORDER, ResearchCardService
from app.mrip.research.types import CARD_VERSION, UnknownSecurity

AS_OF = date(2026, 10, 1)
NOW = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)
UNAVAILABLE_BY_DEFAULT = {"forecast_confidence", "historical_reliability"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any socket connection in these tests is a bug."""
    def refuse(self, *args, **kwargs):
        raise AssertionError("research card tests must not touch the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)


# -- graph ------------------------------------------------------------------

THEME = Node(1, NodeType.THEME, "ai-infrastructure", "AI infrastructure", {"series": {"symbol": "SMH"}})
NVDA = Node(2, NodeType.COMPANY, "NVDA", "NVIDIA", {"series": {"symbol": "NVDA"}})
MSFT = Node(3, NodeType.COMPANY, "MSFT", "Microsoft", {"series": {"symbol": "MSFT"}})
E_DIRECT = Edge(10, 2, 1, RelationType.BENEFITS_FROM, EdgeStatus.VALIDATED, "seed", 1, attributes={"expected_sign": 1})
E_CUSTOMER = Edge(11, 3, 2, RelationType.CUSTOMER_OF, EdgeStatus.HYPOTHESIS, "seed", 1, attributes={"expected_sign": 1})
E_MSFT_THEME = Edge(12, 3, 1, RelationType.BENEFITS_FROM, EdgeStatus.HYPOTHESIS, "seed", 1)
PATH_DIRECT = Path(nodes=(NVDA, THEME), edges=(E_DIRECT,))
PATH_INDIRECT = Path(nodes=(NVDA, MSFT, THEME), edges=(E_CUSTOMER, E_MSFT_THEME))


class FakeGraph:
    def __init__(self, paths: list[Path] | None = None, nodes: tuple[Node, ...] = (NVDA, THEME, MSFT)) -> None:
        self.paths = [PATH_DIRECT, PATH_INDIRECT] if paths is None else paths
        self.nodes = nodes
        self.traverse_calls: list[dict[str, Any]] = []

    def get_node(self, node: NodeKey) -> Node | None:
        for n in self.nodes:
            if n.node_type is node.node_type and n.key == node.key:
                return n
        return None

    def traverse(self, start: NodeKey, *, max_depth: int = 3, direction: Direction = Direction.OUT, **_: Any) -> list[Path]:
        self.traverse_calls.append({"start": start, "max_depth": max_depth, "direction": direction})
        return list(self.paths)


# -- prices -----------------------------------------------------------------

def _bars(seed: int, n: int = 400, drift: float = 0.0004, vol: float = 0.02) -> tuple[PriceBar, ...]:
    rng = np.random.default_rng(seed)
    days = [ts.date() for ts in pd.bdate_range(end=AS_OF, periods=n)]
    closes = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    return tuple(PriceBar(ts=d, open=float(c), high=float(c), low=float(c), close=float(c), volume=1e6)
                 for d, c in zip(days, closes))


class FakePrices:
    def __init__(self, bars_by_symbol: dict[str, tuple[PriceBar, ...]]) -> None:
        self.bars_by_symbol = bars_by_symbol
        self.calls: list[tuple[str, str]] = []

    def series(self, provider: str, symbol: str, start=None, end=None) -> PriceSeries | None:
        self.calls.append((provider, symbol))
        bars = self.bars_by_symbol.get(symbol)
        if not bars:
            return None
        prov = Provenance(provider=provider, gateway="mrip-store", endpoint="mrip_price_bars",
                          fetched_at=NOW, latency=Latency.UNKNOWN)
        return PriceSeries(symbol=symbol, interval="1d", bars=bars, provenance=prov)


def _default_prices() -> FakePrices:
    return FakePrices({"NVDA": _bars(1), "SMH": _bars(2), "MSFT": _bars(3)})


# -- evidence ---------------------------------------------------------------

def _evidence_items() -> list[Evidence]:
    ts = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
    validation = Evidence(
        id=101, src_id=2, dst_id=1, relation_type=RelationType.BENEFITS_FROM, stance=Stance.SUPPORT,
        source_type=SourceType.STATISTICAL_TEST,
        source_uri="mrip://validation/relationship-validation-v1/NVDA->SMH/BENEFITS_FROM",
        available_at=ts, ingested_at=ts, assessed_by="statistics",
        excerpt='{"verdict": "validated", "reasons": ["beta significant"], "metrics": {"p_value": 0.01}}',
    )
    transcript = Evidence(
        id=102, src_id=2, dst_id=1, relation_type=RelationType.BENEFITS_FROM, stance=Stance.SUPPORT,
        source_type=SourceType.EARNINGS_TRANSCRIPT, source_uri="https://example.test/t", available_at=ts,
        ingested_at=ts, assessed_by="laya-test", assessor_confidence=0.6,
    )
    return [validation, transcript]


class FakeEvidence:
    def __init__(self, items: list[Evidence] | None = None, error: Exception | None = None) -> None:
        self.items = _evidence_items() if items is None else items
        self.error = error
        self.refs: list[RelationshipRef] = []

    def list_evidence(self, ref: RelationshipRef, **_: Any) -> list[Evidence]:
        self.refs.append(ref)
        if self.error is not None:
            raise self.error
        return list(self.items)


# -- regime, COT, options, stores ---------------------------------------------

@dataclass
class FakeRegime:
    regime: Any = field(default_factory=lambda: SimpleNamespace(value="NORMAL"))
    vix: float = 16.5
    vix_percentile: float | None = 0.4
    term_backwardation: bool | None = False
    unavailable: tuple[str, ...] = ()
    policy_version: str = "regime-v0-uncalibrated"


class FakeRegimeEngine:
    def at(self, as_of: date) -> FakeRegime:
        return FakeRegime()


class FakeCot:
    def __init__(self, proxy: str = "SMH", percentile_3y: float | None = 0.8) -> None:
        self.market = CotMarket("099001", "Semiconductors", proxy)
        self.percentile_3y = percentile_3y

    def markets(self) -> tuple[CotMarket, ...]:
        return (self.market,)

    def snapshot(self, market: Any, as_of: date | None = None, **_: Any) -> CotSnapshot:
        return CotSnapshot(
            market=self.market, group="managed_money", as_of=as_of or AS_OF, report_date=date(2026, 9, 29),
            available_date=date(2026, 10, 2), net_position=1200.0, net_change=50.0, net_pct_oi=0.2,
            open_interest=9000.0, percentile_1y=0.7, percentile_3y=self.percentile_3y, percentile_5y=0.6,
            zscore=0.9, crowding_score=0.1, crowding_side=None, crowded=False, price_position_divergence=0.5,
        )


def _chain(underlying: str = "NVDA") -> OptionsChainSnapshot:
    exp = date(2026, 10, 16)
    prov = Provenance("cboe", "openbb", "cboe.options.chains", NOW, Latency.DELAYED)
    contracts = (
        OptionContract("C100", exp, 100.0, "call", open_interest=1000, volume=300, implied_volatility=0.2, gamma=0.02),
        OptionContract("P100", exp, 100.0, "put", open_interest=800, volume=150, implied_volatility=0.22, gamma=0.02),
        OptionContract("C105", exp, 105.0, "call", open_interest=500, volume=50, implied_volatility=0.19, gamma=0.01),
    )
    return OptionsChainSnapshot(
        underlying=underlying, underlying_price=100.0, underlying_timestamp=NOW, snapshot_timestamp=NOW,
        oi_effective_date=date(2026, 9, 30), contracts=contracts, provenance=prov,
    )


class FakeSnapshots:
    def __init__(self, chain: OptionsChainSnapshot | None) -> None:
        self.chain = chain

    def latest_before(self, underlying: str, as_of: datetime):
        if self.chain is None or underlying != self.chain.underlying:
            return None
        return self.chain


class FakeOutcomes:
    def __init__(self, resolved: list | None = None) -> None:
        self.resolved = resolved or []
        self.calls: list[dict[str, Any]] = []

    def resolved_pairs(self, prediction_type, *, subject=None):
        self.calls.append({"type": prediction_type, "subject": subject})
        return list(self.resolved)


class FakeCalibrationStore:
    def latest(self, kind):
        return None


class FailingForecast:
    def forecast(self, request):
        raise ForecastUnavailable("no members")

    def version(self) -> str:
        return "failing-v0"


# -- service factory --------------------------------------------------------

def build_service(**overrides: Any) -> ResearchCardService:
    kwargs: dict[str, Any] = {
        "graph": FakeGraph(),
        "evidence_store": FakeEvidence(),
        "price_store": _default_prices(),
        "regime_engine": FakeRegimeEngine(),
        "cot_engine": FakeCot(),
        "snapshot_store": FakeSnapshots(_chain()),
        "outcome_store": FakeOutcomes(),
        "calibration_store": FakeCalibrationStore(),
    }
    kwargs.update(overrides)
    return ResearchCardService(**kwargs)


def _statuses(card: dict[str, Any]) -> dict[str, str]:
    return {name: card[name]["status"] for name in SECTION_ORDER}


# -- happy path ---------------------------------------------------------------

def test_happy_path_composes_every_section_with_provenance():
    card = build_service().build_card("nvda", AS_OF)

    assert card["symbol"] == "NVDA"
    assert card["as_of"] == "2026-10-01"
    assert card["card_version"] == CARD_VERSION
    statuses = _statuses(card)
    for name in SECTION_ORDER:
        expected = "unavailable" if name in UNAVAILABLE_BY_DEFAULT else "available"
        assert statuses[name] == expected, name

    assert card["theme"]["name"] == "AI infrastructure"
    assert card["relationship"]["paths"][0]["nodes"][-1]["key"] == "ai-infrastructure"
    assert card["relationship_confidence"]["validation"]["verdict"] == "validated"
    assert card["relationship_confidence"]["confidence"]["status"] == "unavailable"
    assert card["evidence"]["level"] == "multi_source"
    assert card["evidence"]["total"] == 2
    assert card["cot"]["stance"] == "confirming"
    assert card["vix_regime"]["regime"] == "NORMAL"
    assert isinstance(card["divergence"]["sigma"], float)

    why = {entry["section"]: entry for entry in card["why"]}
    assert set(why) == set(SECTION_ORDER)
    assert why["evidence"]["evidence_ids"] == [101, 102]
    assert why["relationship_confidence"]["evidence_ids"] == [101]
    assert why["potential_12m"]["versions"]
    assert all(isinstance(entry["inputs"], list) for entry in card["why"])


def test_potential_is_a_scenario_distribution_with_ordered_percentiles():
    card = build_service().build_card("NVDA", AS_OF)
    potential = card["potential_12m"]

    assert potential["horizon_days"] == 252
    assert "Not a price target" in potential["definition"]
    assert potential["bear"]["return"] <= potential["base"]["return"] <= potential["bull"]["return"]
    p = potential["percentiles"]
    ordered = [p["P10"], p["P25"], p["P50"], p["P75"], p["P90"]]
    returns = [item["return"] for item in ordered]
    prices = [item["price"] for item in ordered]
    assert returns == sorted(returns)
    assert prices == sorted(prices)
    assert potential["bear"]["return"] == p["P25"]["return"]
    assert potential["base"]["return"] == p["P50"]["return"]
    assert potential["bull"]["return"] == p["P75"]["return"]
    assert potential["bear"]["price"] == pytest.approx(potential["last_price"] * (1 + potential["bear"]["return"]))


def test_divergence_uses_discover_fit_and_reports_pair():
    card = build_service().build_card("NVDA", AS_OF)
    assert card["divergence"]["pair"] == "NVDA->SMH"
    assert card["divergence"]["edge_id"] == 10
    assert card["divergence"]["policy_version"] == "discover-v0-uncalibrated"


def test_unavailable_reasons_are_explicit_for_reliability_and_confidence():
    card = build_service().build_card("NVDA", AS_OF)
    assert card["forecast_confidence"] == {
        "status": "unavailable", "reason": "no fitted forecast calibration stored; forecast is uncalibrated",
    }
    assert card["historical_reliability"]["status"] == "unavailable"
    assert card["historical_reliability"]["n_resolved_outcomes"] == 0


def test_forecast_confidence_stays_unavailable_without_store():
    card = build_service(calibration_store=None).build_card("NVDA", AS_OF)
    assert card["forecast_confidence"]["status"] == "unavailable"
    assert "no calibration store" in card["forecast_confidence"]["reason"]


# -- independent degradation ---------------------------------------------------

def _ok_sections_except(card: dict[str, Any], broken: set[str]) -> None:
    for name in SECTION_ORDER:
        if name in broken or name in UNAVAILABLE_BY_DEFAULT:
            continue
        assert card[name]["status"] == "available", name


def test_no_graph_path_degrades_graph_sections_only():
    card = build_service(graph=FakeGraph(paths=[])).build_card("NVDA", AS_OF)
    # COT is matched through the theme proxy, so it degrades with the graph too.
    graph_sections = {"theme", "relationship", "relationship_confidence", "evidence", "divergence", "cot"}
    for name in graph_sections:
        assert card[name]["status"] == "unavailable", name
        assert card[name]["reason"]
    _ok_sections_except(card, graph_sections)


def test_evidence_store_failure_degrades_evidence_and_confidence_only():
    card = build_service(evidence_store=FakeEvidence(error=RuntimeError("db down"))).build_card("NVDA", AS_OF)
    assert card["evidence"]["status"] == "unavailable"
    assert "db down" in card["evidence"]["reason"]
    assert card["relationship_confidence"]["status"] == "unavailable"
    _ok_sections_except(card, {"evidence", "relationship_confidence"})


def test_no_evidence_is_unavailable_and_confidence_has_no_validation():
    card = build_service(evidence_store=FakeEvidence(items=[])).build_card("NVDA", AS_OF)
    assert card["evidence"]["status"] == "unavailable"
    assert card["relationship_confidence"]["status"] == "unavailable"
    _ok_sections_except(card, {"evidence", "relationship_confidence"})


def test_missing_regime_engine_degrades_vix_only():
    card = build_service(regime_engine=None).build_card("NVDA", AS_OF)
    assert card["vix_regime"]["status"] == "unavailable"
    _ok_sections_except(card, {"vix_regime"})


def test_missing_cot_engine_degrades_cot_only():
    card = build_service(cot_engine=None).build_card("NVDA", AS_OF)
    assert card["cot"]["status"] == "unavailable"
    _ok_sections_except(card, {"cot"})


def test_cot_without_market_mapping_is_unavailable():
    card = build_service(cot_engine=FakeCot(proxy="SPY")).build_card("NVDA", AS_OF)
    assert card["cot"]["status"] == "unavailable"
    assert "no COT market" in card["cot"]["reason"]


def test_cot_without_percentile_is_unavailable_with_observed_context():
    card = build_service(cot_engine=FakeCot(percentile_3y=None)).build_card("NVDA", AS_OF)
    assert card["cot"]["status"] == "unavailable"
    assert card["cot"]["observed"]["market"] == "Semiconductors"


def test_missing_options_snapshot_degrades_options_only():
    card = build_service(snapshot_store=FakeSnapshots(None)).build_card("NVDA", AS_OF)
    assert card["options"]["status"] == "unavailable"
    assert "no stored options snapshot" in card["options"]["reason"]
    _ok_sections_except(card, {"options"})


def test_options_snapshot_for_another_symbol_is_not_used():
    card = build_service(snapshot_store=FakeSnapshots(_chain("SPY"))).build_card("NVDA", AS_OF)
    assert card["options"]["status"] == "unavailable"


def test_missing_theme_proxy_series_degrades_divergence_only():
    prices = FakePrices({"NVDA": _bars(1), "MSFT": _bars(3)})
    card = build_service(price_store=prices).build_card("NVDA", AS_OF)
    assert card["divergence"]["status"] == "unavailable"
    assert "SMH" in card["divergence"]["reason"]
    _ok_sections_except(card, {"divergence"})


def test_short_history_degrades_divergence_but_not_potential():
    prices = FakePrices({"NVDA": _bars(1, n=130), "SMH": _bars(2, n=130), "MSFT": _bars(3)})
    card = build_service(price_store=prices).build_card("NVDA", AS_OF)
    assert card["divergence"]["status"] == "unavailable"
    assert "insufficient history" in card["divergence"]["reason"]
    assert card["potential_12m"]["status"] == "available"


def test_missing_security_prices_degrade_potential_and_divergence():
    prices = FakePrices({"SMH": _bars(2), "MSFT": _bars(3)})
    card = build_service(price_store=prices).build_card("NVDA", AS_OF)
    assert card["potential_12m"]["status"] == "unavailable"
    assert card["divergence"]["status"] == "unavailable"
    _ok_sections_except(card, {"potential_12m", "divergence"})


def test_forecast_failure_degrades_potential_only():
    card = build_service(forecast_provider=FailingForecast()).build_card("NVDA", AS_OF)
    assert card["potential_12m"]["status"] == "unavailable"
    assert "forecast unavailable" in card["potential_12m"]["reason"]
    _ok_sections_except(card, {"potential_12m"})


def test_security_only_in_price_store_has_unavailable_graph_sections():
    graph = FakeGraph(nodes=())
    card = build_service(graph=graph).build_card("NVDA", AS_OF)
    assert card["security"]["graph_node_type"] is None
    assert card["relationship"]["status"] == "unavailable"
    assert "not in the relationship graph" in card["relationship"]["reason"]
    assert graph.traverse_calls == []
    _ok_sections_except(card, {"theme", "relationship", "relationship_confidence", "evidence", "divergence", "cot"})


def test_security_only_in_graph_has_unavailable_price_sections():
    card = build_service(price_store=FakePrices({})).build_card("NVDA", AS_OF)
    assert card["security"]["price_provider"] is None
    assert card["potential_12m"]["status"] == "unavailable"
    assert card["relationship"]["status"] == "available"


def test_unknown_to_price_store_and_graph_raises():
    service = build_service(graph=FakeGraph(nodes=()), price_store=FakePrices({}))
    with pytest.raises(UnknownSecurity):
        service.build_card("ZZZZ", AS_OF)


def test_graph_traversal_uses_both_directions_and_depth_limit():
    graph = FakeGraph()
    build_service(graph=graph).build_card("NVDA", AS_OF)
    call = graph.traverse_calls[-1]
    assert call["direction"] is Direction.BOTH
    assert call["max_depth"] == 3


# -- modeled / observed separation ----------------------------------------------

def test_options_observed_and_modeled_are_separated():
    card = research_card_json(build_service().build_card("NVDA", AS_OF))
    options = card["options"]
    assert options["status"] == "available"
    assert "observed" in options and "modeled" in options
    assert "gex" not in options["observed"]
    assert "gex" not in {k for k in options if k != "modeled"}
    assert options["modeled"]["label"] == MODELED_LABEL
    assert options["modeled"]["modeled"] is True
    assert "gex" in options["modeled"]
    assert "regime" in options["modeled"]
    assert "amplification" in options["modeled"]


def test_no_modeled_option_quantity_leaks_outside_modeled():
    card = research_card_json(build_service().build_card("NVDA", AS_OF))
    modeled_keys = {"gex", "amplification", "gamma_flip", "zero_dte_share"}
    for name in SECTION_ORDER:
        if name == "options":
            continue
        assert not (modeled_keys & set(card[name])), name
    assert not (modeled_keys & set(card["options"]["observed"]))


def test_divergence_and_potential_are_not_labelled_modeled_options():
    card = build_service().build_card("NVDA", AS_OF)
    assert "modeled" not in card["divergence"]
    assert "modeled" not in card["potential_12m"]
    assert card["potential_12m"]["definition"]
