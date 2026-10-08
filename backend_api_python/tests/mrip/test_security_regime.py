"""Security regime (work 020): classifier, discrepancy table, related integration and route contract.

Synthetic price series only; no network. Expectations are derived from the rules in
``app/mrip/security_regime/types.py`` and the discrepancy table in the task, not from the code.
"""
from __future__ import annotations

import socket
from dataclasses import FrozenInstanceError
from datetime import date

import numpy as np
import pandas as pd
import pytest

import app.routes.mrip as mrip_routes
import app.utils.auth as auth
from app.mrip.related.types import UnknownSymbol
from app.mrip.relationships.types import Edge, EdgeStatus, RelationType
from app.mrip.security_regime.features import _trend_labels, classify_security_regime, unavailable_regime
from app.mrip.security_regime.service import (
    DiscrepancyCode,
    SecurityRegimeService,
    find_discrepancies,
    regime_json,
)
from app.mrip.security_regime.types import (
    SECURITY_REGIME_VERSION,
    SecurityRegime,
    SecurityRegimePolicy,
    TrendLabel,
    TrendState,
    VolatilityState,
    VolLabel,
)
from app.mrip.related.service import RelatedService
from tests.mrip.test_api_mrip import FakeSnapshots, _chain
from tests.mrip.test_related import (
    AS_OF,
    FakeGraph,
    FakePrices,
    _bars,
    _edge,
    _prices,
    _row,
    _service,
    _validation,
)

AUTH = {"Authorization": "Bearer test-token"}
END = date(2026, 10, 8)
UP, DOWN, SIDE = TrendLabel.UPTREND, TrendLabel.DOWNTREND, TrendLabel.SIDEWAYS
LOW, NORMAL, HIGH, EXTREME = VolLabel.LOW, VolLabel.NORMAL, VolLabel.HIGH, VolLabel.EXTREME


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(self, *args, **kwargs):
        raise AssertionError("security regime tests must not touch the network")
    monkeypatch.setattr(socket.socket, "connect", refuse)


# -- helpers ----------------------------------------------------------------

def _closes(log_returns: np.ndarray, end: date = END, start_price: float = 100.0) -> pd.Series:
    days = pd.bdate_range(end=pd.Timestamp(end), periods=len(log_returns))
    return pd.Series(start_price * np.exp(np.cumsum(log_returns)), index=days, dtype=float)


def _rng_returns(seed: int, n: int, drift: float = 0.0, vol: float = 0.005) -> np.ndarray:
    return np.random.default_rng(seed).normal(drift, vol, n)


def _regime(trend: TrendLabel | None, vol: VolLabel | None, status: str = "available") -> SecurityRegime:
    """A regime with only the labels the discrepancy engine reads."""
    trend_state = TrendState(
        "available" if trend else "unavailable", trend, 100.0, 100.0, 100.0, 0.0, 1, None,
    ) if trend else TrendState("unavailable", None, None, None, None, None, None, "saknas")
    vol_state = VolatilityState(
        "available" if vol else "unavailable", vol, 20.0, 0.5, None,
    ) if vol else VolatilityState("unavailable", None, None, None, "saknas")
    return SecurityRegime(
        status=status, as_of=END, obs=400, volatility=vol_state, trend=trend_state,
        drawdown_252d_pct=0.0, return_1m_pct=0.0, return_3m_pct=0.0, reason=None,
        policy_version=SECURITY_REGIME_VERSION,
    )


def _edge_for(status: EdgeStatus, expected_sign: int | None) -> Edge:
    attributes = {} if expected_sign is None else {"expected_sign": expected_sign}
    return Edge(id=1, src_id=1, dst_id=2, relation_type=RelationType.CORRELATED_WITH, status=status,
                source="seed", created_version=1, attributes=attributes)


# -- classifier: scenarios ----------------------------------------------------

def test_uptrend_is_classified_up_with_positive_context():
    prices = _closes(_rng_returns(11, 400, drift=0.003))
    regime = classify_security_regime(prices, END)
    assert regime.status == "available" and regime.reason is None
    assert regime.trend.label is UP
    assert regime.trend.close > regime.trend.sma50 > regime.trend.sma200
    assert regime.trend.return_3m_pct > 0
    assert regime.return_3m_pct > 0 and regime.drawdown_252d_pct <= 0


def test_downtrend_is_classified_down():
    prices = _closes(_rng_returns(12, 400, drift=-0.003))
    regime = classify_security_regime(prices, END)
    assert regime.trend.label is DOWN
    assert regime.trend.close < regime.trend.sma50 < regime.trend.sma200
    assert regime.trend.return_3m_pct < 0


def test_close_above_fast_sma_with_fast_below_slow_is_sideways():
    # Last 200 closes average 105 (slow SMA); last 50 average about 100 (fast SMA): fast < slow,
    # so the stack is not up. The close (102) is above the fast SMA, so it is not down either.
    levels = np.concatenate([np.full(250, 120.0), np.full(100, 100.0), np.full(49, 100.0), [102.0]])
    prices = pd.Series(levels, index=pd.bdate_range(end=pd.Timestamp(END), periods=len(levels)))
    regime = classify_security_regime(prices, END)
    assert regime.trend.sma50 < regime.trend.sma200 and regime.trend.close > regime.trend.sma50
    assert regime.trend.label is SIDE


def test_high_vol_spike_at_the_end_is_extreme_volatility():
    returns = np.concatenate([_rng_returns(21, 680, vol=0.004), _rng_returns(22, 20, vol=0.06)])
    regime = classify_security_regime(_closes(returns), END)
    assert regime.volatility.status == "available"
    assert regime.volatility.label is EXTREME
    assert regime.volatility.percentile_3y >= 0.95
    assert regime.volatility.realized_vol_20d_pct > 30.0


def test_calm_period_after_turbulence_is_low_volatility():
    returns = np.concatenate([_rng_returns(31, 500, vol=0.03), _rng_returns(32, 40, vol=0.002)])
    regime = classify_security_regime(_closes(returns), END)
    assert regime.volatility.label is LOW
    assert regime.volatility.percentile_3y < 0.25


def test_short_history_is_unavailable_with_reason_for_both_dimensions():
    regime = classify_security_regime(_closes(_rng_returns(41, 150)), END)
    assert regime.status == "unavailable"
    assert regime.volatility.status == "unavailable" and "för kort" in regime.volatility.reason
    assert regime.trend.status == "unavailable" and "för kort" in regime.trend.reason
    assert regime.reason is not None


def test_history_between_trend_and_volatility_minimums_gives_trend_only():
    regime = classify_security_regime(_closes(_rng_returns(42, 250, drift=0.003)), END)
    assert regime.status == "available"
    assert regime.volatility.status == "unavailable"
    assert regime.trend.status == "available" and regime.trend.label is UP


def test_empty_or_none_prices_are_unavailable_not_errors():
    empty = classify_security_regime(pd.Series(dtype=float), END)
    assert empty.status == "unavailable" and empty.reason == "ingen kurshistorik"


def test_non_positive_price_makes_volatility_unavailable_only():
    returns = _rng_returns(43, 400, drift=0.002)
    prices = _closes(returns)
    prices.iloc[-5] = 0.0
    regime = classify_security_regime(prices, END)
    assert regime.volatility.status == "unavailable" and "ogiltiga kurser" in regime.volatility.reason
    assert regime.trend.status == "available"
    assert regime.status == "available"


def test_as_of_cut_ignores_later_bars():
    prices = _closes(np.concatenate([_rng_returns(51, 300, drift=0.003), _rng_returns(52, 100, drift=-0.02)]))
    cut_day = prices.index[299].date()
    cut = classify_security_regime(prices, cut_day)
    truncated = classify_security_regime(prices.iloc[:300], cut_day)
    assert cut.as_of == cut_day and cut.obs == 300
    assert cut == truncated
    assert cut.trend.label is UP
    assert classify_security_regime(prices, END).trend.label is not UP


def test_since_is_the_trading_days_the_label_has_held_capped_at_252():
    prices = _closes(_rng_returns(61, 600, drift=0.002, vol=0.001))
    regime = classify_security_regime(prices, END)
    assert regime.trend.label is UP
    assert regime.trend.since_days == 252


def test_since_counts_only_the_current_run():
    prices = _closes(np.concatenate([_rng_returns(62, 400, drift=0.003), _rng_returns(63, 30, drift=-0.02, vol=0.001)]))
    regime = classify_security_regime(prices, END)
    labels = _trend_labels(prices, SecurityRegimePolicy()).to_numpy()
    run = 0
    for value in labels[::-1]:
        if value != regime.trend.label:
            break
        run += 1
    assert regime.trend.since_days == run < 252


def test_drawdown_from_252_day_high_and_one_month_return():
    closes = np.concatenate([np.linspace(100.0, 200.0, 200), np.linspace(200.0, 150.0, 60)])
    prices = pd.Series(closes, index=pd.bdate_range(end=pd.Timestamp(END), periods=len(closes)))
    regime = classify_security_regime(prices, END)
    assert regime.drawdown_252d_pct == pytest.approx(-25.0, abs=1e-6)
    expected_1m = round((prices.iloc[-1] / prices.iloc[-22] - 1.0) * 100.0, 4)
    assert regime.return_1m_pct == pytest.approx(expected_1m)


def test_policy_is_frozen_and_versioned():
    policy = SecurityRegimePolicy()
    assert policy.version == "security-regime-v0-uncalibrated"
    with pytest.raises(FrozenInstanceError):
        policy.vol_low_pct = 0.1  # type: ignore[misc]
    assert classify_security_regime(_closes(_rng_returns(71, 400)), END).policy_version == policy.version


def test_unavailable_regime_helper_keeps_reason_and_version():
    regime = unavailable_regime("ingen lagrad kurs")
    assert regime.status == "unavailable" and regime.reason == "ingen lagrad kurs"
    assert regime.policy_version == SECURITY_REGIME_VERSION


# -- discrepancy table ----------------------------------------------------------

TREND_TABLE = [
    # (queried, neighbour, expected_sign, expected_code or None)
    (UP, DOWN, 1, DiscrepancyCode.TREND_MISMATCH),
    (DOWN, UP, 1, DiscrepancyCode.TREND_MISMATCH),
    (UP, UP, 1, None),
    (DOWN, DOWN, 1, None),
    (UP, DOWN, -1, None),
    (DOWN, UP, -1, None),
    (UP, UP, -1, DiscrepancyCode.TREND_MISMATCH),
    (DOWN, DOWN, -1, DiscrepancyCode.TREND_MISMATCH),
    (SIDE, SIDE, 1, None),
    (SIDE, SIDE, -1, None),
    (UP, SIDE, 1, DiscrepancyCode.TREND_DIVERGING),
    (SIDE, DOWN, 1, DiscrepancyCode.TREND_DIVERGING),
    (UP, SIDE, -1, DiscrepancyCode.TREND_DIVERGING),
    (SIDE, DOWN, -1, DiscrepancyCode.TREND_DIVERGING),
]


@pytest.mark.parametrize(("q", "n", "sign", "code"), TREND_TABLE)
@pytest.mark.parametrize("status", [EdgeStatus.VALIDATED, EdgeStatus.HYPOTHESIS])
def test_trend_table_by_sign_and_edge_status(q, n, sign, code, status):
    found = find_discrepancies(_regime(q, NORMAL), _regime(n, NORMAL), _edge_for(status, sign))
    trend = [d for d in found if d.dimension == "trend"]
    if code is None:
        assert trend == []
        return
    assert [d.code for d in trend] == [code]
    d = trend[0]
    if code is DiscrepancyCode.TREND_MISMATCH:
        assert d.severity == ("attention" if status is EdgeStatus.VALIDATED else "info")
    else:
        assert d.severity == "info"
    assert d.queried_label == q.value and d.neighbour_label == n.value


@pytest.mark.parametrize(("q", "n"), [(q, n) for q in (LOW, NORMAL, HIGH, EXTREME) for n in (LOW, NORMAL, HIGH, EXTREME)])
@pytest.mark.parametrize("status", [EdgeStatus.VALIDATED, EdgeStatus.HYPOTHESIS])
def test_volatility_table_by_level_pair_and_edge_status(q, n, status):
    found = [d for d in find_discrepancies(_regime(UP, q), _regime(UP, n), _edge_for(status, 1))
             if d.dimension == "volatility"]
    levels = {LOW: 0, NORMAL: 1, HIGH: 2, EXTREME: 3}
    if abs(levels[q] - levels[n]) < 2:
        assert found == []
        return
    assert [d.code for d in found] == [DiscrepancyCode.VOLATILITY_MISMATCH]
    extreme = EXTREME in (q, n)
    assert found[0].severity == ("attention" if (extreme and status is EdgeStatus.VALIDATED) else "info")


def test_volatility_mismatch_does_not_depend_on_expected_sign():
    for sign in (1, -1, None):
        found = find_discrepancies(_regime(UP, LOW), _regime(UP, HIGH), _edge_for(EdgeStatus.VALIDATED, sign))
        assert [d.code for d in found if d.dimension == "volatility"] == [DiscrepancyCode.VOLATILITY_MISMATCH]


def test_missing_expected_sign_skips_trend_rules_only():
    found = find_discrepancies(_regime(UP, LOW), _regime(DOWN, HIGH), _edge_for(EdgeStatus.VALIDATED, None))
    assert [d.code for d in found] == [DiscrepancyCode.VOLATILITY_MISMATCH]


def test_unavailable_neighbour_yields_no_discrepancies():
    unavailable = SecurityRegime(
        status="unavailable", as_of=None, obs=0,
        volatility=VolatilityState("unavailable", None, None, None, "x"),
        trend=TrendState("unavailable", None, None, None, None, None, None, "x"),
        drawdown_252d_pct=None, return_1m_pct=None, return_3m_pct=None, reason="x",
        policy_version=SECURITY_REGIME_VERSION,
    )
    assert find_discrepancies(_regime(UP, EXTREME), unavailable, _edge_for(EdgeStatus.VALIDATED, 1)) == []


def test_swedish_text_matches_the_brief_example_and_has_no_trade_wording():
    found = find_discrepancies(
        _regime(UP, NORMAL), _regime(DOWN, NORMAL), _edge_for(EdgeStatus.VALIDATED, 1),
        queried_symbol="NVDA", neighbour_symbol="AMZN",
    )
    assert found[0].text == "NVDA är i uppåttrend men AMZN i nedåttrend; sambandet brukar vara samrörande"
    for d in found:
        assert not any(word in d.text.lower() for word in ("köp", "sälj", "buy", "sell"))


def test_regime_json_keeps_labels_as_strings_and_shape():
    regime = classify_security_regime(_closes(_rng_returns(81, 400, drift=0.003)), END)
    out = regime_json(regime)
    assert out["status"] == "available" and out["as_of"] == END.isoformat()
    assert out["trend"]["label"] == "UPTREND" and out["volatility"]["label"] in {"LOW", "NORMAL", "HIGH", "EXTREME"}
    assert set(out) == {"status", "as_of", "obs", "reason", "volatility", "trend", "drawdown_252d_pct",
                        "return_1m_pct", "return_3m_pct", "policy_version"}


# -- related integration -----------------------------------------------------

LEGACY_ROW_KEYS = {"edge_id", "neighbour", "role", "relation_type", "status", "source", "expected_sign",
                   "lag", "divergence", "signal"}
LEGACY_TOP_KEYS = {"symbol", "as_of", "include_hypothesis", "queried", "rows", "meta"}


def test_related_response_is_backward_compatible_and_extended():
    payload = _service([_edge(10, 1, 2, expected_sign=1)], [_validation(1, 2)]).build_related("NVDA", AS_OF, False)
    assert LEGACY_TOP_KEYS <= set(payload)
    assert set(payload) == LEGACY_TOP_KEYS | {"queried_regime"}
    row = _row(payload, 10)
    assert LEGACY_ROW_KEYS <= set(row)
    assert set(row) == LEGACY_ROW_KEYS | {"regime", "discrepancies"}
    assert isinstance(payload["meta"]["counts"]["discrepancies_attention"], int)
    assert payload["meta"]["policy_version"]["security_regime"] == SECURITY_REGIME_VERSION


def test_queried_regime_has_modeled_block_unavailable_without_snapshot_store():
    payload = _service([], []).build_related("NVDA", AS_OF, False)
    queried = payload["queried_regime"]
    assert queried["status"] == "available"
    assert queried["modeled"]["status"] == "unavailable"
    assert queried["modeled"]["label"] == "MODELED / ESTIMATED"
    assert queried["modeled"]["reason"]


def test_trend_mismatch_is_attention_on_validated_edge_and_counted():
    prices = {"NVDA": _bars(_rng_returns(91, 400, drift=0.003)), "MSFT": _bars(_rng_returns(92, 400, drift=-0.003))}
    payload = _service([_edge(10, 1, 2, expected_sign=1)], [_validation(1, 2)], prices=prices).build_related(
        "NVDA", AS_OF, False)
    row = _row(payload, 10)
    assert row["regime"]["trend"]["label"] == "DOWNTREND"
    mismatch = [d for d in row["discrepancies"] if d["code"] == "TREND_MISMATCH"]
    assert len(mismatch) == 1 and mismatch[0]["severity"] == "attention"
    assert payload["queried_regime"]["trend"]["label"] == "UPTREND"
    assert payload["meta"]["counts"]["discrepancies_attention"] >= 1


def test_hypothesis_edge_mismatch_is_capped_at_info():
    prices = {"NVDA": _bars(_rng_returns(93, 400, drift=0.003)), "MSFT": _bars(_rng_returns(94, 400, drift=-0.003))}
    payload = _service([_edge(10, 1, 2, expected_sign=1, status=EdgeStatus.HYPOTHESIS)], [], prices=prices).build_related(
        "NVDA", AS_OF, include_hypothesis=True)
    row = _row(payload, 10)
    assert all(d["severity"] == "info" for d in row["discrepancies"])
    assert payload["meta"]["counts"]["discrepancies_attention"] == 0


def test_neighbour_price_failure_degrades_only_that_row():
    payload = _service([_edge(10, 1, 2, expected_sign=1)], [_validation(1, 2)], price_fail={"MSFT"}).build_related(
        "NVDA", AS_OF, False)
    row = _row(payload, 10)
    assert row["regime"]["status"] == "unavailable" and row["regime"]["reason"]
    assert row["discrepancies"] == []
    assert payload["queried_regime"]["status"] == "available"


def test_neighbour_without_prices_gets_unavailable_regime_with_reason():
    prices = {"NVDA": _bars(_rng_returns(95, 400))}
    payload = _service([_edge(10, 1, 2, expected_sign=1)], [_validation(1, 2)], prices=prices).build_related(
        "NVDA", AS_OF, False)
    row = _row(payload, 10)
    assert row["regime"]["status"] == "unavailable" and row["regime"]["reason"] == "ingen lagrad kurs"
    assert row["regime"]["trend"]["status"] == "unavailable"
    assert row["discrepancies"] == []


def test_modeled_gamma_is_queried_only_and_labelled():
    snapshots = FakeSnapshots(_chain())
    prices = {"SPY": _bars(_rng_returns(96, 400, drift=0.001))}
    service = RelatedService(FakeGraph([]), _empty_evidence(), FakePrices(prices), snapshot_store=snapshots)
    payload = service.build_related("SPY", AS_OF, False)
    modeled = payload["queried_regime"]["modeled"]
    assert modeled["status"] == "available" and modeled["label"] == "MODELED / ESTIMATED"
    assert modeled["gamma_regime"] in {"POSITIVE_GAMMA", "NEUTRAL_GAMMA", "NEGATIVE_GAMMA", "EXTREME_NEGATIVE_GAMMA", "UNKNOWN"}
    assert payload["rows"] == []  # SPY has no edges, so no neighbour can carry a gamma block


def _empty_evidence():
    from tests.mrip.test_related import FakeEvidence
    return FakeEvidence([])


def test_service_classify_none_prices_is_unavailable():
    service = SecurityRegimeService(FakePrices({}))
    regime = service.classify(None, AS_OF)
    assert regime.status == "unavailable" and regime.reason == "ingen lagrad kurs"


def test_build_regime_unknown_symbol_is_raised_only_when_graph_and_prices_miss():
    service = SecurityRegimeService(FakePrices({}), graph=FakeGraph([]))
    with pytest.raises(UnknownSymbol):
        service.build_regime("ZZZZ", AS_OF)


def test_build_regime_known_graph_node_without_prices_is_unavailable_not_404():
    service = SecurityRegimeService(FakePrices({}), graph=FakeGraph([]))
    payload = service.build_regime("NVDA", AS_OF)
    assert payload["queried_regime"]["status"] == "unavailable"
    assert payload["symbol"] == "NVDA"
    assert payload["meta"]["policy_version"] == SECURITY_REGIME_VERSION


# -- routes -------------------------------------------------------------------

@pytest.fixture
def authed(monkeypatch):
    payload = {"sub": "tester", "user_id": 1, "_verified_user_role": "user", "_verified_username": "tester"}
    monkeypatch.setattr(auth, "verify_token", lambda token: payload if token == "test-token" else None)


class FakeRegimeService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, date]] = []

    def build_regime(self, symbol: str, as_of: date) -> dict:
        self.calls.append((symbol, as_of))
        if self.error is not None:
            raise self.error
        return {"symbol": symbol, "as_of": as_of.isoformat(), "queried_regime": {"status": "unavailable"}, "meta": {}}


@pytest.fixture
def regime_route(monkeypatch, authed):
    stub = FakeRegimeService()
    monkeypatch.setattr(mrip_routes, "_security_regime_service", lambda: stub)
    return stub


def test_regime_route_happy_path_normalises_symbol_and_date(client, regime_route):
    resp = client.get("/api/mrip/symbols/nvda/regime?as_of=2026-10-01", headers=AUTH)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["code"] == 1 and body["data"]["queried_regime"] == {"status": "unavailable"}
    assert regime_route.calls == [("nvda", date(2026, 10, 1))]


def test_regime_route_rejects_bad_date_with_400(client, regime_route):
    resp = client.get("/api/mrip/symbols/NVDA/regime?as_of=2026-13-01", headers=AUTH)
    assert resp.status_code == 400 and resp.get_json()["code"] == 0
    assert regime_route.calls == []


def test_regime_route_unknown_symbol_is_404(monkeypatch, client, authed):
    monkeypatch.setattr(mrip_routes, "_security_regime_service", lambda: FakeRegimeService(UnknownSymbol("ZZZZ")))
    resp = client.get("/api/mrip/symbols/ZZZZ/regime", headers=AUTH)
    assert resp.status_code == 404 and resp.get_json()["code"] == 0


def test_regime_route_requires_auth(client, regime_route):
    assert client.get("/api/mrip/symbols/NVDA/regime").status_code == 401
    assert regime_route.calls == []


def test_related_factory_passes_snapshot_store_to_regimes(monkeypatch):
    monkeypatch.setattr(mrip_routes, "_graph", lambda: "graph")
    monkeypatch.setattr(mrip_routes, "_evidence_store", lambda: "evidence")
    monkeypatch.setattr(mrip_routes, "_price_store", lambda: "prices")
    monkeypatch.setattr(mrip_routes, "_snapshot_store", lambda: "snapshots")
    service = mrip_routes._related_service()
    assert service._regimes._snapshots == "snapshots"
    assert service._regimes._prices == "prices"


def test_regime_factory_wires_prices_snapshots_and_graph(monkeypatch):
    monkeypatch.setattr(mrip_routes, "_graph", lambda: "graph")
    monkeypatch.setattr(mrip_routes, "_price_store", lambda: "prices")
    monkeypatch.setattr(mrip_routes, "_snapshot_store", lambda: "snapshots")
    service = mrip_routes._security_regime_service()
    assert isinstance(service, SecurityRegimeService)
    assert (service._prices, service._snapshots, service._graph) == ("prices", "snapshots", "graph")


def test_graph_node_lookup_finds_known_security_and_not_unknown():
    service = SecurityRegimeService(FakePrices({}), graph=FakeGraph([]))
    assert service._in_graph("NVDA") is True
    assert service._in_graph("ZZZZ") is False
