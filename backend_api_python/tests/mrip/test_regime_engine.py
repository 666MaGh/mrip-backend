"""Market Regime Engine on synthetic market data (fake gateway, no network)."""
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import Latency, PriceBar, PriceSeries, Provenance
from app.mrip.forecast.types import QUANTILES, ForecastResult
from app.mrip.outcomes.predictions import forecast_prediction
from app.mrip.calibration.forecast import prediction_attrs, result_attrs
from app.mrip.regime.engine import MarketRegimeEngine, RegimePolicy
from app.mrip.regime.features import Direction, Trend
from app.mrip.stats.regimes import Regime

PROV = Provenance("fake", "test", "fake.history", datetime(2026, 10, 1, tzinfo=timezone.utc), Latency.UNKNOWN)
N = 400
DAYS = pd.bdate_range("2025-01-01", periods=N)


def series(symbol, values, days=DAYS):
    bars = tuple(PriceBar(d.date(), None, None, None, float(v), None) for d, v in zip(days, values))
    return PriceSeries(symbol, "1d", bars, PROV)


def vix_values():
    v = np.full(N, 14.0)
    v[200:260] = np.linspace(14, 45, 60)   # a stress build-up
    v[260:] = np.linspace(45, 18, N - 260)
    return v


class Gateway:
    def __init__(self, data, missing=()):
        self.data, self.missing = data, set(missing)

    def _get(self, symbol):
        if symbol in self.missing or symbol not in self.data:
            raise DataUnavailable(symbol)
        return self.data[symbol]

    def index_history(self, symbol, start=None, end=None):
        return self._get(symbol)

    def price_history(self, symbol, start=None, end=None, interval="1d"):
        return self._get(symbol)


def full_data():
    rng = np.random.default_rng(3)
    spx = 4000 * np.exp(np.cumsum(rng.normal(0.0006, 0.008, N)))
    return {
        "VIX": series("VIX", vix_values()),
        "VIX3M": series("VIX3M", vix_values() * 0.9),          # VIX above VIX3M: backwardation
        "SPX": series("SPX", spx),
        "TNX": series("TNX", np.linspace(40, 64, N)),          # 4.0% -> 6.4% (quoted x10)
        "UUP": series("UUP", np.linspace(28, 22, N)),          # a falling dollar
    }


def test_regime_follows_the_vix_thresholds_and_reports_all_components():
    r = MarketRegimeEngine(Gateway(full_data())).at()
    assert r.as_of == DAYS[-1].date() and r.vix == pytest.approx(18.0) and r.regime is Regime.NORMAL
    assert r.vix_percentile is not None and 0 < r.vix_percentile <= 1
    assert r.vix_change is not None and r.vix_change < 0                     # VIX still falling
    assert r.realized_vol is not None and r.implied_to_realized == pytest.approx(0.18 / r.realized_vol)
    assert r.vix_term_ratio == pytest.approx(1 / 0.9) and r.term_backwardation is True
    assert r.index_trend in (Trend.UPTREND, Trend.MIXED, Trend.DOWNTREND) and r.index_drawdown <= 0
    assert r.rates_yield == pytest.approx(6.4) and r.rates_direction is Direction.RISING
    assert r.usd_direction is Direction.FALLING
    assert r.unavailable == () and set(r.sources) == {"vix", "index", "vix3m", "rates", "usd"}
    assert r.attrs() == {"vix_regime": "NORMAL"} and r.policy_version == "regime-v0-uncalibrated"


def test_stress_period_is_classified_stress_or_crisis_at_that_date():
    engine = MarketRegimeEngine(Gateway(full_data()))
    peak = engine.at(DAYS[259].date())
    assert peak.vix == pytest.approx(45.0) and peak.regime is Regime.CRISIS
    assert engine.at(DAYS[100].date()).regime is Regime.LOW_VOL
    mid = engine.at(DAYS[235].date())
    assert mid.regime in (Regime.ELEVATED, Regime.STRESS)


def test_at_never_uses_data_after_as_of():
    base = full_data()
    target = DAYS[300].date()
    expected = MarketRegimeEngine(Gateway(base)).at(target)
    altered = dict(base)
    for key in ("VIX", "VIX3M", "SPX", "TNX", "UUP"):
        values = np.array([b.close for b in base[key].bars], dtype=float)
        values[301:] = values[301:] * 3 + 5  # rewrite the future
        altered[key] = series(key, values)
    assert MarketRegimeEngine(Gateway(altered)).at(target) == expected


def test_optional_inputs_may_be_missing_but_the_vix_may_not():
    data = full_data()
    r = MarketRegimeEngine(Gateway(data, missing={"VIX3M", "TNX", "UUP"})).at()
    assert set(r.unavailable) == {"vix3m", "rates", "usd"}
    assert (r.vix_term_ratio, r.rates_yield, r.rates_direction, r.usd_direction) == (None, None, None, None)
    assert r.regime is Regime.NORMAL and r.realized_vol is not None
    only_vix = MarketRegimeEngine(Gateway({"VIX": data["VIX"]})).at()
    assert set(only_vix.unavailable) == {"index", "vix3m", "rates", "usd"} and only_vix.realized_vol is None
    with pytest.raises(DataUnavailable, match="VIX history is required"):
        MarketRegimeEngine(Gateway({k: v for k, v in data.items() if k != "VIX"})).at()
    with pytest.raises(DataUnavailable, match="no VIX data on or before"):
        MarketRegimeEngine(Gateway(data)).at(date(2024, 1, 1))


def test_optional_series_are_carried_forward_only_a_few_days():
    data = full_data()
    gappy_days = DAYS[:-6]  # the dollar proxy stopped printing six VIX dates ago
    data["UUP"] = series("UUP", np.linspace(28, 22, len(gappy_days)), gappy_days)
    r = MarketRegimeEngine(Gateway(data), RegimePolicy(ffill_limit=3)).at()
    assert r.usd_direction is None  # beyond the carry limit: unknown, not stale
    assert MarketRegimeEngine(Gateway(data), RegimePolicy(ffill_limit=10)).at().usd_direction is Direction.FALLING


def test_context_is_json_friendly_and_flows_into_predictions_and_calibration_segments():
    r = MarketRegimeEngine(Gateway(full_data())).at()
    ctx = r.context()
    json.dumps(ctx)
    assert ctx["regime"] == "NORMAL" and ctx["term_backwardation"] is True and ctx["rates_direction"] == "RISING"
    q = {k: -0.05 + 0.1 * i / (len(QUANTILES) - 1) for i, k in enumerate(QUANTILES)}
    result = ForecastResult("naive", "naive-v1", "SPY", date(2026, 9, 30), 21, 700.0, q, 60)
    pred = forecast_prediction(result, market_regime=ctx)
    assert pred.payload["market_regime"]["regime"] == "NORMAL"
    assert "market_regime" not in forecast_prediction(result).payload
    assert result_attrs(result, r.regime.value)["vix_regime"] == "NORMAL" and "vix_regime" not in result_attrs(result)
    from app.mrip.outcomes.types import Prediction

    logged = Prediction(1, pred.prediction_type, pred.subject, pred.made_at, pred.horizon_kind, pred.model_version, pred.payload)
    assert prediction_attrs(logged)["vix_regime"] == "NORMAL"
