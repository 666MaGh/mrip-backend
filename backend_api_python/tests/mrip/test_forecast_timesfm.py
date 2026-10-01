"""TimesFMProvider against a fake HTTP session (no network, no model)."""
import numpy as np
import pandas as pd
import pytest

from app.mrip.forecast.timesfm import TimesFMProvider
from app.mrip.forecast.types import QUANTILES, ForecastRequest, ForecastUnavailable


def prices(n=400, seed=1):
    r = np.random.default_rng(seed).normal(0.0002, 0.01, n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2022-01-03", periods=n))


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, handler):
        self.handler, self.calls = handler, []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        return self.handler(json)


def decile_payload(body, offset=0.0, model="google/timesfm-2.5-200m-pytorch"):
    """Quantile paths of log price: the last observed log price + a spread that widens with the step."""
    h, last = body["horizon"], body["series"][-1]
    steps = np.arange(1, h + 1)
    return {
        "model": model,
        "quantiles": {f"{d / 10:.1f}": [last + offset + (d - 5) * 0.004 * np.sqrt(s) for s in steps] for d in range(1, 10)},
    }


def provider(handler=None, **kw):
    return TimesFMProvider("http://tfm:8000/", session=FakeSession(handler or (lambda b: FakeResponse(decile_payload(b)))), **kw)


def test_request_shape_and_conversion_to_simple_return_quantiles():
    p = provider(api_key="k", context_len=128)
    p_series = prices()
    res = p.forecast(ForecastRequest("X", p_series, 21))
    url, body, headers = p._session.calls[0]
    assert url == "http://tfm:8000/v1/forecast" and headers == {"Authorization": "Bearer k"}
    assert body["horizon"] == 21 and len(body["series"]) == 128
    assert body["series"][-1] == pytest.approx(np.log(p_series.iloc[-1]))
    spread = 0.004 * np.sqrt(21)
    assert res.return_quantiles[0.5] == pytest.approx(0.0, abs=1e-12)  # offset 0, decile 5 is the median
    assert res.return_quantiles[0.9] == pytest.approx(np.expm1(4 * spread))
    assert res.return_quantiles[0.1] == pytest.approx(np.expm1(-4 * spread))
    # P25/P75 are interpolated between the deciles that bracket them.
    assert res.return_quantiles[0.25] == pytest.approx(np.expm1(-2.5 * spread))
    assert res.provider == "timesfm" and res.n_obs == 128 and res.warnings
    assert "google/timesfm-2.5-200m-pytorch" in res.provider_version


def test_crossing_raw_quantiles_are_rearranged_not_passed_through():
    def crossing(body):
        payload = decile_payload(body)
        payload["quantiles"]["0.2"], payload["quantiles"]["0.8"] = payload["quantiles"]["0.8"], payload["quantiles"]["0.2"]
        return FakeResponse(payload)

    res = provider(crossing).forecast(ForecastRequest("X", prices(), 5))
    values = [res.return_quantiles[q] for q in QUANTILES]
    assert values == sorted(values)


@pytest.mark.parametrize(
    "handler,message",
    [
        (lambda b: FakeResponse({}, status=503), "service failed"),
        (lambda b: FakeResponse(["x"]), "non-object"),
        (lambda b: FakeResponse({"quantiles": {}}), "no quantiles"),
        (lambda b: FakeResponse({"quantiles": {"0.1": [0.0], "0.9": [0.0]}}), "expected 5"),
        (lambda b: FakeResponse({"quantiles": {"0.2": [float("nan")] * 5, "0.8": [0.0] * 5}}), "non-finite"),
        (lambda b: FakeResponse({"quantiles": {"0.4": [0.0] * 5, "0.6": [0.0] * 5}}), "bracketing"),
    ],
)
def test_service_problems_become_forecast_unavailable(handler, message):
    with pytest.raises(ForecastUnavailable, match=message):
        provider(handler).forecast(ForecastRequest("X", prices(), 5))


def test_needs_enough_history_and_valid_context():
    with pytest.raises(ForecastUnavailable, match="needs at least"):
        provider().forecast(ForecastRequest("X", prices(40), 5))
    with pytest.raises(ValueError):
        TimesFMProvider("http://x", context_len=10)


def test_version_reflects_the_served_model_after_first_call():
    p = provider()
    assert "unqueried" in p.version()
    p.forecast(ForecastRequest("X", prices(), 5))
    assert "timesfm-2.5" in p.version()
