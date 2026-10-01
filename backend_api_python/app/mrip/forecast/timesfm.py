"""TimesFMProvider: HTTP client for the self-hosted TimesFM 2.5 service (ADR-0003).

Only TimesFM weights with a production-compatible licence may be served
(2.5, Apache-2.0); 3.0 weights are research-only and are never used here. The
service (``ops/timesfm``) needs torch, so it runs separately, like LAYA.

Contract: ``POST /v1/forecast {"series": [...], "horizon": h}`` returns
``{"model": id, "quantiles": {"0.1": [h values], ...}}``. We forecast LOG prices
(keeps prices positive and makes returns additive), take the last step's
quantiles, convert to simple returns, and rearrange so quantiles never cross.
Covariates are not used yet; a covariate-aware variant must earn its place in
walk-forward evaluation first.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import requests

from app.mrip.forecast.base import BaseForecastProvider, returns_from_log_quantiles
from app.mrip.forecast.types import QUANTILES, ForecastRequest, ForecastResult, ForecastUnavailable

_MIN_CONTEXT = 64


def _interpolate(levels: Mapping[float, float], q: float) -> float:
    xs = sorted(levels)
    if q in levels:
        return levels[q]
    if q < xs[0] or q > xs[-1]:
        raise ForecastUnavailable(f"service did not return quantiles bracketing {q}")
    return float(np.interp(q, xs, [levels[x] for x in xs]))


class TimesFMProvider(BaseForecastProvider):
    name = "timesfm"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        context_len: int = 512,
        timeout: float = 120.0,
        session: Any | None = None,
    ) -> None:
        if context_len < _MIN_CONTEXT:
            raise ValueError(f"context_len must be >= {_MIN_CONTEXT}")
        self._base = base_url.rstrip("/")
        self._context = context_len
        self._timeout = timeout
        self._session = session or requests.Session()
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model: str | None = None

    def version(self) -> str:
        return f"timesfm-v1({self._model or 'unqueried'},log,ctx={self._context})"

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        self.require(request, _MIN_CONTEXT, "TimesFM")
        log_p = np.log(request.prices.to_numpy())[-self._context :]
        h = request.horizon_days
        payload = self._post({"series": [float(v) for v in log_p], "horizon": h})
        quantiles = payload.get("quantiles")
        if not isinstance(quantiles, dict) or not quantiles:
            raise ForecastUnavailable("TimesFM response has no quantiles")
        self._model = str(payload.get("model") or self._model or "unknown")
        last_step: dict[float, float] = {}
        for key, values in quantiles.items():
            if not isinstance(values, list) or len(values) != h:
                raise ForecastUnavailable(f"TimesFM quantile {key} has {len(values) if isinstance(values, list) else '?'} steps, expected {h}")
            value = float(values[-1])
            if not np.isfinite(value):
                raise ForecastUnavailable("TimesFM returned a non-finite quantile")
            last_step[float(key)] = value
        last_log = float(log_p[-1])
        log_returns = {q: _interpolate(last_step, q) - last_log for q in QUANTILES}
        return ForecastResult(
            provider=self.name,
            provider_version=self.version(),
            symbol=request.symbol,
            as_of=request.as_of,
            horizon_days=h,
            last_price=request.last_price,
            return_quantiles=returns_from_log_quantiles(log_returns),
            n_obs=len(log_p),
            warnings=("foundation-model forecast: unvalidated until it beats the baselines in walk-forward evaluation",),
        )

    def _post(self, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self._session.post(
                self._base + "/v1/forecast", json=body, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # service boundary: network, HTTP status or JSON failure
            raise ForecastUnavailable(f"TimesFM service failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ForecastUnavailable("TimesFM service returned a non-object payload")
        return payload
