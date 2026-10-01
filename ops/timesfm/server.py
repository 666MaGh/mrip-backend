"""MRIP TimesFM 2.5 forecasting service.

LICENSE CONSTRAINT: only Apache-2.0 TimesFM weights may be loaded. TimesFM 3.0
weights are non-commercial / non-production and must never be loaded. The repo
id is therefore hard-coded below; TIMESFM_MODEL_ID may only select among the
explicitly allowed Apache-2.0 ids in ALLOWED_MODELS.

API:
  GET  /health       -> {"status": "ok", "model": <repo id>, "loaded": <bool>}
  POST /v1/forecast  -> {"series": [floats], "horizon": int}
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("timesfm-service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# Allowed model repos -> pinned commit (Apache-2.0, verified via the HF API:
# cardData.license == "apache-2.0"). Nothing else may be loaded.
ALLOWED_MODELS: dict[str, str] = {
    "google/timesfm-2.5-200m-pytorch": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
}
DEFAULT_MODEL_ID = "google/timesfm-2.5-200m-pytorch"

MIN_SERIES = 32
MAX_HORIZON = 256
MAX_ABS_VALUE = 1e30
MAX_BODY_BYTES = 2 * 1024 * 1024
# Decile levels returned by the quantile head (channel 0 is the mean, channels 1..9 are q0.1..q0.9).
QUANTILE_LEVELS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")
POINT_CHANNEL = 5  # the library's point forecast is the q0.5 channel (median)

ERR_INTERNAL = "internal error"


def _resolve_model_id() -> str:
    model_id = os.environ.get("TIMESFM_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID
    if model_id not in ALLOWED_MODELS:
        raise RuntimeError(
            f"TIMESFM_MODEL_ID {model_id!r} is not allowed; only Apache-2.0 models "
            f"{sorted(ALLOWED_MODELS)} may be loaded (TimesFM 3.0 weights are non-commercial)"
        )
    return model_id


def _resolve_max_context() -> int:
    raw = os.environ.get("TIMESFM_MAX_CONTEXT", "1024").strip() or "1024"
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"TIMESFM_MAX_CONTEXT must be an integer, got {raw!r}") from exc
    if not MIN_SERIES <= value <= 16000:
        raise RuntimeError(f"TIMESFM_MAX_CONTEXT must be in {MIN_SERIES}..16000, got {value}")
    return value


MODEL_ID = _resolve_model_id()
MODEL_REVISION = ALLOWED_MODELS[MODEL_ID]
MAX_CONTEXT = _resolve_max_context()
API_KEY = os.environ.get("TIMESFM_API_KEY", "").strip()

_model: Any = None
_lock = threading.Lock()  # serialises model loading and inference


def _load_model() -> Any:
    """Load and compile the model. Caller must hold _lock."""
    global _model
    if _model is not None:
        return _model
    import torch
    import timesfm

    torch.set_float32_matmul_precision("high")
    log.info("loading %s @ %s", MODEL_ID, MODEL_REVISION)
    # torch_compile=False: no C++ toolchain in the slim image, and eager mode is deterministic.
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_compile=False
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=MAX_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    _model = model
    log.info("model ready (max_context=%d, max_horizon=%d)", MAX_CONTEXT, MAX_HORIZON)
    return _model


def _infer(series: list[float], horizon: int) -> dict[str, Any]:
    import torch

    with _lock:
        model = _load_model()
        with torch.inference_mode():
            point, quant = model.forecast(
                horizon=horizon, inputs=[np.asarray(series, dtype=np.float64)]
            )
    # quant: (1, horizon, 10) = [mean, q0.1 .. q0.9]; point: (1, horizon) = q0.5 channel.
    quant = np.asarray(quant, dtype=np.float64)[0]
    point = np.asarray(point, dtype=np.float64)[0]
    if quant.shape != (horizon, 10) or point.shape != (horizon,):
        raise RuntimeError(f"unexpected output shapes {quant.shape} / {point.shape}")
    levels = quant[:, 1:10]
    # fix_quantile_crossing already orders the channels; this is a final deterministic guard.
    levels = np.maximum.accumulate(levels, axis=1)
    if not (np.isfinite(levels).all() and np.isfinite(point).all()):
        raise RuntimeError("model returned non-finite values")
    return {
        "model": MODEL_ID,
        "horizon": horizon,
        "point": [float(v) for v in point],
        "quantiles": {
            level: [float(v) for v in levels[:, i]] for i, level in enumerate(QUANTILE_LEVELS)
        },
    }


class ValidationError(Exception):
    pass


def _validate(payload: Any) -> tuple[list[float], int]:
    if not isinstance(payload, dict):
        raise ValidationError('body must be a JSON object {"series": [...], "horizon": int}')
    series = payload.get("series")
    horizon = payload.get("horizon")
    if not isinstance(series, list):
        raise ValidationError("series must be a list of numbers")
    for i, v in enumerate(series):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValidationError(f"series[{i}] is not a number")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValidationError(f"series[{i}] is not finite (NaN/Infinity not allowed)")
        if abs(v) > MAX_ABS_VALUE:
            raise ValidationError(f"series[{i}] magnitude exceeds {MAX_ABS_VALUE:g}")
    if len(series) < MIN_SERIES:
        raise ValidationError(f"series length {len(series)} is below the minimum of {MIN_SERIES}")
    if len(series) > MAX_CONTEXT:
        raise ValidationError(
            f"series length {len(series)} exceeds the maximum context of {MAX_CONTEXT}"
        )
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValidationError(f"horizon must be an integer in 1..{MAX_HORIZON}")
    if not 1 <= horizon <= MAX_HORIZON:
        raise ValidationError(f"horizon {horizon} is outside 1..{MAX_HORIZON}")
    return [float(v) for v in series], horizon


def _authorized(request: Request) -> bool:
    if not API_KEY:
        return True
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return scheme.lower() == "bearer" and hmac.compare_digest(
        token.strip().encode("utf-8"), API_KEY.encode("utf-8")
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("TIMESFM_PRELOAD", "0").strip() == "1":
        def _preload() -> None:
            with _lock:
                _load_model()

        await run_in_threadpool(_preload)
    yield


app = FastAPI(title="MRIP TimesFM", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error", exc_info=exc)
    return JSONResponse({"detail": ERR_INTERNAL}, status_code=500)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_ID, "loaded": _model is not None}


@app.post("/v1/forecast")
async def forecast(request: Request) -> JSONResponse:
    if not _authorized(request):
        return JSONResponse(
            {"detail": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
        )
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse({"detail": "request body too large"}, status_code=413)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"detail": "body is not valid JSON"}, status_code=422)
    try:
        series, horizon = _validate(payload)
    except ValidationError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    try:
        result = await run_in_threadpool(_infer, series, horizon)
    except Exception:
        log.exception("forecast failed")
        return JSONResponse({"detail": ERR_INTERNAL}, status_code=500)
    return JSONResponse(result)
