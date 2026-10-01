# MRIP TimesFM service

Self-hosted forecasting service for TimesFM 2.5 (200M, CPU). Overlay: `docker-compose.timesfm.yml`
(service `timesfm`, container `mrip-timesfm`, port 8000 on `quantdinger-network`, volume `timesfm_hf`).

```
docker compose -f docker-compose.yml -f docker-compose.research.yml -f docker-compose.timesfm.yml up -d --build timesfm
```

## License constraint

Only the Apache-2.0 weights `google/timesfm-2.5-200m-pytorch` (pinned commit in `server.py`) are loaded.
TimesFM 3.0 weights are non-commercial / non-production and must never be loaded. The repo id is hard-coded;
`TIMESFM_MODEL_ID` may only select among `ALLOWED_MODELS` (currently one entry); anything else aborts startup.
The `timesfm==2.0.2` package contains only the 2.5 model code.

## API

- `GET /health` -> `{"status": "ok", "model": "<repo id>", "loaded": bool}` (no auth).
- `POST /v1/forecast` `{"series": [floats], "horizon": int}` ->
  `{"model", "horizon", "point": [h], "quantiles": {"0.1": [h], ..., "0.9": [h]}}`.
  - Model output is `(h, 10)`: channel 0 is the mean (not returned), channels 1..9 are q0.1..q0.9.
    `point` is the library point forecast, which is channel 5 = the median (identical to `quantiles["0.5"]`).
  - Quantile crossing is fixed by the library (`fix_quantile_crossing`) plus a final monotone guard.
- 422 on: series length < 32 or > `TIMESFM_MAX_CONTEXT`, horizon outside 1..256, non-finite or non-numeric values,
  invalid JSON. 401 on bad/missing bearer token when `TIMESFM_API_KEY` is set. 413 on bodies > 2 MiB.
  500 with a fixed `{"detail": "internal error"}`; tracebacks are only logged.

## Environment

| Var | Default | Meaning |
| --- | --- | --- |
| `TIMESFM_API_KEY` | empty | If set, `/v1/forecast` requires `Authorization: Bearer <key>` |
| `TIMESFM_MAX_CONTEXT` | 1024 | Max series length (32..16000); also the compiled context size |
| `TIMESFM_PRELOAD` | 0 | `1` loads the model at startup instead of on the first forecast |

Inference is serialised with a lock (single worker, ~2 GB RSS). The model always decodes 256 steps and the
response is sliced, so a shorter horizon is an exact prefix of a longer one and output is deterministic.
The first forecast downloads the weights (~0.9 GB) into the `timesfm_hf` volume.
