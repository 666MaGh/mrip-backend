# MRIP: self-hosted TimesFM 2.5 forecasting service. CPU build; the Apache-2.0
# weights (google/timesfm-2.5-200m-pytorch) are downloaded from Hugging Face on
# first forecast into the /hf volume. TimesFM 3.0 weights are non-commercial and
# are never loaded (the repo id is hard-coded in server.py). The timesfm 2.0.x
# package contains only the 2.5 model code.
FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        "timesfm==2.0.2" \
        "numpy==2.3.5" \
        "huggingface_hub==1.33.0" \
        "safetensors==0.7.0" \
        "fastapi==0.142.2" \
        "uvicorn==0.54.0" \
        "starlette==1.7.0" \
        "pydantic==2.13.5"
RUN useradd --uid 10003 --create-home timesfm && mkdir /hf && chown timesfm /hf
WORKDIR /app
COPY --chown=timesfm server.py /app/server.py
USER timesfm
ENV HF_HOME=/hf HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 TIMESFM_PRELOAD=0
EXPOSE 8000
# Single worker on purpose: one model copy in memory, inference serialised by a lock.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
