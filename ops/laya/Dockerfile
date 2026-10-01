# MRIP: self-hosted LAYA inference service (ADR-0004). CPU build; checkpoints are
# downloaded from Hugging Face on first request into the /hf volume.
FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "laya[serve]==0.3.22"
RUN useradd --uid 10002 --create-home laya && mkdir /hf && chown laya /hf
USER laya
ENV HF_HOME=/hf LAYA_HOST=0.0.0.0 LAYA_PRELOAD=0
EXPOSE 8000
CMD ["laya-serve"]
