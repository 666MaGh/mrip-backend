"""Rate-limited, cached HTTP client for SEC EDGAR.

SEC fair-access rules built in: a descriptive User-Agent is mandatory, requests
are capped at 5 per second, 429/5xx responses are retried with backoff (honouring
Retry-After), and every successful response is cached on disk keyed by URL.
The HTTP function, clock and sleep are injectable so tests never touch the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Mapping

from app.mrip.edgar.types import EdgarError

DEFAULT_CACHE_DIR = "/tmp/mrip-edgar-cache"
MAX_REQUESTS_PER_SECOND = 5
_MAX_BACKOFF_SECONDS = 120.0
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: str


HttpGet = Callable[[str, Mapping[str, str], float], HttpResponse]


def default_http_get(url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
    """Production transport (requests). Imported lazily so tests need no network stack."""
    import requests

    response = requests.get(url, headers=dict(headers), timeout=timeout)
    return HttpResponse(status_code=response.status_code, headers=dict(response.headers), body=response.text)


def cache_dir_from_env(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    return Path((env.get("EDGAR_CACHE_DIR") or "").strip() or DEFAULT_CACHE_DIR)


def user_agent_from_env(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return (env.get("EDGAR_USER_AGENT") or "").strip()


class EdgarClient:
    def __init__(
        self,
        user_agent: str,
        cache_dir: Path | str,
        *,
        http: HttpGet = default_http_get,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_requests_per_second: int = MAX_REQUESTS_PER_SECOND,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        agent = (user_agent or "").strip()
        if not agent:
            raise EdgarError(
                "EDGAR_USER_AGENT is empty. SEC requires a descriptive User-Agent, "
                "e.g. 'MRIP private research (you@example.com)'. Set it and retry."
            )
        if max_requests_per_second < 1:
            raise EdgarError("max_requests_per_second must be positive")
        self._headers = {"User-Agent": agent, "Accept-Encoding": "gzip, deflate"}
        self._cache_dir = Path(cache_dir)
        self._http = http
        self._clock = clock
        self._sleep = sleep
        self._max_rps = max_requests_per_second
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._timeout = timeout
        self._request_times: Deque[float] = deque()

    # -- public -----------------------------------------------------------

    def get_text(self, url: str) -> str:
        cached = self._cache_path(url)
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
        body = self._fetch(url)
        self._write_cache(cached, body)
        return body

    def get_json(self, url: str) -> Any:
        try:
            return json.loads(self.get_text(url))
        except json.JSONDecodeError as exc:
            raise EdgarError(f"invalid JSON from {url}") from exc

    def cache_path_for(self, url: str) -> Path:
        return self._cache_path(url)

    # -- internals --------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.txt"

    def _write_cache(self, path: Path, body: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, path)
        os.chmod(path, 0o644)

    def _throttle(self) -> None:
        """Allow at most ``max_requests_per_second`` requests in any rolling one-second window."""
        while True:
            now = self._clock()
            while self._request_times and now - self._request_times[0] >= 1.0:
                self._request_times.popleft()
            if len(self._request_times) < self._max_rps:
                self._request_times.append(now)
                return
            self._sleep(max(0.0, 1.0 - (now - self._request_times[0])))

    def _fetch(self, url: str) -> str:
        for attempt in range(self._max_retries + 1):
            self._throttle()
            try:
                response = self._http(url, self._headers, self._timeout)
            except OSError as exc:
                if attempt == self._max_retries:
                    raise EdgarError(f"network error for {url}: {exc}") from exc
                self._sleep(self._backoff(attempt))
                continue
            if response.status_code == 200:
                return response.body
            if response.status_code in _RETRYABLE_STATUS:
                if attempt == self._max_retries:
                    raise EdgarError(f"HTTP {response.status_code} for {url} after {attempt + 1} attempts")
                delay = _retry_after_seconds(response.headers)
                self._sleep(delay if delay is not None else self._backoff(attempt))
                continue
            raise EdgarError(f"HTTP {response.status_code} for {url}")
        raise EdgarError(f"no response for {url}")  # pragma: no cover - loop always returns or raises

    def _backoff(self, attempt: int) -> float:
        return min(_MAX_BACKOFF_SECONDS, self._backoff_base * (2**attempt))


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Numeric Retry-After only (HTTP-date form falls back to exponential backoff)."""
    raw = None
    for key, value in headers.items():
        if key.lower() == "retry-after":
            raw = value
            break
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    return min(_MAX_BACKOFF_SECONDS, max(0.0, seconds))
