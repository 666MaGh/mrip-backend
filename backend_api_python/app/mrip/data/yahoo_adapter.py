"""YahooChartAdapter: price data from Yahoo Finance (private research only).

Maps Yahoo Finance v8 chart endpoint onto the normalized models in
``app.mrip.data.models``. Nothing outside this module sees Yahoo types.

Data quality and caveats (2026-10-03, Yahoo Finance public chart endpoint):
- Provides daily adjusted closes suitable for total-return-consistent analysis.
- Terms of service restrict use to private research; not licensed for redistribution
  or automated commercial use. This adapter is suitable for Markedet RI Platform's
  internal research and backtesting only.
- User-Agent is declared for transparency and responsible usage.
- timestamps are exchange-local; date conversion uses the exchange timezone from
  the response (typically Europe/Stockholm for OMX-listed securities).
- The adjusted close field is used when present (returned by close=adjclose);
  plain open, high, low are used as-is; volume is the reported share count.
  This ensures consistent total returns across dividend and split events.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

import requests

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import PriceBar, PriceSeries, Latency, Provenance

_GATEWAY = "yahoo-chart"
_ENDPOINT = "v8/finance/chart"
_BASE_URL = "https://query2.finance.yahoo.com"


class YahooChartAdapter:
    """Implements price_history() from FinancialDataGateway on top of Yahoo Finance."""

    terms_note: str = (
        "Yahoo Finance data: private research use only; "
        "not licensed for redistribution or commercial use"
    )

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        retries: int = 2,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 30.0,
        user_agent: str = "mrip-research/1.0 (private research)",
    ) -> None:
        """Initialize the Yahoo adapter.

        Args:
            session: requests.Session instance; if None, creates a new one.
            clock: callable returning current datetime (UTC); for testing.
            retries: number of extra retry attempts on transient failures.
            backoff_seconds: base backoff duration; multiplied exponentially per retry.
            sleep: callable to inject delay; for testing.
            timeout: HTTP request timeout in seconds.
            user_agent: User-Agent header value (identifies this client).
        """
        self._session = session if session is not None else requests.Session()
        self._clock = clock
        self._retries = max(0, retries)
        self._backoff = backoff_seconds
        self._sleep = sleep
        self._timeout = timeout
        self._user_agent = user_agent

    def price_history(
        self, symbol: str, start: date | None = None, end: date | None = None, interval: str = "1d"
    ) -> PriceSeries:
        """Fetch daily price history for a symbol.

        Args:
            symbol: ticker (e.g. "VOLV-B.ST" for Stockholm Exchange).
            start: earliest date (inclusive, UTC); defaults to 1990-01-01.
            end: latest date (inclusive, UTC); defaults to today.
            interval: bar interval; only "1d" (daily) is supported.

        Returns:
            PriceSeries with adjusted close used as the close price when available.

        Raises:
            ValueError: if interval is not "1d".
            DataUnavailable: if the fetch fails, response is malformed, or no usable bars.
        """
        if interval != "1d":
            raise ValueError(f"only 1d bars are supported, got {interval!r}")

        # Default date range: 1990-01-01 to today+1 (so end date is inclusive).
        if start is None:
            start = date(1990, 1, 1)
        if end is None:
            end = self._clock().date()

        period1 = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp())
        period2 = int(
            (datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)).timestamp()
        )

        url = f"{_BASE_URL}/v8/finance/chart/{symbol}"
        params = {
            "interval": "1d",
            "events": "div,splits",
            "period1": period1,
            "period2": period2,
        }
        headers = {"User-Agent": self._user_agent}

        response_data = self._call(
            lambda: self._fetch_json(url, params, headers),
        )

        if isinstance(response_data, dict) and "chart" in response_data:
            chart = response_data["chart"]
            if "error" in chart and chart["error"] is not None:
                error_msg = chart["error"].get("description", "unknown error")
                raise DataUnavailable(f"Yahoo chart endpoint error: {error_msg}")

            result = chart.get("result")
            if not result or not isinstance(result, list) or not result[0]:
                raise DataUnavailable(f"no price data returned for {symbol}")

            bars = self._parse_bars(result[0])
            if not bars:
                raise DataUnavailable(f"no usable price bars for {symbol}")

            meta = result[0].get("meta", {})
            exchange_tz_name = meta.get("exchangeTimezoneName", "UTC")

            return PriceSeries(
                symbol=symbol,
                interval=interval,
                bars=tuple(self._convert_bars_to_dates(bars, exchange_tz_name)),
                provenance=Provenance(
                    provider="yahoo",
                    gateway=_GATEWAY,
                    endpoint=_ENDPOINT,
                    fetched_at=self._clock(),
                    latency=Latency.UNKNOWN,
                ),
            )
        else:
            raise DataUnavailable(f"unexpected response structure for {symbol}")

    def index_history(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> PriceSeries:
        raise DataUnavailable("not supported by the Yahoo adapter")

    def vix_history(self, start: date | None = None, end: date | None = None) -> PriceSeries:
        raise DataUnavailable("not supported by the Yahoo adapter")

    def options_chain(self, symbol: str) -> Any:
        raise DataUnavailable("not supported by the Yahoo adapter")

    def cot(self, market: str, start: date | None = None, end: date | None = None) -> Any:
        raise DataUnavailable("not supported by the Yahoo adapter")

    def macro_series(
        self, series_id: str, start: date | None = None, end: date | None = None
    ) -> Any:
        raise DataUnavailable("not supported by the Yahoo adapter")

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def stockholm_symbol(short_name: str) -> str:
        """Convert a short name to a Yahoo Finance Stockholm symbol.

        Examples:
            "VOLV B" -> "VOLV-B.ST"
            "ERIC B" -> "ERIC-B.ST"
            "ABB" -> "ABB.ST"
            "ALIV SDB" -> "ALIV-SDB.ST"

        Args:
            short_name: a company short name or ticker (with optional class suffix).

        Returns:
            ticker suitable for Yahoo Finance OMX Stockholm queries.

        Raises:
            ValueError: if short_name is empty.
        """
        if not short_name or not short_name.strip():
            raise ValueError("short_name cannot be empty")
        normalized = short_name.strip().upper()
        # Replace spaces with hyphens to handle multi-word parts.
        normalized = normalized.replace(" ", "-")
        if not normalized.endswith(".ST"):
            normalized = f"{normalized}.ST"
        return normalized

    def _fetch_json(
        self, url: str, params: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        """Fetch and parse JSON from a URL.

        Raises:
            DataUnavailable: on HTTP error, JSON parse error, or connection failure.
        """
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
            if resp.status_code >= 500 or resp.status_code == 429:
                # Retry on 5xx and rate limit.
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            if resp.status_code >= 400:
                # Don't retry on other 4xx (404, 400, etc.).
                raise DataUnavailable(f"HTTP {resp.status_code} from Yahoo")
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            if "HTTP" in str(exc):
                raise
            raise DataUnavailable(f"HTTP error from Yahoo: {exc}") from exc
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise DataUnavailable(f"failed to fetch {url}: {exc}") from exc

    def _call(self, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """Execute a callable with exponential backoff retry on transient errors.

        Retries on exceptions and 5xx/429 status codes; other 4xx errors are fatal.
        """
        attempt = 0
        while True:
            try:
                return fn()
            except DataUnavailable as exc:
                # Fatal errors (4xx, malformed JSON, etc.) are not retried.
                if attempt >= self._retries:
                    raise
                # If the error might be transient, retry.
                if "HTTP 4" in str(exc) and "HTTP 429" not in str(exc):
                    raise
                self._sleep(self._backoff * (2**attempt))
                attempt += 1
            except Exception as exc:
                if attempt >= self._retries:
                    raise DataUnavailable(f"Yahoo fetch failed after {attempt + 1} attempt(s): {exc}") from exc
                self._sleep(self._backoff * (2**attempt))
                attempt += 1

    def _parse_bars(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse timestamp and OHLCV data from a Yahoo chart result.

        Returns a list of dicts with keys: timestamp, open, high, low, close, adjclose, volume.
        Skips bars where close is None.
        """
        bars = []
        timestamps = result.get("timestamp", [])
        indicators = result.get("indicators", {})
        quote = (indicators.get("quote") or [{}])[0]
        adjclose_list = ((indicators.get("adjclose") or [{}])[0]).get("adjclose", [])

        for i, ts in enumerate(timestamps):
            close = self._get_value(quote, "close", i)
            if close is None:
                continue

            bar = {
                "timestamp": ts,
                "open": self._get_value(quote, "open", i),
                "high": self._get_value(quote, "high", i),
                "low": self._get_value(quote, "low", i),
                "close": close,
                "adjclose": adjclose_list[i] if i < len(adjclose_list) else None,
                "volume": self._get_value(quote, "volume", i),
            }
            bars.append(bar)

        return bars

    @staticmethod
    def _get_value(obj: dict[str, Any], key: str, index: int) -> Any:
        """Get value at index from a list in obj[key], or None if not present/invalid."""
        lst = obj.get(key)
        if isinstance(lst, list) and index < len(lst):
            return lst[index]
        return None

    def _convert_bars_to_dates(self, bars: list[dict[str, Any]], tz_name: str) -> list[PriceBar]:
        """Convert timestamp bars to PriceBars with date conversion using exchange timezone."""
        try:
            tz = ZoneInfo(tz_name)
        except Exception:
            tz = ZoneInfo("UTC")

        converted = []
        for bar in bars:
            ts_unix = bar.get("timestamp")
            if ts_unix is None:
                continue

            # Convert Unix timestamp to datetime in UTC, then to exchange timezone, extract date.
            dt_utc = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
            dt_local = dt_utc.astimezone(tz)
            bar_date = dt_local.date()

            # Use adjusted close if available, else plain close.
            close = bar.get("adjclose")
            if close is None:
                close = bar.get("close")

            converted.append(
                PriceBar(
                    ts=bar_date,
                    open=bar.get("open"),
                    high=bar.get("high"),
                    low=bar.get("low"),
                    close=close,
                    volume=bar.get("volume"),
                )
            )

        return converted
