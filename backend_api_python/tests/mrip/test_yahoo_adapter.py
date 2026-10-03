"""YahooChartAdapter normalization tests against a fake requests session (no network)."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone, timedelta
from types import SimpleNamespace
from unittest import mock

import pytest

from app.mrip.data.gateway import DataUnavailable
from app.mrip.data.models import Latency
from app.mrip.data.yahoo_adapter import YahooChartAdapter

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


class FakeResponse:
    """Mimics requests.Response for testing."""

    def __init__(self, status_code: int, json_data: dict | None = None, raise_on_json: bool = False):
        self.status_code = status_code
        self._json_data = json_data
        self._raise_on_json = raise_on_json

    def json(self) -> dict:
        if self._raise_on_json:
            raise json.JSONDecodeError("invalid JSON", "", 0)
        if self._json_data is None:
            raise ValueError("no JSON data")
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class FakeSession:
    """Mimics requests.Session for testing."""

    def __init__(self):
        self.requests: list[tuple[str, dict, dict]] = []
        self._response_factory = None

    def set_response_factory(self, factory):
        self._response_factory = factory

    def get(self, url: str, params: dict = None, headers: dict = None, timeout: float = None) -> FakeResponse:
        self.requests.append((url, params or {}, headers or {}))
        if self._response_factory is None:
            return FakeResponse(200, {})
        return self._response_factory(url, params, headers)


def make_adapter(session: FakeSession | None = None, **kwargs) -> YahooChartAdapter:
    """Create an adapter with test defaults."""
    if session is None:
        session = FakeSession()
    return YahooChartAdapter(
        session=session,
        clock=lambda: NOW,
        sleep=lambda s: None,  # no-op sleep for testing
        **kwargs,
    )


def yahoo_chart_response(
    symbol: str = "VOLV-B.ST",
    bars: list[dict] | None = None,
    include_error: bool = False,
    exchange_tz: str = "Europe/Stockholm",
) -> dict:
    """Build a realistic Yahoo chart response."""
    if bars is None:
        bars = [
            {
                "timestamp": 1633104000,  # 2021-10-01 12:00 UTC = 2021-10-01 in Europe/Stockholm
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000000,
                "adjclose": 101.0,
            },
            {
                "timestamp": 1633190400,  # 2021-10-02 12:00 UTC = 2021-10-02 in Europe/Stockholm
                "open": 101.0,
                "high": 103.0,
                "low": 100.5,
                "close": 102.5,
                "volume": 900000,
                "adjclose": 102.5,
            },
            {
                "timestamp": 1633276800,  # 2021-10-03 12:00 UTC = 2021-10-03 in Europe/Stockholm
                "open": 102.0,
                "high": 104.0,
                "low": 101.0,
                "close": None,  # Will be skipped
                "volume": 850000,
                "adjclose": None,
            },
        ]

    # Convert bar dicts to columnar format (as Yahoo returns it).
    timestamps = [b["timestamp"] for b in bars]
    opens = [b.get("open") for b in bars]
    highs = [b.get("high") for b in bars]
    lows = [b.get("low") for b in bars]
    closes = [b.get("close") for b in bars]
    volumes = [b.get("volume") for b in bars]
    adjcloses = [b.get("adjclose") for b in bars]

    result = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": symbol,
                        "currency": "SEK",
                        "exchangeTimezoneName": exchange_tz,
                    },
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": volumes,
                            }
                        ],
                        "adjclose": [{"adjclose": adjcloses}],
                    },
                }
            ]
        }
    }

    if include_error:
        result["chart"]["error"] = {"description": "Symbol not found"}

    return result


def test_price_history_parses_chart_with_adjusted_close():
    """Test that adjusted close is used when available, and null closes are skipped."""
    session = FakeSession()
    session.set_response_factory(
        lambda url, params, headers: FakeResponse(200, yahoo_chart_response(bars=[
            {
                "timestamp": 1633104000,
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000000,
                "adjclose": 101.5,  # Adjusted close differs from close
            },
            {
                "timestamp": 1633190400,
                "open": 101.0,
                "high": 103.0,
                "low": 100.5,
                "close": 102.5,
                "volume": 900000,
                "adjclose": None,  # No adjustment
            },
            {
                "timestamp": 1633276800,
                "open": 102.0,
                "high": 104.0,
                "low": 101.0,
                "close": None,  # Skipped
                "volume": 850000,
                "adjclose": 103.0,
            },
        ]))
    )

    adapter = make_adapter(session)
    series = adapter.price_history("VOLV-B.ST", date(2021, 10, 1), date(2021, 10, 3))

    assert series.symbol == "VOLV-B.ST"
    assert series.interval == "1d"
    assert len(series.bars) == 2  # Third bar skipped
    assert series.bars[0].close == 101.5  # Adjusted close used
    assert series.bars[0].open == 100.0
    assert series.bars[0].high == 102.0
    assert series.bars[0].low == 99.0
    assert series.bars[0].volume == 1000000
    assert series.bars[1].close == 102.5  # Plain close (no adjustment)
    assert series.bars[1].volume == 900000


def test_price_history_converts_timestamps_to_exchange_timezone():
    """Test that timestamps are converted to the exchange timezone date."""
    session = FakeSession()
    # Timestamp 1633104000 is 2021-10-01 12:00 UTC.
    # In Europe/Stockholm (UTC+2 in October), this is 2021-10-01 14:00, so date is 2021-10-01.
    # In UTC, it would be 2021-10-01, but we need to verify we're using the exchange tz.
    session.set_response_factory(
        lambda url, params, headers: FakeResponse(200, yahoo_chart_response(exchange_tz="Europe/Stockholm"))
    )

    adapter = make_adapter(session)
    series = adapter.price_history("VOLV-B.ST")

    assert series.bars[0].ts == date(2021, 10, 1)
    assert series.bars[1].ts == date(2021, 10, 2)


def test_price_history_computes_period1_period2_from_dates():
    """Test that period1/period2 are computed correctly from start/end dates."""
    session = FakeSession()
    session.set_response_factory(lambda url, params, headers: FakeResponse(200, yahoo_chart_response()))

    adapter = make_adapter(session)
    adapter.price_history("VOLV-B.ST", start=date(2021, 10, 1), end=date(2021, 10, 31))

    # Extract the params from the last request.
    assert len(session.requests) == 1
    _, params, _ = session.requests[0]

    # period1 should be the Unix timestamp of 2021-10-01 00:00 UTC.
    period1_expected = int(datetime(2021, 10, 1, 0, 0, tzinfo=timezone.utc).timestamp())
    # period2 should be the Unix timestamp of 2021-11-01 00:00 UTC (end+1 day).
    period2_expected = int(datetime(2021, 11, 1, 0, 0, tzinfo=timezone.utc).timestamp())

    assert params["period1"] == period1_expected
    assert params["period2"] == period2_expected


def test_price_history_defaults_to_1990_to_today_plus_one():
    """Test that default date range is 1990-01-01 to today+1."""
    session = FakeSession()
    session.set_response_factory(lambda url, params, headers: FakeResponse(200, yahoo_chart_response()))

    adapter = make_adapter(session)
    adapter.price_history("VOLV-B.ST")

    _, params, _ = session.requests[0]
    period1_expected = int(datetime(1990, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp())
    # period2 should be tomorrow (relative to NOW).
    tomorrow = NOW.date() + timedelta(days=1)
    period2_expected = int(datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc).timestamp())

    assert params["period1"] == period1_expected
    assert params["period2"] == period2_expected


def test_price_history_rejects_unsupported_interval():
    """Test that unsupported intervals raise ValueError."""
    adapter = make_adapter()
    with pytest.raises(ValueError, match="only 1d bars are supported"):
        adapter.price_history("VOLV-B.ST", interval="1h")


def test_price_history_sends_user_agent_header():
    """Test that the User-Agent header is sent."""
    session = FakeSession()
    session.set_response_factory(lambda url, params, headers: FakeResponse(200, yahoo_chart_response()))

    adapter = make_adapter(session, user_agent="test-agent/1.0")
    adapter.price_history("VOLV-B.ST")

    _, _, headers = session.requests[0]
    assert headers.get("User-Agent") == "test-agent/1.0"


def test_price_history_retries_on_429_with_exponential_backoff():
    """Test that 429 (rate limit) triggers retry with exponential backoff."""
    session = FakeSession()
    sleeps = []

    call_count = [0]

    def response_factory(url, params, headers):
        call_count[0] += 1
        if call_count[0] == 1:
            return FakeResponse(429, {})  # Rate limited
        return FakeResponse(200, yahoo_chart_response())

    session.set_response_factory(response_factory)

    adapter = YahooChartAdapter(
        session=session,
        clock=lambda: NOW,
        sleep=sleeps.append,
        retries=2,
        backoff_seconds=1.0,
    )

    series = adapter.price_history("VOLV-B.ST")
    assert len(series.bars) == 2  # Successfully got data after retry
    assert sleeps == [1.0]  # One backoff before second attempt


def test_price_history_exhausts_retries_on_429():
    """Test that retries are exhausted and DataUnavailable is raised."""
    session = FakeSession()
    session.set_response_factory(lambda url, params, headers: FakeResponse(429, {}))

    adapter = YahooChartAdapter(
        session=session,
        clock=lambda: NOW,
        sleep=lambda s: None,
        retries=2,
    )

    with pytest.raises(DataUnavailable, match="failed after 3 attempt"):
        adapter.price_history("VOLV-B.ST")


def test_price_history_does_not_retry_on_404():
    """Test that 404 errors are not retried."""
    session = FakeSession()
    session.set_response_factory(lambda url, params, headers: FakeResponse(404, {}))

    adapter = YahooChartAdapter(
        session=session,
        clock=lambda: NOW,
        sleep=lambda s: None,
        retries=2,
    )

    with pytest.raises(DataUnavailable, match="HTTP 404"):
        adapter.price_history("ZZZZ-INVALID.ST")


def test_price_history_raises_on_chart_error():
    """Test that chart.error in response raises DataUnavailable with description."""
    session = FakeSession()
    session.set_response_factory(
        lambda url, params, headers: FakeResponse(
            200, yahoo_chart_response(include_error=True)
        )
    )

    adapter = make_adapter(session)
    with pytest.raises(DataUnavailable, match="Symbol not found"):
        adapter.price_history("ZZZZ-INVALID.ST")


def test_price_history_raises_on_empty_result():
    """Test that empty result raises DataUnavailable."""
    session = FakeSession()
    session.set_response_factory(lambda url, params, headers: FakeResponse(200, {"chart": {"result": []}}))

    adapter = make_adapter(session)
    with pytest.raises(DataUnavailable, match="no price data returned"):
        adapter.price_history("VOLV-B.ST")


def test_price_history_raises_on_non_json():
    """Test that non-JSON response raises DataUnavailable."""
    session = FakeSession()
    session.set_response_factory(lambda url, params, headers: FakeResponse(200, raise_on_json=True))

    adapter = make_adapter(session)
    with pytest.raises(DataUnavailable, match="failed to fetch"):
        adapter.price_history("VOLV-B.ST")


def test_price_history_raises_on_no_usable_bars():
    """Test that response with all null closes raises DataUnavailable."""
    session = FakeSession()
    session.set_response_factory(
        lambda url, params, headers: FakeResponse(200, yahoo_chart_response(bars=[
            {"timestamp": 1633104000, "open": 100.0, "high": 102.0, "low": 99.0, "close": None, "volume": 1000000, "adjclose": None},
        ]))
    )

    adapter = make_adapter(session)
    with pytest.raises(DataUnavailable, match="no usable price bars"):
        adapter.price_history("VOLV-B.ST")


def test_provenance_fields():
    """Test that provenance is correctly set."""
    session = FakeSession()
    session.set_response_factory(lambda url, params, headers: FakeResponse(200, yahoo_chart_response()))

    adapter = make_adapter(session)
    series = adapter.price_history("VOLV-B.ST")

    assert series.provenance.provider == "yahoo"
    assert series.provenance.gateway == "yahoo-chart"
    assert series.provenance.endpoint == "v8/finance/chart"
    assert series.provenance.fetched_at == NOW
    assert series.provenance.latency == Latency.UNKNOWN


def test_terms_note_class_attribute():
    """Test that terms_note is set and accessible."""
    assert "private research use only" in YahooChartAdapter.terms_note
    assert "not licensed for redistribution" in YahooChartAdapter.terms_note


def test_unsupported_index_history():
    """Test that index_history raises DataUnavailable."""
    adapter = make_adapter()
    with pytest.raises(DataUnavailable, match="not supported by the Yahoo adapter"):
        adapter.index_history("SPX")


def test_unsupported_vix_history():
    """Test that vix_history raises DataUnavailable."""
    adapter = make_adapter()
    with pytest.raises(DataUnavailable, match="not supported by the Yahoo adapter"):
        adapter.vix_history()


def test_unsupported_options_chain():
    """Test that options_chain raises DataUnavailable."""
    adapter = make_adapter()
    with pytest.raises(DataUnavailable, match="not supported by the Yahoo adapter"):
        adapter.options_chain("VOLV-B.ST")


def test_unsupported_cot():
    """Test that cot raises DataUnavailable."""
    adapter = make_adapter()
    with pytest.raises(DataUnavailable, match="not supported by the Yahoo adapter"):
        adapter.cot("123456")


def test_unsupported_macro_series():
    """Test that macro_series raises DataUnavailable."""
    adapter = make_adapter()
    with pytest.raises(DataUnavailable, match="not supported by the Yahoo adapter"):
        adapter.macro_series("DGS10")


def test_stockholm_symbol_conversions():
    """Test stockholm_symbol static helper."""
    assert YahooChartAdapter.stockholm_symbol("VOLV B") == "VOLV-B.ST"
    assert YahooChartAdapter.stockholm_symbol("ERIC B") == "ERIC-B.ST"
    assert YahooChartAdapter.stockholm_symbol("ABB") == "ABB.ST"
    assert YahooChartAdapter.stockholm_symbol("ALIV SDB") == "ALIV-SDB.ST"


def test_stockholm_symbol_with_leading_trailing_whitespace():
    """Test that stockholm_symbol handles whitespace."""
    assert YahooChartAdapter.stockholm_symbol("  VOLV B  ") == "VOLV-B.ST"


def test_stockholm_symbol_empty_raises_error():
    """Test that stockholm_symbol raises ValueError on empty input."""
    with pytest.raises(ValueError, match="cannot be empty"):
        YahooChartAdapter.stockholm_symbol("")


def test_stockholm_symbol_whitespace_only_raises_error():
    """Test that stockholm_symbol raises ValueError on whitespace-only input."""
    with pytest.raises(ValueError, match="cannot be empty"):
        YahooChartAdapter.stockholm_symbol("   ")


@pytest.mark.skipif(os.getenv("MRIP_LIVE_YAHOO") != "1", reason="requires MRIP_LIVE_YAHOO=1")
def test_live_yahoo_fetch_volv():
    """Live test: fetch real VOLV-B.ST data from 2021-01-01."""
    adapter = YahooChartAdapter()
    series = adapter.price_history("VOLV-B.ST", start=date(2021, 1, 1))

    assert len(series.bars) > 900, f"Expected >900 bars, got {len(series.bars)}"
    assert all(bar.close is not None and bar.close > 0 for bar in series.bars), "Some bars have non-positive close"

    last_date = series.bars[-1].ts
    days_ago = (NOW.date() - last_date).days
    assert days_ago <= 10, f"Last bar is {days_ago} days old, expected <=10"
