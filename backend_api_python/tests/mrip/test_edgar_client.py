from __future__ import annotations

import pytest

from app.mrip.edgar.client import EdgarClient, HttpResponse
from app.mrip.edgar.types import EdgarError

URL = "https://www.sec.gov/files/company_tickers.json"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeHttp:
    def __init__(self, responses: list[HttpResponse | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, str], float]] = []
        self.times: list[float] = []
        self.clock: FakeClock | None = None

    def __call__(self, url: str, headers, timeout: float) -> HttpResponse:
        self.calls.append((url, dict(headers), timeout))
        if self.clock is not None:
            self.times.append(self.clock.now)
        item = self._responses.pop(0) if self._responses else HttpResponse(200, {}, "{}")
        if isinstance(item, Exception):
            raise item
        return item


def _ok(body: str = '{"a": 1}') -> HttpResponse:
    return HttpResponse(200, {}, body)


def test_empty_or_blank_user_agent_is_refused(tmp_path):
    for agent in ("", "   "):
        with pytest.raises(EdgarError, match="EDGAR_USER_AGENT is empty"):
            EdgarClient(agent, tmp_path, http=FakeHttp([]))


def test_user_agent_is_sent_on_every_request(tmp_path):
    http = FakeHttp([_ok()])
    client = EdgarClient("MRIP research (ops@example.com)", tmp_path, http=http, sleep=lambda s: None)
    client.get_text(URL)
    assert http.calls[0][1]["User-Agent"] == "MRIP research (ops@example.com)"


def test_responses_are_cached_by_url_and_not_refetched(tmp_path):
    http = FakeHttp([_ok('{"x": 2}')])
    first = EdgarClient("agent@example.com", tmp_path, http=http, sleep=lambda s: None)
    assert first.get_json(URL) == {"x": 2}
    second_http = FakeHttp([])
    second = EdgarClient("agent@example.com", tmp_path, http=second_http, sleep=lambda s: None)
    assert second.get_json(URL) == {"x": 2}
    assert len(http.calls) == 1
    assert second_http.calls == []
    assert first.cache_path_for(URL).parent == tmp_path


def test_failed_responses_are_not_cached(tmp_path):
    http = FakeHttp([HttpResponse(404, {}, "missing")])
    client = EdgarClient("agent@example.com", tmp_path, http=http, sleep=lambda s: None)
    with pytest.raises(EdgarError, match="HTTP 404"):
        client.get_text(URL)
    assert not client.cache_path_for(URL).exists()
    assert len(http.calls) == 1  # 404 is not retried


def test_rate_limit_never_exceeds_five_requests_in_any_second(tmp_path):
    clock = FakeClock()
    http = FakeHttp([_ok() for _ in range(12)])
    http.clock = clock
    client = EdgarClient(
        "agent@example.com", tmp_path, http=http, clock=clock, sleep=clock.sleep, max_requests_per_second=5
    )
    for index in range(12):
        client.get_text(f"https://www.sec.gov/item/{index}.json")
    times = http.times
    assert len(times) == 12
    for start in range(len(times) - 5):
        assert times[start + 5] - times[start] >= 1.0
    assert clock.sleeps  # throttling actually happened


def test_retries_503_with_exponential_backoff_then_succeeds(tmp_path):
    clock = FakeClock()
    http = FakeHttp([HttpResponse(503, {}, ""), HttpResponse(500, {}, ""), _ok('{"ok": true}')])
    client = EdgarClient("agent@example.com", tmp_path, http=http, clock=clock, sleep=clock.sleep, backoff_base=1.0)
    assert client.get_json(URL) == {"ok": True}
    backoffs = [s for s in clock.sleeps if s >= 1.0]
    assert backoffs[:2] == [1.0, 2.0]


def test_retry_after_header_is_honoured_on_429(tmp_path):
    clock = FakeClock()
    http = FakeHttp([HttpResponse(429, {"Retry-After": "7"}, ""), _ok()])
    client = EdgarClient("agent@example.com", tmp_path, http=http, clock=clock, sleep=clock.sleep)
    client.get_text(URL)
    assert 7.0 in clock.sleeps


def test_gives_up_after_max_retries(tmp_path):
    clock = FakeClock()
    http = FakeHttp([HttpResponse(503, {}, "")] * 3)
    client = EdgarClient(
        "agent@example.com", tmp_path, http=http, clock=clock, sleep=clock.sleep, max_retries=2
    )
    with pytest.raises(EdgarError, match="after 3 attempts"):
        client.get_text(URL)
    assert len(http.calls) == 3


def test_network_errors_are_retried_then_raised(tmp_path):
    clock = FakeClock()
    http = FakeHttp([OSError("reset"), _ok()])
    client = EdgarClient("agent@example.com", tmp_path, http=http, clock=clock, sleep=clock.sleep)
    assert client.get_text(URL) == '{"a": 1}'
    assert len(http.calls) == 2
