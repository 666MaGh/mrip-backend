"""TRADING_ENABLED master gate (MRIP ADR-0001)."""
import json
import os
import subprocess
import sys

import pytest

from app.utils.trading_gate import trading_enabled

BACKEND_ROOT = os.path.join(os.path.dirname(__file__), "..")

HIDDEN_PREFIXES = (
    "/api/credentials",
    "/api/ibkr",
    "/api/alpaca",
    "/api/quick-trade",
    "/api/agent/v1/quick_trade",
    "/api/account",
    "/api/agent/v1/strategies",
    "/api/agent/v1/runtime",
    "/api/agent/v1/trading",
)

# Under /api/strategies only research/authoring rules may remain mounted.
ALLOWED_STRATEGY_PREFIXES = (
    "/api/strategies/generate",
    "/api/strategies/verify",
    "/api/strategies/ai-workspace",
    "/api/strategies/script-",
)

_PROBE = """
import json
from app import create_app
app = create_app("testing")
rules = [r.rule for r in app.url_map.iter_rules()]
client = app.test_client()
print(json.dumps({
    "rules": rules,
    "health": client.get("/health").status_code,
    "credentials": client.get("/api/credentials/list").status_code,
}))
"""


def _probe(trading: str | None) -> dict:
    env = os.environ.copy()
    env["SKIP_STARTUP_HOOKS"] = "1"
    env["OPENAPI_ENABLED"] = "true"
    env.pop("TRADING_ENABLED", None)
    if trading is not None:
        env["TRADING_ENABLED"] = trading
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=BACKEND_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("", False), ("false", False), ("0", False), ("true", True), (" TRUE ", True), ("1", True)],
)
def test_trading_enabled_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TRADING_ENABLED", raising=False)
    else:
        monkeypatch.setenv("TRADING_ENABLED", value)
    assert trading_enabled() is expected


@pytest.mark.parametrize("value", [None, "false"])
def test_trading_routes_hidden_when_disabled(value):
    result = _probe(value)
    assert result["health"] == 200
    for rule in result["rules"]:
        assert not rule.startswith(HIDDEN_PREFIXES), f"trading route exposed: {rule}"
    assert result["credentials"] == 404
    for rule in result["rules"]:
        if rule.startswith("/api/strategies"):
            assert rule.startswith(ALLOWED_STRATEGY_PREFIXES), f"live strategy route exposed: {rule}"
    # Research/authoring routes stay available.
    assert any(r.startswith("/api/strategies/script-sources") for r in result["rules"])
    assert any(r.startswith("/api/backtest") for r in result["rules"])


def test_trading_routes_present_when_enabled():
    result = _probe("true")
    exposed = {p for p in HIDDEN_PREFIXES if any(r.startswith(p) for r in result["rules"])}
    assert "/api/credentials" in exposed
    assert "/api/quick-trade" in exposed
    assert any(r.endswith("/start") and r.startswith("/api/strategies/") for r in result["rules"])
    assert "/api/account" in exposed
    assert "/api/agent/v1/runtime" in exposed
