"""MRIP master trading gate (ADR-0001).

Live trading and broker-facing HTTP routes exist only when ``TRADING_ENABLED``
is explicitly set to a truthy value. The default is disabled so a research
deployment never exposes execution or broker-credential endpoints.
"""
from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def trading_enabled() -> bool:
    """Return True only when TRADING_ENABLED is explicitly truthy."""
    return os.getenv("TRADING_ENABLED", "false").strip().lower() in _TRUTHY
