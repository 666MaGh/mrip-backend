"""SEC EDGAR 10-K relationship import (work 021).

Produces HYPOTHESIS edges with point-in-time SUPPORT evidence. Deterministic
parsing only: no LLM and no numeric computation happens here.
"""
from __future__ import annotations

from app.mrip.edgar.types import PARSER_VERSION

__all__ = ["PARSER_VERSION"]
