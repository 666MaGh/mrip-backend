"""Types and constants for the related-neighbours view (work 019)."""
from __future__ import annotations

RELATED_VERSION = "related-v1"
DISCLAIMER = "Historiska samband, inte prognos eller råd."
SIGNAL_LABEL = "Samband, ej prognos"


class UnknownSymbol(Exception):
    """The symbol is neither a relationship-graph node nor a stored price series."""
