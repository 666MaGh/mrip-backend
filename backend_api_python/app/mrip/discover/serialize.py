"""Strict conversion of observation payloads to JSON-compatible values."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping


def to_jsonable(value: Any) -> Any:
    """Recursively convert a value to JSON-compatible primitives.

    Unknown types raise TypeError instead of being silently stringified.
    """
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else float(value)
    if isinstance(value, Decimal):
        return to_jsonable(float(value))
    if isinstance(value, date):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            norm = to_jsonable(key)
            if not isinstance(norm, str):
                raise TypeError(f"unsupported mapping key type: {type(key).__name__}")
            out[norm] = to_jsonable(item)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    # Numpy scalars and arrays expose tolist() (returns native Python values).
    tolist = getattr(type(value), "tolist", None)
    if callable(tolist) and type(value).__module__.startswith("numpy"):
        return to_jsonable(value.tolist())
    raise TypeError(f"unsupported type for JSON storage: {type(value).__name__}")
