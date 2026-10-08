import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

import numpy as np
import pytest

from app.mrip.discover.serialize import to_jsonable


@dataclass(frozen=True)
class Inner:
    value: float


@dataclass(frozen=True)
class Outer:
    name: str
    inner: Inner
    expiry: date


class Color(Enum):
    RED = "red"


def test_nested_dataclass_converts_to_dict():
    out = to_jsonable(Outer("x", Inner(1.5), date(2026, 10, 9)))
    assert out == {"name": "x", "inner": {"value": 1.5}, "expiry": "2026-10-09"}
    json.dumps(out)


def test_scalar_conversions():
    assert to_jsonable(Color.RED) == "red"
    assert to_jsonable(datetime(2026, 1, 2, 3, 4)) == "2026-01-02T03:04:00"
    assert to_jsonable(Decimal("1.25")) == 1.25
    assert to_jsonable(np.float64(2.5)) == 2.5
    assert to_jsonable(np.int64(3)) == 3
    assert to_jsonable(np.array([1, 2])) == [1, 2]
    assert to_jsonable({"k": ("a", {"b"})}) == {"k": ["a", ["b"]]}


def test_nan_and_inf_become_none():
    assert to_jsonable({"a": float("nan"), "b": [float("inf"), 1.0]}) == {"a": None, "b": [None, 1.0]}


def test_mapping_keys_must_be_strings():
    assert to_jsonable({"k": {"n": 1}}) == {"k": {"n": 1}}
    with pytest.raises(TypeError):
        to_jsonable({(1, 2): "x"})


def test_unknown_type_raises():
    class Opaque:
        pass

    with pytest.raises(TypeError):
        to_jsonable({"x": Opaque()})
