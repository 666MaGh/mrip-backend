"""Segmented calibration with hierarchical back-off.

Segment dimensions are ordered from MOST GENERAL to MOST SPECIFIC, e.g.
``("horizon_kind", "model_version", "vix_regime")``. A key is the ``dim=value``
pairs of a prefix of that list; level 0 (the empty key) is the global segment.
Fitting keeps a segment only when it has enough samples; lookup tries the most
specific segment first and drops dimensions from the right until one exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Mapping, Sequence, TypeVar

P = TypeVar("P")
X = TypeVar("X")


@dataclass(frozen=True, slots=True)
class SegmentEntry(Generic[P]):
    key: str
    level: int  # number of dimensions in the key (0 = global)
    n: int
    params: P


def segment_key(attrs: Mapping[str, str], dims: Sequence[str]) -> str:
    """Canonical key for ``dims`` (empty string for the global segment); raises if an attribute is missing."""
    missing = [d for d in dims if d not in attrs]
    if missing:
        raise KeyError(f"missing segment attribute(s): {missing}")
    return "|".join(f"{d}={attrs[d]}" for d in dims)


def fit_segments(
    items: Sequence[tuple[Mapping[str, str], X]],
    dims: Sequence[str],
    min_samples: int,
    fit_fn: Callable[[list[X]], P],
) -> dict[str, SegmentEntry[P]]:
    """Fit ``fit_fn`` on every segment (every prefix level) that has at least ``min_samples`` items."""
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    entries: dict[str, SegmentEntry[P]] = {}
    for level in range(len(dims) + 1):
        groups: dict[str, list[X]] = {}
        for attrs, item in items:
            groups.setdefault(segment_key(attrs, dims[:level]), []).append(item)
        for key, members in groups.items():
            if len(members) >= min_samples:
                entries[key] = SegmentEntry(key=key, level=level, n=len(members), params=fit_fn(members))
    return entries


def resolve_segment(
    entries: Mapping[str, SegmentEntry[P]], dims: Sequence[str], attrs: Mapping[str, str]
) -> SegmentEntry[P] | None:
    """Most specific fitted segment for ``attrs``, or None when not even the global one exists."""
    for level in range(len(dims), -1, -1):
        try:
            key = segment_key(attrs, dims[:level])
        except KeyError:
            continue  # an unknown attribute makes that level (and more specific ones) unusable
        if key in entries:
            return entries[key]
    return None
