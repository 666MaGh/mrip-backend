from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Mapping


class Kind(str, Enum):
    RELATIONSHIP_DIVERGENCE = "RELATIONSHIP_DIVERGENCE"
    DELAYED_REACTION = "DELAYED_REACTION"
    COT_CROWDING = "COT_CROWDING"
    COT_DIVERGENCE = "COT_DIVERGENCE"
    REGIME_SHIFT = "REGIME_SHIFT"
    TERM_BACKWARDATION = "TERM_BACKWARDATION"
    PEER_DIVERGENCE = "PEER_DIVERGENCE"
    GAMMA_REGIME_CHANGE = "GAMMA_REGIME_CHANGE"
    NEAR_GAMMA_FLIP = "NEAR_GAMMA_FLIP"
    WALL_PROXIMITY = "WALL_PROXIMITY"
    UNUSUAL_OPTIONS_ACTIVITY = "UNUSUAL_OPTIONS_ACTIVITY"
    HIGH_ZERO_DTE = "HIGH_ZERO_DTE"


@dataclass(frozen=True, slots=True)
class Observation:
    kind: Kind
    subject: str
    as_of: date
    headline: str
    magnitude: float
    details: Mapping[str, Any] = field(default_factory=dict)
    data_quality: Mapping[str, Any] = field(default_factory=dict)
    modeled: bool = False
    evidence: Mapping[str, int] | None = None
    reliability: float | None = None
    confidence: float | None = None
    liquidity: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError("magnitude must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class RankedItem:
    observation: Observation
    score: float
    components: Mapping[str, float]
    unknown: tuple[str, ...]
