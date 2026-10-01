"""Typed semantic-decision contract (ADR-0004).

LAYA (or any provider) only makes bounded semantic choices. It never computes
gamma, GEX, volatility, regressions, correlations, z-scores or forecasts.
ABSTAIN is a first-class outcome, not an error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class AbstainReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    NO_EVIDENCE = "no_evidence"


class DecisionError(Exception):
    """The provider failed or returned something that violates the contract."""


@dataclass(frozen=True, slots=True)
class DecisionQuestion:
    """A closed-set question: ``options`` maps each allowed label to its meaning."""

    family: str
    instructions: str
    options: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("a decision question needs at least two options")


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    question: DecisionQuestion
    state: str


@dataclass(frozen=True, slots=True)
class Decision:
    """One decision. ``answer`` is None exactly when ``abstained`` is True.

    ``raw_confidence`` is the provider's own number. It is not our calibrated
    probability; calibration is applied downstream and is what the UI shows.
    """

    family: str
    answer: str | None
    abstained: bool
    raw_confidence: float | None
    model_version: str
    probabilities: Mapping[str, float] = field(default_factory=dict)
    abstain_reason: AbstainReason | None = None

    def __post_init__(self) -> None:
        if self.abstained != (self.answer is None):
            raise ValueError("answer must be None exactly when abstained")
