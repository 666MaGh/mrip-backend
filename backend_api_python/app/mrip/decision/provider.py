"""DecisionProvider interface (ADR-0004)."""
from __future__ import annotations

from typing import Protocol, Sequence

from app.mrip.decision.types import Decision, DecisionRequest


class DecisionProvider(Protocol):
    def predict(self, request: DecisionRequest) -> Decision: ...

    def batch_predict(self, requests: Sequence[DecisionRequest]) -> list[Decision]:
        """Decisions in the same order as ``requests``."""
        ...

    def model_version(self) -> str: ...
