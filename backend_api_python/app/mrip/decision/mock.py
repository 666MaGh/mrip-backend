"""Deterministic DecisionProvider for tests and offline development."""
from __future__ import annotations

from typing import Mapping, Sequence

from app.mrip.decision.types import AbstainReason, Decision, DecisionRequest


class MockDecisionProvider:
    """Answers from a fixed ``family -> label`` script; ``None`` (or unscripted) abstains."""

    def __init__(
        self,
        answers: Mapping[str, str | None] | None = None,
        confidence: float = 0.9,
        version: str = "mock-1",
    ) -> None:
        self._answers = dict(answers or {})
        self._confidence = confidence
        self._version = version
        self.calls: list[DecisionRequest] = []

    def predict(self, request: DecisionRequest) -> Decision:
        self.calls.append(request)
        family = request.question.family
        answer = self._answers.get(family)
        if answer is None:
            return Decision(
                family=family,
                answer=None,
                abstained=True,
                raw_confidence=None,
                model_version=self._version,
                abstain_reason=AbstainReason.NO_EVIDENCE,
            )
        if answer not in request.question.options:
            raise ValueError(f"scripted answer {answer!r} is not an option of {family!r}")
        return Decision(
            family=family,
            answer=answer,
            abstained=False,
            raw_confidence=self._confidence,
            model_version=self._version,
            probabilities={label: (self._confidence if label == answer else 0.0) for label in request.question.options},
        )

    def batch_predict(self, requests: Sequence[DecisionRequest]) -> list[Decision]:
        return [self.predict(r) for r in requests]

    def model_version(self) -> str:
        return self._version
