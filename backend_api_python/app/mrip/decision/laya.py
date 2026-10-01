"""LayaDecisionProvider: HTTP client for a self-hosted ``laya-serve`` (ADR-0004).

LAYA needs torch/transformers, so it runs as its own service and is reached
over HTTP (``POST /v1/systemone`` and ``/v1/systemone/batch``, ``GET /health``).
Response shape (read from laya 0.3.22 source): ``{"answers": {qid: {"type":
"choice", "choice": label, "probabilities": {...}, "answer_confidence": float,
["low_confidence": true]}}, "model": ...}``. Not yet verified against a running
server; see docs/work/003.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import requests

from app.mrip.decision.types import (
    AbstainReason,
    Decision,
    DecisionError,
    DecisionQuestion,
    DecisionRequest,
)


class LayaDecisionProvider:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        model: str | None = None,
        min_confidence: float | None = None,
        timeout: float = 30.0,
        session: Any | None = None,
    ) -> None:
        """``min_confidence`` is LAYA's own opt-in gate; flagged answers become ABSTAIN."""
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._min_confidence = min_confidence
        self._timeout = timeout
        self._session = session or requests.Session()
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._version: str | None = None

    # -- DecisionProvider -------------------------------------------------

    def predict(self, request: DecisionRequest) -> Decision:
        body = self._body(request.question)
        body["state"] = request.state
        payload = self._post("/v1/systemone", body)
        return self._decision(request.question, payload)

    def batch_predict(self, requests_: Sequence[DecisionRequest]) -> list[Decision]:
        decisions: list[Decision | None] = [None] * len(requests_)
        # laya-serve takes one shared question set per batch call: group by question.
        groups: dict[tuple[str, str, tuple[tuple[str, str], ...]], list[int]] = {}
        for index, req in enumerate(requests_):
            q = req.question
            groups.setdefault((q.family, q.instructions, tuple(q.options.items())), []).append(index)
        for indices in groups.values():
            question = requests_[indices[0]].question
            body = self._body(question)
            body["states"] = [requests_[i].state for i in indices]
            payload = self._post("/v1/systemone/batch", body)
            results = payload.get("results")
            if not isinstance(results, list) or len(results) != len(indices):
                raise DecisionError("batch response does not match the number of requests")
            for i, result in zip(indices, results):
                decisions[i] = self._decision(question, result)
        return [d for d in decisions if d is not None]

    def model_version(self) -> str:
        if self._version is None:
            try:
                response = self._session.get(
                    self._base_url + "/health", headers=self._headers, timeout=self._timeout
                )
                response.raise_for_status()
                health = response.json()
            except Exception as exc:
                raise DecisionError(f"laya /health failed: {exc}") from exc
            revisions = health.get("revisions") or {}
            parts = [f"{name}@{rev}" for name, rev in sorted(revisions.items())]
            self._version = "laya:" + (",".join(parts) if parts else "unknown")
        return self._version

    # -- helpers ----------------------------------------------------------

    def _body(self, question: DecisionQuestion) -> dict[str, Any]:
        body: dict[str, Any] = {
            "questions": {
                question.family: {
                    "type": "choice",
                    "instructions": question.instructions,
                    "criteria": dict(question.options),
                }
            }
        }
        if self._model:
            body["model"] = self._model
        if self._min_confidence is not None:
            body["min_confidence"] = self._min_confidence
        return body

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self._session.post(
                self._base_url + path, json=body, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # network, HTTP status or JSON failure at the service boundary
            raise DecisionError(f"laya {path} failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise DecisionError(f"laya {path} returned a non-object payload")
        return payload

    def _decision(self, question: DecisionQuestion, result: Mapping[str, Any]) -> Decision:
        answers = result.get("answers")
        answer = answers.get(question.family) if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise DecisionError(f"no choice answer for {question.family!r} in laya response")
        label = answer.get("choice")
        if label not in question.options:
            raise DecisionError(f"laya returned {label!r}, not an option of {question.family!r}")
        confidence = answer.get("answer_confidence")
        raw = float(confidence) if isinstance(confidence, (int, float)) else None
        probabilities = {
            str(k): float(v) for k, v in (answer.get("probabilities") or {}).items() if isinstance(v, (int, float))
        }
        version = self.model_version()
        if answer.get("low_confidence"):
            return Decision(
                family=question.family,
                answer=None,
                abstained=True,
                raw_confidence=raw,
                model_version=version,
                probabilities=probabilities,
                abstain_reason=AbstainReason.LOW_CONFIDENCE,
            )
        return Decision(
            family=question.family,
            answer=label,
            abstained=False,
            raw_confidence=raw,
            model_version=version,
            probabilities=probabilities,
        )
