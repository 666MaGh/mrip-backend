"""Live LAYA smoke test against a running laya-serve. Opt in with MRIP_LIVE_LAYA_URL."""
import os

import pytest

from app.mrip.decision.laya import LayaDecisionProvider
from app.mrip.decision.types import AbstainReason, DecisionQuestion, DecisionRequest

URL = os.getenv("MRIP_LIVE_LAYA_URL")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not URL, reason="set MRIP_LIVE_LAYA_URL to run live")]

QUESTION = DecisionQuestion(
    family="relationship_type",
    instructions="How does Acme relate to GridCo?",
    options={"supplies": "Acme supplies GridCo", "competes": "Acme competes with GridCo", "unrelated": "no relation"},
)
STATE = "Acme Corp manufactures power transformers and sells them to GridCo, a regional utility."


def test_live_predict_batch_and_version():
    laya = LayaDecisionProvider(URL, timeout=600)
    decision = laya.predict(DecisionRequest(QUESTION, STATE))
    assert decision.answer in QUESTION.options and not decision.abstained
    assert 0.0 < decision.raw_confidence <= 1.0
    assert set(decision.probabilities) == set(QUESTION.options)
    batch = laya.batch_predict([DecisionRequest(QUESTION, STATE)] * 2)
    assert [d.answer for d in batch] == [decision.answer] * 2
    assert laya.model_version().startswith("laya:")


def test_live_min_confidence_gate_becomes_abstain():
    laya = LayaDecisionProvider(URL, min_confidence=1.0, timeout=600)
    decision = laya.predict(DecisionRequest(QUESTION, STATE))
    assert decision.abstained and decision.answer is None
    assert decision.abstain_reason is AbstainReason.LOW_CONFIDENCE
