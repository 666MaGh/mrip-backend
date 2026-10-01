"""Contract tests for DecisionProvider implementations (no network, no model)."""
import pytest

from app.mrip.decision.laya import LayaDecisionProvider
from app.mrip.decision.mock import MockDecisionProvider
from app.mrip.decision.types import AbstainReason, Decision, DecisionError, DecisionQuestion, DecisionRequest

QUESTION = DecisionQuestion(
    family="relationship_type",
    instructions="How does A relate to B?",
    options={"supplies": "A supplies B", "competes": "A competes with B", "unrelated": "no relation"},
)
OTHER = DecisionQuestion(family="materiality", instructions="Material?", options={"yes": "material", "no": "not material"})


def req(question=QUESTION, state="Acme sells transformers to GridCo."):
    return DecisionRequest(question=question, state=state)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, handler):
        self.handler, self.posts, self.gets = handler, [], []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append((url, json, headers))
        return self.handler(url, json)

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        return FakeResponse({"status": "ok", "revisions": {"typed-decisions": "abc123"}})


def laya_answer(family, label, conf=0.8, low=False):
    a = {"type": "choice", "choice": label, "probabilities": {label: conf}, "answer_confidence": conf}
    if low:
        a["low_confidence"] = True
    return {"model": "typed-decisions", "answers": {family: a}}


def make_laya(handler, **kw):
    return LayaDecisionProvider("http://laya:8000/", session=FakeSession(handler), **kw)


# -- shared behaviour -------------------------------------------------------

def test_decision_invariant_answer_none_iff_abstained():
    with pytest.raises(ValueError):
        Decision(family="f", answer="x", abstained=True, raw_confidence=None, model_version="v")
    with pytest.raises(ValueError):
        Decision(family="f", answer=None, abstained=False, raw_confidence=None, model_version="v")


def test_question_needs_two_options():
    with pytest.raises(ValueError):
        DecisionQuestion(family="f", instructions="?", options={"only": "one"})


# -- mock -------------------------------------------------------------------

def test_mock_answers_abstains_and_validates_script():
    mock = MockDecisionProvider({"relationship_type": "supplies", "materiality": None})
    d = mock.predict(req())
    assert (d.answer, d.abstained, d.model_version) == ("supplies", False, "mock-1")
    a = mock.predict(req(OTHER))
    assert a.abstained and a.answer is None and a.abstain_reason is AbstainReason.NO_EVIDENCE
    assert mock.predict(req(DecisionQuestion("unscripted", "?", {"a": "a", "b": "b"}))).abstained
    with pytest.raises(ValueError):
        MockDecisionProvider({"relationship_type": "bogus"}).predict(req())


def test_mock_batch_preserves_order():
    mock = MockDecisionProvider({"relationship_type": "competes", "materiality": "yes"})
    out = mock.batch_predict([req(OTHER), req(), req(OTHER)])
    assert [d.answer for d in out] == ["yes", "competes", "yes"]
    assert mock.model_version() == "mock-1"


# -- LAYA client ------------------------------------------------------------

def test_laya_predict_builds_request_and_returns_typed_decision():
    session_holder = {}

    def handler(url, body):
        session_holder["body"] = (url, body)
        return FakeResponse(laya_answer("relationship_type", "supplies", 0.83))

    laya = make_laya(handler, api_key="k", model="typed-decisions", min_confidence=0.5)
    d = laya.predict(req())

    url, body = session_holder["body"]
    assert url == "http://laya:8000/v1/systemone"
    assert body["state"] == "Acme sells transformers to GridCo."
    assert body["model"] == "typed-decisions" and body["min_confidence"] == 0.5
    q = body["questions"]["relationship_type"]
    assert q["type"] == "choice" and q["criteria"] == dict(QUESTION.options)
    assert laya._session.posts[0][2] == {"Authorization": "Bearer k"}
    assert (d.answer, d.abstained, d.raw_confidence) == ("supplies", False, 0.83)
    assert d.model_version == "laya:typed-decisions@abc123"


def test_laya_low_confidence_becomes_abstain_with_raw_confidence_kept():
    laya = make_laya(lambda u, b: FakeResponse(laya_answer("relationship_type", "supplies", 0.31, low=True)))
    d = laya.predict(req())
    assert d.abstained and d.answer is None
    assert d.abstain_reason is AbstainReason.LOW_CONFIDENCE and d.raw_confidence == 0.31


def test_laya_rejects_label_outside_option_set():
    laya = make_laya(lambda u, b: FakeResponse(laya_answer("relationship_type", "invented")))
    with pytest.raises(DecisionError, match="not an option"):
        laya.predict(req())


@pytest.mark.parametrize(
    "payload",
    [{"answers": {}}, {"answers": {"relationship_type": {"type": "score", "score": 1}}}, ["not", "an", "object"]],
)
def test_laya_malformed_payload_raises(payload):
    with pytest.raises(DecisionError):
        make_laya(lambda u, b: FakeResponse(payload)).predict(req())


def test_laya_http_failure_raises_decision_error_with_cause():
    laya = make_laya(lambda u, b: FakeResponse({}, status=503))
    with pytest.raises(DecisionError, match="/v1/systemone failed") as err:
        laya.predict(req())
    assert isinstance(err.value.__cause__, RuntimeError)


def test_laya_batch_groups_by_question_and_preserves_order():
    calls = []

    def handler(url, body):
        calls.append((url, body))
        (family,) = body["questions"]
        label = "supplies" if family == "relationship_type" else "yes"
        return FakeResponse({"results": [laya_answer(family, label) for _ in body["states"]]})

    laya = make_laya(handler)
    out = laya.batch_predict([req(state="s1"), req(OTHER, "s2"), req(state="s3")])

    assert [d.family for d in out] == ["relationship_type", "materiality", "relationship_type"]
    assert [d.answer for d in out] == ["supplies", "yes", "supplies"]
    assert len(calls) == 2  # one call per distinct question set
    assert calls[0][0].endswith("/v1/systemone/batch") and calls[0][1]["states"] == ["s1", "s3"]


def test_laya_batch_length_mismatch_raises():
    laya = make_laya(lambda u, b: FakeResponse({"results": []}))
    with pytest.raises(DecisionError, match="number of requests"):
        laya.batch_predict([req()])


def test_laya_model_version_is_cached_and_unknown_without_revisions():
    laya = make_laya(lambda u, b: FakeResponse({}))
    assert laya.model_version() == "laya:typed-decisions@abc123"
    laya.model_version()
    assert len(laya._session.gets) == 1
