"""Evidence Engine: contract checks that need no database."""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path as FsPath

import pytest

from app.mrip.decision.mock import MockDecisionProvider
from app.mrip.evidence.assess import STANCE_FAMILY, assess_stance, record_assessed
from app.mrip.evidence.store import EvidenceStore, summarize
from app.mrip.evidence.types import Evidence, EvidenceError, RelationshipRef, SourceType, Stance
from app.mrip.relationships.types import NodeKey, NodeType, RelationType

MIGRATIONS = FsPath(__file__).resolve().parents[2] / "migrations"
REF = RelationshipRef(
    NodeKey(NodeType.COMPANY, "acme"), NodeKey(NodeType.COMPANY, "gridco"), RelationType.SUPPLIES
)
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _values(sql, column):
    m = re.search(rf"{column}\s+VARCHAR\(\d+\)\s+NOT NULL\s+CHECK\s*\(\s*{column}\s+IN\s*\((.*?)\)\s*\)", sql, re.S)
    assert m, column
    return set(re.findall(r"'([^']+)'", m.group(1)))


def test_sql_check_constraints_match_python_enums():
    sql = (MIGRATIONS / "mrip_20261001_evidence.sql").read_text(encoding="utf-8")
    graph_sql = (MIGRATIONS / "mrip_20261001_relationships.sql").read_text(encoding="utf-8")
    assert _values(sql, "stance") == {s.value for s in Stance}
    assert _values(sql, "source_type") == {s.value for s in SourceType}
    assert _values(sql, "relation_type") == _values(graph_sql, "relation_type")


def _ev(i, stance, source_type=SourceType.NEWS, day=1):
    return Evidence(
        id=i, src_id=1, dst_id=2, relation_type=RelationType.SUPPLIES, stance=stance,
        source_type=source_type, source_uri=f"u{i}", available_at=T0 + timedelta(days=day),
        ingested_at=T0, assessed_by="t",
    )


def test_summarize_counts_and_flags_conflict():
    s = summarize([_ev(1, Stance.SUPPORT), _ev(2, Stance.SUPPORT, SourceType.COT, 5), _ev(3, Stance.CONTRADICT)])
    assert s.total == 3 and s.conflicting
    assert s.by_stance == {Stance.SUPPORT: 2, Stance.CONTRADICT: 1, Stance.NEUTRAL: 0}
    assert s.by_source_type == {SourceType.NEWS: 2, SourceType.COT: 1}
    assert s.latest_available_at == T0 + timedelta(days=5)


def test_summarize_without_conflict_and_empty():
    assert not summarize([_ev(1, Stance.SUPPORT), _ev(2, Stance.NEUTRAL)]).conflicting
    empty = summarize([])
    assert empty.total == 0 and empty.latest_available_at is None and not empty.conflicting


class _NoDb:
    def __call__(self):
        raise AssertionError("must not touch the database for invalid arguments")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"source_uri": "  "}, "source_uri"),
        ({"assessed_by": ""}, "assessed_by"),
        ({"available_at": datetime(2026, 9, 1)}, "timezone-aware"),
        ({"assessor_confidence": 1.5}, "within"),
    ],
)
def test_add_requires_provenance_and_valid_arguments(kwargs, message):
    base = dict(source_uri="https://x/y", available_at=T0, assessed_by="manual")
    base.update(kwargs)
    store = EvidenceStore(_NoDb())
    with pytest.raises(EvidenceError, match=message):
        store.add(REF, Stance.SUPPORT, SourceType.NEWS, base.pop("source_uri"), base.pop("available_at"), **base)


def test_retract_requires_reason_and_naive_as_of_rejected():
    store = EvidenceStore(_NoDb())
    with pytest.raises(EvidenceError):
        store.retract(1, " ")
    with pytest.raises(EvidenceError, match="timezone-aware"):
        store.list_evidence(REF, as_of=datetime(2026, 9, 1))


def test_assess_stance_uses_a_closed_three_way_question():
    mock = MockDecisionProvider({STANCE_FAMILY: "support"})
    decision = assess_stance(mock, REF, ("Acme", "GridCo"), "Acme ships transformers to GridCo.")
    assert decision.answer == "support"
    question = mock.calls[0].question
    assert set(question.options) == {s.value for s in Stance}
    assert "Acme supplies GridCo" in question.instructions


class _RecordingStore:
    def __init__(self):
        self.added = []

    def add(self, *args, **kwargs):
        self.added.append((args, kwargs))
        return "stored"


def test_record_assessed_stores_assessed_stance_with_model_provenance():
    store = _RecordingStore()
    result = record_assessed(
        store, MockDecisionProvider({STANCE_FAMILY: "contradict"}, confidence=0.7, version="m-9"),
        REF, ("Acme", "GridCo"), "Acme denies supplying GridCo.", SourceType.NEWS, "https://x/y", T0,
    )
    assert result == "stored"
    (args, kwargs) = store.added[0]
    assert args[1] is Stance.CONTRADICT
    assert kwargs["model_version"] == "m-9" and kwargs["assessor_confidence"] == 0.7
    assert kwargs["excerpt"] == "Acme denies supplying GridCo."


def test_record_assessed_abstain_stores_nothing():
    store = _RecordingStore()
    result = record_assessed(
        store, MockDecisionProvider({STANCE_FAMILY: None}), REF, ("Acme", "GridCo"), "unclear",
        SourceType.NEWS, "https://x/y", T0,
    )
    assert result is None and store.added == []
