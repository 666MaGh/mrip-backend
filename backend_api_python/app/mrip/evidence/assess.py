"""Stance assessment through a DecisionProvider (LAYA when available).

LAYA only classifies whether a passage supports, contradicts or is neutral about
a claim; it computes nothing numeric. ABSTAIN means "do not store": insufficient
evidence is not evidence.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from app.mrip.decision.provider import DecisionProvider
from app.mrip.decision.types import Decision, DecisionQuestion, DecisionRequest
from app.mrip.evidence.store import EvidenceStore
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType, Stance

STANCE_FAMILY = "evidence_stance"


def _claim(name_src: str, name_dst: str, ref: RelationshipRef) -> str:
    return f"{name_src} {ref.relation_type.value.replace('_', ' ').lower()} {name_dst}"


def stance_question(claim: str) -> DecisionQuestion:
    return DecisionQuestion(
        family=STANCE_FAMILY,
        instructions=f"Regarding the claim '{claim}', what does the passage say?",
        options={
            Stance.SUPPORT.value: "The passage supports the claim.",
            Stance.CONTRADICT.value: "The passage contradicts the claim.",
            Stance.NEUTRAL.value: "The passage is relevant but neither supports nor contradicts the claim.",
        },
    )


def assess_stance(
    provider: DecisionProvider,
    relationship: RelationshipRef,
    names: tuple[str, str],
    passage: str,
) -> Decision:
    """Ask the provider about one passage. ``names`` are display names (src, dst)."""
    question = stance_question(_claim(names[0], names[1], relationship))
    return provider.predict(DecisionRequest(question=question, state=passage))


def record_assessed(
    store: EvidenceStore,
    provider: DecisionProvider,
    relationship: RelationshipRef,
    names: tuple[str, str],
    passage: str,
    source_type: SourceType,
    source_uri: str,
    available_at: datetime,
    *,
    source_title: str | None = None,
    publisher: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Evidence | None:
    """Assess a passage and store it as evidence; returns None when the provider abstains."""
    decision = assess_stance(provider, relationship, names, passage)
    if decision.abstained or decision.answer is None:
        return None
    return store.add(
        relationship,
        Stance(decision.answer),
        source_type,
        source_uri,
        available_at,
        assessed_by="decision-provider",
        source_title=source_title,
        publisher=publisher,
        excerpt=passage,
        model_version=decision.model_version,
        assessor_confidence=decision.raw_confidence,
        attributes=attributes,
    )
