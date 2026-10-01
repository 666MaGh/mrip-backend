"""Evidence Engine domain types (work 005).

Evidence says whether a source SUPPORTS, CONTRADICTS or is NEUTRAL about a
relationship. Provenance is mandatory. No composite "evidence score" is
computed here: weights must be validated empirically before use (spec).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from app.mrip.relationships.types import NodeKey, RelationType


class Stance(str, Enum):
    SUPPORT = "support"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


class SourceType(str, Enum):
    ANNUAL_REPORT = "ANNUAL_REPORT"
    QUARTERLY_REPORT = "QUARTERLY_REPORT"
    EARNINGS_TRANSCRIPT = "EARNINGS_TRANSCRIPT"
    INVESTOR_PRESENTATION = "INVESTOR_PRESENTATION"
    REGULATORY_FILING = "REGULATORY_FILING"
    COMPANY_STATEMENT = "COMPANY_STATEMENT"
    MACRO_DATA = "MACRO_DATA"
    COMMODITY_DATA = "COMMODITY_DATA"
    MARKET_DATA = "MARKET_DATA"
    COT = "COT"
    OPTIONS_DATA = "OPTIONS_DATA"
    REGULATORY_DATA = "REGULATORY_DATA"
    NEWS = "NEWS"
    STATISTICAL_TEST = "STATISTICAL_TEST"


class EvidenceError(Exception):
    """Invalid evidence operation (missing provenance, unknown relationship, bad argument)."""


@dataclass(frozen=True, slots=True)
class RelationshipRef:
    """The logical relationship evidence is about (independent of edge versions)."""

    src: NodeKey
    dst: NodeKey
    relation_type: RelationType


@dataclass(frozen=True, slots=True)
class Evidence:
    id: int
    src_id: int
    dst_id: int
    relation_type: RelationType
    stance: Stance
    source_type: SourceType
    source_uri: str
    available_at: datetime
    ingested_at: datetime
    assessed_by: str
    source_title: str | None = None
    publisher: str | None = None
    excerpt: str | None = None
    model_version: str | None = None
    assessor_confidence: float | None = None  # the assessor's raw number, not calibrated
    attributes: Mapping[str, Any] = field(default_factory=dict)
    retracted_at: datetime | None = None
    retract_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    """Counts only. ``conflicting`` is True when both support and contradiction exist."""

    total: int
    by_stance: Mapping[Stance, int]
    by_source_type: Mapping[SourceType, int]
    conflicting: bool
    latest_available_at: datetime | None
