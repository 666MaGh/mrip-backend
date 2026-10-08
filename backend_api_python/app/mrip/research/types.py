"""Research Card domain types (work 017).

A Research Card composes the existing engines for one security. Every section is
independent: a section whose inputs are missing returns ``status: "unavailable"``
with a reason (ABSTAIN) instead of inventing numbers. Modeled option quantities
are kept under ``modeled`` and labelled MODELED / ESTIMATED.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

CARD_VERSION = "research-card-v0-uncalibrated"
EVIDENCE_LEVEL_VERSION = "evidence-level-v0-uncalibrated"
MAX_PATHS = 3


class UnknownSecurity(Exception):
    """The symbol is neither in the price store nor in the relationship graph."""


@dataclass(frozen=True, slots=True)
class SectionOutcome:
    """One card section plus the inputs, versions and evidence that fed it (the "Why?" panel)."""

    body: Mapping[str, Any]
    inputs: tuple[str, ...] = ()
    versions: tuple[str, ...] = ()
    evidence_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class WhyEntry:
    section: str
    inputs: tuple[str, ...]
    versions: tuple[str, ...]
    evidence_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "inputs": list(self.inputs),
            "versions": list(self.versions),
            "evidence_ids": list(self.evidence_ids),
        }


def unavailable(reason: str, *, inputs: tuple[str, ...] = (), versions: tuple[str, ...] = (), **extra: Any) -> SectionOutcome:
    """ABSTAIN: the section has no defensible value. ``extra`` carries observed context only."""
    body: dict[str, Any] = {"status": "unavailable", "reason": reason, **extra}
    return SectionOutcome(body=body, inputs=inputs, versions=versions)
