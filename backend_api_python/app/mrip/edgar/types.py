"""Domain types for the EDGAR 10-K import."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

PARSER_VERSION = "edgar-10k-v0.1-uncalibrated"
ASSESSED_BY = f"edgar:{PARSER_VERSION}"
PUBLISHER = "SEC EDGAR"
FORM_10K = "10-K"

StatementKind = Literal["customer", "supplier"]


class EdgarError(Exception):
    """Invalid EDGAR operation (missing user agent, HTTP failure, malformed response)."""


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    """One ticker from company_tickers.json."""

    ticker: str
    cik: int
    title: str


@dataclass(frozen=True, slots=True)
class FilingRef:
    """A specific 10-K filing and the URL of its primary document."""

    ticker: str
    cik: int
    accession: str  # with dashes, as published
    form: str
    filed: date
    primary_document: str
    url: str


@dataclass(frozen=True, slots=True)
class CandidateStatement:
    """A sentence from a filing that may describe a customer or supplier relationship."""

    sentence: str
    section: str
    percent: float | None
    kind: StatementKind
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NameMatch:
    """Outcome of resolving one mentioned name.

    ``ticker`` is set only for a confident match. ``generic`` names are ignored
    on purpose (e.g. "customers", "the U.S. government").
    """

    name: str
    ticker: str | None
    generic: bool = False


@dataclass(slots=True)
class ImportSummary:
    dry_run: bool
    filers_requested: int = 0
    filings_processed: int = 0
    filings_missing: list[str] = field(default_factory=list)
    unknown_tickers: list[str] = field(default_factory=list)
    statements_found: int = 0
    names_seen: int = 0
    names_matched: int = 0
    names_generic: int = 0
    names_unmatched: dict[str, int] = field(default_factory=dict)
    edges_added: int = 0
    edges_existing: int = 0
    evidence_added: int = 0
    evidence_existing: int = 0
    skipped_self_reference: int = 0
    new_edges: list[dict[str, object]] = field(default_factory=list)

    def top_unmatched(self, limit: int = 15) -> list[tuple[str, int]]:
        return sorted(self.names_unmatched.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
