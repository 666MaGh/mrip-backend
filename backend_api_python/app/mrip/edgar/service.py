"""EDGAR 10-K import service: filings -> candidate statements -> HYPOTHESIS edges + evidence.

Direction convention (same as seeds/ai_infrastructure.json, src=customer, dst=supplier):
  * filer T's 10-K says M is a major customer  -> M CUSTOMER_OF T
  * filer T's 10-K says T depends on supplier M -> T CUSTOMER_OF M

Edges are created as HYPOTHESIS with source "sec-edgar-10k". Existing active edges
are never modified; new evidence is attached to them. Evidence de-duplication is
done by the evidence store (source_uri + excerpt hash), so re-runs are idempotent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from app.mrip.edgar.client import EdgarClient
from app.mrip.edgar.matcher import CompanyIndex
from app.mrip.edgar.parser import parse_10k_html
from app.mrip.edgar.types import (
    ASSESSED_BY,
    FORM_10K,
    PARSER_VERSION,
    PUBLISHER,
    CandidateStatement,
    CompanyRecord,
    EdgarError,
    FilingRef,
    ImportSummary,
)
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType, Stance
from app.mrip.relationships.types import Edge, EdgeStatus, Node, NodeKey, NodeType, RelationType

SOURCE_NAME = "sec-edgar-10k"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class GraphPort(Protocol):
    def get_node(self, node: NodeKey) -> Node | None: ...

    def upsert_node(self, node_type: NodeType, key: str, name: str, attributes: Mapping[str, Any] | None = None) -> Node: ...

    def find_active_edge(self, src: NodeKey, dst: NodeKey, relation_type: RelationType) -> Edge | None: ...

    def add_edge(
        self,
        src: NodeKey,
        dst: NodeKey,
        relation_type: RelationType,
        *,
        source: str,
        status: EdgeStatus = EdgeStatus.HYPOTHESIS,
        attributes: Mapping[str, Any] | None = None,
    ) -> Edge: ...


class EvidencePort(Protocol):
    def add(
        self,
        relationship: RelationshipRef,
        stance: Stance,
        source_type: SourceType,
        source_uri: str,
        available_at: datetime,
        *,
        assessed_by: str,
        source_title: str | None = None,
        publisher: str | None = None,
        excerpt: str | None = None,
        model_version: str | None = None,
        assessor_confidence: float | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Evidence: ...


class SourceClient(Protocol):
    def get_text(self, url: str) -> str: ...

    def get_json(self, url: str) -> Any: ...


def normalize_ticker(raw: str) -> str:
    """Upper-case, with the universe's '.' class separator mapped to SEC's '-' (BRK.B -> BRK-B)."""
    return raw.strip().upper().replace(".", "-")


def submissions_url(cik: int) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik:010d}.json"


def archive_url(cik: int, accession: str, primary_document: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{primary_document}"


def load_company_map(client: SourceClient) -> dict[str, CompanyRecord]:
    """ticker -> CompanyRecord from company_tickers.json (cached by the client)."""
    raw = client.get_json(COMPANY_TICKERS_URL)
    if not isinstance(raw, dict):
        raise EdgarError("company_tickers.json has an unexpected shape")
    records: dict[str, CompanyRecord] = {}
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        ticker = normalize_ticker(str(entry.get("ticker", "")))
        cik = entry.get("cik_str")
        title = str(entry.get("title", "")).strip()
        if not _TICKER_RE.match(ticker) or not isinstance(cik, int) or not title:
            continue
        records.setdefault(ticker, CompanyRecord(ticker=ticker, cik=cik, title=title))
    return records


def find_latest_10k(client: SourceClient, company: CompanyRecord, year: int | None) -> FilingRef | None:
    """Most recent 10-K in the submissions index (optionally filed in ``year``)."""
    data = client.get_json(submissions_url(company.cik))
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    dates = recent.get("filingDate") or []
    documents = recent.get("primaryDocument") or []
    best: FilingRef | None = None
    for form, accession, filed_raw, document in zip(forms, accessions, dates, documents):
        if form != FORM_10K or not document:
            continue
        filed = date.fromisoformat(str(filed_raw))
        if year is not None and filed.year != year:
            continue
        if best is None or filed > best.filed:
            best = FilingRef(
                ticker=company.ticker,
                cik=company.cik,
                accession=str(accession),
                form=form,
                filed=filed,
                primary_document=str(document),
                url=archive_url(company.cik, str(accession), str(document)),
            )
    return best


def _available_at(filed: date) -> datetime:
    return datetime(filed.year, filed.month, filed.day, tzinfo=timezone.utc)


def _relation_endpoints(kind: str, filer: str, counterparty: str) -> tuple[str, str]:
    """(src, dst) for CUSTOMER_OF. Customer statement: counterparty is the customer."""
    if kind == "customer":
        return counterparty, filer
    return filer, counterparty


@dataclass(slots=True)
class _Context:
    filing: FilingRef
    retrieved_at: datetime
    run_started: datetime
    titles: Mapping[str, str]


class EdgarImportService:
    def __init__(
        self,
        graph: GraphPort,
        evidence: EvidencePort,
        client: SourceClient,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._graph = graph
        self._evidence = evidence
        self._client = client
        self._clock = clock

    def run(
        self,
        *,
        filers: Sequence[str],
        universe: Iterable[str],
        year: int | None = None,
        dry_run: bool = False,
    ) -> ImportSummary:
        summary = ImportSummary(dry_run=dry_run)
        companies = load_company_map(self._client)
        universe_set = {normalize_ticker(t) for t in universe}
        summary.unknown_tickers = sorted(t for t in universe_set if t not in companies)
        index = CompanyIndex({t: companies[t].title for t in universe_set if t in companies})
        filer_list = [normalize_ticker(t) for t in filers]
        summary.filers_requested = len(filer_list)
        run_started = self._clock()
        new_pairs: dict[tuple[str, str], Edge | None] = {}

        for ticker in filer_list:
            if ticker not in universe_set or ticker not in companies:
                summary.filings_missing.append(f"{ticker}: not in universe or SEC ticker list")
                continue
            company = companies[ticker]
            filing = find_latest_10k(self._client, company, year)
            if filing is None:
                summary.filings_missing.append(f"{ticker}: no 10-K found" + (f" for {year}" if year else ""))
                continue
            statements = parse_10k_html(self._client.get_text(filing.url))
            summary.filings_processed += 1
            summary.statements_found += len(statements)
            ctx = _Context(
                filing=filing,
                retrieved_at=self._clock(),
                run_started=run_started,
                titles={t: companies[t].title for t in universe_set if t in companies},
            )
            for statement in statements:
                self._handle_statement(summary, ctx, statement, index, universe_set, new_pairs, dry_run)
        return summary

    # -- internals --------------------------------------------------------

    def _handle_statement(
        self,
        summary: ImportSummary,
        ctx: _Context,
        statement: CandidateStatement,
        index: CompanyIndex,
        universe: set[str],
        new_pairs: dict[tuple[str, str], Edge | None],
        dry_run: bool,
    ) -> None:
        filer = ctx.filing.ticker
        for name in statement.names:
            summary.names_seen += 1
            match = index.resolve(name)
            if match.generic:
                summary.names_generic += 1
                continue
            if match.ticker is None:
                summary.names_unmatched[name] = summary.names_unmatched.get(name, 0) + 1
                continue
            summary.names_matched += 1
            if match.ticker == filer:
                summary.skipped_self_reference += 1
                continue
            if match.ticker not in universe:
                continue
            src_ticker, dst_ticker = _relation_endpoints(statement.kind, filer, match.ticker)
            src, dst = NodeKey(NodeType.COMPANY, src_ticker), NodeKey(NodeType.COMPANY, dst_ticker)
            pair = (src_ticker, dst_ticker)
            if pair not in new_pairs:
                new_pairs[pair] = self._ensure_edge(summary, ctx, statement, src, dst, src_ticker, dst_ticker, dry_run)
            if not dry_run:
                self._attach_evidence(summary, ctx, statement, name, src, dst)
            else:
                summary.evidence_added += 1  # dry run: upper bound, nothing is written

    def _ensure_edge(
        self,
        summary: ImportSummary,
        ctx: _Context,
        statement: CandidateStatement,
        src: NodeKey,
        dst: NodeKey,
        src_ticker: str,
        dst_ticker: str,
        dry_run: bool,
    ) -> Edge | None:
        existing = self._graph.find_active_edge(src, dst, RelationType.CUSTOMER_OF)
        if existing is not None:
            summary.edges_existing += 1
            return existing
        summary.edges_added += 1
        summary.new_edges.append(
            {
                "customer": src_ticker if statement.kind == "customer" else dst_ticker,
                "supplier": dst_ticker if statement.kind == "customer" else src_ticker,
                "kind": statement.kind,
                "filer": ctx.filing.ticker,
                "percent": statement.percent,
                "form": ctx.filing.form,
                "filed": ctx.filing.filed.isoformat(),
                "accession": ctx.filing.accession,
                "sentence": statement.sentence[:200],
            }
        )
        if dry_run:
            return None
        for ticker in (src.key, dst.key):
            if self._graph.get_node(NodeKey(NodeType.COMPANY, ticker)) is None:
                name = ctx.titles.get(ticker, ticker)
                self._graph.upsert_node(NodeType.COMPANY, ticker, name, {"series": {"symbol": ticker}})
        return self._graph.add_edge(
            src,
            dst,
            RelationType.CUSTOMER_OF,
            source=SOURCE_NAME,
            status=EdgeStatus.HYPOTHESIS,
            attributes={
                "expected_sign": 1,
                "edgar": {
                    "accession": ctx.filing.accession,
                    "form": ctx.filing.form,
                    "filed": ctx.filing.filed.isoformat(),
                    "percent": statement.percent,
                    "parser_version": PARSER_VERSION,
                },
            },
        )

    def _attach_evidence(
        self,
        summary: ImportSummary,
        ctx: _Context,
        statement: CandidateStatement,
        name: str,
        src: NodeKey,
        dst: NodeKey,
    ) -> None:
        filing = ctx.filing
        stored = self._evidence.add(
            RelationshipRef(src=src, dst=dst, relation_type=RelationType.CUSTOMER_OF),
            Stance.SUPPORT,
            SourceType.REGULATORY_FILING,
            filing.url,
            _available_at(filing.filed),
            assessed_by=ASSESSED_BY,
            source_title=f"SEC 10-K {filing.ticker} {filing.filed.isoformat()}",
            publisher=PUBLISHER,
            excerpt=statement.sentence,
            attributes={
                "kind": statement.kind,
                "section": statement.section,
                "percent": statement.percent,
                "matched_name": name,
                "filer": filing.ticker,
                "accession": filing.accession,
                "form": filing.form,
                "filed": filing.filed.isoformat(),
                "retrieved_at": ctx.retrieved_at.isoformat(),
                "parser_version": PARSER_VERSION,
                "company_statement": True,
                "validated": False,
            },
        )
        if stored.ingested_at >= ctx.run_started:
            summary.evidence_added += 1
        else:
            summary.evidence_existing += 1
