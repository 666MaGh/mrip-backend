from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping

import pytest

from app.mrip.edgar.parser import parse_10k_html
from app.mrip.edgar.service import EdgarImportService, archive_url, submissions_url
from app.mrip.edgar.types import ASSESSED_BY, PARSER_VERSION
from app.mrip.evidence.types import Evidence, RelationshipRef, SourceType, Stance
from app.mrip.relationships.types import Edge, EdgeStatus, Node, NodeKey, NodeType, RelationType

TICKERS = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
    "2": {"cik_str": 1730168, "ticker": "AVGO", "title": "Broadcom Inc."},
    "3": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
    "4": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
}

SUBMISSIONS_NVDA = {
    "filings": {
        "recent": {
            "form": ["10-Q", "10-K", "10-K", "8-K"],
            "accessionNumber": ["0001045810-26-000099", "0001045810-26-000020", "0001045810-25-000023", "0001045810-26-000100"],
            "filingDate": ["2026-05-20", "2026-02-25", "2025-02-26", "2026-06-01"],
            "primaryDocument": ["q.htm", "nvda-20260125.htm", "nvda-20250126.htm", "e.htm"],
        }
    }
}

SUBMISSIONS_MSFT = {
    "filings": {
        "recent": {
            "form": ["10-K"],
            "accessionNumber": ["0000789019-26-000010"],
            "filingDate": ["2026-07-30"],
            "primaryDocument": ["msft-10k.htm"],
        }
    }
}

NVDA_10K = """<html><body>
<h3>Item 1. Business</h3>
<p>Our networking products rely on a single supplier, Broadcom, for certain components.</p>
<p>Our customers include Amazon, Alphabet and Microsoft Corporation for data center systems.</p>
<h3>Concentration of Credit Risk</h3>
<p>Sales to Microsoft Corporation accounted for 16% of our total revenue in fiscal 2026.</p>
<p>Two direct customers accounted for 39% of total accounts receivable as of January 25, 2026.</p>
<p>We depend on TSMC to manufacture substantially all of our semiconductor products.</p>
</body></html>"""

MSFT_10K = """<html><body>
<h3>Concentration of Credit Risk</h3>
<p>Sales to NVIDIA Corporation accounted for 12% of our total revenue this fiscal year.</p>
</body></html>"""


class FakeClock:
    """Each call returns a strictly later instant, so ingestion order is observable."""

    def __init__(self) -> None:
        self._now = datetime(2026, 10, 8, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        from datetime import timedelta

        self._now = self._now + timedelta(seconds=1)
        return self._now


class FakeClient:
    def __init__(self, documents: Mapping[str, Any]) -> None:
        self.documents = dict(documents)
        self.requests: list[str] = []

    def get_text(self, url: str) -> str:
        self.requests.append(url)
        value = self.documents[url]
        return value if isinstance(value, str) else json.dumps(value)

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_text(url))


@dataclass
class FakeGraph:
    nodes: dict[tuple[NodeType, str], Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    writes: int = 0

    def get_node(self, node: NodeKey) -> Node | None:
        return self.nodes.get((node.node_type, node.key))

    def upsert_node(self, node_type, key, name, attributes=None) -> Node:
        self.writes += 1
        node = Node(id=len(self.nodes) + 1, node_type=node_type, key=key, name=name, attributes=dict(attributes or {}))
        self.nodes[(node_type, key)] = node
        return node

    def find_active_edge(self, src: NodeKey, dst: NodeKey, relation_type: RelationType) -> Edge | None:
        s, d = self.nodes.get((src.node_type, src.key)), self.nodes.get((dst.node_type, dst.key))
        if s is None or d is None:
            return None
        for edge in self.edges:
            if edge.retired_version is None and edge.src_id == s.id and edge.dst_id == d.id and edge.relation_type is relation_type:
                return edge
        return None

    def add_edge(self, src, dst, relation_type, *, source, status=EdgeStatus.HYPOTHESIS, attributes=None) -> Edge:
        self.writes += 1
        s, d = self.nodes[(src.node_type, src.key)], self.nodes[(dst.node_type, dst.key)]
        edge = Edge(
            id=len(self.edges) + 1, src_id=s.id, dst_id=d.id, relation_type=relation_type, status=status,
            source=source, created_version=1, attributes=dict(attributes or {}),
        )
        self.edges.append(edge)
        return edge

    def key_of(self, edge: Edge) -> tuple[str, str, str]:
        by_id = {n.id: n for n in self.nodes.values()}
        return by_id[edge.src_id].key, by_id[edge.dst_id].key, edge.relation_type.value


@dataclass
class FakeEvidence:
    clock: FakeClock
    items: list[Evidence] = field(default_factory=list)
    writes: int = 0

    def add(self, relationship: RelationshipRef, stance, source_type, source_uri, available_at, *, assessed_by,
            source_title=None, publisher=None, excerpt=None, model_version=None, assessor_confidence=None,
            attributes=None) -> Evidence:
        for item in self.items:
            if item.source_uri == source_uri and item.excerpt == excerpt and item.relation_type is relationship.relation_type:
                return item
        self.writes += 1
        item = Evidence(
            id=len(self.items) + 1, src_id=0, dst_id=0, relation_type=relationship.relation_type, stance=stance,
            source_type=source_type, source_uri=source_uri, available_at=available_at, ingested_at=self.clock(),
            assessed_by=assessed_by, source_title=source_title, publisher=publisher, excerpt=excerpt,
            attributes=dict(attributes or {}),
        )
        self.items.append(item)
        return item


NVDA_URL = archive_url(1045810, "0001045810-26-000020", "nvda-20260125.htm")
NVDA_2025_URL = archive_url(1045810, "0001045810-25-000023", "nvda-20250126.htm")
MSFT_URL = archive_url(789019, "0000789019-26-000010", "msft-10k.htm")


def _client(extra: Mapping[str, Any] | None = None) -> FakeClient:
    docs: dict[str, Any] = {
        "https://www.sec.gov/files/company_tickers.json": TICKERS,
        submissions_url(1045810): SUBMISSIONS_NVDA,
        submissions_url(789019): SUBMISSIONS_MSFT,
        NVDA_URL: NVDA_10K,
        NVDA_2025_URL: NVDA_10K,
        MSFT_URL: MSFT_10K,
    }
    docs.update(extra or {})
    return FakeClient(docs)


def _service(graph: FakeGraph, evidence: FakeEvidence, client: FakeClient, clock: FakeClock) -> EdgarImportService:
    return EdgarImportService(graph, evidence, client, clock=clock)


def _world_tuple() -> tuple[FakeGraph, FakeEvidence, FakeClock]:
    clock = FakeClock()
    return FakeGraph(), FakeEvidence(clock), clock



def _run(graph, evidence, client, clock, *, filers=("NVDA", "MSFT"), universe=("NVDA", "MSFT", "AVGO", "AMZN", "GOOGL"), dry_run=False, year=None):
    return _service(graph, evidence, client, clock).run(filers=list(filers), universe=list(universe), year=year, dry_run=dry_run)


def test_fixtures_contain_the_expected_statements():
    kinds = [(s.kind, s.names) for s in parse_10k_html(NVDA_10K)]
    assert ("supplier", ("Broadcom",)) in kinds
    assert ("customer", ("Amazon", "Alphabet", "Microsoft Corporation")) in kinds


def test_direction_customer_is_counterparty_customer_of_filer():
    graph, evidence, clock = _world_tuple()
    _run(graph, evidence, _client(), clock)
    edges = {graph.key_of(e) for e in graph.edges}
    # NVDA's 10-K: Microsoft is a major customer -> MSFT CUSTOMER_OF NVDA
    assert ("MSFT", "NVDA", "CUSTOMER_OF") in edges
    # NVDA's 10-K: NVDA depends on supplier Broadcom -> NVDA CUSTOMER_OF AVGO
    assert ("NVDA", "AVGO", "CUSTOMER_OF") in edges
    # MSFT's 10-K: NVIDIA is a major customer -> NVDA CUSTOMER_OF MSFT
    assert ("NVDA", "MSFT", "CUSTOMER_OF") in edges


def test_new_edges_are_hypotheses_with_filing_provenance_and_support_evidence():
    graph, evidence, clock = _world_tuple()
    summary = _run(graph, evidence, _client(), clock)
    edge = next(e for e in graph.edges if graph.key_of(e) == ("MSFT", "NVDA", "CUSTOMER_OF"))
    assert edge.status is EdgeStatus.HYPOTHESIS
    assert edge.source == "sec-edgar-10k"
    assert edge.attributes["expected_sign"] == 1
    assert edge.attributes["edgar"]["accession"] == "0001045810-26-000020"
    assert edge.attributes["edgar"]["form"] == "10-K"
    assert edge.attributes["edgar"]["filed"] == "2026-02-25"
    assert "percent" in edge.attributes["edgar"]
    item = next(i for i in evidence.items if i.excerpt and "Microsoft Corporation accounted for 16%" in i.excerpt)
    assert item.stance is Stance.SUPPORT
    assert item.source_type is SourceType.REGULATORY_FILING
    assert item.source_uri == NVDA_URL
    assert item.available_at == datetime(2026, 2, 25, tzinfo=timezone.utc)
    assert item.assessed_by == ASSESSED_BY == f"edgar:{PARSER_VERSION}"
    assert item.publisher == "SEC EDGAR"
    assert item.source_title == "SEC 10-K NVDA 2026-02-25"
    assert item.attributes["percent"] == 16.0
    assert item.attributes["parser_version"] == "edgar-10k-v0.1-uncalibrated"
    assert summary.edges_added == len(graph.edges)


def test_rerun_is_idempotent_and_adds_no_edges_or_evidence():
    graph, evidence, clock = _world_tuple()
    first = _run(graph, evidence, _client(), clock)
    edges_after_first, evidence_after_first = len(graph.edges), len(evidence.items)
    second = _run(graph, evidence, _client(), clock)
    assert len(graph.edges) == edges_after_first
    assert len(evidence.items) == evidence_after_first
    assert second.edges_added == 0
    assert second.edges_existing == first.edges_added
    assert second.evidence_added == 0
    assert second.evidence_existing == first.evidence_added


def test_existing_edge_is_never_modified_and_new_evidence_is_attached():
    graph, evidence, clock = _world_tuple()
    graph.upsert_node(NodeType.COMPANY, "MSFT", "Microsoft", {"series": {"symbol": "MSFT"}})
    graph.upsert_node(NodeType.COMPANY, "NVDA", "NVIDIA", {"series": {"symbol": "NVDA"}})
    seeded = graph.add_edge(
        NodeKey(NodeType.COMPANY, "MSFT"), NodeKey(NodeType.COMPANY, "NVDA"), RelationType.CUSTOMER_OF,
        source="seed", attributes={"hand": "curated"},
    )
    writes_before = graph.writes
    summary = _run(graph, evidence, _client(), clock)
    assert graph.edges[0].source == "seed"
    assert graph.edges[0].attributes == {"hand": "curated"}
    assert graph.edges[0].status is EdgeStatus.HYPOTHESIS
    assert summary.edges_existing >= 1
    assert any(i.excerpt and "Microsoft Corporation accounted for 16%" in i.excerpt for i in evidence.items)
    assert graph.writes > writes_before  # other new edges were still written


def test_dry_run_writes_nothing_but_reports_candidates():
    graph, evidence, clock = _world_tuple()
    summary = _run(graph, evidence, _client(), clock, dry_run=True)
    assert graph.writes == 0 and graph.nodes == {} and graph.edges == []
    assert evidence.writes == 0 and evidence.items == []
    assert summary.dry_run is True
    assert summary.edges_added >= 3
    assert {e["kind"] for e in summary.new_edges} == {"customer", "supplier"}


def test_unmatched_names_are_counted_and_never_invented():
    graph, evidence, clock = _world_tuple()
    summary = _run(graph, evidence, _client(), clock)
    assert summary.names_unmatched.get("TSMC") == 1
    assert ("TSMC", 1) in summary.top_unmatched()
    assert all(e["customer"] != "TSMC" and e["supplier"] != "TSMC" for e in summary.new_edges)
    assert set(n[1] for n in graph.nodes) <= {"NVDA", "MSFT", "AVGO", "AMZN", "GOOGL"}


def test_nodes_are_created_only_for_universe_tickers():
    graph, evidence, clock = _world_tuple()
    _run(graph, evidence, _client(), clock, universe=("NVDA", "MSFT"))
    assert {key for (_, key) in graph.nodes} <= {"NVDA", "MSFT"}
    assert all(e.attributes.get("edgar") for e in graph.edges)


def test_year_filter_and_missing_filings_are_reported():
    graph, evidence, clock = _world_tuple()
    summary = _run(graph, evidence, _client(), clock, year=2025)
    assert summary.filings_processed == 1  # only NVDA has a 2025 10-K in the fixture
    assert all("NVDA" not in m for m in summary.filings_missing)
    graph, evidence, clock = _world_tuple()
    summary_missing = _run(graph, evidence, _client(), clock, year=2019)
    assert summary_missing.filings_processed == 0
    assert len(summary_missing.filings_missing) == 2


def test_filer_outside_universe_is_skipped_without_requests_for_its_filing():
    graph, evidence, clock = _world_tuple()
    client = _client()
    summary = _run(graph, evidence, client, clock, filers=("NVDA", "XYZ"), universe=("NVDA", "MSFT"))
    assert any("XYZ" in m for m in summary.filings_missing)
    assert not any("XYZ" in url for url in client.requests)


def test_self_reference_is_not_turned_into_an_edge():
    html = "<html><body><p>Sales to NVIDIA Corporation accounted for 9% of our total revenue.</p></body></html>"
    graph, evidence, clock = _world_tuple()
    client = _client({NVDA_URL: html})
    summary = _run(graph, evidence, client, clock, filers=("NVDA",), universe=("NVDA",))
    assert graph.edges == []
    assert summary.skipped_self_reference == 1


def test_latest_10k_is_selected_by_filing_date():
    from app.mrip.edgar.service import find_latest_10k
    from app.mrip.edgar.types import CompanyRecord

    client = _client()
    filing = find_latest_10k(client, CompanyRecord(ticker="NVDA", cik=1045810, title="NVIDIA CORP"), None)
    assert filing is not None
    assert filing.accession == "0001045810-26-000020"
    assert filing.filed == date(2026, 2, 25)
    assert filing.url == NVDA_URL


CEG_CIK = 1868275
CEG_URL = archive_url(CEG_CIK, "0001868275-26-000008", "ceg-10k.htm")
CEG_10K = """<html><body>
<h3>Item 1. Business</h3>
<p>Under the agreement, Microsoft will purchase the output generated from the renewed plant which includes energy, capacity and emissions-free attributes as part of its goal to help power its data center.</p>
</body></html>"""
CEG_TICKERS = {**TICKERS, "5": {"cik_str": CEG_CIK, "ticker": "CEG", "title": "Constellation Energy Corp"}}
SUBMISSIONS_CEG = {
    "filings": {
        "recent": {
            "form": ["10-K"],
            "accessionNumber": ["0001868275-26-000008"],
            "filingDate": ["2026-02-20"],
            "primaryDocument": ["ceg-10k.htm"],
        }
    }
}


def test_counterparty_purchase_from_filer_is_customer_direction():
    # CEG 10-K: Microsoft will purchase the output of the renewed plant -> MSFT CUSTOMER_OF CEG, never CEG CUSTOMER_OF MSFT.
    graph, evidence, clock = _world_tuple()
    client = FakeClient({
        "https://www.sec.gov/files/company_tickers.json": CEG_TICKERS,
        submissions_url(CEG_CIK): SUBMISSIONS_CEG,
        CEG_URL: CEG_10K,
    })
    _run(graph, evidence, client, clock, filers=("CEG",), universe=("CEG", "MSFT"))
    edges = {graph.key_of(e) for e in graph.edges}
    assert ("MSFT", "CEG", "CUSTOMER_OF") in edges
    assert ("CEG", "MSFT", "CUSTOMER_OF") not in edges
