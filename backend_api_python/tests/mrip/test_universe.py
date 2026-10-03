"""Universe module: parsing, fetching, and loading constituent lists.

Tests cover SP500 CSV parsing (with inline data), fetch error handling,
and loader behavior (created/updated/unchanged counts, attribute merging,
dropped constituents, idempotency). Optional database integration test.
"""
from __future__ import annotations

import json
import os
from pathlib import Path as FsPath
from typing import Any

import pytest
import requests

from app.mrip.relationships.types import Node, NodeKey, NodeType
from app.mrip.universe.loader import UniverseLoader, LoadReport
from app.mrip.universe.models import UniverseMember, UniverseError
from app.mrip.universe.sources import fetch_sp500, parse_sp500_csv, MIN_MEMBERS


# ============================================================================
# Parsing tests: parse_sp500_csv with inline CSV
# ============================================================================


class TestParseCSV:
    """Tests for parse_sp500_csv function."""

    def test_parse_valid_csv_with_dotted_symbols(self, monkeypatch: Any) -> None:
        """Parse CSV with dotted symbols (BRK.B, BF.B) kept intact."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_MEMBERS", 2)
        csv_text = (
            "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,"
            "Date added,CIK,Founded\n"
            "BRK.B,Berkshire Hathaway Inc. Class B,Financials,Diversified Financial Services,"
            "Omaha NE,1996-02-29,0001067983,1956\n"
            "BF.B,Brown-Forman Corporation Class B,Consumer Discretionary,Distillers & Vintners,"
            "Louisville KY,1988-01-01,0000012870,1870\n"
            "AAPL,Apple Inc.,Information Technology,Computing Hardware,"
            "Cupertino CA,2005-11-30,0000320193,1976\n"
        )
        members = parse_sp500_csv(csv_text)
        assert len(members) == 3
        assert members[0].symbol == "BRK.B"
        assert members[1].symbol == "BF.B"
        assert members[2].symbol == "AAPL"

    def test_parse_csv_strips_whitespace(self, monkeypatch: Any) -> None:
        """Whitespace is stripped from fields."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_MEMBERS", 1)
        csv_text = (
            "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,"
            "Date added,CIK,Founded\n"
            "  AAPL  ,  Apple Inc.  ,  Information Technology  ,"
            "  Computing Hardware  ,Cupertino CA,2005-11-30,0000320193,1976\n"
        )
        members = parse_sp500_csv(csv_text)
        assert len(members) == 1
        assert members[0].symbol == "AAPL"
        assert members[0].name == "Apple Inc."
        assert members[0].sector == "Information Technology"

    def test_parse_csv_skips_blank_rows(self, monkeypatch: Any) -> None:
        """Rows with blank symbol or name are skipped."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_MEMBERS", 1)
        csv_text = (
            "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,"
            "Date added,CIK,Founded\n"
            "AAPL,Apple Inc.,Information Technology,Computing Hardware,Cupertino CA,"
            "2005-11-30,0000320193,1976\n"
            ",Missing Symbol,Financials,Banking,New York NY,2020-01-01,9999999999,2020\n"
            "MSFT,,Information Technology,Computing Hardware,Redmond WA,"
            "2005-11-30,0000789019,1975\n"
        )
        members = parse_sp500_csv(csv_text)
        assert len(members) == 1
        assert members[0].symbol == "AAPL"

    def test_parse_csv_deduplicates_by_symbol_keeping_first(self, monkeypatch: Any) -> None:
        """Duplicate symbols keep the first occurrence."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_MEMBERS", 1)
        csv_text = (
            "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,"
            "Date added,CIK,Founded\n"
            "AAPL,Apple Inc. First,Information Technology,Computing Hardware,Cupertino CA,"
            "2005-11-30,0000320193,1976\n"
            "AAPL,Apple Inc. Duplicate,Information Technology,Computing Hardware,Cupertino CA,"
            "2005-11-30,0000320193,1976\n"
        )
        members = parse_sp500_csv(csv_text)
        assert len(members) == 1
        assert members[0].name == "Apple Inc. First"

    def test_parse_csv_empty_sector_becomes_none(self, monkeypatch: Any) -> None:
        """Empty or missing sector becomes None."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_MEMBERS", 2)
        csv_text = (
            "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,"
            "Date added,CIK,Founded\n"
            "AAPL,Apple Inc.,,Computing Hardware,Cupertino CA,2005-11-30,0000320193,1976\n"
            "MSFT,Microsoft Inc.,Information Technology,Computing Hardware,Redmond WA,"
            "2005-11-30,0000789019,1975\n"
        )
        members = parse_sp500_csv(csv_text)
        assert members[0].sector is None
        assert members[1].sector == "Information Technology"

    def test_parse_csv_bad_header_raises_error(self) -> None:
        """CSV missing Symbol or Security column raises UniverseError."""
        csv_text = (
            "Ticker,Name,GICS Sector,GICS Sub-Industry,Headquarters Location,"
            "Date added,CIK,Founded\n"
            "AAPL,Apple Inc.,Information Technology,Computing Hardware,Cupertino CA,"
            "2005-11-30,0000320193,1976\n"
        )
        with pytest.raises(UniverseError, match="missing Symbol or Security column"):
            parse_sp500_csv(csv_text)

    def test_parse_csv_too_few_rows_raises_error(self) -> None:
        """Fewer than MIN_MEMBERS members raises UniverseError."""
        # Create a CSV with only 50 rows (less than MIN_MEMBERS).
        lines = ["Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,"
                 "Date added,CIK,Founded"]
        for i in range(50):
            lines.append(
                f"SYM{i:02d},Symbol {i},Sector,SubIndustry,City,"
                f"2020-01-01,{i:010d},2020"
            )
        csv_text = "\n".join(lines)
        with pytest.raises(UniverseError, match="parsed only 50 members"):
            parse_sp500_csv(csv_text)

    def test_parse_csv_min_members_configurable_via_constant(self) -> None:
        """MIN_MEMBERS constant is available for test monkeypatching."""
        assert MIN_MEMBERS == 100
        # This test documents the constant for test fixtures that may reduce it.


# ============================================================================
# Fetch tests: fetch_sp500 with fake session
# ============================================================================


class TestFetchSP500:
    """Tests for fetch_sp500 function."""

    def test_fetch_success_with_valid_csv(self, monkeypatch: Any) -> None:
        """Successful fetch returns parsed members."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_MEMBERS", 2)
        csv_text = (
            "Symbol,Security,GICS Sector,GICS Sub-Industry,Headquarters Location,"
            "Date added,CIK,Founded\n"
            "AAPL,Apple Inc.,Information Technology,Computing Hardware,Cupertino CA,"
            "2005-11-30,0000320193,1976\n"
            "MSFT,Microsoft Inc.,Information Technology,Software & Services,Redmond WA,"
            "1994-11-30,0000789019,1975\n"
        )

        class FakeSession:
            def get(self, url: str, timeout: float) -> Any:
                class FakeResponse:
                    text = csv_text

                    def raise_for_status(self) -> None:
                        pass

                return FakeResponse()

        members = fetch_sp500(session=FakeSession(), url="http://fake")  # type: ignore
        assert len(members) == 2
        assert members[0].symbol == "AAPL"

    def test_fetch_http_error_raises_universe_error(self) -> None:
        """HTTP error status raises UniverseError with chained exception."""

        class FakeSession:
            def get(self, url: str, timeout: float) -> Any:
                raise requests.HTTPError("404 Not Found")

        with pytest.raises(UniverseError, match="failed to fetch"):
            fetch_sp500(session=FakeSession(), url="http://fake")  # type: ignore

    def test_fetch_connection_error_raises_universe_error(self) -> None:
        """Connection error raises UniverseError with chained exception."""

        class FakeSession:
            def get(self, url: str, timeout: float) -> Any:
                raise requests.ConnectionError("Connection refused")

        with pytest.raises(UniverseError, match="failed to fetch"):
            fetch_sp500(session=FakeSession(), url="http://fake")  # type: ignore

    def test_fetch_timeout_raises_universe_error(self) -> None:
        """Timeout raises UniverseError."""

        class FakeSession:
            def get(self, url: str, timeout: float) -> Any:
                raise requests.Timeout("Request timed out")

        with pytest.raises(UniverseError, match="failed to fetch"):
            fetch_sp500(session=FakeSession(), url="http://fake")  # type: ignore


# ============================================================================
# Loader tests: in-memory fake graph
# ============================================================================


class FakeNode:
    """Fake Node for testing (duck-types app.mrip.relationships.types.Node)."""

    def __init__(
        self, id_: int, node_type: NodeType, key: str, name: str, attributes: dict[str, Any] | None = None
    ) -> None:
        self.id = id_
        self.node_type = node_type
        self.key = key
        self.name = name
        self.attributes = attributes or {}

    def __repr__(self) -> str:
        return f"FakeNode(id={self.id}, key={self.key})"


class FakeGraph:
    """In-memory graph for testing (duck-types RelationshipGraph)."""

    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], FakeNode] = {}  # (node_type, key) -> node
        self._next_id = 1

    def upsert_node(
        self, node_type: NodeType, key: str, name: str, attributes: dict[str, Any] | None = None
    ) -> FakeNode:
        """Create or update a node."""
        dict_key = (node_type.value, key)
        if dict_key in self.nodes:
            # Update: modify in place and return a new object with same data (to match RelationshipGraph behavior).
            node = self.nodes[dict_key]
            node.name = name
            node.attributes = dict(attributes or {})
            # Return a copy so the caller can compare old vs new without them being the same object.
            return FakeNode(node.id, node.node_type, node.key, node.name, dict(node.attributes))
        else:
            # Create
            new_node = FakeNode(self._next_id, node_type, key, name, dict(attributes or {}))
            self._next_id += 1
            self.nodes[dict_key] = new_node
            return new_node

    def get_node(self, node_key: NodeKey) -> FakeNode | None:
        """Retrieve a node (returns a copy to match RelationshipGraph behavior)."""
        dict_key = (node_key.node_type.value, node_key.key)
        stored = self.nodes.get(dict_key)
        if stored is None:
            return None
        # Return a copy to prevent callers from modifying stored state.
        return FakeNode(stored.id, stored.node_type, stored.key, stored.name, dict(stored.attributes))

    def list_nodes_in_universe(self, universe: str) -> list[FakeNode]:
        """Return all SECURITY nodes in a universe (returns copies)."""
        nodes = []
        for (node_type, key), node in self.nodes.items():
            if node_type == NodeType.SECURITY.value:
                universes = node.attributes.get("universes", [])
                if universe in universes:
                    # Return a copy to prevent callers from modifying stored state.
                    nodes.append(FakeNode(node.id, node.node_type, node.key, node.name, dict(node.attributes)))
        return nodes

    def _get_stored_node(self, node_key: NodeKey) -> FakeNode | None:
        """Internal: get the actual stored node (for testing)."""
        dict_key = (node_key.node_type.value, node_key.key)
        return self.nodes.get(dict_key)


class TestUniverseLoader:
    """Tests for UniverseLoader with fake in-memory graph."""

    def test_load_creates_nodes(self) -> None:
        """Loading new members creates SECURITY nodes."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        members = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
            UniverseMember(
                universe="SP500",
                symbol="MSFT",
                name="Microsoft Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Software & Services",
                data_symbol="MSFT",
                data_provider="cboe",
            ),
        ]

        report = loader.load(members)
        assert report.created == 2
        assert report.updated == 0
        assert report.unchanged == 0
        assert report.dropped == ()
        assert len(graph.nodes) == 2

    def test_load_updates_existing_node_with_changed_attributes(self) -> None:
        """Loading a member with different attributes marks as updated."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        # First load.
        members1 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]
        report1 = loader.load(members1)
        assert report1.created == 1

        # Second load with different sector.
        members2 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology Updated",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]
        report2 = loader.load(members2)
        assert report2.created == 0
        assert report2.updated == 1
        assert report2.unchanged == 0

    def test_load_unchanged_when_no_changes(self) -> None:
        """Loading identical members marks as unchanged."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        members = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]

        report1 = loader.load(members)
        assert report1.created == 1

        report2 = loader.load(members)
        assert report2.created == 0
        assert report2.updated == 0
        assert report2.unchanged == 1

    def test_load_merges_attributes_preserving_existing_keys(self) -> None:
        """Loading merges attributes and preserves unknown existing keys."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        # First load creates a node with base attributes.
        members1 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]
        loader.load(members1)

        # Manually add an unknown attribute to the stored node.
        stored_node = graph._get_stored_node(NodeKey(NodeType.SECURITY, "US:AAPL"))
        assert stored_node is not None
        stored_node.attributes["custom_field"] = "custom_value"

        # Load again with different sector.
        members2 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Updated Sector",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]
        loader.load(members2)

        # Verify the custom field was preserved.
        updated_node = graph.get_node(NodeKey(NodeType.SECURITY, "US:AAPL"))
        assert updated_node is not None
        assert updated_node.attributes["custom_field"] == "custom_value"
        assert updated_node.attributes["sector"] == "Updated Sector"

    def test_load_merges_universes_list(self) -> None:
        """Loading in multiple universes keeps the union of universes."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        # First load to SP500.
        members_sp500 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]
        loader.load(members_sp500)

        # Load to a different universe (OMXSLC).
        members_omxslc = [
            UniverseMember(
                universe="OMXSLC",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]
        loader.load(members_omxslc)

        # Verify both universes are in the node's attributes.
        node = graph.get_node(NodeKey(NodeType.SECURITY, "US:AAPL"))
        assert node is not None
        universes = node.attributes.get("universes", [])
        assert set(universes) == {"SP500", "OMXSLC"}
        assert universes == sorted(universes)  # Sorted.

    def test_load_drops_constituents_removed_from_universe(self) -> None:
        """Members no longer in the universe lose that universe tag."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        # First load with two members.
        members1 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
            UniverseMember(
                universe="SP500",
                symbol="MSFT",
                name="Microsoft Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Software & Services",
                data_symbol="MSFT",
                data_provider="cboe",
            ),
        ]
        loader.load(members1)

        # Load with only AAPL (MSFT is dropped from SP500).
        members2 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]
        report = loader.load(members2)

        # MSFT loses the SP500 universe tag.
        assert report.dropped == ("US:MSFT",)
        msft_node = graph.get_node(NodeKey(NodeType.SECURITY, "US:MSFT"))
        assert msft_node is not None
        # If MSFT was in multiple universes, only SP500 is removed.
        universes = msft_node.attributes.get("universes", [])
        assert "SP500" not in universes

    def test_load_raises_on_mixed_universes(self) -> None:
        """ValueError if members belong to different universes."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        members = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
            UniverseMember(
                universe="OMXSLC",
                symbol="MSFT",
                name="Microsoft Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Software & Services",
                data_symbol="MSFT",
                data_provider="cboe",
            ),
        ]
        with pytest.raises(ValueError, match="same universe"):
            loader.load(members)

    def test_load_raises_on_empty_members(self) -> None:
        """ValueError if members is empty."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        with pytest.raises(ValueError, match="must not be empty"):
            loader.load([])

    def test_load_idempotent(self) -> None:
        """Loading the same members twice produces all-unchanged on second run."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        members = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
            UniverseMember(
                universe="SP500",
                symbol="MSFT",
                name="Microsoft Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Software & Services",
                data_symbol="MSFT",
                data_provider="cboe",
            ),
        ]

        report1 = loader.load(members)
        assert report1.created == 2
        assert report1.updated == 0
        assert report1.unchanged == 0

        report2 = loader.load(members)
        assert report2.created == 0
        assert report2.updated == 0
        assert report2.unchanged == 2
        assert report2.dropped == ()


# ============================================================================
# Database integration test (opt-in)
# ============================================================================


MIGRATION = FsPath(__file__).resolve().parents[2] / "migrations" / "mrip_20261001_relationships.sql"

pytestmark_db = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL"),
]


@pytest.fixture()
def graph_db():  # type: ignore
    """Set up a real graph for database tests."""
    from app.utils import db
    from app.mrip.relationships.graph import RelationshipGraph

    # Skip if database is not available.
    if os.getenv("MRIP_TEST_DB") != "1":
        pytest.skip("MRIP_TEST_DB not set")

    # Apply migrations and reset state.
    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(db._BOOTSTRAP_LEDGER_SQL)
        conn.commit()
        cur.close()

    with db.get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(MIGRATION.read_text(encoding="utf-8"))
        cur.execute("TRUNCATE mrip_rel_edges, mrip_rel_nodes RESTART IDENTITY CASCADE")
        cur.execute("UPDATE mrip_rel_graph_state SET version = 0 WHERE id = 1")
        conn.commit()
        cur.close()

    return RelationshipGraph(db.get_db_connection)


@pytest.mark.skipif(os.getenv("MRIP_TEST_DB") != "1", reason="set MRIP_TEST_DB=1 with a throwaway DATABASE_URL")
class TestUniverseLoaderDB:
    """Database integration tests for UniverseLoader."""

    def test_load_and_retrieve_via_graph(self, graph_db: Any) -> None:  # type: ignore
        """Load members and retrieve them via graph.list_nodes_in_universe."""
        loader = UniverseLoader(graph_db)

        members = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
            UniverseMember(
                universe="SP500",
                symbol="MSFT",
                name="Microsoft Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Software & Services",
                data_symbol="MSFT",
                data_provider="cboe",
            ),
            UniverseMember(
                universe="SP500",
                symbol="GOOGL",
                name="Alphabet Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Internet Services & Infrastructure",
                data_symbol="GOOGL",
                data_provider="cboe",
            ),
        ]

        report = loader.load(members)
        assert report.created == 3
        assert report.updated == 0
        assert report.unchanged == 0

        # Retrieve via graph.
        nodes = graph_db.list_nodes_in_universe("SP500")
        assert len(nodes) == 3
        keys = {n.key for n in nodes}
        assert keys == {"US:AAPL", "US:MSFT", "US:GOOGL"}

    def test_reload_with_one_member_removed(self, graph_db: Any) -> None:  # type: ignore
        """Reload with one member removed drops it from the universe."""
        loader = UniverseLoader(graph_db)

        # Initial load.
        members1 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
            UniverseMember(
                universe="SP500",
                symbol="MSFT",
                name="Microsoft Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Software & Services",
                data_symbol="MSFT",
                data_provider="cboe",
            ),
        ]
        report1 = loader.load(members1)
        assert report1.created == 2

        # Reload with only AAPL.
        members2 = [
            UniverseMember(
                universe="SP500",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="US",
                currency="USD",
                sector="Information Technology",
                sub_industry="Computing Hardware",
                data_symbol="AAPL",
                data_provider="cboe",
            ),
        ]
        report2 = loader.load(members2)
        assert report2.created == 0
        assert report2.updated == 0
        assert report2.unchanged == 1
        assert report2.dropped == ("US:MSFT",)

        # Verify via graph: MSFT has no universes or empty list.
        nodes = graph_db.list_nodes_in_universe("SP500")
        assert len(nodes) == 1
        assert nodes[0].key == "US:AAPL"

        # MSFT still exists but has no universes.
        msft_node = graph_db.get_node(NodeKey(NodeType.SECURITY, "US:MSFT"))
        assert msft_node is not None
        universes = msft_node.attributes.get("universes", [])
        assert "SP500" not in universes
