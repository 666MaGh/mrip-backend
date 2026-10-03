"""OMX Stockholm Large Cap universe: parsing, loading, and integration tests.

Tests cover OMX CSV parsing (with inline data), packaged file loading,
loader behavior with multiple universes, and optional database integration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path as FsPath
from typing import Any

import pytest

from app.mrip.relationships.types import Node, NodeKey, NodeType
from app.mrip.universe.loader import UniverseLoader, LoadReport
from app.mrip.universe.models import UniverseMember, UniverseError
from app.mrip.universe.sources import parse_omx_csv, load_omxslc, MIN_OMX_MEMBERS


# ============================================================================
# Parsing tests: parse_omx_csv with inline CSV
# ============================================================================


class TestParseOmxCSV:
    """Tests for parse_omx_csv function."""

    def test_parse_valid_csv_with_space_symbols(self, monkeypatch: Any) -> None:
        """Parse CSV with space-separated symbols converted to hyphenated form."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 2)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
            "ERIC B,Ericsson B,Information Technology,SEK\n"
            "ABB,ABB Ltd,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert len(members) == 3
        assert members[0].symbol == "VOLV-B"
        assert members[1].symbol == "ERIC-B"
        assert members[2].symbol == "ABB"

    def test_parse_csv_skips_comment_and_blank_lines(self, monkeypatch: Any) -> None:
        """Comment lines (starting with '#') and blank lines are skipped."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "# This is a comment line\n"
            "Symbol,Name,Sector,Currency\n"
            "\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
            "# Another comment\n"
            "\n"
            "ABB,ABB Ltd,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert len(members) == 2
        assert members[0].symbol == "VOLV-B"
        assert members[1].symbol == "ABB"

    def test_parse_csv_data_symbol_with_st_suffix(self, monkeypatch: Any) -> None:
        """data_symbol is symbol + '.ST' (Yahoo format)."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].data_symbol == "VOLV-B.ST"

    def test_parse_csv_node_key_is_xsto_colon_symbol(self, monkeypatch: Any) -> None:
        """node_key is 'XSTO:{symbol}'."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].node_key == "XSTO:VOLV-B"

    def test_parse_csv_aliv_sdb_conversion(self, monkeypatch: Any) -> None:
        """'ALIV SDB' is converted to 'ALIV-SDB'."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "ALIV SDB,Autoliv SDB,Consumer Discretionary,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].symbol == "ALIV-SDB"
        assert members[0].data_symbol == "ALIV-SDB.ST"

    def test_parse_csv_strips_whitespace(self, monkeypatch: Any) -> None:
        """Whitespace is stripped from fields."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "  VOLV B  ,  Volvo B  ,  Industrials  ,  SEK  \n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].symbol == "VOLV-B"
        assert members[0].name == "Volvo B"
        assert members[0].sector == "Industrials"
        assert members[0].currency == "SEK"

    def test_parse_csv_currency_default(self, monkeypatch: Any) -> None:
        """Currency defaults to 'SEK' when blank or missing."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].currency == "SEK"

    def test_parse_csv_currency_explicit(self, monkeypatch: Any) -> None:
        """Currency from CSV when provided."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,EUR\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].currency == "EUR"

    def test_parse_csv_blank_sector_becomes_none(self, monkeypatch: Any) -> None:
        """Empty or missing sector becomes None."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 2)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,,SEK\n"
            "ERIC B,Ericsson B,Information Technology,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].sector is None
        assert members[1].sector == "Information Technology"

    def test_parse_csv_sub_industry_none(self, monkeypatch: Any) -> None:
        """sub_industry is always None for OMX."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].sub_industry is None

    def test_parse_csv_isin_none(self, monkeypatch: Any) -> None:
        """isin is always None for OMX."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].isin is None

    def test_parse_csv_universe_is_omxslc(self, monkeypatch: Any) -> None:
        """universe is always 'OMXSLC'."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].universe == "OMXSLC"

    def test_parse_csv_exchange_is_xsto(self, monkeypatch: Any) -> None:
        """exchange is always 'XSTO' (Nasdaq Stockholm)."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].exchange == "XSTO"

    def test_parse_csv_data_provider_is_yahoo(self, monkeypatch: Any) -> None:
        """data_provider is always 'yahoo'."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert members[0].data_provider == "yahoo"

    def test_parse_csv_skips_blank_rows(self, monkeypatch: Any) -> None:
        """Rows with blank symbol or name are skipped."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
            ",Missing Symbol,Industrials,SEK\n"
            "ABB,,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert len(members) == 1
        assert members[0].symbol == "VOLV-B"

    def test_parse_csv_deduplicates_by_symbol_keeping_first(self, monkeypatch: Any) -> None:
        """Duplicate symbols keep the first occurrence."""
        monkeypatch.setattr("app.mrip.universe.sources.MIN_OMX_MEMBERS", 1)
        csv_text = (
            "Symbol,Name,Sector,Currency\n"
            "VOLV B,Volvo B First,Industrials,SEK\n"
            "VOLV B,Volvo B Duplicate,Industrials,SEK\n"
        )
        members = parse_omx_csv(csv_text)
        assert len(members) == 1
        assert members[0].name == "Volvo B First"

    def test_parse_csv_bad_header_missing_symbol(self) -> None:
        """CSV missing Symbol column raises UniverseError."""
        csv_text = (
            "Ticker,Name,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        with pytest.raises(UniverseError, match="missing Symbol or Name column"):
            parse_omx_csv(csv_text)

    def test_parse_csv_bad_header_missing_name(self) -> None:
        """CSV missing Name column raises UniverseError."""
        csv_text = (
            "Symbol,Title,Sector,Currency\n"
            "VOLV B,Volvo B,Industrials,SEK\n"
        )
        with pytest.raises(UniverseError, match="missing Symbol or Name column"):
            parse_omx_csv(csv_text)

    def test_parse_csv_too_few_rows_raises_error(self) -> None:
        """Fewer than MIN_OMX_MEMBERS members raises UniverseError."""
        lines = ["Symbol,Name,Sector,Currency"]
        for i in range(49):
            lines.append(f"SYM{i:02d},Symbol {i},Sector,SEK")
        csv_text = "\n".join(lines)
        with pytest.raises(UniverseError, match="parsed only 49 members"):
            parse_omx_csv(csv_text)

    def test_parse_csv_min_omx_members_constant_available(self) -> None:
        """MIN_OMX_MEMBERS constant is available for test monkeypatching."""
        assert MIN_OMX_MEMBERS == 50
        # This test documents the constant for test fixtures that may reduce it.


# ============================================================================
# Packaged file loading test
# ============================================================================


class TestLoadOmxslc:
    """Tests for load_omxslc function (packaged data file)."""

    def test_load_packaged_file_returns_members(self) -> None:
        """load_omxslc() successfully loads the packaged omxslc.csv file."""
        members = load_omxslc()
        assert len(members) >= 90

    def test_packaged_members_have_unique_node_keys(self) -> None:
        """All loaded members have unique node_key values."""
        members = load_omxslc()
        keys = {m.node_key for m in members}
        assert len(keys) == len(members)

    def test_packaged_data_symbols_end_with_st(self) -> None:
        """Every data_symbol ends with '.ST'."""
        members = load_omxslc()
        for member in members:
            assert member.data_symbol.endswith(".ST"), f"data_symbol {member.data_symbol} does not end with .ST"

    def test_packaged_contains_expected_tickers(self) -> None:
        """Packaged list contains specific well-known tickers."""
        members = load_omxslc()
        symbols = {m.symbol for m in members}
        expected = {"VOLV-B", "ERIC-B", "ATCO-A", "SEB-A", "HM-B"}
        assert expected.issubset(symbols), f"missing expected symbols: {expected - symbols}"

    def test_packaged_members_have_sectors(self) -> None:
        """Every loaded member has a sector (not None)."""
        members = load_omxslc()
        for member in members:
            assert member.sector is not None, f"member {member.symbol} has no sector"

    def test_packaged_all_members_in_omxslc_universe(self) -> None:
        """All loaded members have universe='OMXSLC'."""
        members = load_omxslc()
        for member in members:
            assert member.universe == "OMXSLC"

    def test_packaged_all_members_xsto_exchange(self) -> None:
        """All loaded members have exchange='XSTO'."""
        members = load_omxslc()
        for member in members:
            assert member.exchange == "XSTO"


# ============================================================================
# Loader tests with fake graph (multiple universes)
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


class TestUniverseLoaderMultiverse:
    """Tests for UniverseLoader with multiple universes (SP500 and OMXSLC)."""

    def test_load_omxslc_after_sp500_creates_separate_nodes(self) -> None:
        """Loading OMXSLC after SP500 creates separate nodes due to different exchanges."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        # Load SP500 member.
        sp500_members = [
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
        loader.load(sp500_members)

        # Load OMX member (different exchange, even with same display name).
        omx_members = [
            UniverseMember(
                universe="OMXSLC",
                symbol="AAPL",
                name="Apple Inc.",
                exchange="XSTO",
                currency="SEK",
                sector="Information Technology",
                sub_industry=None,
                data_symbol="AAPL.ST",
                data_provider="yahoo",
            ),
        ]
        loader.load(omx_members)

        # Verify separate nodes were created due to different keys (US:AAPL vs XSTO:AAPL).
        assert len(graph.nodes) == 2
        assert graph.get_node(NodeKey(NodeType.SECURITY, "US:AAPL")) is not None
        assert graph.get_node(NodeKey(NodeType.SECURITY, "XSTO:AAPL")) is not None

    def test_security_in_two_universes_keeps_both_tags(self) -> None:
        """A security in two universes keeps both universe tags after both loads."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        # Simulate a security present in both SP500 and OMXSLC with same node key.
        # First, manually create a shared node key scenario (not realistic but tests merging).
        member1 = UniverseMember(
            universe="SP500",
            symbol="SHARED",
            name="Shared Security",
            exchange="US",
            currency="USD",
            sector="Information Technology",
            sub_industry="Computing Hardware",
            data_symbol="SHARED",
            data_provider="cboe",
        )
        loader.load([member1])

        # Now load a member with the same node_key but different universe.
        member2 = UniverseMember(
            universe="OMXSLC",
            symbol="SHARED",
            name="Shared Security",
            exchange="US",  # Same exchange intentionally
            currency="USD",
            sector="Information Technology",
            sub_industry=None,
            data_symbol="SHARED.ST",
            data_provider="yahoo",
        )
        loader.load([member2])

        # Verify both universes are tagged.
        node = graph.get_node(NodeKey(NodeType.SECURITY, "US:SHARED"))
        assert node is not None
        universes = node.attributes.get("universes", [])
        assert set(universes) == {"SP500", "OMXSLC"}

    def test_reload_omxslc_with_one_member_removed_drops_and_preserves_sp500(self) -> None:
        """Reloading OMXSLC with one member removed drops it from OMXSLC but preserves SP500."""
        graph = FakeGraph()
        loader = UniverseLoader(graph)

        # Create a shared node (same key, different universes).
        member_both = UniverseMember(
            universe="SP500",
            symbol="SHARED",
            name="Shared Security",
            exchange="US",
            currency="USD",
            sector="Information Technology",
            sub_industry="Computing Hardware",
            data_symbol="SHARED",
            data_provider="cboe",
        )
        loader.load([member_both])

        # Add it to OMXSLC too.
        member_omx = UniverseMember(
            universe="OMXSLC",
            symbol="SHARED",
            name="Shared Security",
            exchange="US",
            currency="USD",
            sector="Information Technology",
            sub_industry=None,
            data_symbol="SHARED.ST",
            data_provider="yahoo",
        )
        loader.load([member_omx])

        # Verify both universes present.
        node = graph.get_node(NodeKey(NodeType.SECURITY, "US:SHARED"))
        assert node is not None
        universes = node.attributes.get("universes", [])
        assert set(universes) == {"SP500", "OMXSLC"}

        # Reload OMXSLC without the shared member.
        other_omx = UniverseMember(
            universe="OMXSLC",
            symbol="OTHER",
            name="Other Security",
            exchange="US",
            currency="USD",
            sector="Industrials",
            sub_industry=None,
            data_symbol="OTHER.ST",
            data_provider="yahoo",
        )
        report = loader.load([other_omx])

        # SHARED should be dropped from OMXSLC only.
        assert "US:SHARED" in report.dropped

        # SHARED should still exist but without OMXSLC tag (SP500 remains).
        node = graph.get_node(NodeKey(NodeType.SECURITY, "US:SHARED"))
        assert node is not None
        universes = node.attributes.get("universes", [])
        assert universes == ["SP500"]

        # OTHER should be created.
        other_node = graph.get_node(NodeKey(NodeType.SECURITY, "US:OTHER"))
        assert other_node is not None
        universes = other_node.attributes.get("universes", [])
        assert universes == ["OMXSLC"]


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
class TestOmxUniverseLoaderDB:
    """Database integration tests for OMX universe loader."""

    def test_load_packaged_omxslc_via_real_graph(self, graph_db: Any) -> None:  # type: ignore
        """Load packaged OMXSLC members via real RelationshipGraph and verify counts."""
        loader = UniverseLoader(graph_db)
        members = load_omxslc()

        report = loader.load(members)
        assert report.created == len(members)
        assert report.updated == 0
        assert report.unchanged == 0

        # Retrieve via graph.
        nodes = graph_db.list_nodes_in_universe("OMXSLC")
        assert len(nodes) == len(members)

        # Verify all symbols are present.
        node_keys = {n.key for n in nodes}
        for member in members:
            assert member.node_key in node_keys
