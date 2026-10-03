"""Universe loader: upsert members into the relationship graph.

The loader atomically loads a universe's constituents into the relationship
graph (graph.py), merging attributes and tracking membership across universes.
Members are tracked via "universes" attribute (a list of universe names).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.mrip.relationships.types import Node, NodeKey, NodeType
from app.mrip.universe.models import UniverseMember


@dataclass(frozen=True, slots=True)
class LoadReport:
    """Result of a UniverseLoader.load operation.

    Attributes:
        created: Count of nodes created.
        updated: Count of nodes with changed name or attributes.
        unchanged: Count of nodes with identical name and attributes.
        dropped: Sorted tuple of node keys that lost this universe membership.
    """

    created: int
    updated: int
    unchanged: int
    dropped: tuple[str, ...]


class UniverseLoader:
    """Loads universe constituents into a RelationshipGraph.

    A RelationshipGraph duck-typed as having:
        - upsert_node(node_type: NodeType, key: str, name: str, attributes: dict | None) -> Node
        - get_node(NodeKey) -> Node | None
        - list_nodes_in_universe(universe: str) -> list[Node]
    """

    def __init__(self, graph: Any) -> None:
        """Initialize with a RelationshipGraph."""
        self._graph = graph

    def load(self, members: Sequence[UniverseMember]) -> LoadReport:
        """Load members of a single universe into the graph.

        All members must belong to the same universe. Atomically:
        1. For each member, upsert a SECURITY node with merged attributes.
        2. Count created (new), updated (changed), and unchanged nodes.
        3. Find nodes that held this universe but are not in members; remove the
           universe tag from their "universes" list.

        Raises:
            ValueError: if members is empty or members have mismatched universes.
        """
        if not members:
            raise ValueError("members must not be empty")

        members_list = list(members)
        universes = {m.universe for m in members_list}
        if len(universes) != 1:
            raise ValueError(f"all members must belong to the same universe; got {universes}")

        universe_name = universes.pop()
        created = 0
        updated = 0
        unchanged = 0

        # Build a map of node_key -> member for efficient lookup.
        member_map = {m.node_key: m for m in members_list}

        # Upsert each member.
        for member in members_list:
            existing = self._graph.get_node(NodeKey(NodeType.SECURITY, member.node_key))
            new_attributes = self._build_attributes(member, existing)

            node = self._graph.upsert_node(
                NodeType.SECURITY,
                member.node_key,
                member.name,
                new_attributes,
            )

            if existing is None:
                created += 1
            elif existing.name != node.name or _attributes_differ(existing.attributes, node.attributes):
                updated += 1
            else:
                unchanged += 1

        # Find members that still have this universe but are not in the load.
        existing_members = self._graph.list_nodes_in_universe(universe_name)
        dropped_keys: list[str] = []

        for existing_node in existing_members:
            if existing_node.key not in member_map:
                # Remove this universe from the node's universes list.
                attrs = dict(existing_node.attributes or {})
                universes_list = attrs.get("universes", [])
                if universe_name in universes_list:
                    updated_universes = [u for u in universes_list if u != universe_name]
                    if updated_universes:
                        attrs["universes"] = updated_universes
                    else:
                        attrs.pop("universes", None)
                    self._graph.upsert_node(
                        NodeType.SECURITY,
                        existing_node.key,
                        existing_node.name,
                        attrs,
                    )
                    dropped_keys.append(existing_node.key)

        return LoadReport(
            created=created,
            updated=updated,
            unchanged=unchanged,
            dropped=tuple(sorted(dropped_keys)),
        )

    def _build_attributes(self, member: UniverseMember, existing: Node | None) -> Mapping[str, Any]:
        """Build merged attributes for a member, preserving existing unknown keys.

        Merges:
        - "series": {symbol, provider} from member
        - "universes": union of existing universes and the member's universe
        - sector, sub_industry, exchange, currency, isin from member
        """
        attrs: dict[str, Any] = {}

        # Preserve unknown existing attributes.
        if existing is not None:
            attrs.update(existing.attributes or {})

        # Update series.
        attrs["series"] = {"symbol": member.data_symbol, "provider": member.data_provider}

        # Merge universes (union, sorted).
        existing_universes = set(attrs.get("universes") or [])
        existing_universes.add(member.universe)
        attrs["universes"] = sorted(existing_universes)

        # Update sector (None -> omit from dict to avoid null clutter).
        if member.sector is not None:
            attrs["sector"] = member.sector
        else:
            attrs.pop("sector", None)

        # Update sub_industry.
        if member.sub_industry is not None:
            attrs["sub_industry"] = member.sub_industry
        else:
            attrs.pop("sub_industry", None)

        # Update exchange and currency.
        attrs["exchange"] = member.exchange
        attrs["currency"] = member.currency

        # Update isin.
        if member.isin is not None:
            attrs["isin"] = member.isin
        else:
            attrs.pop("isin", None)

        return attrs


def _attributes_differ(old: Mapping[str, Any] | None, new: Mapping[str, Any] | None) -> bool:
    """Check if two attribute dicts differ (order-independent for JSON comparison)."""
    old_json = json.dumps(dict(old or {}), sort_keys=True)
    new_json = json.dumps(dict(new or {}), sort_keys=True)
    return old_json != new_json
