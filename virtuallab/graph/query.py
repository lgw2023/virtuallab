"""Query helpers for the VirtualLab graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import networkx as nx


@dataclass
class QueryService:
    """Provide structured access patterns on top of ``networkx``."""

    graph: nx.MultiDiGraph

    def by_type(self, node_type: str, **filters: Any) -> Iterable[Dict[str, Any]]:
        """Return nodes that match ``node_type`` and ``filters``."""

        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != node_type:
                continue
            if all(data.get(key) == value for key, value in filters.items()):
                yield {"id": node_id, **data}

    def neighbors(
        self, node_id: str, *, hop: int = 1, edge_types: Optional[Iterable[str]] = None
    ) -> Iterable[Dict[str, Any]]:
        """Yield neighboring nodes for ``node_id`` up to ``hop`` steps."""

        if node_id not in self.graph:
            return
        visited = {node_id}
        frontier = {node_id}
        for _ in range(hop):
            next_frontier = set()
            for current in frontier:
                for _, neighbor, edge_data in self.graph.out_edges(current, data=True):
                    if edge_types and edge_data.get("type") not in edge_types:
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    yield {"id": neighbor, **self.graph.nodes[neighbor]}
            frontier = next_frontier

    def timeline(
        self,
        *,
        scope: Optional[Dict[str, Any]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> Iterable[Dict[str, Any]]:
        """Return nodes ordered by their ``executed_at``/``created_at`` fields."""

        nodes = []
        include_types = set(include or [])
        plan_scope = scope.get("plan_id") if scope else None

        plan_members: Optional[set[str]]
        if plan_scope:
            plan_members = self._collect_plan_members(plan_scope)
        else:
            plan_members = None

        for node_id, data in self.graph.nodes(data=True):
            if include_types and data.get("type") not in include_types:
                continue
            if plan_members is not None and node_id not in plan_members:
                continue
            timestamp = data.get("executed_at") or data.get("created_at")
            nodes.append((timestamp, {"id": node_id, **data}))
        for _, payload in sorted(nodes, key=lambda item: item[0] or ""):
            yield payload

    def _collect_plan_members(self, plan_id: str) -> set[str]:
        """Return the set of node identifiers associated with ``plan_id``."""

        members: set[str] = set()
        if plan_id not in self.graph:
            return members

        members.add(plan_id)
        subtasks: set[str] = set()

        for node_id, data in self.graph.nodes(data=True):
            if data.get("plan_id") == plan_id:
                members.add(node_id)
                if data.get("type") == "Subtask":
                    subtasks.add(node_id)

        for subtask_id in subtasks:
            for _, target_id, edge_data in self.graph.out_edges(subtask_id, data=True):
                if edge_data.get("type") == "CONTAINS":
                    members.add(target_id)

        return members
