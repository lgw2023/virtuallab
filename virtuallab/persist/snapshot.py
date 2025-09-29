"""Graph snapshot management."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import networkx as nx


@dataclass
class SnapshotManager:
    """Maintain in-memory snapshots of the graph for quick rollback."""

    history: List[nx.MultiDiGraph] = field(default_factory=list)

    def snapshot(self, graph: nx.MultiDiGraph) -> None:
        """Persist a shallow copy of ``graph`` in ``history``."""

        self.history.append(graph.copy())

    def rollback(self) -> nx.MultiDiGraph:
        """Return the most recent snapshot."""

        if not self.history:
            raise RuntimeError("No snapshots available")
        return self.history.pop()
