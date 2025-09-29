"""Graph export utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx


@dataclass
class GraphExporter:
    """Serialize the in-memory graph to a portable representation."""

    graph: nx.MultiDiGraph

    def export(self, *, format: Literal["graphml", "json"] = "json") -> str:
        """Export the graph to the requested ``format``."""

        if format == "graphml":  # pragma: no cover - placeholder
            return "<graphml />"
        if format == "json":  # pragma: no cover - placeholder
            return "{}"
        raise ValueError(f"Unsupported export format: {format}")
