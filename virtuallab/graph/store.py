"""In-memory NetworkX based storage for the VirtualLab knowledge graph."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import networkx as nx

from .model import EdgeSpec, GraphDelta, NodeSpec, NodeType


@dataclass
class GraphStore:
    """Lightweight wrapper around :class:`networkx.MultiDiGraph`.

    The store currently keeps the entire graph in memory. Persistence and
    synchronization features will be layered on top of this abstraction in
    subsequent iterations.
    """

    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)

    def add_node(self, node: NodeSpec) -> None:
        """Add ``node`` to the underlying graph."""

        self.graph.add_node(node.id, type=node.type.value, **node.attributes)

    def add_edge(self, edge: EdgeSpec) -> None:
        """Add ``edge`` to the underlying graph."""

        self.graph.add_edge(
            edge.source,
            edge.target,
            key=edge.type.value,
            type=edge.type.value,
            **edge.attributes,
        )

    def apply_delta(self, delta: GraphDelta) -> None:
        """Apply a :class:`GraphDelta` to the graph."""

        for node in delta.added_nodes:
            self.add_node(node)
        for edge in delta.added_edges:
            self.add_edge(edge)
        for node in delta.updated_nodes:
            if node.id in self.graph:
                self.graph.nodes[node.id].update(node.attributes)

    def get_node(self, node_id: str) -> Optional[NodeSpec]:
        """Retrieve a node and return it as a :class:`NodeSpec` if present."""

        if node_id not in self.graph:
            return None
        attrs = dict(self.graph.nodes[node_id])
        node_type = NodeType(attrs.pop("type"))
        return NodeSpec(id=node_id, type=node_type, attributes=attrs)

    def nodes(self) -> Iterable[str]:
        """Iterate over node identifiers."""

        return self.graph.nodes

    def edges(self) -> Iterable[tuple[str, str, dict]]:
        """Iterate over edges with their data."""

        for source, target, data in self.graph.edges(data=True):
            yield source, target, dict(data)
