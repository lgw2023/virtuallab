"""Tests covering :mod:`virtuallab.graph.model`."""

from __future__ import annotations

import pytest

from virtuallab.graph.model import (
    EdgeSpec,
    EdgeType,
    GraphDelta,
    GraphSchema,
    NodeSpec,
    NodeType,
    coerce_edge_payload,
    coerce_node_payload,
)


def test_node_spec_update_merges_values():
    node = NodeSpec(id="n1", type=NodeType.PLAN, attributes={"name": "plan"})
    node.update({"status": "active"})
    assert node.attributes == {"name": "plan", "status": "active"}


def test_graph_schema_required_fields():
    fields = tuple(GraphSchema.required_fields_for(NodeType.PLAN))
    assert {"name", "goal", "owner", "status"}.issubset(fields)


def test_graph_schema_edge_fields_default():
    fields = tuple(GraphSchema.edge_fields_for(EdgeType.USES_DATA))
    assert fields == ("role",)


def test_coerce_node_payload_validates_identifier():
    with pytest.raises(ValueError):
        coerce_node_payload(NodeType.PLAN, attributes={"name": "p", "goal": "g", "owner": "o", "status": "draft"})


def test_coerce_node_payload_generates_spec_with_defaults():
    node = coerce_node_payload(
        NodeType.PLAN,
        attributes={"id": "plan_1", "name": "p", "goal": "g", "owner": "o", "status": "draft"},
    )
    assert node.id == "plan_1"
    assert node.attributes["labels"] == []


def test_coerce_edge_payload_populates_default_fields():
    edge = coerce_edge_payload(EdgeType.ASSOCIATED_WITH, source="a", target="b", attributes={"score": 0.5})
    assert isinstance(edge, EdgeSpec)
    assert edge.attributes["score"] == 0.5


def test_graph_delta_defaults_iterables():
    delta = GraphDelta()
    assert list(delta.added_nodes) == []
    assert list(delta.added_edges) == []
