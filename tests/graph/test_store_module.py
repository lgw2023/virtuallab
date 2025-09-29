"""Tests for :mod:`virtuallab.graph.store`."""

from __future__ import annotations

from virtuallab.graph.model import EdgeSpec, EdgeType, GraphDelta, NodeSpec, NodeType
from virtuallab.graph.store import GraphStore


def make_plan_node(identifier: str) -> NodeSpec:
    return NodeSpec(id=identifier, type=NodeType.PLAN, attributes={"name": identifier})


def test_graph_store_adds_nodes_and_edges():
    store = GraphStore()
    node_a = make_plan_node("plan_a")
    node_b = make_plan_node("plan_b")
    edge = EdgeSpec(source=node_a.id, target=node_b.id, type=EdgeType.ASSOCIATED_WITH)

    store.add_node(node_a)
    store.add_node(node_b)
    store.add_edge(edge)

    retrieved = store.get_node("plan_a")
    assert retrieved is not None
    assert retrieved.id == "plan_a"
    assert retrieved.attributes["name"] == "plan_a"

    edges = list(store.edges())
    assert edges[0][0] == node_a.id and edges[0][1] == node_b.id


def test_graph_store_apply_delta_updates_nodes():
    store = GraphStore()
    original = make_plan_node("plan_a")
    store.add_node(original)

    updated = NodeSpec(id="plan_a", type=NodeType.PLAN, attributes={"status": "active"})
    delta = GraphDelta(updated_nodes=[updated])
    store.apply_delta(delta)

    node = store.get_node("plan_a")
    assert node is not None
    assert node.attributes["status"] == "active"
