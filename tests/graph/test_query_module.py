"""Tests for :mod:`virtuallab.graph.query`."""

from __future__ import annotations

import itertools

import networkx as nx

from virtuallab.graph.query import QueryService


def build_graph():
    graph = nx.MultiDiGraph()
    graph.add_node("plan_1", type="Plan", created_at="2024-01-01T00:00:00")
    graph.add_node("subtask_1", type="Subtask", plan_id="plan_1", created_at="2024-01-02T00:00:00")
    graph.add_node(
        "step_1",
        type="Step",
        subtask_id="subtask_1",
        executed_at="2024-01-03T00:00:00",
        tool="python",
    )
    graph.add_node("note_1", type="Note", created_at="2024-01-04T00:00:00")
    graph.add_edge("plan_1", "subtask_1", key="CONTAINS", type="CONTAINS")
    graph.add_edge("subtask_1", "step_1", key="CONTAINS", type="CONTAINS")
    graph.add_edge("note_1", "step_1", key="ASSOCIATED_WITH", type="ASSOCIATED_WITH")
    return graph


def test_by_type_filters_results():
    service = QueryService(build_graph())
    results = list(service.by_type("Step", tool="python"))
    assert results and results[0]["id"] == "step_1"


def test_neighbors_respects_hop_and_edge_types():
    service = QueryService(build_graph())
    results = list(service.neighbors("plan_1", hop=2, edge_types=["CONTAINS"]))
    ids = {item["id"] for item in results}
    assert ids == {"subtask_1", "step_1"}


def test_neighbors_missing_node_yields_empty_iterator():
    service = QueryService(build_graph())
    assert list(service.neighbors("missing")) == []


def test_timeline_orders_by_timestamp():
    service = QueryService(build_graph())
    items = list(service.timeline(include=["Plan", "Step", "Note"]))
    ids_in_order = [item["id"] for item in items]
    assert ids_in_order == ["plan_1", "step_1", "note_1"]


def test_timeline_scoped_to_plan_members():
    graph = build_graph()
    graph.add_node("step_2", type="Step", subtask_id="subtask_2", executed_at="2024-01-05T00:00:00")
    service = QueryService(graph)
    scoped = list(service.timeline(scope={"plan_id": "plan_1"}))
    assert all(item["id"] != "step_2" for item in scoped)


def test_collect_plan_members_includes_subtasks_and_steps():
    service = QueryService(build_graph())
    members = service._collect_plan_members("plan_1")
    assert {"plan_1", "subtask_1", "step_1"}.issubset(members)
