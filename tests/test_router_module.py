"""Integration-style tests for :mod:`virtuallab.router`."""

from __future__ import annotations

from virtuallab.api import VirtualLabApp
from virtuallab.graph.model import EdgeType, NodeType


def test_router_dispatches_registered_handler_creates_real_plan():
    """Ensure the router dispatches to the actual ``create_plan`` handler."""

    app = VirtualLabApp()

    result = app.router.dispatch(
        "create_plan",
        {"name": "Router test plan", "goal": "exercise real handler"},
    )

    plan_id = result["result"]["plan_id"]
    stored_plan = app.graph_store.get_node(plan_id)

    assert stored_plan is not None
    assert stored_plan.type is NodeType.PLAN
    assert stored_plan.attributes["name"] == "Router test plan"
    assert stored_plan.attributes["goal"] == "exercise real handler"


def test_router_dispatches_add_subtask_and_persists_linkage():
    """Dispatch through the router to create a subtask linked to a plan."""

    app = VirtualLabApp()

    plan_result = app.router.dispatch("create_plan", {"name": "parent"})
    plan_id = plan_result["result"]["plan_id"]

    subtask_result = app.router.dispatch(
        "add_subtask",
        {"plan_id": plan_id, "name": "child"},
    )

    subtask_id = subtask_result["result"]["subtask_id"]
    stored_subtask = app.graph_store.get_node(subtask_id)

    assert stored_subtask is not None
    assert stored_subtask.type is NodeType.SUBTASK
    assert stored_subtask.attributes["plan_id"] == plan_id
    assert stored_subtask.attributes["name"] == "child"

    # Ensure the graph has the relationship that ``add_subtask`` creates.
    edges = list(app.graph_store.graph.edges(plan_id, data=True))
    assert any(edge[1] == subtask_id and edge[2]["type"] == EdgeType.CONTAINS.value for edge in edges)


def test_router_dispatch_missing_action_raises():
    app = VirtualLabApp()

    try:
        app.router.dispatch("unknown_action", {})
    except KeyError as exc:
        assert "unknown_action" in str(exc)
    else:  # pragma: no cover - defensive fail if dispatch does not raise
        raise AssertionError("dispatch should raise KeyError for unknown actions")

