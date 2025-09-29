import pytest

from virtuallab.api import VirtualLabApp
from virtuallab.exec.runner import StepRunner
from virtuallab.graph.model import EdgeType, NodeType
from virtuallab.knowledge import SummaryService


class _StubStepAdapter:
    def run(self, *, step_id: str, payload: dict) -> dict:
        return {
            "step_id": step_id,
            "output": f"ran:{payload.get('text', '')}",
            "status": "completed",
            "run_id": "run_stub",
        }


class _StubSummarizer:
    def summarize(self, *, text: str, style: str | None = None) -> str:
        prefix = f"[{style}] " if style else ""
        return prefix + text[:50]


@pytest.fixture()
def app():
    runner = StepRunner()
    runner.register_adapter("engineer", _StubStepAdapter())
    summary_service = SummaryService(adapter=_StubSummarizer())
    return VirtualLabApp(step_runner=runner, summary_service=summary_service)


def _create_plan(app: VirtualLabApp, **overrides) -> str:
    params = {"name": "Plan", "goal": "Goal", "owner": "owner"}
    params.update(overrides)
    response = app.handle({"action": "create_plan", "params": params})
    return response["result"]["plan_id"]


def _add_subtask(app: VirtualLabApp, plan_id: str, **overrides) -> str:
    params = {"plan_id": plan_id, "name": "Subtask", "status": "pending"}
    params.update(overrides)
    response = app.handle({"action": "add_subtask", "params": params})
    return response["result"]["subtask_id"]


def _add_step(app: VirtualLabApp, subtask_id: str, **overrides) -> str:
    params = {
        "subtask_id": subtask_id,
        "name": "Step",
        "tool": "engineer",
        "inputs": {"param": 1},
    }
    params.update(overrides)
    response = app.handle({"action": "add_step", "params": params})
    return response["result"]["step_id"]


def test_create_plan_registers_node(app: VirtualLabApp) -> None:
    response = app.handle(
        {
            "action": "create_plan",
            "params": {"name": "Anomaly Detection", "goal": "Find issues", "owner": "qa"},
        }
    )

    assert response["ok"] is True
    plan_id = response["result"]["plan_id"]
    node = app.graph_store.get_node(plan_id)
    assert node is not None
    assert node.type is NodeType.PLAN
    assert node.attributes["name"] == "Anomaly Detection"
    assert app.event_bus.events[-1].action == "create_plan"
    assert response["graph_delta"]["added_nodes"][0]["id"] == plan_id


def test_add_subtask_requires_existing_plan(app: VirtualLabApp) -> None:
    with pytest.raises(KeyError, match="does not exist"):
        app.handle({"action": "add_subtask", "params": {"plan_id": "missing"}})


def test_add_subtask_creates_relationship(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)

    subtask = app.graph_store.get_node(subtask_id)
    assert subtask is not None
    assert subtask.type is NodeType.SUBTASK
    assert subtask.attributes["plan_id"] == plan_id

    edge_data = app.graph_store.graph.get_edge_data(plan_id, subtask_id)
    assert edge_data is not None
    assert EdgeType.CONTAINS.value in edge_data


def test_add_step_requires_existing_subtask(app: VirtualLabApp) -> None:
    with pytest.raises(KeyError, match="does not exist"):
        app.handle({"action": "add_step", "params": {"subtask_id": "missing"}})


def test_add_step_creates_edge(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)
    step_id = _add_step(app, subtask_id)

    step = app.graph_store.get_node(step_id)
    assert step is not None
    assert step.type is NodeType.STEP
    assert step.attributes["subtask_id"] == subtask_id

    edge_data = app.graph_store.graph.get_edge_data(subtask_id, step_id)
    assert edge_data is not None
    assert EdgeType.CONTAINS.value in edge_data


def test_link_requires_existing_nodes(app: VirtualLabApp) -> None:
    with pytest.raises(KeyError, match="Source node"):
        app.handle(
            {
                "action": "link",
                "params": {"source": "missing", "target": "also-missing", "type": EdgeType.DERIVES.value},
            }
        )


def test_link_adds_edge(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    data_response = app.handle(
        {
            "action": "add_data",
            "params": {"id": "data_manual", "payload_ref": "s3://bucket/file", "format": "json", "source": "manual"},
        }
    )
    data_id = data_response["result"]["data_id"]

    response = app.handle(
        {
            "action": "link",
            "params": {
                "source": plan_id,
                "target": data_id,
                "type": EdgeType.ASSOCIATED_WITH.value,
                "attributes": {"score": 0.5},
            },
        }
    )

    edge_payload = response["result"]["edge"]
    assert edge_payload == {"source": plan_id, "target": data_id, "type": EdgeType.ASSOCIATED_WITH.value}

    edge_data = app.graph_store.graph.get_edge_data(plan_id, data_id)
    assert edge_data is not None
    assert EdgeType.ASSOCIATED_WITH.value in edge_data


def test_query_by_type_returns_expected_nodes(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)
    _add_step(app, subtask_id)

    response = app.handle(
        {
            "action": "query",
            "params": {"kind": "by_type", "type": NodeType.SUBTASK.value, "filters": {"plan_id": plan_id}},
        }
    )

    items = response["result"]["items"]
    assert any(item["id"] == subtask_id for item in items)


def test_query_neighbors_filters_by_edge_type(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)
    _add_step(app, subtask_id)

    response = app.handle(
        {
            "action": "query",
            "params": {
                "kind": "neighbors",
                "node_id": plan_id,
                "hop": 1,
                "edge_types": [EdgeType.CONTAINS.value],
            },
        }
    )

    items = response["result"]["items"]
    assert [item["id"] for item in items] == [subtask_id]


def test_timeline_query_orders_by_timestamp(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)

    # Override timestamps for deterministic ordering.
    app.graph_store.graph.nodes[plan_id]["created_at"] = "2024-01-01T00:00:00+00:00"
    app.graph_store.graph.nodes[subtask_id]["created_at"] = "2024-01-02T00:00:00+00:00"

    response = app.handle(
        {
            "action": "query",
            "params": {
                "kind": "timeline",
                "scope": None,
                "include": [NodeType.PLAN.value, NodeType.SUBTASK.value],
            },
        }
    )

    items = response["result"]["items"]
    assert [item["id"] for item in items] == [plan_id, subtask_id]


def test_handle_requires_action_key(app: VirtualLabApp) -> None:
    with pytest.raises(KeyError, match="action"):
        app.handle({})


def test_run_step_updates_step_node(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)
    step_id = _add_step(app, subtask_id)

    response = app.handle(
        {
            "action": "run_step",
            "params": {
                "step_id": step_id,
                "payload": {"text": "do something"},
            },
        }
    )

    result = response["result"]
    assert result["status"] == "completed"
    assert result["run_id"] == "run_stub"
    assert result["output"] == "ran:do something"

    node = app.graph_store.get_node(step_id)
    assert node is not None
    assert node.attributes["status"] == "completed"
    assert node.attributes["last_run_output"] == "ran:do something"
    assert node.attributes["run_id"] == "run_stub"

    updated_nodes = response["graph_delta"]["updated_nodes"]
    assert updated_nodes and updated_nodes[0]["id"] == step_id


def test_summarize_returns_summary(app: VirtualLabApp) -> None:
    response = app.handle(
        {
            "action": "summarize",
            "params": {"text": "This is a long text", "style": "bullet"},
        }
    )

    result = response["result"]
    assert result["summary"].startswith("[bullet]")
    assert response["graph_delta"] == {"added_nodes": [], "added_edges": [], "updated_nodes": []}

