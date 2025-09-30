from unittest import mock

import pytest

from virtuallab.api import VirtualLabApp
from virtuallab.config import get_env
from virtuallab.exec.adapters import engineer
from virtuallab.graph.model import EdgeType, NodeType
from virtuallab.graph.rules import AutoLinkCandidate, AutoLinkContext, AutoLinkProposal


@pytest.fixture()
def app():
    app = VirtualLabApp()
    stub_client = mock.Mock()
    stub_client.run.side_effect = lambda prompt, tools=None: f"ran:{prompt}:{len(tools or [])}"
    app.step_runner.register_adapter("engineer", engineer.EngineerAdapter(client=stub_client))
    return app


@pytest.fixture(scope="module")
def ensure_openai_env() -> None:
    if not get_env("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")
    if not get_env("OPENAI_API_MODEL"):
        pytest.skip("OPENAI_API_MODEL is not configured")


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


def test_summarize_action_invokes_llm(app: VirtualLabApp, ensure_openai_env: None) -> None:
    response = app.handle(
        {
            "action": "summarize",
            "params": {
                "text": "Summarize the experiment results highlighting key findings.",
                "style": "concise",
            },
        }
    )

    summary = response["result"]["summary"]
    assert isinstance(summary, str)
    assert summary.strip(), "Expected summarization to return non-empty content"
    assert summary.strip() not in {
        "Summarize the experiment results highlighting key findings.",
        "[concise] Summarize the experiment results highlighting key findings.",
    }


def test_record_note_creates_note_node(app: VirtualLabApp) -> None:
    response = app.handle(
        {
            "action": "record_note",
            "params": {"content": "Observation", "tags": ["daily", "lab"]},
        }
    )

    note_id = response["result"]["note_id"]
    node = app.graph_store.get_node(note_id)
    assert node is not None
    assert node.type is NodeType.NOTE
    assert node.attributes["content"] == "Observation"
    assert response["result"]["graph_delta"]["added_nodes"][0]["id"] == note_id


def test_record_note_links_to_targets(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)

    response = app.handle(
        {
            "action": "record_note",
            "params": {
                "content": "Plan review",
                "linked_to": [plan_id],
                "edge_attributes": {"score": 0.9},
            },
        }
    )

    note_id = response["result"]["note_id"]
    edge_delta = response["graph_delta"]["added_edges"][0]
    assert edge_delta == {
        "source": note_id,
        "target": plan_id,
        "type": EdgeType.ASSOCIATED_WITH.value,
        "attributes": {"score": 0.9},
    }

    edge_data = app.graph_store.graph.get_edge_data(note_id, plan_id)
    assert edge_data is not None
    assert EdgeType.ASSOCIATED_WITH.value in edge_data


def test_record_note_rejects_unknown_link_targets(app: VirtualLabApp) -> None:
    with pytest.raises(KeyError, match="does not exist"):
        app.handle(
            {
                "action": "record_note",
                "params": {"content": "Unknown link", "linked_to": ["missing"]},
            }
        )


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


def test_auto_link_generates_follow_edges(app: VirtualLabApp, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)
    first_step = _add_step(app, subtask_id, name="Collect data")
    second_step = _add_step(app, subtask_id, name="Train model")

    assert app.auto_link_service is not None

    def _propose_links(*, context: AutoLinkContext) -> AutoLinkProposal:
        steps = [node for node in context.nodes if node.get("type") == NodeType.STEP.value]
        steps.sort(key=lambda node: node.get("attributes", {}).get("created_at", ""))
        if len(steps) < 2:
            return AutoLinkProposal(candidates=())
        candidate = AutoLinkCandidate(
            source=steps[0]["id"],
            target=steps[1]["id"],
            type=EdgeType.FOLLOWS,
            rationale="Ordered execution",
            confidence=0.75,
            attributes={"position": 1},
        )
        return AutoLinkProposal(candidates=[candidate], analysis="deterministic test linkage")

    monkeypatch.setattr(app.auto_link_service.adapter, "propose_links", _propose_links, raising=False)

    response = app.handle(
        {
            "action": "auto_link",
            "params": {"scope": {"plan_id": plan_id}, "rules": ["temporal", "causal"]},
        }
    )

    applied = response["result"]["applied"]
    assert applied
    link = applied[0]
    assert link["source"] == first_step
    assert link["target"] == second_step
    assert link["type"] == EdgeType.FOLLOWS.value
    assert link["attributes"]["position"] == 1
    assert response["graph_delta"]["added_edges"][0]["source"] == first_step

    edge_data = app.graph_store.graph.get_edge_data(first_step, second_step)
    assert edge_data is not None
    assert EdgeType.FOLLOWS.value in edge_data


def test_auto_link_skips_duplicates(app: VirtualLabApp, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)
    _add_step(app, subtask_id, name="Collect data")
    _add_step(app, subtask_id, name="Train model")

    assert app.auto_link_service is not None

    def _propose_links(*, context: AutoLinkContext) -> AutoLinkProposal:
        steps = [node for node in context.nodes if node.get("type") == NodeType.STEP.value]
        steps.sort(key=lambda node: node.get("attributes", {}).get("created_at", ""))
        if len(steps) < 2:
            return AutoLinkProposal(candidates=())
        candidate = AutoLinkCandidate(
            source=steps[0]["id"],
            target=steps[1]["id"],
            type=EdgeType.FOLLOWS,
        )
        return AutoLinkProposal(candidates=[candidate])

    monkeypatch.setattr(app.auto_link_service.adapter, "propose_links", _propose_links, raising=False)

    first_response = app.handle({"action": "auto_link", "params": {"scope": {"plan_id": plan_id}}})
    assert first_response["result"]["applied"]

    second_response = app.handle({"action": "auto_link", "params": {"scope": {"plan_id": plan_id}}})
    assert second_response["result"]["applied"] == []
    skipped = second_response["result"].get("skipped")
    assert skipped and skipped[0]["reason"] == "duplicate"
    assert second_response["graph_delta"]["added_edges"] == []


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


def test_timeline_scope_includes_plan_node(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    _add_subtask(app, plan_id)

    response = app.handle(
        {
            "action": "query",
            "params": {
                "kind": "timeline",
                "scope": {"plan_id": plan_id},
                "include": [NodeType.PLAN.value, NodeType.SUBTASK.value],
            },
        }
    )

    items = response["result"]["items"]
    assert items[0]["id"] == plan_id


def test_timeline_scope_includes_plan_hierarchy(app: VirtualLabApp) -> None:
    plan_id = _create_plan(app)
    subtask_id = _add_subtask(app, plan_id)
    first_step = _add_step(app, subtask_id, name="Step A")
    second_step = _add_step(app, subtask_id, name="Step B")

    graph = app.graph_store.graph
    graph.nodes[plan_id]["created_at"] = "2024-01-01T00:00:00+00:00"
    graph.nodes[subtask_id]["created_at"] = "2024-01-02T00:00:00+00:00"
    graph.nodes[first_step]["created_at"] = "2024-01-03T00:00:00+00:00"
    graph.nodes[second_step]["created_at"] = "2024-01-04T00:00:00+00:00"

    response = app.handle(
        {
            "action": "query",
            "params": {
                "kind": "timeline",
                "scope": {"plan_id": plan_id},
                "include": [
                    NodeType.PLAN.value,
                    NodeType.SUBTASK.value,
                    NodeType.STEP.value,
                ],
            },
        }
    )

    items = response["result"]["items"]
    ids = [item["id"] for item in items]
    assert ids == [plan_id, subtask_id, first_step, second_step]


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
    assert result["run_id"].startswith("run_")
    assert result["output"] == "ran:do something:0"
    assert result["details"]["tool"] == "engineer"
    assert result["details"]["step_id"] == step_id

    node = app.graph_store.get_node(step_id)
    assert node is not None
    assert node.attributes["status"] == "completed"
    assert node.attributes["last_run_output"] == "ran:do something:0"
    assert node.attributes["run_id"] == result["run_id"]

    updated_nodes = response["graph_delta"]["updated_nodes"]
    assert updated_nodes and updated_nodes[0]["id"] == step_id


def test_summarize_returns_summary(app: VirtualLabApp) -> None:
    response = app.handle(
        {
            "action": "summarize",
            "params": {"text": "This is a long text", "style": "bullet"},
        }
    )
    print(f"\n@ test_summarize_returns_summary: response = {response}")
    result = response["result"]
    summary = result["summary"]
    print(f"\n@ test_summarize_returns_summary: summary = {summary}")

    assert isinstance(summary, str)
    assert summary.strip(), "Expected summarization to return non-empty content"
    assert summary.strip() != "This is a long text"
    assert result["style"] == "bullet"
    print(f"\n@ test_summarize_returns_summary: result['style'] = {result['style']}")
    assert response["graph_delta"] == {"added_nodes": [], "added_edges": [], "updated_nodes": []}
    print(f"\n@ test_summarize_returns_summary: response['graph_delta'] = {response['graph_delta']}")

