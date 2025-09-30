"""Tests for :mod:`virtuallab.graph.rules`."""

from __future__ import annotations

import json

from virtuallab.graph.model import EdgeType, GraphDelta, NodeSpec, NodeType
from virtuallab.graph.rules import (
    AutoLinkCandidate,
    AutoLinkContext,
    AutoLinkResult,
    AutoLinkService,
    OpenAIAutoLinkAdapter,
)
from virtuallab.graph.store import GraphStore


def build_store() -> GraphStore:
    store = GraphStore()
    plan = NodeSpec(id="plan_1", type=NodeType.PLAN, attributes={"name": "Plan"})
    data = NodeSpec(
        id="data_1",
        type=NodeType.DATA,
        attributes={"payload_ref": "ref", "format": "json", "source": "lab"},
    )
    store.add_node(plan)
    store.add_node(data)
    return store


def test_autolink_service_builds_context_from_store():
    store = build_store()
    store.graph.add_edge("plan_1", "data_1", key="USES_DATA", type="USES_DATA")

    captured_payload: dict | None = None

    async def fake_completion(prompt: str, **kwargs):
        nonlocal captured_payload
        _, payload_text = prompt.split("Graph context:\n", 1)
        captured_payload = json.loads(payload_text)
        return json.dumps({"links": []})

    adapter = OpenAIAutoLinkAdapter(completion_func=fake_completion)
    service = AutoLinkService(adapter=adapter)

    result = service.generate(graph_store=store, scope={"plan": "plan_1"}, rules=["temporal"])

    assert result.context.scope == {"plan": "plan_1"}
    assert {node["id"] for node in result.context.nodes} == {"plan_1", "data_1"}
    assert any(edge["type"] == "USES_DATA" for edge in result.context.edges)
    assert {rule.name for rule in result.context.rules} == {"temporal"}
    assert captured_payload is not None
    assert {node["id"] for node in captured_payload["nodes"]} == {"plan_1", "data_1"}


def test_candidate_payload_includes_optional_fields():
    candidate = AutoLinkCandidate(
        source="plan_1",
        target="data_1",
        type=EdgeType.USES_DATA,
        rationale="needs data",
        confidence=0.8,
        attributes={"role": "input"},
    )
    payload = candidate.to_payload()
    assert payload["rationale"] == "needs data"
    assert payload["attributes"]["role"] == "input"


def test_autolink_service_filters_missing_and_duplicate_edges():
    store = build_store()
    store.graph.add_edge("plan_1", "data_1", key="USES_DATA", type="USES_DATA")

    async def fake_completion(prompt: str, **kwargs):
        response = {
            "links": [
                {"source": "plan_1", "target": "data_1", "type": "USES_DATA"},
                {"source": "plan_1", "target": "unknown", "type": "USES_DATA"},
                {
                    "source": "data_1",
                    "target": "plan_1",
                    "type": "DEPENDS_ON",
                    "rationale": "reverse",
                    "confidence": 0.9,
                },
            ],
            "analysis": "analysis",
        }
        return json.dumps(response)

    adapter = OpenAIAutoLinkAdapter(completion_func=fake_completion)
    service = AutoLinkService(adapter=adapter)

    result = service.generate(graph_store=store, scope=None, rules=["dependency", "Temporal", "logic"])

    assert [edge.type for edge in result.delta.added_edges] == [EdgeType.DEPENDS_ON]
    assert [edge.attributes["rationale"] for edge in result.delta.added_edges] == ["reverse"]
    assert len(result.applied) == 1
    assert result.analysis == "analysis"
    assert result.skipped and {item["reason"] for item in result.skipped} == {"duplicate", "missing-node"}
    assert {rule.name for rule in result.context.rules} == {"dependency", "temporal"}


def test_autolink_result_payload_serialisable():
    context = AutoLinkContext(scope=None, nodes=[], edges=[], rules=())
    result = AutoLinkResult(
        context=context,
        applied=[AutoLinkCandidate(source="a", target="b", type=EdgeType.ASSOCIATED_WITH)],
        skipped=[{"candidate": {"source": "x"}, "reason": "duplicate"}],
        delta=GraphDelta(),
        analysis="analysis",
    )
    payload = result.to_payload()
    assert payload["applied"][0]["source"] == "a"
    assert payload["analysis"] == "analysis"
