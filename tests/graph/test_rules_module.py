"""Tests for :mod:`virtuallab.graph.rules`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import networkx as nx

from virtuallab.graph.model import EdgeType, GraphDelta, NodeSpec, NodeType
from virtuallab.graph.rules import (
    AutoLinkCandidate,
    AutoLinkContext,
    AutoLinkProposal,
    AutoLinkResult,
    AutoLinkService,
    DEFAULT_RULES,
    RuleDescription,
)


class StubAdapter:
    """Adapter returning a predefined proposal for inspection."""

    def __init__(self, proposal: AutoLinkProposal) -> None:
        self.proposal = proposal
        self.seen_contexts: list[AutoLinkContext] = []

    def propose_links(self, *, context: AutoLinkContext) -> AutoLinkProposal:
        self.seen_contexts.append(context)
        return self.proposal


def build_store():
    from virtuallab.graph.store import GraphStore

    store = GraphStore()
    plan = NodeSpec(id="plan_1", type=NodeType.PLAN, attributes={"name": "Plan"})
    data = NodeSpec(id="data_1", type=NodeType.DATA, attributes={"payload_ref": "ref", "format": "json", "source": "lab"})
    store.add_node(plan)
    store.add_node(data)
    return store


def test_autolink_context_payload_shape():
    context = AutoLinkContext(scope={"plan": "plan_1"}, nodes=[{"id": "n"}], edges=[], rules=list(DEFAULT_RULES.values()))
    payload = context.to_prompt_payload()
    assert payload["scope"] == {"plan": "plan_1"}
    assert payload["nodes"] == [{"id": "n"}]


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

    proposal = AutoLinkProposal(
        candidates=[
            AutoLinkCandidate(source="plan_1", target="data_1", type=EdgeType.USES_DATA),
            AutoLinkCandidate(source="plan_1", target="unknown", type=EdgeType.USES_DATA),
            AutoLinkCandidate(
                source="data_1",
                target="plan_1",
                type=EdgeType.DEPENDS_ON,
                rationale="reverse",
                confidence=0.9,
            ),
        ],
        analysis="analysis",
    )
    adapter = StubAdapter(proposal)
    service = AutoLinkService(adapter=adapter)

    result = service.generate(graph_store=store, scope=None, rules=["dependency", "Temporal", "logic"])

    # Only the third candidate should be applied because the first is duplicate and second missing target.
    assert [edge.type for edge in result.delta.added_edges] == [EdgeType.DEPENDS_ON]
    assert len(result.applied) == 1
    assert result.skipped and {item["reason"] for item in result.skipped} == {"duplicate", "missing-node"}

    # Ensure rules were normalised and context captured.
    assert {rule.name for rule in result.context.rules} == {"dependency", "temporal"}
    assert adapter.seen_contexts and adapter.seen_contexts[0].nodes


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
