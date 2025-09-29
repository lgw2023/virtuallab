"""Auto-linking services powered by large language models."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
)

from .model import EdgeSpec, EdgeType, GraphDelta

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .store import GraphStore


CompletionFn = Callable[..., Awaitable[str | AsyncIterator[str]]]


def _run_sync(coro: Awaitable[str]) -> str:
    """Execute ``coro`` synchronously with a defensive event-loop guard."""

    try:
        return asyncio.run(coro)
    except RuntimeError as exc:  # pragma: no cover - running loop edge case
        if "asyncio.run() cannot be called" in str(exc):
            raise RuntimeError(
                "AutoLinkAdapter cannot execute because an event loop is already running. "
                "Provide a pre-executed completion function instead."
            ) from exc
        raise


async def _ensure_text(result: str | AsyncIterator[str]) -> str:
    """Normalise streaming completion responses to a plain string."""

    if isinstance(result, str):
        return result

    chunks: list[str] = []
    async for chunk in result:
        chunks.append(chunk)
    return "".join(chunks)


@dataclass(frozen=True)
class RuleDescription:
    """Metadata describing an auto-linking rule family."""

    name: str
    description: str
    suggested_edge_types: tuple[EdgeType, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Return a serialisable representation for prompting."""

        return {
            "name": self.name,
            "description": self.description,
            "suggested_edge_types": [edge_type.value for edge_type in self.suggested_edge_types],
        }


DEFAULT_RULES: dict[str, RuleDescription] = {
    "temporal": RuleDescription(
        name="temporal",
        description=(
            "Identify chronological relationships between steps. Use FOLLOWS when one step should "
            "execute after another, and DERIVES when later work directly refines the earlier output."
        ),
        suggested_edge_types=(EdgeType.FOLLOWS, EdgeType.DERIVES),
    ),
    "dependency": RuleDescription(
        name="dependency",
        description=(
            "Detect prerequisite or resource dependencies. Use DEPENDS_ON for logical/process "
            "dependencies and USES_DATA when a node consumes a specific dataset."
        ),
        suggested_edge_types=(EdgeType.DEPENDS_ON, EdgeType.USES_DATA),
    ),
    "causal": RuleDescription(
        name="causal",
        description=(
            "Surface cause-effect links grounded in evidence. Use CAUSED_BY for causal explanations "
            "and PRODUCES for explicit outcome generation relationships."
        ),
        suggested_edge_types=(EdgeType.CAUSED_BY, EdgeType.PRODUCES),
    ),
}

RULE_ALIASES: Mapping[str, str] = {
    "logic": "dependency",
    "dependencies": "dependency",
}


@dataclass
class AutoLinkContext:
    """Snapshot of the current graph state provided to adapters."""

    scope: Mapping[str, Any] | None
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    rules: list[RuleDescription]

    def to_prompt_payload(self) -> dict[str, Any]:
        """Return a JSON-friendly payload used for prompting LLMs."""

        return {
            "scope": dict(self.scope or {}),
            "nodes": self.nodes,
            "edges": self.edges,
            "rules": [rule.to_payload() for rule in self.rules],
        }


@dataclass
class AutoLinkCandidate:
    """Candidate edge proposed by an adapter."""

    source: str
    target: str
    type: EdgeType
    rationale: str | None = None
    confidence: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Convert the candidate to a serialisable structure."""

        payload = {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
        }
        if self.rationale is not None:
            payload["rationale"] = self.rationale
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.attributes:
            payload["attributes"] = dict(self.attributes)
        return payload


@dataclass
class AutoLinkProposal:
    """Structured response returned by auto-link adapters."""

    candidates: Sequence[AutoLinkCandidate]
    analysis: str | None = None


@dataclass
class AutoLinkResult:
    """Result of applying auto-link proposals to the graph."""

    context: AutoLinkContext
    applied: list[AutoLinkCandidate]
    skipped: list[dict[str, Any]]
    delta: GraphDelta
    analysis: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a serialisable representation for API consumers."""

        return {
            "scope": dict(self.context.scope or {}),
            "rules": [rule.name for rule in self.context.rules],
            "applied": [candidate.to_payload() for candidate in self.applied],
            "skipped": self.skipped,
            "analysis": self.analysis,
        }


class AutoLinkAdapter(Protocol):
    """Protocol defining the behaviour of auto-link adapters."""

    def propose_links(self, *, context: AutoLinkContext) -> AutoLinkProposal:
        """Produce edge candidates for ``context``."""


@dataclass
class AutoLinkService:
    """Coordinate auto-link proposals and apply safe filtering."""

    adapter: AutoLinkAdapter
    rule_catalogue: Mapping[str, RuleDescription] = field(default_factory=lambda: dict(DEFAULT_RULES))

    def generate(
        self,
        *,
        graph_store: "GraphStore",
        scope: Mapping[str, Any] | None = None,
        rules: Iterable[str] | None = None,
    ) -> AutoLinkResult:
        """Generate a :class:`GraphDelta` of inferred edges for ``graph_store``."""

        context = self._build_context(graph_store=graph_store, scope=scope, rules=rules)
        proposal = self.adapter.propose_links(context=context)
        applied: list[AutoLinkCandidate] = []
        skipped: list[dict[str, Any]] = []
        added_edges: list[EdgeSpec] = []

        graph = graph_store.graph
        existing_edges = {
            (source, target, data.get("type"))
            for source, target, data in graph.edges(data=True)
        }

        for candidate in proposal.candidates:
            key = (candidate.source, candidate.target, candidate.type.value)
            payload = candidate.to_payload()
            if candidate.source not in graph or candidate.target not in graph:
                skipped.append({"candidate": payload, "reason": "missing-node"})
                continue
            if key in existing_edges:
                skipped.append({"candidate": payload, "reason": "duplicate"})
                continue

            attributes = dict(candidate.attributes)
            if candidate.rationale and "rationale" not in attributes:
                attributes["rationale"] = candidate.rationale
            if candidate.confidence is not None and "confidence" not in attributes:
                attributes["confidence"] = candidate.confidence

            edge = EdgeSpec(
                source=candidate.source,
                target=candidate.target,
                type=candidate.type,
                attributes=attributes,
            )
            applied.append(candidate)
            added_edges.append(edge)
            existing_edges.add(key)

        delta = GraphDelta(added_edges=tuple(added_edges))
        return AutoLinkResult(
            context=context,
            applied=applied,
            skipped=skipped,
            delta=delta,
            analysis=proposal.analysis,
        )

    # -- internal helpers -------------------------------------------------

    def _build_context(
        self,
        *,
        graph_store,
        scope: Mapping[str, Any] | None,
        rules: Iterable[str] | None,
    ) -> AutoLinkContext:
        selected_rules = list(self._normalise_rules(rules))
        nodes = [
            {
                "id": node_id,
                "type": data.get("type"),
                "attributes": {k: v for k, v in data.items() if k != "type"},
            }
            for node_id, data in graph_store.graph.nodes(data=True)
        ]
        edges = [
            {
                "source": source,
                "target": target,
                "type": data.get("type"),
                "attributes": {k: v for k, v in data.items() if k != "type"},
            }
            for source, target, data in graph_store.graph.edges(data=True)
        ]
        return AutoLinkContext(scope=scope, nodes=nodes, edges=edges, rules=selected_rules)

    def _normalise_rules(self, rules: Iterable[str] | None) -> Iterable[RuleDescription]:
        if rules is None:
            return self.rule_catalogue.values()

        seen: set[str] = set()
        for rule_name in rules:
            if not isinstance(rule_name, str):
                continue
            key = rule_name.lower()
            key = RULE_ALIASES.get(key, key)
            if key in seen:
                continue
            seen.add(key)
            if key in self.rule_catalogue:
                yield self.rule_catalogue[key]


@dataclass
class OpenAIAutoLinkAdapter(AutoLinkAdapter):
    """Adapter that prompts an OpenAI-compatible model for linking suggestions."""

    completion_func: CompletionFn | None = None
    system_prompt: str = (
        "You are an expert research workflow analyst. Infer high-quality knowledge graph links "
        "that reflect temporal order, dependencies, and causal relationships."
    )
    completion_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._json_locator = None
        if self.completion_func is None:
            try:  # pragma: no cover - optional dependency
                from virtuallab.exec.adapters.openai_model import gpt_4o_mini_complete

                self.completion_func = gpt_4o_mini_complete
            except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - optional deps
                raise RuntimeError(
                    "OpenAIAutoLinkAdapter requires an OpenAI-compatible completion function"
                ) from exc
        try:  # pragma: no cover - optional dependency
            from virtuallab.exec.adapters.openai_model import locate_json_string_body_from_string

            self._json_locator = locate_json_string_body_from_string
        except (ModuleNotFoundError, ImportError):  # pragma: no cover - fallback when unavailable
            self._json_locator = lambda text: text

    def propose_links(self, *, context: AutoLinkContext) -> AutoLinkProposal:
        if self.completion_func is None:  # pragma: no cover - defensive guard
            raise RuntimeError("Completion function is not configured")

        prompt = self._build_prompt(context)

        async def _invoke() -> str:
            raw_result = await self.completion_func(
                prompt,
                system_prompt=self.system_prompt,
                response_format={"type": "json_object"},
                **self.completion_kwargs,
            )
            return await _ensure_text(raw_result)

        response_text = _run_sync(_invoke())
        json_blob = self._json_locator(response_text) if self._json_locator else response_text

        try:
            data = json.loads(json_blob)
        except json.JSONDecodeError:
            return AutoLinkProposal(candidates=())

        links = data.get("links", [])
        if not isinstance(links, list):
            return AutoLinkProposal(candidates=())

        candidates: list[AutoLinkCandidate] = []
        for item in links:
            if not isinstance(item, Mapping):
                continue
            source = item.get("source")
            target = item.get("target")
            edge_type = item.get("type")
            if not isinstance(source, str) or not isinstance(target, str) or not isinstance(edge_type, str):
                continue
            try:
                type_enum = EdgeType(edge_type)
            except ValueError:
                continue

            confidence_value = item.get("confidence")
            confidence: float | None
            if isinstance(confidence_value, (int, float)):
                confidence = float(confidence_value)
            elif isinstance(confidence_value, str):
                try:
                    confidence = float(confidence_value)
                except ValueError:
                    confidence = None
            else:
                confidence = None

            attributes = item.get("attributes")
            if isinstance(attributes, Mapping):
                attr_payload = dict(attributes)
            else:
                attr_payload = {}

            candidates.append(
                AutoLinkCandidate(
                    source=source,
                    target=target,
                    type=type_enum,
                    rationale=item.get("rationale") if isinstance(item.get("rationale"), str) else None,
                    confidence=confidence,
                    attributes=attr_payload,
                )
            )

        analysis = data.get("analysis") or data.get("reasoning")
        if isinstance(analysis, str):
            analysis_text = analysis
        else:
            analysis_text = None
        return AutoLinkProposal(candidates=candidates, analysis=analysis_text)

    def _build_prompt(self, context: AutoLinkContext) -> str:
        payload = context.to_prompt_payload()
        instructions = [
            "Analyse the VirtualLab graph context and propose directed edges that satisfy the requested rule families.",
            "Only suggest links that add meaningful structure. Omit duplicates that already exist.",
            "Use the provided edge type vocabulary exactly as-is (case sensitive).",
            "Return a JSON object with fields: links (array of edge objects with source, target, type, optional rationale, confidence, attributes) and analysis (short summary).",
            "Graph context:",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ]
        return "\n".join(instructions)


__all__ = [
    "AutoLinkAdapter",
    "AutoLinkCandidate",
    "AutoLinkContext",
    "AutoLinkProposal",
    "AutoLinkResult",
    "AutoLinkService",
    "OpenAIAutoLinkAdapter",
    "RuleDescription",
]

