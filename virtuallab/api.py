"""Public API surface for VirtualLab."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Iterable, Mapping, Optional, Sequence

from virtuallab.graph.ids import new_id, utc_now
from virtuallab.graph.model import EdgeSpec, EdgeType, GraphDelta, NodeSpec, NodeType
from virtuallab.graph.store import GraphStore
from virtuallab.graph.query import QueryService
from virtuallab.graph.rules import AutoLinkProposal, AutoLinkService, OpenAIAutoLinkAdapter
from virtuallab.obs.events import EventBus
from virtuallab.router import ActionRouter
from virtuallab.exec.runner import StepRunner
from virtuallab.knowledge import OpenAILLMSummarizerAdapter, SummaryService


class _NodeHandle:
    """Lightweight view over a node stored in the VirtualLab graph."""

    __slots__ = ("app", "id")

    def __init__(self, app: "VirtualLabApp", node_id: str) -> None:
        self.app = app
        self.id = node_id

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.__class__.__name__}(id={self.id!r})"

    def node(self) -> NodeSpec:
        node = self.app.graph_store.get_node(self.id)
        if node is None:  # pragma: no cover - defensive
            raise KeyError(f"Node '{self.id}' no longer exists")
        return node


class PlanHandle(_NodeHandle):
    """Facade exposing plan centric helpers."""

    def add_subtask(self, **params) -> "SubtaskHandle":
        subtask_id, _ = self.app.add_subtask(plan=self.id, **params)
        return SubtaskHandle(self.app, subtask_id)

    def register_data(
        self,
        *,
        link_type: str | EdgeType = EdgeType.USES_DATA,
        link_attributes: Mapping[str, object] | None = None,
        **params,
    ) -> "DataHandle":
        data_id, _ = self.app.add_data(**params)
        self.app.link(source=self.id, target=data_id, type=link_type, attributes=link_attributes)
        return DataHandle(self.app, data_id)

    def link(self, target: "_NodeRef", *, type: str | EdgeType = EdgeType.ASSOCIATED_WITH, attributes: Mapping[str, object] | None = None) -> None:
        self.app.link(source=self.id, target=target, type=type, attributes=attributes)

    def timeline(self, *, include: Sequence[str] | None = None) -> list[dict]:
        return self.app.query(
            kind="timeline",
            scope={"plan_id": self.id},
            include=list(include) if include is not None else None,
        )["items"]


class SubtaskHandle(_NodeHandle):
    """Expose helpers for subtask centric graph operations."""

    def add_step(self, **params) -> "StepHandle":
        step_id, _ = self.app.add_step(subtask=self.id, **params)
        return StepHandle(self.app, step_id)


class StepHandle(_NodeHandle):
    """Convenience wrapper for step execution and inspection."""

    def run(self, *, payload: Mapping[str, object] | None = None, tool: str | None = None) -> dict:
        return self.app.run_step(step=self.id, payload=dict(payload or {}), tool=tool)


class DataHandle(_NodeHandle):
    """Data node utilities for linking artefacts into the plan."""

    def link_to(self, target: "_NodeRef", *, type: str | EdgeType = EdgeType.ASSOCIATED_WITH, attributes: Mapping[str, object] | None = None) -> None:
        self.app.link(source=self.id, target=target, type=type, attributes=attributes)


_NodeRef = str | _NodeHandle


@dataclass
class VirtualLabApp:
    """Container wiring together the core VirtualLab subsystems."""

    graph_store: GraphStore = field(default_factory=GraphStore)
    event_bus: EventBus = field(default_factory=EventBus)
    router: ActionRouter = field(default_factory=ActionRouter)
    step_runner: StepRunner = field(default_factory=StepRunner)
    summary_service: SummaryService | None = None
    auto_link_service: AutoLinkService | None = None

    def __post_init__(self) -> None:
        if self.summary_service is None:
            try:
                adapter = OpenAILLMSummarizerAdapter()
            except (ModuleNotFoundError, ImportError):  # pragma: no cover - optional deps
                class _EchoSummarizer:
                    def summarize(self, *, text: str, style: str | None = None) -> str:
                        return text if style is None else f"[{style}] {text}"

                adapter = _EchoSummarizer()
            self.summary_service = SummaryService(adapter=adapter)
        if self.auto_link_service is None:
            try:
                adapter = OpenAIAutoLinkAdapter()
            except RuntimeError:
                class _NoOpAutoLinkAdapter:
                    def propose_links(self, *, context):
                        return AutoLinkProposal(candidates=())

                self.auto_link_service = AutoLinkService(adapter=_NoOpAutoLinkAdapter())
            else:
                self.auto_link_service = AutoLinkService(adapter=adapter)
        self._register_default_actions()
        self._register_default_step_adapters()

    def handle(self, payload: dict) -> dict:
        """Dispatch an API payload and return a canonical response."""

        action = payload.get("action")
        if not action:
            raise KeyError("payload must include 'action'")
        params = payload.get("params", {})
        result = self.router.dispatch(action, params)
        event = self.event_bus.emit(level="info", msg=f"Executed action '{action}'", action=action)
        response = {
            "ok": True,
            "result": result.get("result", {}),
            "events": [event.__dict__],
            "graph_delta": self._serialize_graph_delta(result.get("graph_delta")),
        }
        return response

    def _register_default_actions(self) -> None:
        self.router.register("create_plan", self._handle_create_plan)
        self.router.register("add_subtask", self._handle_add_subtask)
        self.router.register("add_step", self._handle_add_step)
        self.router.register("add_data", self._handle_add_data)
        self.router.register("link", self._handle_link)
        self.router.register("auto_link", self._handle_auto_link)
        self.router.register("query", self._handle_query)
        self.router.register("run_step", self._handle_run_step)
        self.router.register("summarize", self._handle_summarize)
        self.router.register("record_note", self._handle_record_note)
        for action in (
            "export_graph",
            "snapshot",
            "rollback",
        ):
            self.router.register(action, self._not_implemented_action(action))

    def _register_default_step_adapters(self) -> None:
        if "engineer" in self.step_runner.adapters:
            return
        try:
            from virtuallab.exec.adapters.engineer import EngineerAdapter

            self.step_runner.register_adapter("engineer", EngineerAdapter())
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            # The Engineer adapter depends on the external ``smolagents`` package.
            # When it is unavailable we silently skip registration so that callers
            # can supply their own lightweight adapters during testing.
            return

    def _not_implemented_action(self, action: str):
        def _handler(_: dict) -> dict:
            return {
                "result": {"message": f"Action '{action}' is not implemented yet."},
                "graph_delta": None,
            }

        return _handler

    def _handle_create_plan(self, params: dict) -> dict:
        plan_id, delta = self.create_plan(
            name=params.get("name", ""),
            goal=params.get("goal", ""),
            owner=params.get("owner", "system"),
            status=params.get("status", "draft"),
            labels=params.get("labels"),
            plan_id=params.get("id"),
            attributes=params.get("attributes"),
        )
        return {"result": {"plan_id": plan_id}, "graph_delta": delta}

    def _handle_add_subtask(self, params: dict) -> dict:
        subtask_id, delta = self.add_subtask(
            plan=params.get("plan_id"),
            subtask_id=params.get("id"),
            name=params.get("name", ""),
            status=params.get("status", "pending"),
            priority=params.get("priority", "normal"),
            labels=params.get("labels"),
            attributes=params.get("attributes"),
        )
        return {"result": {"subtask_id": subtask_id}, "graph_delta": delta}

    def _handle_add_step(self, params: dict) -> dict:
        step_id, delta = self.add_step(
            subtask=params.get("subtask_id"),
            step_id=params.get("id"),
            name=params.get("name", ""),
            tool=params.get("tool", ""),
            inputs=params.get("inputs", {}),
            status=params.get("status", "pending"),
            labels=params.get("labels"),
            run_id=params.get("run_id"),
            attributes=params.get("attributes"),
        )
        return {"result": {"step_id": step_id}, "graph_delta": delta}

    def _handle_add_data(self, params: dict) -> dict:
        data_id, delta = self.add_data(
            data_id=params.get("id"),
            payload_ref=params.get("payload_ref"),
            format=params.get("format"),
            source=params.get("source"),
            labels=params.get("labels"),
            attributes=params.get("attributes"),
        )
        return {"result": {"data_id": data_id}, "graph_delta": delta}

    def _handle_record_note(self, params: dict) -> dict:
        note_id, delta, payload = self.record_note(
            note_id=params.get("id"),
            content=params.get("content"),
            tags=params.get("tags"),
            linked_to=params.get("linked_to"),
            labels=params.get("labels"),
            edge_type=params.get("edge_type"),
            edge_attributes=params.get("edge_attributes"),
        )
        payload.update({"note_id": note_id, "graph_delta": self._serialize_graph_delta(delta)})
        return {"result": payload, "graph_delta": delta}

    def _handle_link(self, params: dict) -> dict:
        edge, delta = self.link(
            source=params.get("source"),
            target=params.get("target"),
            type=params.get("type", EdgeType.ASSOCIATED_WITH.value),
            attributes=params.get("attributes"),
            ensure_exists=True,
        )
        return {"result": {"edge": edge}, "graph_delta": delta}

    def _handle_auto_link(self, params: dict) -> dict:
        return self.auto_link(
            scope=params.get("scope"),
            rules=params.get("rules"),
        )

    def _handle_query(self, params: dict) -> dict:
        payload = self.query(
            kind=params.get("kind", "by_type"),
            type=params.get("type"),
            filters=params.get("filters", {}),
            scope=params.get("scope"),
            include=params.get("include"),
            node_id=params.get("node_id"),
            hop=params.get("hop", 1),
            edge_types=params.get("edge_types"),
        )
        return {"result": payload, "graph_delta": None}

    def _handle_run_step(self, params: dict) -> dict:
        payload = self.run_step(
            step=params.get("step_id"),
            payload=params.get("payload"),
            tool=params.get("tool"),
            run_id=params.get("run_id"),
        )
        return {"result": payload["result"], "graph_delta": payload["graph_delta"]}

    def _handle_summarize(self, params: dict) -> dict:
        payload = self.summarize(
            text=params.get("text"),
            style=params.get("style"),
            target=params.get("target_id"),
        )
        return payload

    # ------------------------------------------------------------------
    # High level orchestration primitives
    # ------------------------------------------------------------------

    def plan(self, **params) -> PlanHandle:
        plan_id, _ = self.create_plan(**params)
        return PlanHandle(self, plan_id)

    def create_plan(
        self,
        *,
        name: str = "",
        goal: str = "",
        owner: str = "system",
        status: str = "draft",
        labels: Iterable[str] | None = None,
        plan_id: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> tuple[str, GraphDelta]:
        plan_id = plan_id or new_id("plan")
        now = utc_now()
        attrs = {
            "name": name,
            "goal": goal,
            "owner": owner,
            "status": status,
            "created_at": now,
            "updated_at": now,
            "labels": list(labels or []),
        }
        if attributes:
            attrs.update(attributes)
        node = NodeSpec(id=plan_id, type=NodeType.PLAN, attributes=attrs)
        delta = GraphDelta(added_nodes=[node])
        self.graph_store.apply_delta(delta)
        return plan_id, delta

    def subtask(self, plan: _NodeRef, **params) -> SubtaskHandle:
        subtask_id, _ = self.add_subtask(plan=plan, **params)
        return SubtaskHandle(self, subtask_id)

    def add_subtask(
        self,
        *,
        plan: _NodeRef | None = None,
        plan_id: str | None = None,
        name: str = "",
        status: str = "pending",
        priority: str = "normal",
        labels: Iterable[str] | None = None,
        subtask_id: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> tuple[str, GraphDelta]:
        owner_id = self._resolve_id(plan_id or plan)
        if not owner_id:
            raise KeyError("'plan' is required")
        plan_node = self.graph_store.get_node(owner_id)
        if plan_node is None or plan_node.type is not NodeType.PLAN:
            raise KeyError(f"Plan '{owner_id}' does not exist")

        subtask_id = subtask_id or new_id("subtask")
        now = utc_now()
        attrs = {
            "plan_id": owner_id,
            "name": name,
            "status": status,
            "priority": priority,
            "created_at": now,
            "updated_at": now,
            "labels": list(labels or []),
        }
        if attributes:
            attrs.update(attributes)
        node = NodeSpec(id=subtask_id, type=NodeType.SUBTASK, attributes=attrs)
        edge = EdgeSpec(source=owner_id, target=subtask_id, type=EdgeType.CONTAINS, attributes={})
        delta = GraphDelta(added_nodes=[node], added_edges=[edge])
        self.graph_store.apply_delta(delta)
        return subtask_id, delta

    def step(self, subtask: _NodeRef, **params) -> StepHandle:
        step_id, _ = self.add_step(subtask=subtask, **params)
        return StepHandle(self, step_id)

    def add_step(
        self,
        *,
        subtask: _NodeRef | None = None,
        subtask_id: str | None = None,
        name: str = "",
        tool: str = "",
        inputs: Mapping[str, object] | None = None,
        status: str = "pending",
        labels: Iterable[str] | None = None,
        run_id: str | None = None,
        step_id: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> tuple[str, GraphDelta]:
        owner_id = self._resolve_id(subtask_id or subtask)
        if not owner_id:
            raise KeyError("'subtask' is required")
        subtask_node = self.graph_store.get_node(owner_id)
        if subtask_node is None or subtask_node.type is not NodeType.SUBTASK:
            raise KeyError(f"Subtask '{owner_id}' does not exist")

        step_id = step_id or new_id("step")
        now = utc_now()
        attrs = {
            "subtask_id": owner_id,
            "name": name,
            "tool": tool,
            "inputs": dict(inputs or {}),
            "status": status,
            "run_id": run_id,
            "created_at": now,
            "updated_at": now,
            "labels": list(labels or []),
        }
        if attributes:
            attrs.update(attributes)
        node = NodeSpec(id=step_id, type=NodeType.STEP, attributes=attrs)
        edge = EdgeSpec(source=owner_id, target=step_id, type=EdgeType.CONTAINS, attributes={})
        delta = GraphDelta(added_nodes=[node], added_edges=[edge])
        self.graph_store.apply_delta(delta)
        return step_id, delta

    def add_data(
        self,
        *,
        payload_ref: str | None = None,
        format: str | None = None,
        source: str | None = None,
        labels: Iterable[str] | None = None,
        data_id: str | None = None,
        attributes: Mapping[str, object] | None = None,
    ) -> tuple[str, GraphDelta]:
        data_id = data_id or new_id("data")
        now = utc_now()
        attrs = {
            "payload_ref": payload_ref,
            "format": format,
            "source": source,
            "created_at": now,
            "updated_at": now,
            "labels": list(labels or []),
        }
        if attributes:
            attrs.update(attributes)
        node = NodeSpec(id=data_id, type=NodeType.DATA, attributes=attrs)
        delta = GraphDelta(added_nodes=[node])
        self.graph_store.apply_delta(delta)
        return data_id, delta

    def record_note(
        self,
        *,
        content: str | None,
        note_id: str | None = None,
        tags: Sequence[str] | str | None = None,
        linked_to: Sequence[_NodeRef] | _NodeRef | None = None,
        labels: Iterable[str] | None = None,
        edge_type: str | EdgeType | None = None,
        edge_attributes: Mapping[str, object] | None = None,
    ) -> tuple[str, GraphDelta, dict]:
        if not content:
            raise KeyError("'content' is required")

        note_id = note_id or new_id("note")
        tag_values = [str(item) for item in self._normalize_sequence(tags)]
        linked_ids = [
            resolved
            for resolved in (self._resolve_id(item) for item in self._normalize_sequence(linked_to))
            if resolved
        ]

        for target_id in linked_ids:
            if not self.graph_store.get_node(target_id):
                raise KeyError(f"Linked target '{target_id}' does not exist")

        now = utc_now()
        attrs = {
            "content": content,
            "tags": tag_values,
            "linked_to": linked_ids,
            "created_at": now,
            "updated_at": now,
            "labels": list(labels or []),
        }
        node = NodeSpec(id=note_id, type=NodeType.NOTE, attributes=attrs)

        edge_type_value = edge_type or EdgeType.ASSOCIATED_WITH.value
        edge = EdgeType(edge_type_value)
        edges = [
            EdgeSpec(source=note_id, target=target_id, type=edge, attributes=dict(edge_attributes or {}))
            for target_id in linked_ids
        ]

        delta = GraphDelta(added_nodes=[node], added_edges=edges)
        self.graph_store.apply_delta(delta)

        result_payload = {
            "tags": tag_values,
            "linked_to": linked_ids,
        }
        return note_id, delta, result_payload

    def link(
        self,
        *,
        source: _NodeRef,
        target: _NodeRef,
        type: str | EdgeType = EdgeType.ASSOCIATED_WITH,
        attributes: Mapping[str, object] | None = None,
        ensure_exists: bool = False,
    ) -> tuple[dict, GraphDelta]:
        source_id = self._resolve_id(source)
        target_id = self._resolve_id(target)
        if not source_id or not target_id:
            raise KeyError("'source' and 'target' are required")
        if ensure_exists:
            if not self.graph_store.get_node(source_id):
                raise KeyError(f"Source node '{source_id}' does not exist")
            if not self.graph_store.get_node(target_id):
                raise KeyError(f"Target node '{target_id}' does not exist")

        edge_type = EdgeType(type) if isinstance(type, str) else type
        edge = EdgeSpec(
            source=source_id,
            target=target_id,
            type=edge_type,
            attributes=dict(attributes or {}),
        )
        delta = GraphDelta(added_edges=[edge])
        self.graph_store.apply_delta(delta)
        payload = {"source": source_id, "target": target_id, "type": edge_type.value}
        return payload, delta

    def auto_link(
        self,
        *,
        scope: Mapping[str, object] | None,
        rules: Sequence[str] | str | None,
    ) -> dict:
        if self.auto_link_service is None:
            raise RuntimeError("Auto-link service is not configured")

        if scope is not None and not isinstance(scope, Mapping):
            raise TypeError("'scope' must be a mapping if provided")

        if rules is None:
            rule_list: list[str] | None = None
        elif isinstance(rules, str):
            rule_list = [rules]
        else:
            rule_list = [str(item) for item in rules]

        result = self.auto_link_service.generate(
            graph_store=self.graph_store,
            scope=scope,
            rules=rule_list,
        )
        delta = result.delta
        self.graph_store.apply_delta(delta)
        payload = result.to_payload()
        return {"result": payload, "graph_delta": delta}

    def query(
        self,
        *,
        kind: str,
        type: str | None = None,
        filters: Mapping[str, object] | None = None,
        scope: Mapping[str, object] | None = None,
        include: Sequence[str] | None = None,
        node_id: str | None = None,
        hop: int = 1,
        edge_types: Sequence[str] | None = None,
    ) -> dict:
        query = QueryService(self.graph_store.graph)
        if kind == "by_type":
            if not type:
                raise KeyError("'type' is required for by_type queries")
            results = list(query.by_type(type, **(filters or {})))
        elif kind == "timeline":
            results = list(query.timeline(scope=scope, include=include))
        elif kind == "neighbors":
            if not node_id:
                raise KeyError("'node_id' is required for neighbors queries")
            results = list(query.neighbors(node_id, hop=hop, edge_types=edge_types))
        else:
            raise ValueError(f"Unsupported query kind: {kind}")
        return {"items": results}

    def run_step(
        self,
        *,
        step: _NodeRef,
        payload: Mapping[str, object] | None = None,
        tool: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        step_id = self._resolve_id(step)
        if not step_id:
            raise KeyError("'step' is required")
        step_node = self.graph_store.get_node(step_id)
        if step_node is None or step_node.type is not NodeType.STEP:
            raise KeyError(f"Step '{step_id}' does not exist")

        chosen_tool = tool or step_node.attributes.get("tool")
        if not chosen_tool:
            raise KeyError("A 'tool' is required to execute the step")

        raw_payload = dict(payload or {})
        execution_details = self.step_runner.run(tool=chosen_tool, step_id=step_id, payload=raw_payload)
        print(f"execution_details: {json.dumps(execution_details, indent=2, ensure_ascii=False)}")
        execution_record = dict(execution_details)
        execution_record.setdefault("step_id", step_id)
        execution_record.setdefault("tool", chosen_tool)

        actual_run_id = execution_record.get("run_id") or run_id
        if not actual_run_id:
            actual_run_id = new_id("run")
            execution_record["run_id"] = actual_run_id

        status = execution_record.get("status") or "completed"
        timestamp = utc_now()

        updates: dict[str, object] = {
            "status": status,
            "run_id": actual_run_id,
            "updated_at": timestamp,
            "executed_at": timestamp,
            "last_run_tool": chosen_tool,
        }
        if "output" in execution_record:
            updates["last_run_output"] = execution_record["output"]
        if "error" in execution_record:
            updates["last_run_error"] = execution_record["error"]
        if "metrics" in execution_record:
            updates["last_run_metrics"] = execution_record["metrics"]

        updated_node = NodeSpec(id=step_node.id, type=step_node.type, attributes=updates)
        delta = GraphDelta(updated_nodes=[updated_node])
        self.graph_store.apply_delta(delta)

        result_payload = {
            "step_id": step_id,
            "run_id": actual_run_id,
            "status": status,
            "output": execution_record.get("output"),
            "brief_output": execution_record.get("brief_output"),
            "details": execution_record,
        }
        return {"result": result_payload, "graph_delta": delta}

    def summarize(
        self,
        *,
        text: str | None,
        style: str | None = None,
        target: _NodeRef | None = None,
    ) -> dict:
        if self.summary_service is None:
            raise RuntimeError("Summary service is not configured")
        if not text:
            raise KeyError("'text' is required")

        summary_result = self.summary_service.summarize(text=text, style=style)

        delta: Optional[GraphDelta]
        if target is not None:
            target_id = self._resolve_id(target)
            node = self.graph_store.get_node(target_id)
            if node is None:
                raise KeyError(f"Node '{target_id}' does not exist")
            updates = {
                "summary": summary_result["summary"],
                "summary_style": summary_result["style"],
                "updated_at": utc_now(),
            }
            updated_node = NodeSpec(id=node.id, type=node.type, attributes=updates)
            delta = GraphDelta(updated_nodes=[updated_node])
            self.graph_store.apply_delta(delta)
        else:
            delta = None

        return {"result": summary_result, "graph_delta": delta}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_id(node: _NodeRef | None) -> str | None:
        if node is None:
            return None
        if isinstance(node, _NodeHandle):
            return node.id
        return str(node)

    @staticmethod
    def _normalize_sequence(value: Sequence[_NodeRef] | _NodeRef | None) -> list[_NodeRef]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [item for item in value]
        return [value]

    def _serialize_graph_delta(self, delta: Optional[GraphDelta]) -> dict:
        if delta is None:
            return {"added_nodes": [], "added_edges": [], "updated_nodes": []}

        def _serialize_node(node: NodeSpec) -> dict:
            return {"id": node.id, "type": node.type.value, "attributes": dict(node.attributes)}

        def _serialize_edge(edge: EdgeSpec) -> dict:
            return {
                "source": edge.source,
                "target": edge.target,
                "type": edge.type.value,
                "attributes": dict(edge.attributes),
            }

        return {
            "added_nodes": [_serialize_node(node) for node in delta.added_nodes],
            "added_edges": [_serialize_edge(edge) for edge in delta.added_edges],
            "updated_nodes": [_serialize_node(node) for node in delta.updated_nodes],
        }


_APP = VirtualLabApp()


def VirtualLab_tool(payload: dict) -> dict:
    """Entry point exposed to external callers."""

    return _APP.handle(payload)
