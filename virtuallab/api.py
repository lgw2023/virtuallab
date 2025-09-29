"""Public API surface for VirtualLab."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from virtuallab.graph.ids import new_id, utc_now
from virtuallab.graph.model import EdgeSpec, EdgeType, GraphDelta, NodeSpec, NodeType
from virtuallab.graph.store import GraphStore
from virtuallab.graph.query import QueryService
from virtuallab.obs.events import EventBus
from virtuallab.router import ActionRouter
from virtuallab.exec.runner import StepRunner


@dataclass
class VirtualLabApp:
    """Container wiring together the core VirtualLab subsystems."""

    graph_store: GraphStore = field(default_factory=GraphStore)
    event_bus: EventBus = field(default_factory=EventBus)
    router: ActionRouter = field(default_factory=ActionRouter)
    step_runner: StepRunner = field(default_factory=StepRunner)

    def __post_init__(self) -> None:
        self._register_default_actions()

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
        self.router.register("query", self._handle_query)
        for action in (
            "run_step",
            "summarize",
            "record_note",
            "auto_link",
            "export_graph",
            "snapshot",
            "rollback",
        ):
            self.router.register(action, self._not_implemented_action(action))

    def _not_implemented_action(self, action: str):
        def _handler(_: dict) -> dict:
            return {
                "result": {"message": f"Action '{action}' is not implemented yet."},
                "graph_delta": None,
            }

        return _handler

    def _handle_create_plan(self, params: dict) -> dict:
        plan_id = params.get("id") or new_id("plan")
        now = utc_now()
        node = NodeSpec(
            id=plan_id,
            type=NodeType.PLAN,
            attributes={
                "name": params.get("name", ""),
                "goal": params.get("goal", ""),
                "owner": params.get("owner", "system"),
                "status": params.get("status", "draft"),
                "created_at": now,
                "updated_at": now,
                "labels": params.get("labels", []),
            },
        )
        self.graph_store.add_node(node)
        delta = GraphDelta(added_nodes=[node])
        return {"result": {"plan_id": plan_id}, "graph_delta": delta}

    def _handle_add_subtask(self, params: dict) -> dict:
        plan_id = params.get("plan_id")
        if not plan_id:
            raise KeyError("'plan_id' is required")
        if not self.graph_store.get_node(plan_id):
            raise KeyError(f"Plan '{plan_id}' does not exist")
        subtask_id = params.get("id") or new_id("subtask")
        now = utc_now()
        node = NodeSpec(
            id=subtask_id,
            type=NodeType.SUBTASK,
            attributes={
                "plan_id": plan_id,
                "name": params.get("name", ""),
                "status": params.get("status", "pending"),
                "priority": params.get("priority", "normal"),
                "created_at": now,
                "updated_at": now,
                "labels": params.get("labels", []),
            },
        )
        edge = EdgeSpec(
            source=plan_id,
            target=subtask_id,
            type=EdgeType.CONTAINS,
            attributes={},
        )
        self.graph_store.add_node(node)
        self.graph_store.add_edge(edge)
        delta = GraphDelta(added_nodes=[node], added_edges=[edge])
        return {"result": {"subtask_id": subtask_id}, "graph_delta": delta}

    def _handle_add_step(self, params: dict) -> dict:
        subtask_id = params.get("subtask_id")
        if not subtask_id:
            raise KeyError("'subtask_id' is required")
        if not self.graph_store.get_node(subtask_id):
            raise KeyError(f"Subtask '{subtask_id}' does not exist")
        step_id = params.get("id") or new_id("step")
        now = utc_now()
        node = NodeSpec(
            id=step_id,
            type=NodeType.STEP,
            attributes={
                "subtask_id": subtask_id,
                "name": params.get("name", ""),
                "tool": params.get("tool", ""),
                "inputs": params.get("inputs", {}),
                "status": params.get("status", "pending"),
                "run_id": params.get("run_id"),
                "created_at": now,
                "updated_at": now,
                "labels": params.get("labels", []),
            },
        )
        edge = EdgeSpec(
            source=subtask_id,
            target=step_id,
            type=EdgeType.CONTAINS,
            attributes={},
        )
        self.graph_store.add_node(node)
        self.graph_store.add_edge(edge)
        delta = GraphDelta(added_nodes=[node], added_edges=[edge])
        return {"result": {"step_id": step_id}, "graph_delta": delta}

    def _handle_add_data(self, params: dict) -> dict:
        data_id = params.get("id") or new_id("data")
        now = utc_now()
        node = NodeSpec(
            id=data_id,
            type=NodeType.DATA,
            attributes={
                "payload_ref": params.get("payload_ref"),
                "format": params.get("format"),
                "source": params.get("source"),
                "created_at": now,
                "updated_at": now,
                "labels": params.get("labels", []),
            },
        )
        self.graph_store.add_node(node)
        delta = GraphDelta(added_nodes=[node])
        return {"result": {"data_id": data_id}, "graph_delta": delta}

    def _handle_link(self, params: dict) -> dict:
        source = params.get("source")
        target = params.get("target")
        if not source or not target:
            raise KeyError("'source' and 'target' are required")
        if not self.graph_store.get_node(source):
            raise KeyError(f"Source node '{source}' does not exist")
        if not self.graph_store.get_node(target):
            raise KeyError(f"Target node '{target}' does not exist")
        edge_type = EdgeType(params.get("type", EdgeType.ASSOCIATED_WITH.value))
        edge = EdgeSpec(source=source, target=target, type=edge_type, attributes=params.get("attributes", {}))
        self.graph_store.add_edge(edge)
        delta = GraphDelta(added_edges=[edge])
        return {"result": {"edge": {"source": source, "target": target, "type": edge.type.value}}, "graph_delta": delta}

    def _handle_query(self, params: dict) -> dict:
        query = QueryService(self.graph_store.graph)
        kind = params.get("kind", "by_type")
        if kind == "by_type":
            node_type = params.get("type")
            if not node_type:
                raise KeyError("'type' is required for by_type queries")
            results = list(query.by_type(node_type, **params.get("filters", {})))
        elif kind == "timeline":
            results = list(query.timeline(scope=params.get("scope"), include=params.get("include")))
        elif kind == "neighbors":
            node_id = params.get("node_id")
            if not node_id:
                raise KeyError("'node_id' is required for neighbors queries")
            results = list(
                query.neighbors(
                    node_id,
                    hop=params.get("hop", 1),
                    edge_types=params.get("edge_types"),
                )
            )
        else:
            raise ValueError(f"Unsupported query kind: {kind}")
        return {"result": {"items": results}, "graph_delta": None}

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
