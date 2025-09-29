"""Adapter for the Engineer autonomous agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Protocol


def _load_smolagents_dependencies() -> dict[str, Any]:
    """Import smolagents lazily and surface a clear error when missing."""

    try:
        from smolagents import CodeAgent, OpenAIServerModel, WebSearchTool
        from smolagents.tools import PythonInterpreterTool
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "The 'smolagents' package is required to use the Engineer adapter. "
            "Install it with 'pip install smolagents'."
        ) from exc

    return {
        "CodeAgent": CodeAgent,
        "OpenAIServerModel": OpenAIServerModel,
        "WebSearchTool": WebSearchTool,
        "PythonInterpreterTool": PythonInterpreterTool,
    }


class EngineerClient(Protocol):
    """Minimal protocol describing the Engineer agent interface."""

    def run(self, prompt: str, *, tools: list[str] | None = None) -> str:
        """Execute ``prompt`` using the Engineer agent."""


@dataclass
class SmolagentsEngineerClient:
    """Implementation of :class:`EngineerClient` backed by smolagents."""

    model: Any | None = None
    tool_factories: Dict[str, Callable[[], Any]] = field(default_factory=dict)
    default_tools: Iterable[str] = field(default_factory=lambda: ("web_search",))
    stream_outputs: bool = False

    def __post_init__(self) -> None:
        deps = _load_smolagents_dependencies()
        self._CodeAgent = deps["CodeAgent"]
        self._OpenAIServerModel = deps["OpenAIServerModel"]
        self._tool_registry: Dict[str, Callable[[], Any]] = {
            "web_search": deps["WebSearchTool"],
            "python": deps["PythonInterpreterTool"],
        }
        if self.tool_factories:
            self._tool_registry.update(self.tool_factories)
        if self.model is None:
            self.model = self._create_default_model()

    def _create_default_model(self) -> Any:
        """Instantiate :class:`OpenAIServerModel` using environment variables."""

        model_id = os.environ.get("LLM_MODEL")
        api_base = os.environ.get("LLM_MODEL_URL")
        api_key = os.environ.get("LLM_MODEL_API_KEY")
        if not model_id or not api_base or not api_key:
            raise EnvironmentError(
                "LLM_MODEL, LLM_MODEL_URL, and LLM_MODEL_API_KEY environment variables "
                "must be set when 'model' is not provided to SmolagentsEngineerClient."
            )
        return self._OpenAIServerModel(model_id=model_id, api_base=api_base, api_key=api_key)

    def _resolve_tools(self, tools: list[str] | None) -> list[Any]:
        if isinstance(tools, str):
            tool_names = [tools]
        else:
            tool_names = list(tools) if tools else list(self.default_tools)
        resolved = []
        for name in tool_names:
            factory = self._tool_registry.get(name)
            if factory is None:
                raise KeyError(f"Unknown Engineer tool '{name}'")
            resolved.append(factory())
        return resolved

    def run(self, prompt: str, *, tools: list[str] | None = None) -> str:
        agent = self._CodeAgent(
            tools=self._resolve_tools(tools),
            model=self.model,
            stream_outputs=self.stream_outputs,
        )
        result = agent.run(prompt)
        return result.strip() if isinstance(result, str) else str(result)


@dataclass
class EngineerAdapter:
    """Adapter that bridges :class:`StepRunner` with the Engineer agent."""

    client: EngineerClient = field(default_factory=SmolagentsEngineerClient)

    def run(self, *, step_id: str, payload: dict) -> dict:
        """Execute the step using the Engineer agent."""

        prompt: str = payload.get("text", "")
        tools = payload.get("tools") or []
        output = self.client.run(prompt, tools=tools)
        return {"step_id": step_id, "output": output}
