"""Adapter for the Engineer autonomous agent."""
from __future__ import annotations

import httpx
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Protocol

from ...config import get_env


def _load_smolagents_dependencies() -> dict[str, Any]:
    """Import smolagents lazily and surface a clear error when missing."""

    try:
        from smolagents import ToolCallingAgent, OpenAIServerModel, WebSearchTool
        from ...tools.bash_code_run_tool import BashCodeRunTool
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ModuleNotFoundError(
            "The 'smolagents' package is required to use the Engineer adapter. "
            "Install it with 'pip install smolagents'."
        ) from exc

    return {
        "ToolCallingAgent": ToolCallingAgent,
        "OpenAIServerModel": OpenAIServerModel,
        "WebSearchTool": WebSearchTool(),
        "ShellBashInterpreterTool": BashCodeRunTool(working_dir="/Users/liguowei/ubuntu/virtuallab/genome"),
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
        self._ToolCallingAgent = deps["ToolCallingAgent"]
        self._OpenAIServerModel = deps["OpenAIServerModel"]
        self._tool_registry: Dict[str, Callable[[], Any]] = {
            "web_search": deps["WebSearchTool"],
            "shell_bash": deps["ShellBashInterpreterTool"],
        }
        if self.tool_factories:
            self._tool_registry.update(self.tool_factories)
        self._proxy_url = get_env("http_proxy_port") or None
        self._owns_model = self.model is None
        if self._owns_model:
            self.model = self._create_default_model(proxy_url=self._proxy_url)

    def _create_default_model(self, *, proxy_url: str | None) -> Any:
        """Instantiate :class:`OpenAIServerModel` using environment variables."""

        model_id = get_env("LLM_MODEL")
        api_base = get_env("LLM_MODEL_URL")
        api_key = get_env("LLM_MODEL_API_KEY")
        if not model_id or not api_base or not api_key:
            raise EnvironmentError(
                "LLM_MODEL, LLM_MODEL_URL, and LLM_MODEL_API_KEY environment variables "
                "must be set when 'model' is not provided to SmolagentsEngineerClient."
            )
        client_kwargs: dict[str, Any] = {}
        if proxy_url:
            client_kwargs["http_client"] = httpx.Client(proxy=proxy_url, verify=False)
        return self._OpenAIServerModel(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            client_kwargs=client_kwargs,
        )

    def _resolve_tools(self, tools: list[str] | None) -> list[Any]:
        if isinstance(tools, str):
            tool_names = [tools]
        else:
            tool_names = list(tools) if tools else list(self.default_tools)
        resolved = []
        for name in tool_names:
            factory_tool = self._tool_registry.get(name)
            if factory_tool is None:
                raise KeyError(f"Unknown Engineer tool '{name}'")
            resolved.append(factory_tool)
        return resolved

    def run(self, prompt: str, *, tools: list[str] | None = None) -> dict:
        attempts = 1
        if self._owns_model and self._proxy_url:
            attempts = 2

        last_error: Exception | None = None
        for attempt in range(attempts):
            agent = self._ToolCallingAgent(
                tools=self._resolve_tools(tools),
                model=self.model,
                stream_outputs=self.stream_outputs,
                max_steps=50,
            )
            try:
                full_result = agent.run(prompt, return_full_result=True)
                result = full_result.output
                brief_info = f"""
                please extract the brief information from the following full result.
                the brief information should be in a concise, readable and the following format:
                {{
                    "task step": <goal and action of the task step in one sentence>,
                    "tools used": <list ofkey tools used in the task step>,
                    "resulted files": <list of key resulted files>
                }}
                """
                prompt = """
                Here is the full result:
                """ + str(full_result.steps)
                from .openai_model import openai_complete
                brief_output = openai_complete(prompt, system_prompt=brief_info)
            except Exception as exc:  # pragma: no cover - network failures
                last_error = exc
                if attempt == 0 and attempts > 1:
                    self._proxy_url = None
                    self.model = self._create_default_model(proxy_url=None)
                    continue
                # return str(exc)
                return {"full_result": str(exc), "brief_result": str(exc)}

            return {"full_result": result, "brief_result": brief_output}

        # return "" if last_error is None else str(last_error)
        return {"full_result": "", "brief_result": ""} if last_error is None else {"full_result": str(last_error), "brief_result": str(last_error)} 


@dataclass
class EngineerAdapter:
    """Adapter that bridges :class:`StepRunner` with the Engineer agent."""

    client: EngineerClient = field(default_factory=SmolagentsEngineerClient)

    def run(self, *, step_id: str, payload: dict) -> dict:
        """Execute the step using the Engineer agent."""

        prompt: str = payload.get("text", "")
        tools = payload.get("tools") or []
        output = self.client.run(prompt, tools=tools)

        return {"step_id": step_id, "output": output["full_result"], "brief_output": output["brief_result"], "tools": tools}
