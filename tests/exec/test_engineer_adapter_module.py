"""Tests for the Engineer adapter wiring."""

from __future__ import annotations

from unittest import mock

import builtins
import importlib
import pathlib
import sys
import types

import pytest

import virtuallab.exec as exec_pkg

ADAPTERS_PACKAGE = "virtuallab.exec.adapters"


def load_engineer_module():
    adapters_path = pathlib.Path(exec_pkg.__file__).resolve().parent / "adapters"
    sys.modules.pop(ADAPTERS_PACKAGE, None)
    package = types.ModuleType(ADAPTERS_PACKAGE)
    package.__path__ = [str(adapters_path)]  # type: ignore[attr-defined]
    sys.modules[ADAPTERS_PACKAGE] = package
    return importlib.import_module("virtuallab.exec.adapters.engineer")


engineer = load_engineer_module()


def test_load_smolagents_dependencies_provides_clear_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("smolagents"):
            raise ModuleNotFoundError("smolagents missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError) as exc:
        engineer._load_smolagents_dependencies()

    assert "smolagents" in str(exc.value)


def make_stub_dependencies():
    class StubAgent:
        def __init__(self, *, tools, model, stream_outputs):
            self.tools = tools
            self.model = model
            self.stream_outputs = stream_outputs

        def run(self, prompt: str) -> str:
            return f"ran:{prompt}:{len(self.tools)}"

    class StubModel:
        def __init__(self, *, model_id, api_base, api_key, client_kwargs):
            self.config = (model_id, api_base, api_key, client_kwargs)

    return {
        "ToolCallingAgent": StubAgent,
        "OpenAIServerModel": StubModel,
        "WebSearchTool": lambda: "search",
        "ShellBashInterpreterTool": lambda: "shell_bash",
    }


def test_engineer_client_resolves_tools_and_runs(monkeypatch):
    monkeypatch.setattr(engineer, "_load_smolagents_dependencies", make_stub_dependencies)
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_MODEL_URL", "http://example")
    monkeypatch.setenv("LLM_MODEL_API_KEY", "key")

    client = engineer.SmolagentsEngineerClient(stream_outputs=False)

    output = client.run("do work", tools=["web_search", "shell_bash"])
    assert output == "ran:do work:2"


def test_engineer_client_rejects_unknown_tool(monkeypatch):
    monkeypatch.setattr(engineer, "_load_smolagents_dependencies", make_stub_dependencies)
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_MODEL_URL", "http://example")
    monkeypatch.setenv("LLM_MODEL_API_KEY", "key")

    client = engineer.SmolagentsEngineerClient()

    with pytest.raises(KeyError):
        client.run("prompt", tools=["nonexistent"])


def test_engineer_adapter_returns_structured_response(monkeypatch):
    stub_client = mock.Mock()
    stub_client.run.return_value = "result"

    adapter = engineer.EngineerAdapter(client=stub_client)
    response = adapter.run(step_id="s", payload={"text": "prompt", "tools": []})

    assert response == {"step_id": "s", "output": "result"}
    stub_client.run.assert_called_once_with("prompt", tools=[])


def test_engineer_client_resolves_tools_and_runs_real():
    client = engineer.SmolagentsEngineerClient(stream_outputs=False)
    output = client.run("list all files under current dir", tools=["web_search", "shell_bash"])
    assert isinstance(output, str)