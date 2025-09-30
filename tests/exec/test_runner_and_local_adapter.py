"""Tests for execution runners and adapters."""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

import pytest

import virtuallab.api as api_module
import virtuallab.exec as exec_pkg
from virtuallab.exec.runner import StepRunner


def load_local_module():
    adapters_path = pathlib.Path(exec_pkg.__file__).resolve().parent / "adapters"
    package_name = "virtuallab.exec.adapters"
    sys.modules.pop(package_name, None)
    package = types.ModuleType(package_name)
    package.__path__ = [str(adapters_path)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    return importlib.import_module("virtuallab.exec.adapters.local")


local_adapter_module = load_local_module()


def test_step_runner_runs_virtual_lab_step_via_local_adapter(monkeypatch):
    app = api_module.VirtualLabApp()
    monkeypatch.setattr(api_module, "_APP", app)

    local_adapter = local_adapter_module.LocalFuncAdapter()
    app.step_runner.register_adapter("local", local_adapter)

    def complete_step(payload: dict) -> dict:
        step_id = payload["step_id"]
        inputs = payload.get("inputs", {})
        total = sum(value for value in inputs.values() if isinstance(value, (int, float)))
        return {
            "step_id": step_id,
            "status": "completed",
            "output": {"sum": total, "inputs": inputs},
            "metrics": {"input_count": len(inputs)},
        }

    local_adapter.register("complete_step", complete_step)

    plan_response = api_module.VirtualLab_tool({
        "action": "create_plan",
        "params": {"name": "Integration plan"},
    })
    plan_id = plan_response["result"]["plan_id"]

    subtask_response = api_module.VirtualLab_tool({
        "action": "add_subtask",
        "params": {"plan_id": plan_id, "name": "Integration task"},
    })
    subtask_id = subtask_response["result"]["subtask_id"]

    step_response = api_module.VirtualLab_tool({
        "action": "add_step",
        "params": {
            "subtask_id": subtask_id,
            "name": "Compute sum",
            "tool": "local",
        },
    })
    step_id = step_response["result"]["step_id"]

    run_response = api_module.VirtualLab_tool({
        "action": "run_step",
        "params": {
            "step_id": step_id,
            "tool": "local",
            "payload": {
                "function": "complete_step",
                "step_id": step_id,
                "inputs": {"a": 1, "b": 2},
            },
        },
    })

    assert run_response["result"]["status"] == "completed"
    assert run_response["result"]["output"] == {
        "sum": 3,
        "inputs": {"a": 1, "b": 2},
    }
    assert run_response["result"]["details"]["metrics"] == {"input_count": 2}

    step_node = app.graph_store.get_node(step_id)
    assert step_node is not None
    assert step_node.attributes["status"] == "completed"
    assert step_node.attributes["last_run_output"] == {
        "sum": 3,
        "inputs": {"a": 1, "b": 2},
    }


def test_step_runner_missing_adapter():
    runner = StepRunner()
    with pytest.raises(KeyError):
        runner.run(tool="unknown", step_id="s", payload={})


def test_local_func_adapter_runs_virtual_lab_tool(monkeypatch):
    app = api_module.VirtualLabApp()
    monkeypatch.setattr(api_module, "_APP", app)

    adapter = local_adapter_module.LocalFuncAdapter()
    adapter.register("VirtualLab_tool", api_module.VirtualLab_tool)

    response = adapter.run(
        step_id="local-1",
        payload={
            "function": "VirtualLab_tool",
            "action": "create_plan",
            "params": {"name": "Direct call"},
        },
    )

    assert response["ok"] is True
    plan_id = response["result"]["plan_id"]
    assert app.graph_store.get_node(plan_id) is not None


def test_local_func_adapter_missing_function_metadata():
    adapter = local_adapter_module.LocalFuncAdapter()
    with pytest.raises(KeyError):
        adapter.run(step_id="local", payload={})


def test_local_func_adapter_missing_registration():
    adapter = local_adapter_module.LocalFuncAdapter()
    with pytest.raises(KeyError):
        adapter.run(step_id="local", payload={"function": "nope"})
