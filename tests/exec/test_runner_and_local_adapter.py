"""Tests for execution runners and adapters."""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

import pytest

from virtuallab.exec.runner import StepRunner
import virtuallab.exec as exec_pkg


def load_local_module():
    adapters_path = pathlib.Path(exec_pkg.__file__).resolve().parent / "adapters"
    package_name = "virtuallab.exec.adapters"
    sys.modules.pop(package_name, None)
    package = types.ModuleType(package_name)
    package.__path__ = [str(adapters_path)]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    return importlib.import_module("virtuallab.exec.adapters.local")


local_adapter_module = load_local_module()


def test_step_runner_dispatches_registered_adapter():
    class EchoAdapter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def run(self, *, step_id: str, payload: dict) -> dict:
            self.calls.append((step_id, payload))
            return {"step_id": step_id, "payload": payload}

    runner = StepRunner()
    adapter = EchoAdapter()
    runner.register_adapter("echo", adapter)

    result = runner.run(tool="echo", step_id="s1", payload={"value": 1})

    assert result == {"step_id": "s1", "payload": {"value": 1}}
    assert adapter.calls == [("s1", {"value": 1})]


def test_step_runner_missing_adapter():
    runner = StepRunner()
    with pytest.raises(KeyError):
        runner.run(tool="unknown", step_id="s", payload={})


def test_local_func_adapter_executes_registered_callable():
    adapter = local_adapter_module.LocalFuncAdapter()

    def process(payload: dict) -> dict:
        return {"step_id": payload["step_id"], "double": payload["value"] * 2}

    adapter.register("double", process)

    result = adapter.run(step_id="local-1", payload={"function": "double", "step_id": "local-1", "value": 3})
    assert result == {"step_id": "local-1", "double": 6}


def test_local_func_adapter_missing_function_metadata():
    adapter = local_adapter_module.LocalFuncAdapter()
    with pytest.raises(KeyError):
        adapter.run(step_id="local", payload={})


def test_local_func_adapter_missing_registration():
    adapter = local_adapter_module.LocalFuncAdapter()
    with pytest.raises(KeyError):
        adapter.run(step_id="local", payload={"function": "nope"})
