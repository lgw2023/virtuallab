"""Adapter for executing local Python callables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict


@dataclass
class LocalFuncAdapter:
    """Adapter that executes registered local Python callables."""

    registry: Dict[str, Callable[[dict], dict]] = field(default_factory=dict)

    def register(self, name: str, func: Callable[[dict], dict]) -> None:
        """Register a callable under ``name``."""

        self.registry[name] = func

    def run(self, *, step_id: str, payload: dict) -> dict:
        """Execute the registered callable matching ``payload['function']``."""

        function_name = payload.get("function")
        if not function_name:
            raise KeyError("payload must include 'function'")
        if function_name not in self.registry:
            raise KeyError(f"No local function registered for '{function_name}'")
        func = self.registry[function_name]
        return func(payload)
