"""Step execution orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol


class StepAdapter(Protocol):
    """Protocol for execution adapters used by :class:`StepRunner`."""

    def run(self, *, step_id: str, payload: dict) -> dict:
        """Execute the step identified by ``step_id`` and return raw results."""


@dataclass
class StepRunner:
    """Coordinates execution requests across registered adapters."""

    adapters: Dict[str, StepAdapter] = field(default_factory=dict)

    def register_adapter(self, name: str, adapter: StepAdapter) -> None:
        """Register an adapter under ``name``."""

        self.adapters[name] = adapter

    def run(self, *, tool: str, step_id: str, payload: dict) -> dict:
        """Dispatch execution to the adapter associated with ``tool``."""

        if tool not in self.adapters:
            raise KeyError(f"No adapter registered for tool '{tool}'")
        return self.adapters[tool].run(step_id=step_id, payload=payload)
