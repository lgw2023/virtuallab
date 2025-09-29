"""Adapter for the Engineer autonomous agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EngineerClient(Protocol):
    """Minimal protocol describing the Engineer agent interface."""

    def run(self, prompt: str, *, tools: list[str] | None = None) -> str:
        """Execute ``prompt`` using the Engineer agent."""


@dataclass
class EngineerAdapter:
    """Adapter that bridges :class:`StepRunner` with the Engineer agent."""

    client: EngineerClient

    def run(self, *, step_id: str, payload: dict) -> dict:  # pragma: no cover - placeholder
        """Execute the step using the Engineer agent."""

        prompt: str = payload.get("text", "")
        tools = payload.get("tools") or []
        output = self.client.run(prompt, tools=tools)
        return {"step_id": step_id, "output": output}
