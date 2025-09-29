"""Adapter for the OpenAIServerModel endpoint."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class OpenAIModelClient(Protocol):
    """Protocol describing the OpenAI server model client."""

    def chat(self, messages: list[dict], functions: list[dict] | None = None) -> dict:
        """Send a chat completion request."""


@dataclass
class OpenAIModelAdapter:
    """Adapter for interacting with the OpenAIServerModel."""

    client: OpenAIModelClient

    def run(self, *, step_id: str, payload: dict) -> dict:  # pragma: no cover - placeholder
        messages = payload.get("messages", [])
        functions = payload.get("functions")
        response = self.client.chat(messages, functions=functions)
        return {"step_id": step_id, "response": response}
