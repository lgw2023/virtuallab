"""Action routing for the VirtualLab API."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Protocol


class ActionHandler(Protocol):
    """Protocol representing a callable action handler."""

    def __call__(self, params: dict) -> dict:  # pragma: no cover - interface
        ...


@dataclass
class ActionRouter:
    """Dispatch actions to their registered handlers."""

    registry: Dict[str, ActionHandler] = field(default_factory=dict)

    def register(self, action: str, handler: ActionHandler) -> None:
        """Register ``handler`` under ``action``."""

        self.registry[action] = handler

    def dispatch(self, action: str, params: dict) -> dict:
        """Execute the handler associated with ``action``."""

        if action not in self.registry:
            raise KeyError(f"Unknown action: {action}")
        return self.registry[action](params)
