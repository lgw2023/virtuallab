"""Auto-linking rule definitions for VirtualLab."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .model import GraphDelta


class RuleContext(Protocol):
    """Protocol describing the information required by auto-link rules."""

    def iter_steps(self) -> Iterable[str]:
        """Return identifiers of step nodes within the current scope."""

    def relationships(self) -> Iterable[tuple[str, str]]:
        """Return existing directed relationships for deduplication."""


@dataclass
class AutoLinkRule:
    """Base class for auto-linking rules."""

    name: str

    def apply(self, context: RuleContext) -> GraphDelta:
        """Produce a :class:`GraphDelta` for the provided ``context``."""

        raise NotImplementedError


class TemporalLinkRule(AutoLinkRule):
    """Placeholder implementation for temporal linking logic."""

    def __init__(self) -> None:
        super().__init__(name="temporal")

    def apply(self, context: RuleContext) -> GraphDelta:  # pragma: no cover - placeholder
        return GraphDelta()


class LogicalLinkRule(AutoLinkRule):
    """Placeholder implementation for dependency detection."""

    def __init__(self) -> None:
        super().__init__(name="logic")

    def apply(self, context: RuleContext) -> GraphDelta:  # pragma: no cover - placeholder
        return GraphDelta()


class CausalLinkRule(AutoLinkRule):
    """Placeholder implementation for causal inference."""

    def __init__(self) -> None:
        super().__init__(name="causal")

    def apply(self, context: RuleContext) -> GraphDelta:  # pragma: no cover - placeholder
        return GraphDelta()


RULE_REGISTRY = {
    "temporal": TemporalLinkRule,
    "logic": LogicalLinkRule,
    "causal": CausalLinkRule,
}
