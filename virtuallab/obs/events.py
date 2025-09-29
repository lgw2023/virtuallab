"""Event bus primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from virtuallab.graph.ids import utc_now


@dataclass
class Event:
    """Simple event structure stored in the event bus."""

    ts: str
    level: str
    msg: str
    action: str | None = None
    actor: str | None = None
    target_ids: List[str] = field(default_factory=list)
    extras: dict | None = None


@dataclass
class EventBus:
    """Append-only in-memory event bus."""

    events: List[Event] = field(default_factory=list)

    def emit(
        self,
        *,
        level: str,
        msg: str,
        action: str | None = None,
        actor: str | None = None,
        target_ids: Iterable[str] | None = None,
        extras: dict | None = None,
    ) -> Event:
        """Create and store a new :class:`Event`."""

        event = Event(
            ts=utc_now(),
            level=level,
            msg=msg,
            action=action,
            actor=actor,
            target_ids=list(target_ids or []),
            extras=extras,
        )
        self.events.append(event)
        return event

    def history(self) -> Iterable[Event]:
        """Return the chronological event history."""

        return tuple(self.events)
