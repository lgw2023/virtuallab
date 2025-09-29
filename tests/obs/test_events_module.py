"""Tests for :mod:`virtuallab.obs.events`."""

from __future__ import annotations

from virtuallab.obs.events import EventBus


def test_event_bus_emit_and_history(monkeypatch):
    bus = EventBus()
    event = bus.emit(level="info", msg="Test", action="act", target_ids=["node"], extras={"detail": 1})

    assert event.msg == "Test"
    history = list(bus.history())
    assert history == [event]
