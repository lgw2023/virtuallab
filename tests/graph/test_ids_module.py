"""Tests for :mod:`virtuallab.graph.ids`."""

from __future__ import annotations

from virtuallab.graph import ids


def test_new_id_prefix_and_uniqueness():
    identifier = ids.new_id("plan")
    assert identifier.startswith("plan_")
    assert identifier != ids.new_id("plan")


def test_utc_now_returns_iso_format():
    timestamp = ids.utc_now()
    assert "T" in timestamp and timestamp.endswith("Z") is False
