"""Tests for persistence helpers."""

from __future__ import annotations

import networkx as nx
import pytest

from virtuallab.persist.snapshot import SnapshotManager
from virtuallab.persist.export import GraphExporter


def test_snapshot_manager_roundtrip():
    manager = SnapshotManager()
    graph = nx.MultiDiGraph()
    graph.add_node("n1", type="Plan")
    manager.snapshot(graph)

    restored = manager.rollback()
    assert list(restored.nodes) == ["n1"]

    with pytest.raises(RuntimeError):
        manager.rollback()


def test_graph_exporter_rejects_unknown_format():
    exporter = GraphExporter(graph=nx.MultiDiGraph())
    with pytest.raises(ValueError):
        exporter.export(format="unsupported")
