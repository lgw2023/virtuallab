"""Tests for :mod:`virtuallab.router`."""

from __future__ import annotations

import pytest

from virtuallab.router import ActionRouter


def test_router_dispatches_registered_handler():
    router = ActionRouter()

    def handler(params: dict) -> dict:
        return {"result": params["value"] * 2}

    router.register("double", handler)
    assert router.dispatch("double", {"value": 21}) == {"result": 42}


def test_router_dispatch_missing_action():
    router = ActionRouter()
    with pytest.raises(KeyError):
        router.dispatch("unknown", {})
