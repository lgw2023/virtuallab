"""Tests for :mod:`virtuallab.config`."""

from __future__ import annotations

from unittest import mock

from virtuallab import config


def test_get_env_returns_default_when_missing(monkeypatch):
    config._load_environment.cache_clear()  # reset cache to isolate tests
    monkeypatch.delenv("VIRTUALLAB_FAKE", raising=False)
    assert config.get_env("VIRTUALLAB_FAKE", default="fallback") == "fallback"


def test_get_env_ensures_environment_loaded(monkeypatch):
    config._load_environment.cache_clear()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    with mock.patch.object(config, "load_dotenv", autospec=True) as load_mock:
        load_mock.side_effect = lambda *args, **kwargs: calls.append((args, kwargs))
        monkeypatch.setenv("VIRTUALLAB_ENV_FLAG", "42")
        assert config.get_env("VIRTUALLAB_ENV_FLAG") == "42"

    # ``load_dotenv`` should have been invoked exactly once through the cached loader.
    assert len(calls) == 1
