"""Tests for :mod:`virtuallab.config`."""

from __future__ import annotations

from pathlib import Path

from virtuallab import config


PROJECT_ROOT = Path(config.__file__).resolve().parents[1]


def test_get_env_reads_from_project_dotenv(monkeypatch):
    """The helper should pull values from the real project ``.env`` file."""

    config._load_environment.cache_clear()
    monkeypatch.delenv("LLM_MODEL", raising=False)

    value = config.get_env("LLM_MODEL")

    # ``LLM_MODEL`` is defined in the repository level ``.env`` file.
    assert value == "qwen3-max" or value == "deepseek-ai/DeepSeek-V3.1-Terminus"


def test_get_env_prefers_process_environment(monkeypatch):
    """Explicit environment variables should win over the file contents."""

    config._load_environment.cache_clear()
    monkeypatch.setenv("LLM_MODEL", "in-memory")

    assert config.get_env("LLM_MODEL") == "in-memory"


def test_get_env_can_reload_after_cache_clear(monkeypatch):
    """Clearing the cache allows the loader to pick up updated ``.env`` values."""

    env_file = PROJECT_ROOT / ".env"
    original_contents = env_file.read_text()
    key = "VIRTUALLAB_TEST_TEMP"

    try:
        # Append a temporary setting and verify it is discovered.
        env_file.write_text(f"{original_contents}\n{key}=first\n")
        config._load_environment.cache_clear()
        monkeypatch.delenv(key, raising=False)
        assert config.get_env(key) == "first"

        # Update the value and show that cache clearing is required to see it.
        env_file.write_text(f"{original_contents}\n{key}=second\n")
        monkeypatch.delenv(key, raising=False)
        assert config.get_env(key) is None

        config._load_environment.cache_clear()
        monkeypatch.delenv(key, raising=False)
        assert config.get_env(key) == "second"
    finally:
        env_file.write_text(original_contents)
        monkeypatch.delenv(key, raising=False)
        config._load_environment.cache_clear()


def test_get_env_returns_default_when_missing(monkeypatch):
    """Missing keys should fall back to the provided default value."""

    config._load_environment.cache_clear()
    monkeypatch.delenv("VIRTUALLAB_DOES_NOT_EXIST", raising=False)

    assert config.get_env("VIRTUALLAB_DOES_NOT_EXIST", default="fallback") == "fallback"
