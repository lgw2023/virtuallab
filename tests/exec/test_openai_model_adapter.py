"""Tests for the OpenAI adapter helpers."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, List

import pytest


def _ensure_httpx_stub() -> None:
    if "httpx" in sys.modules:
        return

    httpx_module = ModuleType("httpx")

    class _AsyncClient:  # pragma: no cover - trivial helper
        def __init__(self, **_: Any) -> None:
            pass

    httpx_module.AsyncClient = _AsyncClient  # type: ignore[attr-defined]
    sys.modules["httpx"] = httpx_module


def _ensure_numpy_stub() -> None:
    if "numpy" in sys.modules:
        return

    numpy_module = ModuleType("numpy")

    def _array(values: Any) -> Any:  # pragma: no cover - trivial helper
        return values

    numpy_module.array = _array  # type: ignore[attr-defined]
    sys.modules["numpy"] = numpy_module


def _ensure_tenacity_stub() -> None:
    if "tenacity" in sys.modules:
        return

    tenacity_module = ModuleType("tenacity")

    class _RetryCondition:  # pragma: no cover - trivial helper
        def __or__(self, other: Any) -> "_RetryCondition":
            return self

    def _retry(*args: Any, **kwargs: Any):  # pragma: no cover - trivial helper
        def _wrapper(func: Any) -> Any:
            return func

        return _wrapper

    def _stop_after_attempt(*_: Any, **__: Any) -> Any:  # pragma: no cover - trivial helper
        return None

    def _wait_exponential(*_: Any, **__: Any) -> Any:  # pragma: no cover - trivial helper
        return None

    def _retry_if_exception_type(*_: Any, **__: Any) -> _RetryCondition:  # pragma: no cover
        return _RetryCondition()

    def _before_sleep_log(*_: Any, **__: Any) -> Any:  # pragma: no cover - helper
        return None

    tenacity_module.retry = _retry  # type: ignore[attr-defined]
    tenacity_module.stop_after_attempt = _stop_after_attempt  # type: ignore[attr-defined]
    tenacity_module.wait_exponential = _wait_exponential  # type: ignore[attr-defined]
    tenacity_module.retry_if_exception_type = _retry_if_exception_type  # type: ignore[attr-defined]
    tenacity_module.before_sleep_log = _before_sleep_log  # type: ignore[attr-defined]
    sys.modules["tenacity"] = tenacity_module


def _ensure_openai_stub() -> None:
    if "openai" in sys.modules:
        return

    openai_module = ModuleType("openai")

    class _BaseError(Exception):
        pass

    class APIConnectionError(_BaseError):
        pass

    class APITimeoutError(_BaseError):
        pass

    class RateLimitError(_BaseError):
        pass

    class _EmbeddingsNamespace:
        async def create(self, **_: Any) -> Any:  # pragma: no cover - unused helper
            return SimpleNamespace(data=[])

    class AsyncOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))
            self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=None)))
            self.embeddings = _EmbeddingsNamespace()

        async def __aenter__(self) -> "AsyncOpenAI":  # pragma: no cover - unused
            return self

        async def __aexit__(self, *exc_info: Any) -> None:  # pragma: no cover - unused
            return None

        async def close(self) -> None:
            return None

    openai_module.APIConnectionError = APIConnectionError
    openai_module.APITimeoutError = APITimeoutError
    openai_module.AsyncOpenAI = AsyncOpenAI
    openai_module.RateLimitError = RateLimitError
    sys.modules["openai"] = openai_module


_ensure_httpx_stub()
_ensure_numpy_stub()
_ensure_tenacity_stub()
_ensure_openai_stub()

import virtuallab.exec.adapters.openai_model as openai_adapter


@pytest.fixture
def anyio_backend() -> str:
    """Force the anyio plugin to use the asyncio backend only."""

    return "asyncio"


class DummyTokenTracker:
    """Simple token tracker stub that stores the usage payloads it receives."""

    def __init__(self) -> None:
        self.records: List[dict[str, int]] = []

    def add_usage(self, counts: dict[str, int]) -> None:
        self.records.append(counts)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)


class _FakeStreamingResponse:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.aclose_called = False

    def __aiter__(self):  # pragma: no cover - exercised indirectly
        async def _iterate():
            for chunk in self._chunks:
                await asyncio.sleep(0)
                yield chunk

        return _iterate()

    async def aclose(self) -> None:
        self.aclose_called = True


@pytest.mark.anyio("asyncio")
async def test_openai_complete_if_cache_returns_content(monkeypatch):
    monkeypatch.setenv("OPENAI_API_MODEL", "test-model")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_API_URL", "https://example.com")

    recorded_kwargs: dict[str, Any] = {}

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> _FakeResponse:
            nonlocal recorded_kwargs
            recorded_kwargs = kwargs
            return _FakeResponse("hello\\u4f60")

    class FakeClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())
            self.beta = SimpleNamespace(chat=SimpleNamespace(completions=None))
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    fake_client = FakeClient()
    monkeypatch.setattr(
        openai_adapter,
        "create_openai_async_client",
        lambda **_: fake_client,
    )

    result = await openai_adapter.openai_complete_if_cache(
        model="model-id",
        prompt="user prompt",
        system_prompt="system message",
        history_messages=[{"role": "assistant", "content": "prev"}],
    )

    assert result == "hello你"
    assert recorded_kwargs["model"] == "model-id"
    assert recorded_kwargs["messages"][0] == {"role": "system", "content": "system message"}
    assert fake_client.closed is True


@pytest.mark.anyio("asyncio")
async def test_openai_complete_if_cache_streaming_tracks_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_MODEL", "stream-model")
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    tracker = DummyTokenTracker()

    chunks = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="A"))],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="B"))],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12),
        ),
    ]

    streaming_response = _FakeStreamingResponse(chunks)

    class FakeCompletions:
        async def create(self, **_: Any) -> _FakeStreamingResponse:
            return streaming_response

    class FakeClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())
            self.beta = SimpleNamespace(chat=SimpleNamespace(completions=None))
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    fake_client = FakeClient()
    monkeypatch.setattr(
        openai_adapter,
        "create_openai_async_client",
        lambda **_: fake_client,
    )

    iterator = await openai_adapter.openai_complete_if_cache(
        model="stream-model",
        prompt="stream",
        token_tracker=tracker,
    )

    assert hasattr(iterator, "__aiter__")

    collected = []
    async for chunk in iterator:
        collected.append(chunk)

    assert collected == ["A", "B"]
    assert tracker.records == [
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    ]
    assert streaming_response.aclose_called is True
    assert fake_client.closed is True


@pytest.mark.anyio("asyncio")
async def test_openai_complete_passes_keyword_extraction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_MODEL", "env-model")

    captured: dict[str, Any] = {}

    async def fake_openai_complete_if_cache(model: str, prompt: str, **kwargs: Any) -> str:
        captured["model"] = model
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(
        openai_adapter,
        "openai_complete_if_cache",
        fake_openai_complete_if_cache,
    )

    result = await openai_adapter.openai_complete(
        prompt="p",
        system_prompt="s",
        history_messages=[{"role": "user", "content": "hi"}],
        keyword_extraction=True,
    )

    assert result == "ok"
    assert captured["model"] == "env-model"
    assert captured["kwargs"]["system_prompt"] == "s"
    assert captured["kwargs"]["history_messages"] == [{"role": "user", "content": "hi"}]
    assert "keyword_extraction" not in captured["kwargs"]
    assert "response_format" not in captured["kwargs"]


@pytest.mark.anyio("asyncio")
async def test_nvidia_openai_complete_keyword_extraction(monkeypatch):
    async def fake_openai_complete_if_cache(model: str, prompt: str, **_: Any) -> str:
        assert model == "nvidia/llama-3.1-nemotron-70b-instruct"
        assert prompt == "ask"
        return "noise {\"answer\": 1} trailing"

    monkeypatch.setattr(
        openai_adapter,
        "openai_complete_if_cache",
        fake_openai_complete_if_cache,
    )

    result = await openai_adapter.nvidia_openai_complete(
        prompt="ask",
        keyword_extraction=True,
    )

    assert result == "noise {\"answer\": 1} trailing"


def test_locate_json_string_body_from_string():
    text = "prefix {\"key\": 5} suffix"
    assert (
        openai_adapter.locate_json_string_body_from_string(text)
        == "{\"key\": 5}"
    )
