"""Tests for the OpenAI adapter helpers using the real AsyncOpenAI client."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

import virtuallab.exec.adapters.openai_model as openai_adapter


@pytest.fixture
def anyio_backend() -> str:
    """Force the anyio plugin to use the asyncio backend only."""

    return "asyncio"


class DummyTokenTracker:
    """Simple token tracker stub that stores the usage payloads it receives."""

    def __init__(self) -> None:
        self.records: list[dict[str, int]] = []

    def add_usage(self, counts: dict[str, int]) -> None:
        self.records.append(counts)


@pytest.mark.anyio("asyncio")
async def test_openai_complete_if_cache_sends_messages_and_tracks_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    captured_request: dict[str, Any] = {}
    tracker = DummyTokenTracker()

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content.decode())
        response_payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "model-id",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello\\u4f60"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }
        return httpx.Response(200, json=response_payload)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def fake_create_openai_async_client(*, api_key=None, base_url=None, client_configs=None):
        return openai_adapter.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

    monkeypatch.setattr(
        openai_adapter, "create_openai_async_client", fake_create_openai_async_client
    )

    result = await openai_adapter.openai_complete_if_cache(
        model="model-id",
        prompt="user prompt",
        system_prompt="system message",
        history_messages=[{"role": "assistant", "content": "prev"}],
        token_tracker=tracker,
        base_url="https://mock.api/v1",
    )

    assert result == "hello你"
    assert captured_request["url"] == "https://mock.api/v1/chat/completions"
    assert captured_request["body"]["model"] == "model-id"
    assert captured_request["body"]["messages"] == [
        {"role": "system", "content": "system message"},
        {"role": "assistant", "content": "prev"},
        {"role": "user", "content": "user prompt"},
    ]
    assert tracker.records == [
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    ]
    assert http_client.is_closed


@pytest.mark.anyio("asyncio")
async def test_openai_complete_if_cache_streaming_tracks_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    tracker = DummyTokenTracker()
    captured_request: dict[str, Any] = {}

    chunks = [
        {
            "id": "chunk-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "model-id",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "A"}},
            ],
        },
        {
            "id": "chunk-2",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "model-id",
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "B"}},
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content.decode())
        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        payload += "data: [DONE]\n\n"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload.encode("utf-8"),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def fake_create_openai_async_client(*, api_key=None, base_url=None, client_configs=None):
        return openai_adapter.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

    monkeypatch.setattr(
        openai_adapter, "create_openai_async_client", fake_create_openai_async_client
    )

    iterator = await openai_adapter.openai_complete_if_cache(
        model="model-id",
        prompt="stream",
        stream=True,
        token_tracker=tracker,
        base_url="https://mock.api/v1",
    )

    collected = []
    async for chunk in iterator:
        collected.append(chunk)

    assert collected == ["A", "B"]
    assert tracker.records == [
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    ]
    assert captured_request["body"]["stream"] is True
    assert http_client.is_closed


@pytest.mark.anyio("asyncio")
async def test_openai_complete_keyword_extraction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    captured_request: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_request["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "created": 0,
                "model": "env-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def fake_create_openai_async_client(*, api_key=None, base_url=None, client_configs=None):
        return openai_adapter.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

    monkeypatch.setattr(
        openai_adapter, "create_openai_async_client", fake_create_openai_async_client
    )

    result = await openai_adapter.openai_complete(
        prompt="p",
        system_prompt="s",
        history_messages=[{"role": "user", "content": "hi"}],
        keyword_extraction=True,
        base_url="https://mock.api/v1",
    )

    assert result == "ok"
    assert "response_format" not in captured_request["body"]
    assert captured_request["body"]["messages"][0] == {
        "role": "system",
        "content": "s",
    }


@pytest.mark.anyio("asyncio")
async def test_nvidia_openai_complete_keyword_extraction(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    captured_request: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_request["url"] = str(request.url)
        captured_request["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-3",
                "object": "chat.completion",
                "created": 0,
                "model": "nvidia/llama-3.1-nemotron-70b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "noise {\"answer\": 1} trailing",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def fake_create_openai_async_client(*, api_key=None, base_url=None, client_configs=None):
        return openai_adapter.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )

    monkeypatch.setattr(
        openai_adapter, "create_openai_async_client", fake_create_openai_async_client
    )

    result = await openai_adapter.nvidia_openai_complete(
        prompt="ask",
        keyword_extraction=True,
    )

    assert result == "noise {\"answer\": 1} trailing"
    assert captured_request["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert captured_request["body"]["model"] == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_locate_json_string_body_from_string():
    text = "prefix {\"key\": 5} suffix"
    assert (
        openai_adapter.locate_json_string_body_from_string(text)
        == "{\"key\": 5}"
    )
