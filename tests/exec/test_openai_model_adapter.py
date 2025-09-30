"""Integration tests for the OpenAI adapter helpers."""

from __future__ import annotations

import pytest

import virtuallab.exec.adapters.openai_model as openai_adapter
from virtuallab.config import get_env


@pytest.fixture
def anyio_backend() -> str:
    """Force the anyio plugin to use the asyncio backend only."""

    return "asyncio"


@pytest.fixture(scope="module")
def openai_config() -> dict[str, str]:
    """Ensure that required OpenAI configuration is present for the integration tests."""

    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is not configured")
    model = get_env("OPENAI_API_MODEL")
    if not model:
        pytest.skip("OPENAI_API_MODEL is not configured")
    base_url = get_env("OPENAI_API_URL")
    return {"model": model, "base_url": base_url} if base_url else {"model": model}


class DummyTokenTracker:
    """Simple token tracker stub that stores the usage payloads it receives."""

    def __init__(self) -> None:
        self.records: list[dict[str, int]] = []

    def add_usage(self, counts: dict[str, int]) -> None:
        self.records.append(counts)


@pytest.mark.anyio("asyncio")
async def test_openai_complete_returns_content(openai_config: dict[str, str]) -> None:
    """The completion helper should return a non-empty string from the live service."""

    response = await openai_adapter.openai_complete(
        prompt="Respond with a short confirmation for integration testing.",
        system_prompt="You are verifying connectivity for the VirtualLab test suite.",
    )

    assert isinstance(response, str)
    assert response.strip(), "Expected the LLM to return non-empty content"


@pytest.mark.anyio("asyncio")
async def test_openai_complete_streaming_yields_chunks(
    openai_config: dict[str, str]
) -> None:
    """Streaming completions should yield chunks and optionally track usage."""

    tracker = DummyTokenTracker()
    iterator = await openai_adapter.openai_complete_if_cache(
        openai_config["model"],
        prompt="Provide a two sentence update streamed token-by-token.",
        stream=True,
        token_tracker=tracker,
        base_url=openai_config.get("base_url"),
    )

    collected: list[str] = []
    async for chunk in iterator:
        collected.append(chunk)

    combined = "".join(collected).strip()
    assert combined, "Expected streamed response to contain content"
    if tracker.records:
        assert all("total_tokens" in record for record in tracker.records)


@pytest.mark.anyio("asyncio")
async def test_openai_complete_keyword_extraction_live(openai_config: dict[str, str]) -> None:
    """Keyword extraction flag should produce a response from the live endpoint."""

    prompt = (
        "Extract keywords for the following topics and respond in JSON:"
        " data pipelines, experiment tracking, anomaly detection."
    )
    response = await openai_adapter.openai_complete(
        prompt=prompt,
        system_prompt="Return machine readable JSON keyed by 'keywords'.",
        keyword_extraction=True,
        base_url=openai_config.get("base_url"),
    )

    assert isinstance(response, str)
    assert response.strip(), "Expected keyword extraction to yield content"


def test_locate_json_string_body_from_string():
    text = "prefix {\"key\": 5} suffix"
    assert (
        openai_adapter.locate_json_string_body_from_string(text)
        == "{\"key\": 5}"
    )
