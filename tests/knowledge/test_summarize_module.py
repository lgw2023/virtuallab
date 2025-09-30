"""Tests for :mod:`virtuallab.knowledge.summarize`."""

from __future__ import annotations

import asyncio

import pytest

from virtuallab.config import get_env
from virtuallab.knowledge.summarize import (
    OpenAILLMSummarizerAdapter,
    SummaryService,
    _ensure_text,
)


async def async_text_generator(chunks: list[str]):
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


def test_ensure_text_handles_async_iterator():
    async def run():
        return await _ensure_text(async_text_generator(["a", "b", "c"]))

    result = asyncio.run(run())
    assert result == "abc"


@pytest.fixture(scope="module")
def ensure_openai_env() -> None:
    if not get_env("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")
    if not get_env("OPENAI_API_MODEL"):
        pytest.skip("OPENAI_API_MODEL is not configured")


def test_openai_llm_summarizer_integration(ensure_openai_env: None):
    adapter = OpenAILLMSummarizerAdapter()

    summary = adapter.summarize(text="Important findings", style="bullet")
    print(f"\n@ test_openai_llm_summarizer_integration: {summary}")
    assert isinstance(summary, str)
    assert summary.strip()
    assert summary.strip() not in {
        "Important findings",
        "[bullet] Important findings",
    }


def test_summary_service_wraps_adapter(ensure_openai_env: None):
    adapter = OpenAILLMSummarizerAdapter()
    service = SummaryService(adapter=adapter)

    payload = service.summarize(text="content", style="concise")
    print(f"\n@ test_summary_service_wraps_adapter: {payload}")
    assert isinstance(payload["summary"], str)
    assert payload["summary"].strip()
    assert payload["summary"].strip() not in {"content", "[concise] content"}
    assert payload["style"] == "concise"
