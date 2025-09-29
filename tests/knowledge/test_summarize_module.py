"""Tests for :mod:`virtuallab.knowledge.summarize`."""

from __future__ import annotations

import asyncio

import pytest

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


def test_openai_llm_summarizer_builds_prompt_and_invokes_completion(monkeypatch):
    calls: list[dict[str, object]] = []

    async def completion(prompt: str, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        return "summary"

    adapter = OpenAILLMSummarizerAdapter(completion_func=completion)
    summary = adapter.summarize(text="Important findings", style="bullet")

    assert "Important findings" in calls[0]["prompt"]
    assert "bullet" in calls[0]["prompt"]
    assert summary == "summary"


def test_summary_service_wraps_adapter():
    class StubAdapter:
        def summarize(self, *, text: str, style: str | None = None) -> str:
            return f"processed:{text}:{style}"

    service = SummaryService(adapter=StubAdapter())
    payload = service.summarize(text="content", style=None)
    assert payload == {"summary": "processed:content:None", "style": None}
