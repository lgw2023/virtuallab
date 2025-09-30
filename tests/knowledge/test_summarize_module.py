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


@pytest.fixture()
def recorded_completion(monkeypatch):
    calls: list[dict[str, object]] = []

    async def completion(prompt: str, *, system_prompt: str, **kwargs):
        calls.append({"prompt": prompt, "system_prompt": system_prompt, "kwargs": kwargs})
        return "summary"

    monkeypatch.setattr(
        "virtuallab.exec.adapters.openai_model.gpt_4o_mini_complete", completion
    )
    return calls


def test_openai_llm_summarizer_builds_prompt_and_invokes_completion(recorded_completion):
    adapter = OpenAILLMSummarizerAdapter()

    summary = adapter.summarize(text="Important findings", style="bullet")

    assert summary == "summary"
    assert len(recorded_completion) == 1
    prompt = recorded_completion[0]["prompt"]
    assert "Important findings" in prompt
    assert "bullet" in prompt
    assert recorded_completion[0]["system_prompt"].startswith("You are an expert")


def test_summary_service_wraps_adapter(recorded_completion):
    adapter = OpenAILLMSummarizerAdapter()
    service = SummaryService(adapter=adapter)

    payload = service.summarize(text="content", style="concise")

    assert payload == {"summary": "summary", "style": "concise"}
