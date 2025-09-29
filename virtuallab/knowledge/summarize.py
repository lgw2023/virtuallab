"""Summarization utilities for VirtualLab."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol


class SummarizerAdapter(Protocol):
    """Protocol representing an external summarization capability."""

    def summarize(self, *, text: str, style: str | None = None) -> str:
        """Summarize ``text`` according to ``style``."""


CompletionFn = Callable[..., Awaitable[str | AsyncIterator[str]]]


def _run_sync(coro: Awaitable[str]) -> str:
    """Execute ``coro`` synchronously, creating an event loop if required."""

    try:
        return asyncio.run(coro)
    except RuntimeError as exc:  # pragma: no cover - running loop edge case
        if "asyncio.run() cannot be called" in str(exc):
            raise RuntimeError(
                "OpenAILLMSummarizerAdapter cannot execute because an event loop is already running. "
                "Provide a pre-executed summary instead."
            ) from exc
        raise


async def _ensure_text(result: str | AsyncIterator[str]) -> str:
    """Normalise OpenAI streaming responses to a plain string."""

    if isinstance(result, str):
        return result

    chunks: list[str] = []
    async for chunk in result:
        chunks.append(chunk)
    return "".join(chunks)


@dataclass
class OpenAILLMSummarizerAdapter:
    """Adapter that delegates summarisation to an OpenAI-compatible model."""

    completion_func: CompletionFn | None = None
    system_prompt: str = (
        "You are an expert research assistant who produces concise, structured summaries."
    )
    completion_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.completion_func is None:
            from virtuallab.exec.adapters.openai_model import gpt_4o_mini_complete

            self.completion_func = gpt_4o_mini_complete

    def summarize(self, *, text: str, style: str | None = None) -> str:
        prompt = self._build_prompt(text=text, style=style)

        async def _invoke() -> str:
            if self.completion_func is None:  # pragma: no cover - defensive guard
                raise RuntimeError("LLM completion function is not configured")
            raw_result = await self.completion_func(
                prompt,
                system_prompt=self.system_prompt,
                **self.completion_kwargs,
            )
            return await _ensure_text(raw_result)

        return _run_sync(_invoke())

    def _build_prompt(self, *, text: str, style: str | None) -> str:
        instructions = [
            "Summarize the following content for archival in the VirtualLab knowledge base.",
        ]
        if style:
            instructions.append(f"Format the summary using the '{style}' style when possible.")
        instructions.extend(
            [
                "Focus on the key findings, decisions, and recommended next steps.",
                "Text:",
                text,
                "Summary:",
            ]
        )
        return "\n".join(instructions)


@dataclass
class SummaryService:
    """High-level interface coordinating summarization workflows."""

    adapter: SummarizerAdapter

    def summarize(self, *, text: str, style: str | None = None) -> dict:
        """Produce a structured summary result."""

        summary = self.adapter.summarize(text=text, style=style)
        return {"summary": summary, "style": style}
