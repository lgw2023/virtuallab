"""Summarization utilities for VirtualLab."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SummarizerAdapter(Protocol):
    """Protocol representing an external summarization capability."""

    def summarize(self, *, text: str, style: str | None = None) -> str:
        """Summarize ``text`` according to ``style``."""


@dataclass
class SummaryService:
    """High-level interface coordinating summarization workflows."""

    adapter: SummarizerAdapter

    def summarize(self, *, text: str, style: str | None = None) -> dict:
        """Produce a structured summary result."""

        summary = self.adapter.summarize(text=text, style=style)
        return {"summary": summary, "style": style}
