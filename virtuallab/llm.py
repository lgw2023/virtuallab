"""Compatibility layer for OpenAI helpers.

All OpenAI-related helpers now live in
:mod:`virtuallab.exec.adapters.openai_model`.  This module re-exports the
public API to maintain backwards compatibility with existing imports.
"""

from .exec.adapters.openai_model import (
    GPTKeywordExtractionFormat,
    InvalidResponseError,
    VERBOSE_DEBUG,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    locate_json_string_body_from_string,
    nvidia_openai_complete,
    openai_complete,
    openai_complete_if_cache,
    openai_embed,
    safe_unicode_decode,
    verbose_debug,
    wrap_embedding_func_with_attrs,
    create_openai_async_client,
)

__all__ = [
    "GPTKeywordExtractionFormat",
    "InvalidResponseError",
    "VERBOSE_DEBUG",
    "create_openai_async_client",
    "gpt_4o_complete",
    "gpt_4o_mini_complete",
    "locate_json_string_body_from_string",
    "nvidia_openai_complete",
    "openai_complete",
    "openai_complete_if_cache",
    "openai_embed",
    "safe_unicode_decode",
    "verbose_debug",
    "wrap_embedding_func_with_attrs",
]
