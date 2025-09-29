"""OpenAI adapter and asynchronous client helpers.

This module combines the adapter abstraction used by the execution
runtime together with the asynchronous helper utilities that were
previously provided in :mod:`virtuallab.llm`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Protocol, Union

import httpx
import numpy as np
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ...config import get_env

if sys.version_info < (3, 9):  # pragma: no cover - Python version guard
    from typing import AsyncIterator
else:  # pragma: no cover - Python version guard
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

VERBOSE_DEBUG = get_env("VERBOSE", "false").lower() == "true"


def verbose_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log ``msg`` with optional truncation controlled via ``VERBOSE``.

    When ``VERBOSE`` is truthy the full message is logged. Otherwise the
    formatted message is truncated to the first 100 characters to avoid
    overwhelming the logs when dealing with large responses.
    """

    if VERBOSE_DEBUG:
        logger.info(msg, *args, **kwargs)
        return

    if args:
        formatted_msg = msg % args
    else:
        formatted_msg = msg
    truncated_msg = (
        formatted_msg[:100] + "..." if len(formatted_msg) > 100 else formatted_msg
    )
    logger.info(truncated_msg, **kwargs)


try:  # pragma: no cover - optional dependency management
    import pipmaster as pm  # type: ignore
except Exception:  # pragma: no cover - pipmaster may not be available
    pm = None  # type: ignore[assignment]


def _ensure_openai_package() -> None:
    """Ensure the ``openai`` package is importable."""

    try:
        import openai  # noqa: F401  # pylint: disable=unused-import
        return
    except ImportError:
        pass

    if pm is not None and hasattr(pm, "install"):
        try:
            if not pm.is_installed("openai"):
                pm.install("openai")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("pipmaster failed to install openai: %s", exc)

    try:
        import openai  # noqa: F401  # pylint: disable=unused-import
        return
    except ImportError:
        pass

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    except Exception as exc:  # pragma: no cover - installation failure
        raise ImportError("The 'openai' package is required but could not be installed.") from exc

    import openai  # noqa: F401  # pylint: disable=unused-import  # noqa: E402


_ensure_openai_package()

from openai import (  # type: ignore  # noqa: E402
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)


def wrap_embedding_func_with_attrs(*, embedding_dim: int, max_token_size: int):
    """Attach embedding metadata to the wrapped function."""

    def decorator(func):
        setattr(func, "embedding_dim", embedding_dim)
        setattr(func, "max_token_size", max_token_size)
        return func

    return decorator


def locate_json_string_body_from_string(text: str) -> str:
    """Extract a JSON object substring from ``text`` if present."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text

    candidate = text[start : end + 1]
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        return text


def safe_unicode_decode(data: bytes) -> str:
    """Decode unicode escape sequences while tolerating bad input."""

    try:
        return data.decode("unicode_escape")
    except Exception:
        return data.decode("utf-8", errors="ignore")


GPTKeywordExtractionFormat = {"type": "json_object"}


class InvalidResponseError(Exception):
    """Custom exception used to trigger tenacity retries."""


class OpenAIModelClient(Protocol):
    """Protocol describing an OpenAI-compatible chat client."""

    def chat(self, messages: list[dict], functions: list[dict] | None = None) -> dict:
        """Send a chat completion request."""


@dataclass
class OpenAIModelAdapter:
    """Adapter for interacting with an OpenAI-compatible chat endpoint."""

    client: OpenAIModelClient

    def run(self, *, step_id: str, payload: dict) -> dict:  # pragma: no cover - passthrough
        messages = payload.get("messages", [])
        functions = payload.get("functions")
        response = self.client.chat(messages, functions=functions)
        return {"step_id": step_id, "response": response}


def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] | None = None,
) -> AsyncOpenAI:
    """Create an :class:`AsyncOpenAI` client using explicit or environment configuration."""

    if not api_key:
        api_key = get_env("OPENAI_API_KEY")

    if client_configs is None:
        client_configs = {}

    proxy_port = get_env("http_proxy_port")
    merged_configs: dict[str, Any] = {
        **client_configs,
        "api_key": api_key,
        "http_client": httpx.AsyncClient(proxy=proxy_port, verify=False)
        if proxy_port
        else None,
    }

    if base_url is not None:
        merged_configs["base_url"] = base_url
    else:
        merged_configs["base_url"] = get_env(
            "OPENAI_API_URL", "https://api.openai.com/v1"
        )

    return AsyncOpenAI(**merged_configs)


@retry(
    stop=stop_after_attempt(50),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InvalidResponseError)
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def openai_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Execute a chat completion request with optional streaming support."""

    if history_messages is None:
        history_messages = []

    if not model:
        model = get_env("OPENAI_API_MODEL")
    if not model:
        raise ValueError(f"OPENAI_API_MODEL is not set or is invalid: {model}")

    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)

    client_configs = kwargs.pop("openai_client_configs", {})
    openai_async_client = create_openai_async_client(
        api_key=api_key or get_env("OPENAI_API_KEY"),
        base_url=base_url or get_env("OPENAI_API_URL"),
        client_configs=client_configs,
    )

    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    messages = kwargs.pop("messages", messages)

    logger.debug(
        "messages:\n%s",
        "\n".join(f"{key}: {value}" for m in messages for key, value in m.items()),
    )

    try:
        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
    except APIConnectionError as exc:
        logger.error("OpenAI API Connection Error: %s", exc)
        logger.error("model=%s, messages=%s, %s", model, messages, kwargs)
        await openai_async_client.close()
        raise
    except RateLimitError as exc:
        logger.error("OpenAI API Rate Limit Error: %s", exc)
        logger.error("model=%s, messages=%s, %s", model, messages, kwargs)
        await openai_async_client.close()
        raise
    except APITimeoutError as exc:
        logger.error("OpenAI API Timeout Error: %s", exc)
        logger.error("model=%s, messages=%s, %s", model, messages, kwargs)
        await openai_async_client.close()
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("OpenAI API Call Failed, Got: %s", exc)
        logger.error("model=%s, messages=%s, %s", model, messages, kwargs)
        await openai_async_client.close()
        raise

    if hasattr(response, "__aiter__"):

        async def inner() -> AsyncIterator[str]:
            iteration_started = False
            final_chunk_usage = None

            try:
                iteration_started = True
                async for chunk in response:  # type: ignore[assignment]
                    if hasattr(chunk, "usage") and chunk.usage:
                        final_chunk_usage = chunk.usage
                        logger.info("Received usage info in streaming chunk: %s", chunk.usage)

                    if not hasattr(chunk, "choices") or not chunk.choices:
                        logger.warning("Received chunk without choices: %s", chunk)
                        continue

                    if not hasattr(chunk.choices[0], "delta") or not hasattr(
                        chunk.choices[0].delta, "content"
                    ):
                        continue

                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    if "\\u" in content:
                        content = safe_unicode_decode(content.encode("utf-8"))

                    yield content

                if token_tracker and final_chunk_usage:
                    token_counts = {
                        "prompt_tokens": getattr(final_chunk_usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            final_chunk_usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
                    }
                    token_tracker.add_usage(token_counts)
                    logger.info(
                        "Streaming token usage (from API): %s",
                        token_counts,
                    )
                elif token_tracker:
                    logger.info("No usage information available in streaming response")
            except Exception as exc:  # pragma: no cover - streaming cleanup
                logger.error("Error in stream response: %s", exc)
                if (
                    iteration_started
                    and hasattr(response, "aclose")
                    and callable(getattr(response, "aclose", None))
                ):
                    try:
                        await response.aclose()
                        logger.info("Successfully closed stream response after error")
                    except Exception as close_error:  # pragma: no cover - cleanup warning
                        logger.warning(
                            "Failed to close stream response: %s",
                            close_error,
                        )
                await openai_async_client.close()
                raise
            finally:
                if (
                    iteration_started
                    and hasattr(response, "aclose")
                    and callable(getattr(response, "aclose", None))
                ):
                    try:
                        await response.aclose()
                        logger.info("Successfully closed stream response")
                    except Exception as close_error:  # pragma: no cover - cleanup warning
                        logger.warning(
                            "Failed to close stream response in finally block: %s",
                            close_error,
                        )

                try:
                    await openai_async_client.close()
                    logger.info("Successfully closed OpenAI client for streaming response")
                except Exception as client_close_error:  # pragma: no cover - cleanup warning
                    logger.warning(
                        "Failed to close OpenAI client in streaming finally block: %s",
                        client_close_error,
                    )

        return inner()

    try:
        if (
            not response
            or not response.choices
            or not hasattr(response.choices[0], "message")
            or not hasattr(response.choices[0].message, "content")
        ):
            logger.error("Invalid response from OpenAI API")
            await openai_async_client.close()
            raise InvalidResponseError("Invalid response from OpenAI API")

        content = response.choices[0].message.content

        if not content or content.strip() == "":
            logger.error("Received empty content from OpenAI API")
            await openai_async_client.close()
            raise InvalidResponseError("Received empty content from OpenAI API")

        if "\\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))

        if token_tracker and hasattr(response, "usage"):
            token_counts = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
            token_tracker.add_usage(token_counts)

        logger.info("Response content len: %s", len(content))
        verbose_debug("Response: %s", response)

        return content
    finally:
        await openai_async_client.close()


async def openai_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool | None = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Proxy to :func:`openai_complete_if_cache` using the default model."""

    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = get_env("OPENAI_API_MODEL")
    if not model_name:
        raise ValueError(
            f"llm_model_name is not set or is invalid: {model_name}\n{kwargs}"
        )
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool | None = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Completion helper targeting the ``gpt-4o`` model."""

    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool | None = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Completion helper targeting the ``gpt-4o-mini`` model."""

    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = GPTKeywordExtractionFormat
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def nvidia_openai_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool | None = False,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Completion helper targeting NVIDIA's hosted OpenAI-compatible models."""

    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    if keyword_extraction:
        return locate_json_string_body_from_string(result)  # type: ignore[arg-type]
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
    client_configs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Generate embeddings with retries using the OpenAI embeddings endpoint."""

    min_delay = 4
    max_delay = 60
    multiplier = 1
    max_attempts = 50

    openai_async_client = create_openai_async_client(
        api_key=api_key or get_env("EMBEDDING_BINDING_API_KEY"),
        base_url=base_url or get_env("EMBEDDING_BINDING_URL"),
        client_configs=client_configs,
    )

    expected_dim = None
    try:
        expected_dim_env = get_env("EMBEDDING_DIM")
        if expected_dim_env:
            expected_dim = int(expected_dim_env)
    except Exception:
        expected_dim = None

    def _is_response_ok(resp: Any) -> bool:
        try:
            if resp is None or not hasattr(resp, "data"):
                return False
            data = resp.data
            if not isinstance(data, list) or len(data) != len(texts):
                return False
            for dp in data:
                emb = getattr(dp, "embedding", None)
                if emb is None or not hasattr(emb, "__len__") or len(emb) == 0:
                    return False
                if expected_dim is not None and len(emb) != expected_dim:
                    return False
            return True
        except Exception:
            return False

    last_error: Exception | None = None
    async with openai_async_client:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await openai_async_client.embeddings.create(
                    model=get_env("EMBEDDING_MODEL") or model,
                    input=texts,
                    encoding_format="float",
                )

                if _is_response_ok(response):
                    return np.array([dp.embedding for dp in response.data])

                last_error = RuntimeError(
                    "Abnormal embedding response: count/shape mismatch."
                )
                raise last_error

            except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                last_error = exc
            except Exception as exc:  # pragma: no cover - defensive logging
                last_error = exc

            if attempt < max_attempts:
                delay = multiplier * (2 ** (attempt - 1))
                delay = max(min_delay, min(delay, max_delay))
                logger.warning(
                    "[Embedding Attempt %s/%s] failed: %s. Retrying in %ss...",
                    attempt,
                    max_attempts,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "[Embedding Attempt %s/%s] failed: %s. Giving up.",
                    attempt,
                    max_attempts,
                    last_error,
                )
                raise last_error
