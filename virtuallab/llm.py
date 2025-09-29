import sys
import httpx
import asyncio
import os
import logging

logger = logging.getLogger()

VERBOSE_DEBUG = os.getenv("VERBOSE", "false").lower() == "true"

def verbose_debug(msg: str, *args, **kwargs):
    """Function for outputting detailed debug information.
    When VERBOSE_DEBUG=True, outputs the complete message.
    When VERBOSE_DEBUG=False, outputs only the first 50 characters.

    Args:
        msg: The message format string
        *args: Arguments to be formatted into the message
        **kwargs: Keyword arguments passed to logger.info()
    """
    if VERBOSE_DEBUG:
        logger.info(msg, *args, **kwargs)
    else:
        # Format the message with args first
        if args:
            formatted_msg = msg % args
        else:
            formatted_msg = msg
        # Then truncate the formatted message
        truncated_msg = (
            formatted_msg[:100] + "..." if len(formatted_msg) > 100 else formatted_msg
        )
        logger.info(truncated_msg, **kwargs)

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from ..utils import (
    wrap_embedding_func_with_attrs,
    locate_json_string_body_from_string,
    safe_unicode_decode,
    logger,
)
from ..types import GPTKeywordExtractionFormat

import numpy as np
from typing import Any, Union


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism"""
    pass


def create_openai_async_client(
    api_key: str | None = None,
    base_url: str | None = None,
    client_configs: dict[str, Any] = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client with the given configuration.

    Args:
        api_key: OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        base_url: Base URL for the OpenAI API. If None, uses the default OpenAI API URL.
        client_configs: Additional configuration options for the AsyncOpenAI client.
            These will override any default configurations but will be overridden by
            explicit parameters (api_key, base_url).

    Returns:
        An AsyncOpenAI client instance.
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if client_configs is None:
        client_configs = {}

    # Create a merged config dict with precedence: explicit params > client_configs > defaults
    merged_configs = {
        **client_configs,
        "api_key": api_key,
        "http_client": httpx.AsyncClient(proxy=os.getenv("http_proxy_port"), verify=False) if os.getenv("http_proxy_port") else None,
    }

    if base_url is not None:
        merged_configs["base_url"] = base_url
    else:
        merged_configs["base_url"] = os.getenv(
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
) -> str:
    """Complete a prompt using OpenAI's API with caching support.

    Args:
        model: The OpenAI model to use.
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        base_url: Optional base URL for the OpenAI API.
        api_key: Optional OpenAI API key. If None, uses the OPENAI_API_KEY environment variable.
        **kwargs: Additional keyword arguments to pass to the OpenAI API.
            Special kwargs:
            - openai_client_configs: Dict of configuration options for the AsyncOpenAI client.
                These will be passed to the client constructor but will be overridden by
                explicit parameters (api_key, base_url).
            - hashing_kv: Will be removed from kwargs before passing to OpenAI.
            - keyword_extraction: Will be removed from kwargs before passing to OpenAI.

    Returns:
        The completed text or an async iterator of text chunks if streaming.

    Raises:
        InvalidResponseError: If the response from OpenAI is invalid or empty.
        APIConnectionError: If there is a connection error with the OpenAI API.
        RateLimitError: If the OpenAI API rate limit is exceeded.
        APITimeoutError: If the OpenAI API request times out.
    """
    if history_messages is None:
        history_messages = []

    if not model:
        model = os.getenv("OPENAI_API_MODEL")
    if not model:
        raise ValueError(f"OPENAI_API_MODEL is not set or is invalid: {model}")

    # Set openai logger level to INFO when VERBOSE_DEBUG is off
    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("openai").setLevel(logging.INFO)

    # Extract client configuration options
    client_configs = kwargs.pop("openai_client_configs", {})
    # Create the OpenAI client
    openai_async_client = create_openai_async_client(
        api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"), client_configs=client_configs
    )

    # Remove special kwargs that shouldn't be passed to OpenAI
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    # Prepare messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    messages = kwargs.pop("messages", messages)

    logger.debug(f"messages:\n{'\n'.join(f'{key}: {value}' for m in messages for key, value in m.items())}")

    try:
        # Don't use async with context manager, use client directly
        # response = await openai_async_client.chat.completions.create(
        #         model=model, messages=messages, **kwargs
        #     )
        if "response_format" in kwargs:
            response = await openai_async_client.beta.chat.completions.parse(
                model=model, messages=messages, **kwargs
            )
        else:
            response = await openai_async_client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
    except APIConnectionError as e:
        logger.error(f"OpenAI API Connection Error: {e}")
        logger.error(f"model={model}, messages={messages}, {kwargs}")
        await openai_async_client.close()  # Ensure client is closed
        raise
    except RateLimitError as e:
        logger.error(f"OpenAI API Rate Limit Error: {e}")
        logger.error(f"model={model}, messages={messages}, {kwargs}")
        await openai_async_client.close()  # Ensure client is closed
        raise
    except APITimeoutError as e:
        logger.error(f"OpenAI API Timeout Error: {e}")
        logger.error(f"model={model}, messages={messages}, {kwargs}")
        await openai_async_client.close()  # Ensure client is closed
        raise
    except Exception as e:
        logger.error(f"OpenAI API Call Failed, Got: {e}")
        logger.error(f"model={model}, messages={messages}, {kwargs}")
        await openai_async_client.close()  # Ensure client is closed
        raise

    if hasattr(response, "__aiter__"):

        async def inner():
            # Track if we've started iterating
            iteration_started = False
            final_chunk_usage = None

            try:
                iteration_started = True
                async for chunk in response:
                    # Check if this chunk has usage information (final chunk)
                    if hasattr(chunk, "usage") and chunk.usage:
                        final_chunk_usage = chunk.usage
                        logger.info(
                            f"Received usage info in streaming chunk: {chunk.usage}"
                        )

                    # Check if choices exists and is not empty
                    if not hasattr(chunk, "choices") or not chunk.choices:
                        logger.warning(f"Received chunk without choices: {chunk}")
                        continue

                    # Check if delta exists and has content
                    if not hasattr(chunk.choices[0], "delta") or not hasattr(
                        chunk.choices[0].delta, "content"
                    ):
                        # This might be the final chunk, continue to check for usage
                        continue

                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue
                    if r"\u" in content:
                        content = safe_unicode_decode(content.encode("utf-8"))

                    yield content

                # After streaming is complete, track token usage
                if token_tracker and final_chunk_usage:
                    # Use actual usage from the API
                    token_counts = {
                        "prompt_tokens": getattr(final_chunk_usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            final_chunk_usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
                    }
                    token_tracker.add_usage(token_counts)
                    logger.info(f"Streaming token usage (from API): {token_counts}")
                elif token_tracker:
                    logger.info("No usage information available in streaming response")
            except Exception as e:
                logger.error(f"Error in stream response: {str(e)}")
                # Try to clean up resources if possible
                if (
                    iteration_started
                    and hasattr(response, "aclose")
                    and callable(getattr(response, "aclose", None))
                ):
                    try:
                        await response.aclose()
                        logger.info("Successfully closed stream response after error")
                    except Exception as close_error:
                        logger.warning(
                            f"Failed to close stream response: {close_error}"
                        )
                # Ensure client is closed in case of exception
                await openai_async_client.close()
                raise
            finally:
                # Ensure resources are released even if no exception occurs
                if (
                    iteration_started
                    and hasattr(response, "aclose")
                    and callable(getattr(response, "aclose", None))
                ):
                    try:
                        await response.aclose()
                        logger.info("Successfully closed stream response")
                    except Exception as close_error:
                        logger.warning(
                            f"Failed to close stream response in finally block: {close_error}"
                        )

                # This prevents resource leaks since the caller doesn't handle closing
                try:
                    await openai_async_client.close()
                    logger.info(
                        "Successfully closed OpenAI client for streaming response"
                    )
                except Exception as client_close_error:
                    logger.warning(
                        f"Failed to close OpenAI client in streaming finally block: {client_close_error}"
                    )

        return inner()

    else:
        try:
            if (
                not response
                or not response.choices
                or not hasattr(response.choices[0], "message")
                or not hasattr(response.choices[0].message, "content")
            ):
                logger.error("Invalid response from OpenAI API")
                await openai_async_client.close()  # Ensure client is closed
                raise InvalidResponseError("Invalid response from OpenAI API")

            content = response.choices[0].message.content

            if not content or content.strip() == "":
                logger.error("Received empty content from OpenAI API")
                await openai_async_client.close()  # Ensure client is closed
                raise InvalidResponseError("Received empty content from OpenAI API")

            if r"\u" in content:
                content = safe_unicode_decode(content.encode("utf-8"))

            if token_tracker and hasattr(response, "usage"):
                token_counts = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(
                        response.usage, "completion_tokens", 0
                    ),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }
                token_tracker.add_usage(token_counts)

            logger.info(f"Response content len: {len(content)}")
            verbose_debug(f"Response: {response}")

            return content
        finally:
            # Ensure client is closed in all cases for non-streaming responses
            await openai_async_client.close()


async def openai_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["response_format"] = "json"
    model_name = os.getenv("OPENAI_API_MODEL") # kwargs["hashing_kv"].global_config["llm_model_name"]
    if not model_name:
        raise ValueError(f"llm_model_name is not set or is invalid: {model_name}\n{kwargs}")
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
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
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
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
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",  # context length 128k
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    if keyword_extraction:  
        return locate_json_string_body_from_string(result)
    return result

# 你原有的装饰器依旧保留
@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def openai_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
    client_configs: dict[str, Any] = None,
) -> np.ndarray:
    """Generate embeddings for a list of texts using OpenAI's API, with up to 10 extra retries
    on failures or abnormal responses (max 11 attempts total).
    """

    # ---- 可配置的退避参数（与你的 tenacity 设置一致）----
    min_delay = 4
    max_delay = 60
    multiplier = 1
    max_attempts = 50  # 第一次 + 额外重试 49 次

    # 创建客户端
    # print(
    #     "openai_async_client = create_openai_async_client("
    #     f"api_key={os.getenv('EMBEDDING_BINDING_API_KEY')}, "
    #     f"base_url={os.getenv('EMBEDDING_BINDING_URL')}, "
    #     f"client_configs={client_configs})"
    # )
    openai_async_client = create_openai_async_client(
        api_key=api_key or os.getenv("EMBEDDING_BINDING_API_KEY"),
        base_url=base_url or os.getenv("EMBEDDING_BINDING_URL"),
        client_configs=client_configs,
    )

    # 如果环境里提供了期望维度，就用于严格校验；否则只做基本校验
    expected_dim = None
    try:
        # 如果你愿意，也可以换成读取装饰器注入的属性
        expected_dim_env = os.getenv("EMBEDDING_DIM")
        if expected_dim_env:
            expected_dim = int(expected_dim_env)
    except Exception:
        expected_dim = None

    def _is_response_ok(resp) -> bool:
        """判断 response 是否‘正常’：
        - resp 存在且有 data
        - data 数量与输入 texts 对齐
        - 每个 embedding 是非空向量；若设置了 expected_dim，则维度需匹配
        """
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

    last_error = None
    async with openai_async_client:
        for attempt in range(1, max_attempts + 1):
            try:
                # 为了避免打印过长的 input，只展示每段前 10 个字符
                # preview = [x[:10] for x in texts]
                # print(
                #     "response = await openai_async_client.embeddings.create("
                #     f"model={os.getenv('EMBEDDING_MODEL') or model}, "
                #     f"input={preview}, encoding_format='float')"
                # )
                response = await openai_async_client.embeddings.create(
                    model=os.getenv("EMBEDDING_MODEL") or model,
                    input=texts,
                    encoding_format="float",
                )

                if _is_response_ok(response):
                    # 转为 numpy 数组并返回
                    return np.array([dp.embedding for dp in response.data])

                # 返回结构异常：不抛异常也重试
                last_error = RuntimeError(
                    "Abnormal embedding response: count/shape mismatch."
                )
                raise last_error

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                # 明确可重试的错误
                last_error = e
            except Exception as e:
                # 其他错误也纳入重试（以满足“直到不报错且结果正常”）
                last_error = e

            # 如果还有机会重试，则按指数退避等待
            if attempt < max_attempts:
                # 指数退避：multiplier * 2^(attempt-1)，并夹在 [min_delay, max_delay]
                delay = multiplier * (2 ** (attempt - 1))
                delay = max(min_delay, min(delay, max_delay))
                logger.warning(
                    f"[Embedding Attempt {attempt}/{max_attempts}] failed: {last_error}. Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            else:
                # 到达最大尝试次数，抛出最后一次错误
                logger.error(
                    f"[Embedding Attempt {attempt}/{max_attempts}] failed: {last_error}. Giving up."
                )
                raise last_error
