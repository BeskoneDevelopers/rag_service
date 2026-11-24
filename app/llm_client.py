"""OpenAI-compatible llama.cpp client; httpx is replaceable by tests."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

import httpx
from anyio import CancelScope, move_on_after


@dataclass
class GenerationUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMUnavailableError(Exception):
    pass


class LLMTimeoutError(Exception):
    pass


class LLMServiceError(Exception):
    pass


def _settings():
    from app.config import settings

    return settings


def model_name() -> str:
    return _settings().llm_model


def _llama_url() -> httpx.URL:
    try:
        return httpx.URL(_settings().llm_chat_url)
    except httpx.InvalidURL as exc:
        raise LLMUnavailableError("Некорректный адрес llama-server") from exc


def _payload(prompt: str, *, stream: bool, cache_prompt: bool = True) -> dict:
    payload = {
        "model": model_name(),
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": _settings().llm_max_tokens,
        "cache_prompt": cache_prompt,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _connection_dropped(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, (httpx.ReadError, httpx.WriteError)):
        return True
    return isinstance(exc, httpx.RemoteProtocolError) and str(exc).startswith(
        (
            "Server disconnected",
            "peer unexpectedly closed connection",
            "peer closed connection without sending complete message body",
        )
    )


def _http_error(exc: httpx.HTTPError, *, disconnected: bool) -> Exception:
    if disconnected:
        return LLMUnavailableError("Соединение с llama-server потеряно")
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError("Истекло время ожидания llama-server")
    if isinstance(exc, (httpx.ConnectError, httpx.ProxyError, httpx.UnsupportedProtocol)):
        return LLMUnavailableError("Не удалось подключиться к llama-server")
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (429, 503):
            return LLMUnavailableError("llama-server временно недоступен или занят")
        return LLMServiceError(f"llama-server вернул HTTP {status}")
    return LLMServiceError("Некорректный ответ llama-server")


def _parse_response(raw: str | bytes) -> dict:
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise LLMServiceError("Некорректный JSON от llama-server") from exc
    if not isinstance(data, dict) or "error" in data:
        raise LLMServiceError("Некорректный ответ llama-server")
    return data


def _get_choice(data: dict, *, stream: bool) -> dict | None:
    choices = data.get("choices")
    if not isinstance(choices, list):
        raise LLMServiceError("Нет списка choices в ответе llama-server")
    if not choices and stream and "usage" in data:
        return None
    if not choices or not isinstance(choices[0], dict):
        raise LLMServiceError("Некорректный choices от llama-server")
    return choices[0]


def _check_finish_reason(reason: object) -> bool:
    if reason is None:
        return False
    if reason == "length":
        raise LLMServiceError("Достигнут лимит токенов: ответ не завершён")
    if reason != "stop":
        raise LLMServiceError("Генерация не завершилась штатно")
    return True


def _set_usage(data: dict, usage: GenerationUsage | None) -> None:
    if usage is None or not isinstance(data.get("usage"), dict):
        return
    counts = data["usage"]
    if type(counts.get("prompt_tokens")) is int:
        usage.prompt_tokens = counts["prompt_tokens"]
    if type(counts.get("completion_tokens")) is int:
        usage.completion_tokens = counts["completion_tokens"]


@asynccontextmanager
async def _request_llama(url: httpx.URL, payload: dict) -> AsyncIterator[httpx.Response]:
    timeout = httpx.Timeout(_settings().llm_timeout_seconds, connect=5.0)
    stack = AsyncExitStack()
    try:
        client = await stack.enter_async_context(httpx.AsyncClient(timeout=timeout))
        response = await stack.enter_async_context(client.stream("POST", url, json=payload))
        response.raise_for_status()
        yield response
    finally:
        # Cancellation must still close the upstream response before releasing the lock.
        with CancelScope(shield=True), move_on_after(5):
            await stack.aclose()


async def generate_answer(prompt: str) -> str:
    try:
        async with asyncio.timeout(_settings().llm_generation_timeout_seconds):
            return await _generate_answer(prompt)
    except TimeoutError as exc:
        raise LLMTimeoutError("Превышен абсолютный лимит генерации") from exc


async def _generate_answer(prompt: str) -> str:
    url = _llama_url()
    payload = _payload(prompt, stream=False)
    for retry_allowed in (True, False):
        try:
            async with _request_llama(url, payload) as response:
                data = _parse_response(await response.aread())
                choice = _get_choice(data, stream=False)
                assert choice is not None
                _check_finish_reason(choice.get("finish_reason"))
                message = choice.get("message")
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    raise LLMServiceError("Некорректный текст ответа llama-server")
                return message["content"]
        except httpx.HTTPError as exc:
            disconnected = _connection_dropped(exc)
            if not (retry_allowed and disconnected):
                raise _http_error(exc, disconnected=disconnected) from exc

    raise LLMServiceError("Не удалось завершить запрос к llama-server")


async def stream_answer(
    prompt: str, *, usage: GenerationUsage | None = None, cache_prompt: bool = True
) -> AsyncGenerator[str, None]:
    url = _llama_url()
    payload = _payload(prompt, stream=True, cache_prompt=cache_prompt)
    emitted_token = False
    deadline = asyncio.get_running_loop().time() + _settings().llm_generation_timeout_seconds
    for retry_allowed in (True, False):
        try:
            async with AsyncExitStack() as stack:
                async with asyncio.timeout_at(deadline):
                    response = await stack.enter_async_context(_request_llama(url, payload))
                finished = False
                lines = response.aiter_lines()
                while True:
                    # Do not keep a cancellation scope open across an async-generator yield.
                    async with asyncio.timeout_at(deadline):
                        try:
                            line = await anext(lines)
                        except StopAsyncIteration:
                            break
                    if not line.startswith("data:"):
                        continue
                    raw = line.removeprefix("data:").strip()
                    if raw == "[DONE]":
                        if not finished:
                            raise LLMServiceError("Поток llama-server завершён без finish_reason")
                        return
                    data = _parse_response(raw)
                    _set_usage(data, usage)
                    choice = _get_choice(data, stream=True)
                    if choice is None:
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        raise LLMServiceError("Некорректный формат потока llama-server")
                    content = delta.get("content")
                    if content is not None and not isinstance(content, str):
                        raise LLMServiceError("Некорректный текст в потоке llama-server")
                    if content:
                        emitted_token = True
                        yield content
                    if _check_finish_reason(choice.get("finish_reason")):
                        finished = True
                raise LLMServiceError("Поток llama-server оборвался без [DONE]")
        except TimeoutError as exc:
            raise LLMTimeoutError("Превышен абсолютный лимит генерации") from exc
        except httpx.HTTPError as exc:
            disconnected = _connection_dropped(exc)
            if not (retry_allowed and disconnected and not emitted_token):
                raise _http_error(exc, disconnected=disconnected) from exc
