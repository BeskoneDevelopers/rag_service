import json
import os
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

OLLAMA_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
OLLAMA_MODEL = "qwen3:1.7b"


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


def _ollama_url() -> httpx.URL:
    try:
        return httpx.URL(os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"))
    except httpx.InvalidURL as exc:
        raise LLMUnavailableError("Некорректный адрес Ollama") from exc


def _payload(prompt: str, *, stream: bool) -> dict[str, str | bool]:
    return {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": stream,
        "think": False,
    }


def _connection_dropped(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, (httpx.ReadError, httpx.WriteError)):
        return True
    # HTTPX reports premature HTTP/1.1 EOF and malformed HTTP as RemoteProtocolError.
    return isinstance(exc, httpx.RemoteProtocolError) and str(exc).startswith(
        (
            "Server disconnected",
            "peer unexpectedly closed connection",
            "peer closed connection without sending complete message body",
        )
    )


def _http_error(exc: httpx.HTTPError, *, disconnected: bool) -> Exception:
    if disconnected:
        return LLMUnavailableError("Соединение с Ollama потеряно")
    if isinstance(exc, httpx.ConnectTimeout):
        return LLMTimeoutError("Истекло время подключения к Ollama")
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError("Истекло время ожидания ответа Ollama")
    if isinstance(exc, (httpx.ConnectError, httpx.ProxyError)):
        return LLMUnavailableError("Не удалось подключиться к Ollama")
    if isinstance(exc, httpx.UnsupportedProtocol):
        return LLMUnavailableError("Некорректный адрес Ollama")
    if isinstance(exc, httpx.HTTPStatusError):
        return LLMServiceError(f"Ollama вернула HTTP {exc.response.status_code}")
    return LLMServiceError("Некорректный ответ Ollama")


def _parse_response(raw: str | bytes) -> dict[str, object]:
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise LLMServiceError("Некорректный JSON в ответе Ollama") from exc

    if not isinstance(data, dict):
        raise LLMServiceError("Некорректный формат ответа Ollama")
    if "error" in data:
        raise LLMServiceError("Ollama сообщила об ошибке генерации")
    return data


def _response_text(value: object) -> str:
    if not isinstance(value, str):
        raise LLMServiceError("Некорректный формат ответа Ollama")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LLMServiceError("Некорректный текст в ответе Ollama") from exc
    return value


@asynccontextmanager
async def _request_ollama(
    url: httpx.URL, payload: dict[str, str | bool]
) -> AsyncIterator[httpx.Response]:
    async with (
        httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client,
        client.stream("POST", url, json=payload) as response,
    ):
        response.raise_for_status()
        yield response


async def generate_answer(prompt: str) -> str:
    url = _ollama_url()
    payload = _payload(prompt, stream=False)
    for retry_allowed in (True, False):
        try:
            async with _request_ollama(url, payload) as response:
                data = _parse_response(await response.aread())
                if data.get("done", True) is not True:
                    raise LLMServiceError("Ответ Ollama не завершён")
                return _response_text(data.get("response"))
        except httpx.HTTPError as exc:
            disconnected = _connection_dropped(exc)
            if not (retry_allowed and disconnected):
                raise _http_error(exc, disconnected=disconnected) from exc


async def stream_answer(
    prompt: str, *, usage: GenerationUsage | None = None
) -> AsyncGenerator[str, None]:
    url = _ollama_url()
    payload = _payload(prompt, stream=True)
    emitted_token = False
    for retry_allowed in (True, False):
        try:
            async with _request_ollama(url, payload) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    data = _parse_response(line)
                    done = data.get("done")
                    if not isinstance(done, bool):
                        raise LLMServiceError("Некорректный формат потока Ollama")
                    token = _response_text(data.get("response", "" if done else None))
                    if done and usage is not None:
                        prompt_tokens = data.get("prompt_eval_count")
                        completion_tokens = data.get("eval_count")
                        if type(prompt_tokens) is int:
                            usage.prompt_tokens = prompt_tokens
                        if type(completion_tokens) is int:
                            usage.completion_tokens = completion_tokens
                    if token:
                        emitted_token = True
                        yield token
                    if done:
                        return

                raise LLMServiceError("Поток Ollama завершился без подтверждения окончания")
        except httpx.HTTPError as exc:
            disconnected = _connection_dropped(exc)
            if not (retry_allowed and disconnected and not emitted_token):
                raise _http_error(exc, disconnected=disconnected) from exc
