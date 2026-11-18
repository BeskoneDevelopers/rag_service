import json
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app import llm_client, rag_pipeline
from app.routers import ask

pytestmark = pytest.mark.anyio
QUESTION = {"company": "test", "question": "Вопрос?"}
SOURCES = [{"document_id": 1}]
FIRST_CHUNK = b'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n'
LAST_CHUNK = (
    b'data: {"choices":[{"delta":{"content":" last"},"finish_reason":"stop"}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":32,"completion_tokens":5}}\n\n'
    b'data: [DONE]\n\n'
)
SUCCESS = {"choices": [{"message": {"content": "first last"}, "finish_reason": "stop"}]}


class LlamaBody(httpx.AsyncByteStream):
    def __init__(self, *parts):
        self.parts = parts
        self.closed = False

    async def __aiter__(self):
        for part in self.parts:
            if isinstance(part, Exception):
                raise part
            yield part

    async def aclose(self):
        self.closed = True


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def api(monkeypatch, anyio_backend):
    @asynccontextmanager
    async def generation_lock():
        yield

    app = FastAPI()
    app.include_router(ask.router)
    app.dependency_overrides[ask.get_current_user] = lambda: {"company": "test"}
    retrieved = {"context": "Контекст", "sources": SOURCES}
    for module in (ask, rag_pipeline):
        monkeypatch.setattr(module, "retrieve_context", AsyncMock(return_value=retrieved))
        monkeypatch.setattr(module, "llm_generation_lock", generation_lock)
    async with AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_llama(monkeypatch):
    monkeypatch.setattr(
        llm_client,
        "_settings",
        lambda: SimpleNamespace(
            llm_chat_url="http://llama.test/v1/chat/completions",
            llm_model="qwen3.5-2b",
            llm_timeout_seconds=30,
        ),
    )

    def install(*replies):
        requests = []

        def handle(request):
            reply = replies[len(requests)]
            requests.append(request)
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(
            llm_client.httpx,
            "AsyncClient",
            partial(AsyncClient, transport=httpx.MockTransport(handle)),
        )
        return requests

    return install


def read_events(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for frame in response.text.strip().split("\n\n"):
        event, data = frame.splitlines()
        events.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def check_error(response, status, detail):
    assert response.status_code == status
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
async def test_retries_transport_drop_before_answer(api, mock_llama, path):
    streaming = path.endswith("/stream")
    broken = LlamaBody(httpx.ReadError("connection reset"))
    success = (
        httpx.Response(200, content=FIRST_CHUNK + LAST_CHUNK)
        if streaming
        else httpx.Response(200, json=SUCCESS)
    )
    requests = mock_llama(httpx.Response(200, stream=broken), success)
    response = await api.post(path, json=QUESTION)
    assert broken.closed
    assert len(requests) == 2
    assert all(json.loads(request.content)["stream"] is streaming for request in requests)
    if streaming:
        assert read_events(response) == [
            ("token", {"text": "first"}),
            ("token", {"text": " last"}),
            ("sources", SOURCES),
        ]
    else:
        assert response.status_code == 200
        assert response.json() == {"answer": "first last", "sources": SOURCES}


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
async def test_retries_disconnect_at_most_once(api, mock_llama, path):
    requests = mock_llama(
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
    )
    response = await api.post(path, json=QUESTION)
    check_error(response, 503, "Соединение с llama-server потеряно")
    assert len(requests) == 2


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
@pytest.mark.parametrize(
    ("reply", "status", "detail"),
    [
        (httpx.ConnectError("refused"), 503, "Не удалось подключиться к llama-server"),
        (httpx.ConnectTimeout("connect"), 504, "Истекло время ожидания llama-server"),
        (httpx.ReadTimeout("idle"), 504, "Истекло время ожидания llama-server"),
        (httpx.Response(503), 503, "llama-server временно недоступен или занят"),
        (httpx.Response(404), 502, "llama-server вернул HTTP 404"),
        (httpx.RemoteProtocolError("illegal status line"), 502, "Некорректный ответ llama-server"),
        (httpx.Response(200, content=b"{bad json}\n"), 502, "Некорректный JSON от llama-server"),
        (httpx.Response(200, content=b"[]\n"), 502, "Некорректный ответ llama-server"),
        (httpx.Response(200, content=b'{"error":"private details"}\n'), 502, "Некорректный ответ llama-server"),
    ],
)
async def test_other_llama_failures_do_not_retry(api, mock_llama, path, reply, status, detail):
    if path.endswith("/stream") and isinstance(reply, httpx.Response):
        raw = reply.content
        if raw == b"{bad json}\n":
            reply = httpx.Response(200, content=b"data: {bad json}\n\n")
        elif raw == b"[]\n":
            reply = httpx.Response(200, content=b"data: []\n\n")
        elif raw.startswith(b'{"error":'):
            reply = httpx.Response(200, content=b'data: {"error":"private details"}\n\n')
    requests = mock_llama(reply)
    response = await api.post(path, json=QUESTION)
    check_error(response, status, detail)
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("tail", "detail"),
    [
        (httpx.ReadError("connection reset"), "Соединение с llama-server потеряно"),
        (httpx.ReadTimeout("idle"), "Истекло время ожидания llama-server"),
        (b'data: {"error":"model stopped"}\n\n', "Некорректный ответ llama-server"),
        (
            b'data: {"choices":[{"delta":{"content":15},"finish_reason":null}]}\n\n',
            "Некорректный текст в потоке llama-server",
        ),
        (b"", "Поток llama-server оборвался без [DONE]"),
        (RuntimeError("private implementation details"), "Не удалось завершить поток ответа"),
    ],
)
async def test_partial_stream_ends_with_error_without_retry(api, mock_llama, tail, detail):
    body = LlamaBody(FIRST_CHUNK, tail)
    requests = mock_llama(httpx.Response(200, stream=body))
    response = await api.post("/ask/stream", json=QUESTION)
    assert read_events(response) == [("token", {"text": "first"}), ("error", {"detail": detail})]
    assert len(requests) == 1
    assert body.closed


async def test_usage_is_read_from_stream(api, mock_llama):
    mock_llama(httpx.Response(200, content=FIRST_CHUNK + LAST_CHUNK))
    usage = llm_client.GenerationUsage()
    result = [part async for part in llm_client.stream_answer("Question", usage=usage)]
    assert result == ["first", " last"]
    assert usage.prompt_tokens == 32
    assert usage.completion_tokens == 5
