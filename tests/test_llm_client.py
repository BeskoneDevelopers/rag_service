import json
from contextlib import asynccontextmanager
from functools import partial
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
FIRST_CHUNK = b'{"response":"first","done":false}\n'


class OllamaBody(httpx.AsyncByteStream):
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
    async with AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def mock_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.test/api/generate")

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


def check_error(response, path, status, detail):
    if path == "/ask/stream":
        assert read_events(response) == [("error", {"detail": detail})]
    else:
        assert response.status_code == status
        assert response.json() == {"detail": detail}


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
async def test_retries_transport_drop_before_answer(api, mock_ollama, path):
    streaming = path.endswith("/stream")
    broken = OllamaBody(httpx.ReadError("connection reset"))
    if streaming:
        success = httpx.Response(200, content=FIRST_CHUNK + b'{"response":" last","done":true}\n')
    else:
        success = httpx.Response(200, json={"response": "first last", "done": True})
    requests = mock_ollama(httpx.Response(200, stream=broken), success)

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
async def test_retries_disconnect_at_most_once(api, mock_ollama, path):
    requests = mock_ollama(
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
    )

    response = await api.post(path, json=QUESTION)

    check_error(response, path, 503, "Соединение с Ollama потеряно")
    assert len(requests) == 2


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
@pytest.mark.parametrize(
    ("reply", "status", "detail"),
    [
        (httpx.ConnectError("refused"), 503, "Не удалось подключиться к Ollama"),
        (httpx.ConnectTimeout("connect"), 504, "Истекло время подключения к Ollama"),
        (httpx.ReadTimeout("idle"), 504, "Истекло время ожидания ответа Ollama"),
        (httpx.Response(404), 502, "Ollama вернула HTTP 404"),
        (httpx.Response(500), 502, "Ollama вернула HTTP 500"),
        (httpx.RemoteProtocolError("illegal status line"), 502, "Некорректный ответ Ollama"),
        (httpx.Response(200, content=b"{bad json}\n"), 502, "Некорректный JSON в ответе Ollama"),
        (httpx.Response(200, content=b"[]\n"), 502, "Некорректный формат ответа Ollama"),
        (
            httpx.Response(200, content=b'{"error":"private backend details"}\n'),
            502,
            "Ollama сообщила об ошибке генерации",
        ),
    ],
)
async def test_other_ollama_failures_do_not_retry(api, mock_ollama, path, reply, status, detail):
    requests = mock_ollama(reply)

    response = await api.post(path, json=QUESTION)

    check_error(response, path, status, detail)
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("tail", "detail"),
    [
        (httpx.ReadError("connection reset"), "Соединение с Ollama потеряно"),
        (httpx.ReadTimeout("idle"), "Истекло время ожидания ответа Ollama"),
        (b'{"error":"model stopped"}\n', "Ollama сообщила об ошибке генерации"),
        (b'{"response":15,"done":false}\n', "Некорректный формат ответа Ollama"),
        (b"", "Поток Ollama завершился без подтверждения окончания"),
        (RuntimeError("private implementation details"), "Не удалось завершить поток ответа"),
    ],
)
async def test_partial_stream_ends_with_error_without_retry(api, mock_ollama, tail, detail):
    body = OllamaBody(FIRST_CHUNK, tail)
    requests = mock_ollama(httpx.Response(200, stream=body))

    response = await api.post("/ask/stream", json=QUESTION)

    assert read_events(response) == [("token", {"text": "first"}), ("error", {"detail": detail})]
    assert len(requests) == 1
    assert body.closed
