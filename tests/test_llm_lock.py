import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app import llm_lock, rag_pipeline
from app.routers import ask

pytestmark = pytest.mark.anyio


class MemoryRedis:
    def __init__(self):
        self.token = None
        self.expires_at = 0
        self.set_started = anyio.Event()
        self.set_result = None
        self.set_error = None
        self.renew_error = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    def owner(self):
        return self.token if anyio.current_time() < self.expires_at else None

    async def set(self, key, token, *, nx, ex):
        assert key == llm_lock.LOCK_KEY and nx
        if self.owner() is not None:
            return None
        self.token = token
        self.expires_at = anyio.current_time() + ex
        self.set_started.set()
        if self.set_result:
            await self.set_result.wait()
        if self.set_error:
            raise self.set_error
        return True

    async def eval(self, script, numkeys, key, token, *args):
        assert numkeys == 1 and key == llm_lock.LOCK_KEY
        if script == llm_lock.RENEW_SCRIPT and self.renew_error:
            raise self.renew_error
        if token != self.owner():
            return 0
        if script == llm_lock.RENEW_SCRIPT:
            (ttl,) = args
            self.expires_at = anyio.current_time() + ttl
        else:
            assert script == llm_lock.RELEASE_SCRIPT
            self.token = None
        return 1


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def redis(monkeypatch):
    backend = MemoryRedis()
    monkeypatch.setattr(llm_lock, "Redis", SimpleNamespace(from_url=lambda *a, **kw: backend))
    return backend


@pytest.fixture
def app(monkeypatch):
    app = FastAPI()
    app.include_router(ask.router)
    app.dependency_overrides[ask.get_current_user] = lambda: {"company": "test"}
    result = {"context": "Контекст", "sources": [{"document_id": 1}]}
    monkeypatch.setattr(ask, "retrieve_context", AsyncMock(return_value=result))
    monkeypatch.setattr(rag_pipeline, "retrieve_context", AsyncMock(return_value=result))
    return app


QUESTION = {"company": "test", "question": "Вопрос?"}


@pytest.mark.parametrize("stream_first", [False, True])
async def test_ask_endpoints_exclude_each_other(app, redis, monkeypatch, stream_first):
    started = anyio.Event()
    finish = anyio.Event()

    async def generate(prompt):
        started.set()
        await finish.wait()
        return "Ответ"

    async def stream(prompt):
        yield await generate(prompt)

    monkeypatch.setattr(rag_pipeline, "generate_answer", generate)
    monkeypatch.setattr(ask, "stream_answer", stream)
    first, second = ("/ask/stream", "/ask") if stream_first else ("/ask", "/ask/stream")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:

        async def check_first():
            response = await client.post(first, json=QUESTION)
            assert response.status_code == 200
            if stream_first:
                assert response.text == (
                    'event: token\ndata: {"text": "Ответ"}\n\n'
                    'event: sources\ndata: [{"document_id": 1}]\n\n'
                )
            else:
                assert response.json()["answer"] == "Ответ"

        with anyio.fail_after(5):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(check_first)
                await started.wait()
                response = await client.post(second, json=QUESTION)
                assert response.status_code == 503
                assert response.headers["content-type"] == "application/json"
                finish.set()

    assert redis.owner() is None


async def test_lock_survives_initial_ttl(redis, monkeypatch):
    monkeypatch.setattr(llm_lock, "LOCK_TTL", 1)
    async with llm_lock.llm_generation_lock():
        owner = redis.owner()
        await anyio.sleep(1.1)
        assert redis.owner() == owner
        with pytest.raises(llm_lock.LLMBusyError):
            async with llm_lock.llm_generation_lock():
                pytest.fail("Acquired a lock held by an active request")
    assert redis.owner() is None


async def test_lock_rejects_ownership_from_another_worker(redis):
    await redis.set(llm_lock.LOCK_KEY, "other-worker", nx=True, ex=llm_lock.LOCK_TTL)
    with pytest.raises(llm_lock.LLMBusyError):
        async with llm_lock.llm_generation_lock():
            pytest.fail("Acquired a lock held by another worker")
    assert redis.owner() == "other-worker"


async def test_local_generation_remains_exclusive_if_redis_drops_the_key(redis):
    async with llm_lock.llm_generation_lock():
        redis.token = None
        with pytest.raises(llm_lock.LLMBusyError):
            async with llm_lock.llm_generation_lock():
                pytest.fail("Acquired a lock before the active generation stopped")


async def test_lock_cancels_generation_before_ttl_if_renewal_stalls(redis, monkeypatch):
    monkeypatch.setattr(llm_lock, "LOCK_TTL", 1)

    async def stuck_renewal(token):
        await anyio.sleep_forever()

    monkeypatch.setattr(llm_lock, "renew_llm_lock", stuck_renewal)
    with anyio.fail_after(5), pytest.raises(llm_lock.LLMLockUnavailableError):
        async with llm_lock.llm_generation_lock():
            try:
                await anyio.sleep_forever()
            finally:
                assert redis.owner() is not None

    assert redis.owner() is None


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
async def test_redis_renewal_failure_cancels_generation(app, redis, monkeypatch, path):
    monkeypatch.setattr(llm_lock, "LOCK_TTL", 1)
    redis.renew_error = RedisConnectionError("Redis unavailable")
    closed = anyio.Event()

    async def generate(prompt):
        try:
            await anyio.sleep_forever()
        finally:
            closed.set()

    async def stream(prompt):
        yield "Часть ответа"
        await generate(prompt)

    monkeypatch.setattr(rag_pipeline, "generate_answer", generate)
    monkeypatch.setattr(ask, "stream_answer", stream)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with anyio.fail_after(5):
            response = await client.post(path, json=QUESTION)

    assert closed.is_set()
    assert redis.owner() is None
    if path == "/ask/stream":
        assert response.status_code == 200
        assert "event: error" in response.text
        assert "event: sources" not in response.text
    else:
        assert response.status_code == 503


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
async def test_failed_set_cleans_up_ambiguous_acquisition(app, redis, monkeypatch, path):
    redis.set_error = RedisConnectionError("Lost the SET response")
    generate = AsyncMock()

    async def stream(prompt):
        yield await generate(prompt)

    monkeypatch.setattr(rag_pipeline, "generate_answer", generate)
    monkeypatch.setattr(ask, "stream_answer", stream)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(path, json=QUESTION)

    assert response.status_code == 503
    assert redis.owner() is None
    generate.assert_not_called()


@pytest.mark.parametrize("phase", ["set", "headers", "token"])
async def test_stream_disconnect_closes_generation(app, redis, monkeypatch, phase):
    disconnect = anyio.Event()
    started = anyio.Event()
    closed = anyio.Event()
    if phase == "set":
        redis.set_result = anyio.Event()

    async def stream(prompt):
        started.set()
        try:
            yield "Часть ответа"
            await anyio.sleep_forever()
        finally:
            assert redis.owner() is not None
            closed.set()

    monkeypatch.setattr(ask, "stream_answer", stream)
    body = json.dumps(QUESTION).encode()

    async def receive():
        nonlocal body
        if body:
            message = {"type": "http.request", "body": body}
            body = b""
            return message
        if phase == "set":
            await redis.set_started.wait()
            redis.set_result.set()
        else:
            await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if phase == "headers" or message.get("body", b"").startswith(b"event: token"):
            disconnect.set()
            await anyio.sleep_forever()

    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.4"},
        "path": "/ask/stream",
        "method": "POST",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
    }
    with anyio.fail_after(5):
        await app(scope, receive, send)

    assert redis.owner() is None
    if phase in ("headers", "token"):
        assert closed.is_set()
        assert started.is_set()
    else:
        assert not started.is_set()
