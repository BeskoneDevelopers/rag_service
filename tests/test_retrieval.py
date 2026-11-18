import json
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from celery.exceptions import TimeoutError as CeleryTimeoutError
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from kombu.exceptions import OperationalError
from sqlalchemy.dialects import postgresql

from app import rag_pipeline, retrieval, tasks
from app.routers import ask

pytestmark = pytest.mark.anyio
QUESTION = {"company": "tenant-a", "question": "What changed?"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def api(monkeypatch):
    @asynccontextmanager
    async def generation_lock():
        yield

    app = FastAPI()
    app.include_router(ask.router)
    app.dependency_overrides[ask.get_current_user] = lambda: {"company": "tenant-a"}
    for module in (ask, rag_pipeline):
        monkeypatch.setattr(module, "llm_generation_lock", generation_lock)

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def retrieval_result(monkeypatch):
    def chunk(number, content):
        return SimpleNamespace(
            id=number + 100,
            document_id=number + 10,
            document=SimpleNamespace(filename=f"doc-{number}.pdf"),
            page_number=number + 1,
            section="Overview",
            company="tenant-a",
            content=content,
        )

    ranked = [
        (chunk(1, "First fact"), 0.71),
        (chunk(2, "Threshold fact"), retrieval.SIMILARITY_THRESHOLD),
        (chunk(3, "Below threshold"), 0.449),
    ]
    session = Mock()
    monkeypatch.setattr(tasks, "sessionlocal", lambda: nullcontext(session))
    retrieve_chunks = Mock(return_value=ranked)
    monkeypatch.setattr(tasks, "retrieve_chunks", retrieve_chunks)
    result = tasks.retrieve_context_task.run("tenant-a", QUESTION["question"], 3)
    retrieve_chunks.assert_called_once_with(session, "tenant-a", QUESTION["question"], 3)
    return result


def test_retrieval_uses_company_vector_cosine_order_and_top_k(monkeypatch):
    embed = Mock(return_value=[[0.1, 0.2]])
    monkeypatch.setattr(retrieval, "embed_texts", embed)
    session = Mock()
    first = object()
    session.execute.return_value.all.return_value = [(first, 0.25)]

    assert retrieval.retrieve_chunks(session, "tenant-a", "question", 3) == [(first, 0.75)]
    embed.assert_called_once_with(["question"])
    setup, query = [call.args[0] for call in session.execute.call_args_list]
    assert str(setup) == "SET LOCAL hnsw.iterative_scan = strict_order"
    compiled = query.compile(dialect=postgresql.dialect())
    assert "WHERE chunk.company" in str(compiled)
    assert "ORDER BY" in str(compiled) and "<=>" in str(compiled) and "LIMIT" in str(compiled)
    assert list(compiled.params.values()).count([0.1, 0.2]) >= 1
    assert "tenant-a" in compiled.params.values()
    assert 3 in compiled.params.values()


def test_threshold_is_inclusive_and_sources_match_context(retrieval_result):
    assert retrieval_result["sources"] == [
        {
            "number": 1,
            "chunk_id": 101,
            "document_id": 11,
            "filename": "doc-1.pdf",
            "page_number": 2,
            "section": "Overview",
            "company": "tenant-a",
            "score": 0.71,
        },
        {
            "number": 2,
            "chunk_id": 102,
            "document_id": 12,
            "filename": "doc-2.pdf",
            "page_number": 3,
            "section": "Overview",
            "company": "tenant-a",
            "score": 0.45,
        },
    ]
    context = retrieval_result["context"]
    assert "Источник [1]\nДокумент ID: 11" in context
    assert "Источник [2]\nДокумент ID: 12" in context
    assert "Below threshold" not in context


def stub_retrieval(monkeypatch, result):
    for module in (ask, rag_pipeline):
        monkeypatch.setattr(module, "retrieve_context", AsyncMock(return_value=result))


def events(response):
    return [
        (event.removeprefix("event: "), json.loads(data.removeprefix("data: ")))
        for event, data in (frame.splitlines() for frame in response.text.strip().split("\n\n"))
    ]


@pytest.mark.parametrize("stream", [False, True])
async def test_no_relevant_context_returns_fallback_without_llm(api, monkeypatch, stream):
    monkeypatch.setattr(tasks, "sessionlocal", lambda: nullcontext(Mock()))
    monkeypatch.setattr(tasks, "retrieve_chunks", Mock(return_value=[(object(), 0.449)]))
    retrieved = tasks.retrieve_context_task.run("tenant-a", QUESTION["question"], 5)
    assert retrieved == {"context": "", "sources": []}
    stub_retrieval(monkeypatch, retrieved)
    generate = Mock(side_effect=AssertionError("LLM was called without context"))
    monkeypatch.setattr(ask, "stream_answer", generate)
    monkeypatch.setattr(rag_pipeline, "generate_answer", generate)

    async with api:
        response = await api.post("/ask/stream" if stream else "/ask", json=QUESTION)

    assert response.status_code == 200
    fallback = "Не знаю: в документах этого нет"
    if stream:
        assert events(response) == [("token", {"text": fallback}), ("sources", [])]
    else:
        assert response.json() == {"answer": fallback, "sources": []}
    generate.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("citation", ["2", "99"])
async def test_answer_citations_are_checked_against_sources(
    api, monkeypatch, retrieval_result, stream, citation
):
    stub_retrieval(monkeypatch, retrieval_result)
    answer = f"Facts [1][{citation}]"

    async def generate(prompt):
        assert "Источник [1]" in prompt and "Источник [2]" in prompt
        return answer

    async def stream_answer(prompt):
        text = await generate(prompt)
        yield text[:10]
        yield text[10:]

    monkeypatch.setattr(rag_pipeline, "generate_answer", generate)
    monkeypatch.setattr(ask, "stream_answer", stream_answer)
    async with api:
        response = await api.post("/ask/stream" if stream else "/ask", json=QUESTION)

    if stream:
        assert response.status_code == 200
        chunks = events(response)
        assert "".join(data["text"] for name, data in chunks if name == "token") == answer
        if citation == "2":
            assert chunks[-1] == ("sources", retrieval_result["sources"])
        else:
            assert [name for name, _ in chunks] == ["token", "token", "error"]
    elif citation == "2":
        assert response.status_code == 200
        assert response.json() == {"answer": answer, "sources": retrieval_result["sources"]}
    else:
        assert response.status_code == 502
        assert "неизвестный источник" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
@pytest.mark.parametrize(
    ("failure", "status"),
    [
        (OperationalError("broker down"), 503),
        (CeleryTimeoutError("wait"), 504),
        (ValueError("internal worker details"), 502),
    ],
)
async def test_celery_failure_becomes_http_error_before_sse(
    api, monkeypatch, path, failure, status
):
    task = Mock()
    task.get.side_effect = failure
    send = Mock(return_value=task)
    if status == 503:
        send.side_effect = failure
    monkeypatch.setattr(rag_pipeline.celery_app, "send_task", send)

    async with api:
        response = await api.post(path, json=QUESTION)

    assert response.status_code == status
    assert "detail" in response.json()
    assert str(failure) not in response.text
