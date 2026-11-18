import secrets
import time
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi.testclient import TestClient

from app.auth import get_current_user
from app.llm_client import LLMServiceError, LLMTimeoutError, LLMUnavailableError
from app.main import app
from app.routers.documents import get_session

client = TestClient(app)


def test_ask_without_token():
    response = client.post(
        "/ask",
        json={
            "company": "qwq",
            "question": "Как называется дипломный проект?",
            "top_k": 3,
        },
    )

    assert response.status_code == 401


def test_ask_forbidden_company():
    def fake_current_user():
        return {
            "sub": "test_user",
            "company": "qwq",
        }

    app.dependency_overrides[get_current_user] = fake_current_user

    try:
        response = client.post(
            "/ask",
            json={
                "company": "company_b",
                "question": "Какие документы доступны?",
                "top_k": 3,
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 403


def test_ask_invalid_signature(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", secrets.token_urlsafe(32))

    token = jwt.encode(
        {
            "sub": "test_user",
            "company": "qwq",
            "exp": int(time.time()) + 1800,
        },
        secrets.token_urlsafe(32),
        algorithm="HS256",
    )

    response = client.post(
        "/ask",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "company": "qwq",
            "question": "Как называется дипломный проект?",
            "top_k": 3,
        },
    )

    assert response.status_code == 401


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
@pytest.mark.parametrize("expired", [False, True])
def test_signed_token_enforces_company_and_expiry(monkeypatch, path, expired):
    secret = secrets.token_urlsafe(32)
    monkeypatch.setenv("JWT_SECRET", secret)
    token = jwt.encode(
        {
            "sub": "test_user",
            "company": "company_a",
            "exp": int(time.time()) + (-60 if expired else 60),
        },
        secret,
        algorithm="HS256",
    )

    response = client.post(
        path,
        headers={"Authorization": f"Bearer {token}"},
        json={"company": "company_b", "question": "Чужие документы?"},
    )

    assert response.status_code == (401 if expired else 403)


def test_upload_forbidden_company():
    def fake_current_user():
        return {
            "sub": "test_user",
            "company": "qwq",
        }

    app.dependency_overrides[get_current_user] = fake_current_user

    try:
        response = client.post(
            "/documents",
            data={"company": "company_b"},
            files={
                "file": (
                    "test_auth.md",
                    b"Test access control",
                    "text/markdown",
                )
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 403


def test_get_foreign_document():
    def fake_current_user():
        return {
            "sub": "test_user",
            "company": "qwq",
        }

    fake_session = AsyncMock()
    fake_session.scalar.return_value = None

    app.dependency_overrides[get_current_user] = fake_current_user
    app.dependency_overrides[get_session] = lambda: fake_session

    try:
        response = client.get("/documents/13")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_session, None)

    assert response.status_code == 404

    stmt = fake_session.scalar.await_args.args[0]
    assert "company" in str(stmt.whereclause)
    assert "qwq" in stmt.compile().params.values()


def test_ask_llm_unavailable(monkeypatch):
    def fake_current_user():
        return {
            "sub": "test_user",
            "company": "qwq",
        }

    async def fake_ask_rag(company, question, top_k):
        raise LLMUnavailableError("Ollama недоступна")

    app.dependency_overrides[get_current_user] = fake_current_user

    monkeypatch.setattr(
        "app.routers.ask.ask_rag",
        fake_ask_rag,
    )

    try:
        response = client.post(
            "/ask",
            json={
                "company": "qwq",
                "question": "Как называется дипломный проект?",
                "top_k": 3,
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 503


def test_ask_llm_timeout(monkeypatch):
    def fake_current_user():
        return {"sub": "test_user", "company": "qwq"}

    async def fake_ask_rag(company, question, top_k):
        raise LLMTimeoutError("Время ожидания истекло")

    app.dependency_overrides[get_current_user] = fake_current_user
    monkeypatch.setattr("app.routers.ask.ask_rag", fake_ask_rag)

    try:
        response = client.post(
            "/ask",
            json={
                "company": "qwq",
                "question": "Как называется дипломный проект?",
                "top_k": 3,
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 504


def test_ask_llm_service_error(monkeypatch):
    def fake_current_user():
        return {"sub": "test_user", "company": "qwq"}

    async def fake_ask_rag(company, question, top_k):
        raise LLMServiceError("Ошибка сервиса генерации ответа")

    app.dependency_overrides[get_current_user] = fake_current_user
    monkeypatch.setattr("app.routers.ask.ask_rag", fake_ask_rag)

    try:
        response = client.post(
            "/ask",
            json={
                "company": "qwq",
                "question": "Как называется дипломный проект?",
                "top_k": 3,
            },
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 502
