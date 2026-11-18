from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from kombu.exceptions import OperationalError
from sqlalchemy.dialects import postgresql

from app import tasks
from app.routers import documents


@pytest.fixture
def uploads(monkeypatch, tmp_path):
    session = AsyncMock()
    enqueue = Mock()
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(documents.process_document, "apply_async", enqueue)
    app = FastAPI()
    app.include_router(documents.router)
    app.dependency_overrides[documents.get_session] = lambda: session
    app.dependency_overrides[documents.get_current_user] = lambda: {"company": "tenant-a"}
    with TestClient(app) as client:
        yield client, session, enqueue


def upload(client, name="notes.md"):
    return client.post(
        "/documents",
        data={"company": "tenant-a"},
        files={"file": (name, b"# Notes\n\nSample.", "text/markdown")},
    )


def test_duplicate_is_returned_and_extra_upload_removed(uploads, tmp_path):
    client, session, enqueue = uploads
    session.scalar.side_effect = [123, None]
    session.execute.return_value = Mock(
        one=Mock(return_value=SimpleNamespace(id=123, status="pending"))
    )

    first = upload(client)
    duplicate = upload(client, "different-name.md")

    assert first.status_code == duplicate.status_code == 202
    assert first.json() == {"id": 123, "status": "pending", "duplicate": False}
    assert duplicate.json() == {"id": 123, "status": "pending", "duplicate": True}
    assert len(list(tmp_path.iterdir())) == 1
    enqueue.assert_called_once_with(args=(123,), retry=False)
    inserts = [
        str(call.args[0].compile(dialect=postgresql.dialect()))
        for call in session.scalar.call_args_list
    ]
    assert all(
        "ON CONFLICT ON CONSTRAINT uq_document_company_hash DO NOTHING" in sql for sql in inserts
    )


def test_enqueue_failure_can_retry_same_document(uploads, tmp_path):
    client, session, enqueue = uploads
    session.scalar.side_effect = [123, None, 123]
    session.execute.return_value = Mock(
        one=Mock(return_value=SimpleNamespace(id=123, status="failed"))
    )
    enqueue.side_effect = [OperationalError("redis unavailable"), None]

    failed = upload(client)
    retried = upload(client)

    assert failed.status_code == 503
    assert retried.status_code == 202
    assert retried.json() == {"id": 123, "status": "pending", "duplicate": True}
    assert len(list(tmp_path.iterdir())) == 1
    assert [call.kwargs["args"] for call in enqueue.call_args_list] == [(123,), (123,)]


def test_indexing_error_rolls_back_and_marks_failed(monkeypatch):
    document = SimpleNamespace(id=123, status="pending", file_path="bad.pdf")
    session = Mock()
    session.get.return_value = document
    monkeypatch.setattr(tasks, "sessionlocal", lambda: nullcontext(session))
    monkeypatch.setattr(tasks, "parser_document", Mock(side_effect=ValueError("bad PDF")))

    with pytest.raises(ValueError, match="bad PDF"):
        tasks.process_document.run(123)

    assert document.status == "failed"
    session.rollback.assert_called_once()
    assert session.commit.call_count == 2
    session.add.assert_not_called()


def test_repeated_celery_delivery_skips_completed_document(monkeypatch):
    session = Mock()
    session.get.return_value = SimpleNamespace(status="completed")
    monkeypatch.setattr(tasks, "sessionlocal", lambda: nullcontext(session))
    parse = Mock(side_effect=AssertionError("Completed document was indexed twice"))
    monkeypatch.setattr(tasks, "parser_document", parse)

    assert tasks.process_document.run(123) == {"processed": 123}
    parse.assert_not_called()
