from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import reindex_documents as script


@pytest.mark.parametrize("apply", [False, True])
def test_reindex_selects_company_and_only_inactive_docx_and_markdown(monkeypatch, apply):
    session = Mock()
    session.scalars.return_value = [1, 2, 3]
    docs = [
        SimpleNamespace(id=1, filename="Awards & Recognitions.docx", status="completed"),
        SimpleNamespace(id=2, filename="page.pdf", status="completed"),
        SimpleNamespace(id=3, filename="notes.md", status="processing"),
    ]
    session.get.side_effect = docs
    enqueue = Mock(return_value=SimpleNamespace(id="task-id"))
    monkeypatch.setattr(script, "sessionlocal", lambda: nullcontext(session))
    monkeypatch.setattr(script.process_document, "apply_async", enqueue)
    results = script.reindex("company", None, apply)
    query = session.scalars.call_args.args[0]
    assert "company" in query.compile().params.values()
    assert len(results) == 1 and results[0]["id"] == 1
    if apply:
        enqueue.assert_called_once_with(args=(1,), retry=False)
        assert docs[0].status == "pending"
    else:
        enqueue.assert_not_called()
        session.commit.assert_not_called()


def test_failed_reindex_enqueue_can_be_retried(monkeypatch):
    session = Mock()
    session.scalars.return_value = [1]
    doc = SimpleNamespace(id=1, filename="notes.md", status="completed")
    session.get.return_value = doc
    monkeypatch.setattr(script, "sessionlocal", lambda: nullcontext(session))
    monkeypatch.setattr(
        script.process_document, "apply_async", Mock(side_effect=RuntimeError("down"))
    )
    with pytest.raises(RuntimeError, match="down"):
        script.reindex("company", "notes.md", True)
    assert doc.status == "failed"
