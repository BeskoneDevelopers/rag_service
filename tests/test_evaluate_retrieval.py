import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from scripts import evaluate_retrieval as script


def test_raw_report_distinguishes_hits_missing_references_and_offtopic(tmp_path, monkeypatch):
    benchmark = tmp_path / "questions.csv"
    benchmark.write_text("fixture", encoding="utf-8")
    questions = [
        ("benchmark", "company", "Fact?", {"doc"}),
        ("benchmark", "company", "Missing?", {"missing"}),
        ("offtopic", "company", "Noise?", set()),
    ]
    session = Mock()
    session.execute.return_value = [("company", "Doc.pdf")]
    monkeypatch.setattr(script, "load_questions", lambda *args: questions)
    monkeypatch.setattr(script, "sessionlocal", lambda: nullcontext(session))
    monkeypatch.setattr(script, "embed_texts", lambda texts: [[0.1]] * len(texts))
    chunk = SimpleNamespace(document=SimpleNamespace(filename="Doc.pdf"))
    monkeypatch.setattr(
        script, "search_chunks", Mock(side_effect=[[(chunk, 0.6)], [], [(chunk, 0.1)]])
    )
    report = tmp_path / "retrieval.json"
    script.evaluate(benchmark, 42, 20, report)
    body = json.loads(report.read_text(encoding="utf-8"))
    assert body["reference_questions"] == 1
    assert body["missing_reference"] == 1
    assert body["hits5"] == body["hits5_filtered"] == 1
    assert body["results"][0]["expected_available"] == ["doc"]
    assert body["results"][1]["score"] is None
    assert body["results"][2]["kind"] == "offtopic"
    assert len(body["csv_sha256"]) == 64
