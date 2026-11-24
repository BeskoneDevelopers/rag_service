import csv
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from scripts import measure_ttft as ttft


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize("text", ["", " \n", "<thi", "<think>internal", "<think>x</think> \n"])
def test_reasoning_is_not_first_answer_text(text):
    assert not ttft.has_answer_text(text)


@pytest.mark.anyio
async def test_ttft_clock_starts_on_content_after_reasoning(monkeypatch):
    @asynccontextmanager
    async def lock():
        yield

    async def stream(prompt, *, usage, cache_prompt):
        assert cache_prompt is False
        for content in [" ", "<thi", "nk>reasoning", "</think>", "Answer [1]"]:
            yield content
        usage.prompt_tokens = 42
        usage.completion_tokens = 8

    clock = iter([1.0, 1.01, 1.03])
    monkeypatch.setattr(ttft, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(ttft, "llm_generation_lock", lock)
    monkeypatch.setattr(ttft, "stream_answer", stream)
    monkeypatch.setattr(
        ttft,
        "retrieve_context",
        AsyncMock(return_value={"context": "Fact", "sources": [{"number": 1}]}),
    )
    result = await ttft.measure("company", "question", 3)
    assert result == {
        "status": "ok",
        "sources": 1,
        "ttft_ms": 30.0,
        "llm_ttft_ms": 20.0,
        "prompt_tokens": 42,
        "completion_tokens": 8,
    }


@pytest.mark.anyio
async def test_each_top_k_uses_unique_questions_and_preserves_failures(monkeypatch, tmp_path):
    questions = [f"Relevant question {n}?" for n in range(9)]
    measure = AsyncMock(
        side_effect=[
            {"status": "ok"},
            *[{"status": "ok", "ttft_ms": 100.0, "llm_ttft_ms": 50.0} for _ in range(8)],
            RuntimeError("test failure"),
        ]
    )
    monkeypatch.setattr(ttft, "measure", measure)
    output = tmp_path / "ttft.csv"
    rows = await ttft.run("company", questions, "Separate warmup?", output, "test-build")
    calls = [call.args for call in measure.call_args_list]
    assert calls[0] == ("company", "Separate warmup?", 2)
    assert [q for _, q, _ in calls[1:]] == questions
    assert [k for _, _, k in calls[1:]] == [2, 3, 5] * 3
    with output.open(encoding="utf-8", newline="") as file:
        saved = list(csv.DictReader(file))
    assert len(saved) == 9 and saved[-1]["status"] == "error"
    assert saved[-1]["error_type"] == "RuntimeError"
    assert rows[-1]["status"] == "error"
    assert "| 5 | 2 |" in output.with_suffix(".md").read_text(encoding="utf-8")
    metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["cache_prompt"] is False
    with pytest.raises(FileExistsError):
        await ttft.run("company", questions, None, output, "test-build")


@pytest.mark.parametrize(
    "questions,warmup",
    [
        (["q"] * 9, None),
        ([f"q{i}" for i in range(8)], None),
        ([f"q{i}" for i in range(9)], " Q0 "),
        ([f"q{i}" for i in range(8)] + [" "], None),
    ],
)
def test_reused_or_insufficient_questions_are_rejected(questions, warmup):
    with pytest.raises(ValueError):
        ttft.validate_questions(questions, warmup)
