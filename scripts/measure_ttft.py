import argparse
import asyncio
import csv
import json
import platform
import re
from contextlib import aclosing
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

from app.config import settings
from app.llm_client import GenerationUsage, model_name, stream_answer
from app.llm_lock import llm_generation_lock
from app.prompts import build_prompt, has_invalid_citation
from app.rag_pipeline import retrieve_context

TOP_K = (2, 3, 5)


def has_answer_text(text: str) -> bool:
    # Ignore both complete reasoning blocks and tags split across SSE deltas.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    for length in range(1, len("<think>")):
        if text.endswith("<think>"[:length]):
            text = text[:-length]
            break
    return bool(text.strip())


async def measure(company: str, question: str, top_k: int) -> dict:
    started = perf_counter()
    retrieved = await retrieve_context(company, question, top_k)
    source_count = len(retrieved["sources"])
    if not retrieved["context"]:
        return {"status": "no_context", "sources": 0, "prompt_tokens": 0, "completion_tokens": 0}

    prompt = build_prompt(question, retrieved["context"])
    usage = GenerationUsage()
    first_token_at = None
    answer_parts = []
    async with llm_generation_lock():
        generation_started = perf_counter()
        async with aclosing(stream_answer(prompt, usage=usage, cache_prompt=False)) as tokens:
            async for token in tokens:
                answer_parts.append(token)
                if first_token_at is None and has_answer_text("".join(answer_parts)):
                    first_token_at = perf_counter()

    status = "ok"
    if first_token_at is None:
        status = "no_token"
    elif has_invalid_citation("".join(answer_parts), source_count):
        status = "invalid_citation"
    elif usage.prompt_tokens is None or usage.completion_tokens is None:
        status = "missing_usage"

    return {
        "status": status,
        "sources": source_count,
        "ttft_ms": round((first_token_at - started) * 1000, 1) if first_token_at else None,
        "llm_ttft_ms": round((first_token_at - generation_started) * 1000, 1)
        if first_token_at
        else None,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
    }


def validate_questions(questions: list[str], warmup_question: str | None) -> None:
    normalized = [" ".join(q.split()).casefold() for q in questions]
    if len(questions) < 9 or len(questions) % len(TOP_K):
        raise ValueError("Supply 9 or more unique questions, a multiple of 3")
    if any(not q for q in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("Every measured question must be non-empty and unique")
    if warmup_question is not None:
        warmup = " ".join(warmup_question.split()).casefold()
        if not warmup or warmup in normalized:
            raise ValueError("The warmup question must be separate from measured questions")


def write_summary(rows: list[dict], path: Path) -> None:
    lines = [
        "| top_k | Успешных замеров | Медиана TTFT, мс | Медиана LLM TTFT, мс |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for top_k in TOP_K:
        valid = [r for r in rows if r["top_k"] == top_k and r["status"] == "ok"]
        if valid:
            lines.append(
                f"| {top_k} | {len(valid)} | {median(r['ttft_ms'] for r in valid):.1f} | "
                f"{median(r['llm_ttft_ms'] for r in valid):.1f} |"
            )
        else:
            lines.append(f"| {top_k} | 0 | — | — |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(
    company: str,
    questions: list[str],
    warmup_question: str | None,
    output: Path,
    server_version: str,
) -> list[dict]:
    validate_questions(questions, warmup_question)
    if model_name() != "qwen3.5-2b" or settings.embeddings_model != "bge-m3":
        raise ValueError("Final benchmark requires qwen3.5-2b and bge-m3 aliases")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as file:
        return await _run(company, questions, warmup_question, file, output, server_version)


async def _run(company, questions, warmup_question, file, output, server_version):
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "llama_server_version": server_version,
        "cache_prompt": False,
        "llm_model": model_name(),
        "embeddings_model": settings.embeddings_model,
        "max_tokens": settings.llm_max_tokens,
        "generation_timeout_seconds": settings.llm_generation_timeout_seconds,
        "warmup_question": warmup_question,
        "question_assignment": "round-robin 2,3,5; each question used exactly once",
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "model",
            "company",
            "question",
            "top_k",
            "status",
            "sources",
            "ttft_ms",
            "llm_ttft_ms",
            "prompt_tokens",
            "completion_tokens",
            "error_type",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    file.flush()
    if warmup_question:
        warmup_result = await measure(company, warmup_question, TOP_K[0])
        metadata["warmup_result"] = warmup_result
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if warmup_result["status"] != "ok":
            raise ValueError(f"Warmup failed: {warmup_result['status']}")
    rows = []
    for index, question in enumerate(questions):
        top_k = TOP_K[index % len(TOP_K)]
        try:
            result = await measure(company, question, top_k)
        except Exception as exc:
            result = {"status": "error", "error_type": type(exc).__name__}
        row = {
            "model": model_name(),
            "company": company,
            "question": question,
            "top_k": top_k,
            **result,
        }
        writer.writerow(row)
        file.flush()
        rows.append(row)
    write_summary(rows, output.with_suffix(".md"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure unique RAG queries on llama-server")
    parser.add_argument("--company", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", action="append")
    source.add_argument(
        "--questions-file", type=Path, help="JSON array of unique relevant questions"
    )
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--warmup-question")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--server-version", required=True, help="Actual llama-server --version output"
    )
    args = parser.parse_args()
    if args.warmup and not args.warmup_question:
        parser.error("--warmup requires a separate --warmup-question")
    try:
        questions = (
            json.loads(args.questions_file.read_text(encoding="utf-8"))
            if args.questions_file
            else args.question
        )
        if not isinstance(questions, list) or any(not isinstance(q, str) for q in questions):
            raise ValueError("Questions must be a JSON array of strings")
        rows = asyncio.run(
            run(args.company, questions, args.warmup_question, args.output, args.server_version)
        )
        if any(row["status"] != "ok" for row in rows):
            parser.exit(1, f"Some measurements failed; see {args.output}\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")


if __name__ == "__main__":
    main()
