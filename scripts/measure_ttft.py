import argparse
import asyncio
import csv
import sys
from contextlib import aclosing
from time import perf_counter

from app.llm_client import OLLAMA_MODEL, GenerationUsage, stream_answer
from app.llm_lock import llm_generation_lock
from app.prompts import build_prompt, has_invalid_citation
from app.rag_pipeline import retrieve_context

TOP_K = (2, 3, 5)


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
        async with aclosing(stream_answer(prompt, usage=usage)) as tokens:
            async for token in tokens:
                if first_token_at is None:
                    first_token_at = perf_counter()
                answer_parts.append(token)

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


async def run(company: str, questions: list[str], warmup: bool) -> None:
    if warmup:
        await measure(company, questions[0], TOP_K[0])
    writer = csv.DictWriter(
        sys.stdout,
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
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for question in questions:
        for top_k in TOP_K:
            result = await measure(company, question, top_k)
            writer.writerow(
                {
                    "model": OLLAMA_MODEL,
                    "company": company,
                    "question": question,
                    "top_k": top_k,
                    **result,
                }
            )
            sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure RAG TTFT and actual Ollama token counts")
    parser.add_argument("--company", required=True)
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.company, args.question, args.warmup))
