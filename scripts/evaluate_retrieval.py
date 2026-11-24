import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

from sqlalchemy import select

from app.embeddings import embed_texts
from app.models import Document
from app.retrieval import SIMILARITY_THRESHOLD, search_chunks
from app.worker_db import sessionlocal

THRESHOLDS = sorted({0.35, 0.40, SIMILARITY_THRESHOLD, 0.50})


def normalize_filename(filename: str) -> str:
    return Path(filename).stem.strip().casefold()


OFFTOPIC_QUESTIONS = [
    "Who painted The Starry Night?",
    "How do I bake chocolate brownies?",
    "How many moons does Neptune have?",
    "Who won the 1986 FIFA World Cup?",
    "What is the capital of Iceland?",
]


def load_questions(path: Path, seed: int = 42, questions_per_company: int = 20):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"Enterprise Name", "Query", "Supporting Facts"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        rows = list(reader)

    groups = defaultdict(list)

    for row in rows:
        company = row["Enterprise Name"]
        question = row["Query"].strip()

        if not question:
            continue

        facts = json.loads(row["Supporting Facts"] or "[]")

        expected_files = {
            normalize_filename(fact["filename"])
            for fact in facts
            if isinstance(fact, dict) and fact.get("filename")
        }

        groups[company].append((question, expected_files))

    rng = random.Random(seed)
    questions = []

    for company, company_questions in groups.items():
        selected = rng.sample(
            company_questions,
            min(questions_per_company, len(company_questions)),
        )

        for question, expected_files in selected:
            questions.append(("benchmark", company, question, expected_files))

    companies = sorted(set(groups) | {"CloudWay-24"})

    for company in companies:
        for question in OFFTOPIC_QUESTIONS:
            questions.append(("offtopic", company, question, set()))

    return questions


def evaluate(path: Path, seed: int, questions_per_company: int, report_path: Path | None = None):
    questions = load_questions(path, seed, questions_per_company)
    with path.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()

    print(f"CSV SHA-256: {digest}")
    print(f"Sampling: seed={seed}, up to {questions_per_company} questions per company")
    print(f"Total questions: {len(questions)}")
    print("Generating embeddings...")

    vectors = embed_texts([question for _, _, question, _ in questions])

    results = []

    with sessionlocal() as session:
        available_docs = defaultdict(set)

        for company, filename in session.execute(
            select(Document.company, Document.filename).where(Document.status == "completed")
        ):
            available_docs[company].add(normalize_filename(filename))

        for (kind, company, question, expected), vector in zip(questions, vectors, strict=True):
            chunks = search_chunks(
                session,
                company,
                vector,
                top_k=5,
            )

            score = float(chunks[0][1]) if chunks else None

            found_files = {normalize_filename(chunk.document.filename) for chunk, _ in chunks}

            filtered_files = {
                normalize_filename(chunk.document.filename)
                for chunk, similarity in chunks
                if similarity >= SIMILARITY_THRESHOLD
            }

            expected_available = expected & available_docs[company]

            results.append(
                {
                    "kind": kind,
                    "company": company,
                    "question": question,
                    "score": score,
                    "expected": expected,
                    "expected_available": expected_available,
                    "found": found_files,
                    "hit5": bool(expected_available & found_files),
                    "hit5_filtered": bool(expected_available & filtered_files),
                }
            )

    print("\n=== SIMILARITY SUMMARY ===")

    for kind in ["benchmark", "offtopic"]:
        values = [
            item["score"] for item in results if item["kind"] == kind and item["score"] is not None
        ]

        missing = sum(item["kind"] == kind and item["score"] is None for item in results)

        print(f"\n{kind.upper()}")
        print(f"Count: {len(values)}")
        print(f"No results: {missing}")

        if values:
            print(f"Min: {min(values):.3f}")
            print(f"Median: {median(values):.3f}")
            print(f"Max: {max(values):.3f}")

    print("\n=== THRESHOLD COMPARISON ===")
    print("Threshold | Rejected benchmark | Accepted offtopic")

    for threshold in THRESHOLDS:
        rejected_benchmark = sum(
            item["kind"] == "benchmark" and (item["score"] is None or item["score"] < threshold)
            for item in results
        )

        accepted_offtopic = sum(
            item["kind"] == "offtopic" and item["score"] is not None and item["score"] >= threshold
            for item in results
        )

        print(f"{threshold:9.2f} | {rejected_benchmark:18} | {accepted_offtopic:17}")

    print("\n=== LOWEST BENCHMARK SCORES ===")

    benchmark = sorted(
        (item for item in results if item["kind"] == "benchmark"),
        key=lambda item: item["score"] if item["score"] is not None else -1,
    )

    for item in benchmark[:10]:
        print(f"{str(item['score']):18} | {item['company']:21} | {item['question'][:75]}")

    print("\n=== HIGHEST OFFTOPIC SCORES ===")

    offtopic = sorted(
        (item for item in results if item["kind"] == "offtopic"),
        key=lambda item: item["score"] if item["score"] is not None else -1,
        reverse=True,
    )

    for item in offtopic[:10]:
        print(f"{str(item['score']):18} | {item['company']:21} | {item['question'][:75]}")

    print("\n=== RETRIEVAL HIT@5 ===")

    benchmark = [item for item in results if item["kind"] == "benchmark"]

    valid = [item for item in benchmark if item["expected_available"]]

    missing_reference = len(benchmark) - len(valid)

    hits = sum(item["hit5"] for item in valid)
    filtered_hits = sum(item["hit5_filtered"] for item in valid)

    print(f"Benchmark questions: {len(benchmark)}")
    print(f"Questions with reference PDF: {len(valid)}")
    print(f"Missing reference PDF: {missing_reference}")

    if valid:
        print(f"Hit@5: {hits}/{len(valid)} = {hits / len(valid):.1%}")
        print(
            f"Hit@5 with threshold {SIMILARITY_THRESHOLD:.2f}: "
            f"{filtered_hits}/{len(valid)} = {filtered_hits / len(valid):.1%}"
        )

    print("\n=== MISSED QUESTIONS ===")

    missed = [item for item in valid if not item["hit5_filtered"]]

    for item in missed[:10]:
        print(f"\nCompany: {item['company']}")
        print(f"Question: {item['question']}")
        print(f"Expected: {sorted(item['expected_available'])}")
        print(f"Found: {sorted(item['found'])}")

    if report_path:
        report = {
            "csv_sha256": digest,
            "seed": seed,
            "questions_per_company": questions_per_company,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "reference_questions": len(valid),
            "missing_reference": missing_reference,
            "hits5": hits,
            "hits5_filtered": filtered_hits,
            "results": [
                {k: sorted(v) if isinstance(v, set) else v for k, v in item.items()}
                for item in results
            ],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval on an indexed RAG dataset")
    parser.add_argument("--benchmark", type=Path, default=Path("/tmp/rag_benchmark.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--questions-per-company", type=int, default=20)
    parser.add_argument("--report", type=Path, help="Save actual per-question retrieval results")
    args = parser.parse_args()
    if args.questions_per_company < 1:
        parser.error("--questions-per-company must be positive")
    try:
        evaluate(args.benchmark, args.seed, args.questions_per_company, args.report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
