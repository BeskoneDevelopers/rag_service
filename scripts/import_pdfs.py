"""Upload all 236 RAG-Multi-Corpus PDFs to a local development RAG API.

This local-only helper signs a JWT for each dataset company from JWT_SECRET,
which must already be available to the process. It never prints the secret.
"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import httpx
import jwt

EXPECTED_COUNTS = {
    "Aventro Motors": 50,
    "Cendara University": 40,
    "CloudWay-24": 37,
    "Velvera Technologies": 38,
    "ZX Bank": 71,
}


def find_pdfs(dataset: Path) -> list[tuple[str, Path]]:
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    pdfs: list[tuple[str, Path]] = []
    for company in EXPECTED_COUNTS:
        pdf_dir = dataset / company / "pdf"
        if not pdf_dir.is_dir():
            raise ValueError(f"Missing PDF directory: {pdf_dir}")
        pdfs.extend((company, pdf) for pdf in sorted(pdf_dir.glob("*.pdf")))
    return pdfs


def _token(secret: str, company: str) -> str:
    return jwt.encode(
        {
            "sub": "local-corpus-import",
            "company": company,
            "exp": int(time.time()) + 24 * 3600,
        },
        secret,
        algorithm="HS256",
    )


def upload_all(
    client: httpx.Client, pdfs: list[tuple[str, Path]], secret: str
) -> tuple[list[dict], float]:
    started = time.monotonic()
    records: list[dict] = []
    for index, (company, file_path) in enumerate(pdfs, start=1):
        token = _token(secret, company)
        with file_path.open("rb") as file:
            response = client.post(
                "/documents",
                headers={"Authorization": f"Bearer {token}"},
                data={"company": company},
                files={"file": (file_path.name, file, "application/pdf")},
            )
        response.raise_for_status()
        if response.status_code != 202:
            raise ValueError(f"Expected 202 from /documents, got {response.status_code}")
        body = response.json()
        if type(body.get("id")) is not int:
            raise ValueError(f"Missing document ID for {file_path.name}")
        records.append(
            {
                "id": body["id"],
                "company": company,
                "filename": file_path.name,
                "status": body["status"],
                "duplicate": body["duplicate"],
            }
        )
        print(f"[{index}/{len(pdfs)}] {company}: {file_path.name} -> {body['status']}", flush=True)
    return records, time.monotonic() - started


def wait_for_index(
    client: httpx.Client,
    records: list[dict],
    secret: str,
    interval: float,
    timeout_seconds: float,
) -> float:
    started = time.monotonic()
    pending = {(r["company"], r["id"]) for r in records}
    while pending:
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"Timed out with {len(pending)} unfinished documents")
        for company, doc_id in list(pending):
            response = client.get(
                f"/documents/{doc_id}",
                headers={"Authorization": f"Bearer {_token(secret, company)}"},
            )
            response.raise_for_status()
            status = response.json()["status"]
            if status == "failed":
                raise RuntimeError(f"Indexing failed for {company} document ID={doc_id}")
            if status == "completed":
                pending.remove((company, doc_id))
        print(f"Indexed {len(records) - len(pending)}/{len(records)}", flush=True)
        if pending:
            time.sleep(interval)
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to RAG-Multi-Corpus/datasets")
    parser.add_argument("--url", default="http://web:8000", help="URL reachable by importer")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        if args.interval <= 0 or args.timeout <= 0:
            raise ValueError("Interval and timeout must be positive")
        pdfs = find_pdfs(args.dataset)
        counts = Counter(company for company, _ in pdfs)
        print(f"Found {len(pdfs)} PDFs: {dict(counts)}")
        if counts != Counter(EXPECTED_COUNTS):
            raise ValueError("Dataset does not contain the expected 236 PDFs")
        if args.dry_run:
            return

        secret = os.environ.get("JWT_SECRET")
        if not secret:
            raise ValueError("JWT_SECRET must be set in the importing process")
        with httpx.Client(base_url=args.url, timeout=120) as client:
            records, upload_seconds = upload_all(client, pdfs, secret)
            print(f"Uploaded in {upload_seconds:.1f}s")
            indexing_seconds = (
                wait_for_index(client, records, secret, args.interval, args.timeout)
                if args.wait
                else None
            )
        report = {
            "total": len(records),
            "duplicate": sum(bool(r["duplicate"]) for r in records),
            "upload_seconds": round(upload_seconds, 2),
            "wait_seconds": round(indexing_seconds, 2) if indexing_seconds is not None else None,
            "documents": records,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Report written to {args.report}")
        print(
            f"Done: {report['total']} uploads, {report['duplicate']} duplicates. "
            f"Wait: {report['wait_seconds']}s"
        )
    except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError) as exc:
        parser.exit(1, f"Import failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
