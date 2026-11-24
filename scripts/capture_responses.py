"""Save actual /ask JSON and SSE bytes from a running local service, without JWTs."""

import argparse
import json
from pathlib import Path

import httpx

from app.config import settings
from scripts.import_pdfs import _token


def capture(client: httpx.Client, company: str, question: str, top_k: int, output: Path) -> None:
    # Each run gets a new directory; never overwrite earlier observations.
    output.mkdir(parents=True, exist_ok=False)
    payload = {"company": company, "question": question, "top_k": top_k}
    headers = {"Authorization": f"Bearer {_token(settings.jwt_secret.get_secret_value(), company)}"}
    (output / "request.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    response = client.post("/ask", json=payload, headers=headers)
    (output / "ask.json").write_bytes(response.content)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body.get("answer"), str) or not body.get("sources"):
        raise ValueError("/ask did not return an answer with real sources")
    with client.stream("POST", "/ask/stream", json=payload, headers=headers) as response:
        response.raise_for_status()
        with (output / "ask.sse").open("xb") as file:
            for part in response.iter_bytes():
                file.write(part)
                file.flush()
    frames = (output / "ask.sse").read_text(encoding="utf-8").strip().split("\n\n")
    if not frames or not frames[-1].startswith("event: sources\n"):
        raise ValueError("SSE did not end with sources; partial raw recording is preserved")
    sources = json.loads(frames[-1].split("data: ", 1)[1])
    if not sources:
        raise ValueError("SSE returned no real sources")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://web:8000")
    parser.add_argument("--company", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        with httpx.Client(base_url=args.url, timeout=180) as client:
            capture(client, args.company, args.question, args.top_k, args.output)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        parser.exit(1, f"Capture failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
