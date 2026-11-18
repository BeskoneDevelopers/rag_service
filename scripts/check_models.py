"""Smoke-check two standalone llama-server backends from within a Docker container."""

import argparse
import json
from urllib.parse import urlsplit

import httpx

from app.embeddings import _parse_embeddings


def check_http_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid HTTP URL: {url}")


def check_models(
    chat_url: str,
    embedding_url: str,
    *,
    chat_model: str,
    embedding_model: str,
) -> None:
    check_http_url(chat_url)
    check_http_url(embedding_url)
    timeout = httpx.Timeout(120.0, connect=5.0)
    with httpx.Client(timeout=timeout) as client:
        for label, url in (("chat", chat_url), ("embeddings", embedding_url)):
            health_url = url.split("/v1/")[0] + "/health"
            response = client.get(health_url)
            response.raise_for_status()
            print(f"{label} /health: {response.json()}")

        response = client.post(
            embedding_url,
            json={
                "model": embedding_model,
                "input": ["bank annual report", "university course catalog"],
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        vectors = _parse_embeddings(response.json(), expected=2)
        print(f"embedding response: {len(vectors)} vectors, {len(vectors[0])} dimensions")

        response = client.post(
            chat_url,
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "stream": False,
            },
        )
        response.raise_for_status()
        body = response.json()
        choices = body.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            raise ValueError("Chat response has no choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chat response has no text")
        print(f"chat response: {content[:150]!r}")
        print(f"token usage: {json.dumps(body.get('usage', {}), ensure_ascii=False)}")
        print("OK: both models responded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chat-url",
        default="http://host.docker.internal:8081/v1/chat/completions",
    )
    parser.add_argument(
        "--embeddings-url",
        default="http://host.docker.internal:8082/v1/embeddings",
    )
    parser.add_argument("--chat-model", default="qwen3.5-2b")
    parser.add_argument("--embeddings-model", default="bge-m3")
    args = parser.parse_args()
    try:
        check_models(
            args.chat_url,
            args.embeddings_url,
            chat_model=args.chat_model,
            embedding_model=args.embeddings_model,
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Model check failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
