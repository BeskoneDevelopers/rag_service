"""Embedding client for the standalone llama-server running outside Docker."""

import math

import httpx

EMBEDDING_DIMENSIONS = 1024


class EmbeddingServiceError(Exception):
    pass


def _settings():
    # Avoid requiring database settings merely to import the embedding module.
    from app.config import settings

    return settings


def _parse_embeddings(body: object, expected: int) -> list[list[float]]:
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise EmbeddingServiceError("Некорректный ответ embedding-сервера")
    if len(body["data"]) != expected:
        raise EmbeddingServiceError("Количество embeddings не совпадает с запросом")

    vectors: list[list[float] | None] = [None] * expected
    for item in body["data"]:
        if not isinstance(item, dict) or type(item.get("index")) is not int:
            raise EmbeddingServiceError("Неизвестный индекс embedding")
        index = item["index"]
        values = item.get("embedding")
        if not 0 <= index < expected or vectors[index] is not None:
            raise EmbeddingServiceError("Повторный или недопустимый индекс embedding")
        if not isinstance(values, list) or len(values) != EMBEDDING_DIMENSIONS:
            raise EmbeddingServiceError("Неверная размерность embedding")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise EmbeddingServiceError("Embedding содержит некорректные числа")
        vectors[index] = [float(v) for v in values]

    if any(vector is None for vector in vectors):
        raise EmbeddingServiceError("Отсутствует embedding из batch")
    return [vector for vector in vectors if vector is not None]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    cfg = _settings()
    timeout = httpx.Timeout(cfg.embeddings_timeout_seconds, connect=5.0)
    result: list[list[float]] = []
    try:
        with httpx.Client(timeout=timeout) as client:
            for start in range(0, len(texts), cfg.embeddings_batch_size):
                batch = texts[start : start + cfg.embeddings_batch_size]
                response = client.post(
                    cfg.embeddings_url,
                    json={
                        "model": cfg.embeddings_model,
                        "input": batch,
                        "encoding_format": "float",
                    },
                )
                response.raise_for_status()
                result.extend(_parse_embeddings(response.json(), len(batch)))
    except (httpx.HTTPError, ValueError) as exc:
        raise EmbeddingServiceError(
            "Embedding-сервер временно недоступен или ответил с ошибкой"
        ) from exc
    return result
