from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def get_model() -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = get_model()
    vector = model.encode(texts, batch_size=8).tolist()
    return vector
