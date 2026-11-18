import os

import pytest

from app import embeddings

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/test")
os.environ.setdefault("DATABASE_SYNC_URL", "postgresql+psycopg://localhost/test")
os.environ.setdefault("DEBUG", "false")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


@pytest.fixture(autouse=True)
def forbid_embedding_model(monkeypatch):
    def fail_if_loaded():
        pytest.fail("Tests must stub embeddings")

    monkeypatch.setattr(embeddings, "get_model", fail_if_loaded)
