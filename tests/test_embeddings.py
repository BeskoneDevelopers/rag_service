import json
from types import SimpleNamespace

import httpx
import pytest

from app import embeddings


@pytest.fixture(autouse=True)
def configuration(monkeypatch):
    monkeypatch.setattr(
        embeddings,
        "_settings",
        lambda: SimpleNamespace(
            embeddings_url="http://test/v1/embeddings",
            embeddings_model="bge-m3",
            embeddings_batch_size=2,
            embeddings_timeout_seconds=30,
        ),
    )


def test_batch_order_and_batch_size(monkeypatch):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        data = [
            {"index": i, "embedding": [float(len(text))] * 1024}
            for i, text in enumerate(body["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client
    monkeypatch.setattr(
        embeddings.httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    vectors = embeddings.embed_texts(["a", "hello", "xyz"])
    assert [vector[0] for vector in vectors] == [1.0, 5.0, 3.0]
    assert [request["input"] for request in requests] == [["a", "hello"], ["xyz"]]
    assert all(request["model"] == "bge-m3" for request in requests)


@pytest.mark.parametrize(
    "data",
    [
        [{"index": 0, "embedding": [1.0] * 10}],
        [{"index": 8, "embedding": [1.0] * 1024}],
        [{"index": 0, "embedding": [float("nan")] * 1024}],
        [],
    ],
)
def test_invalid_embeddings_are_rejected(data):
    with pytest.raises(embeddings.EmbeddingServiceError):
        embeddings._parse_embeddings({"data": data}, 1)


def test_empty_input():
    assert embeddings.embed_texts([]) == []
