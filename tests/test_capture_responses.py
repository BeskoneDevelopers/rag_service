import json

import httpx
import pytest

from scripts.capture_responses import capture


@pytest.mark.parametrize("terminal", ["sources", "error"])
def test_capture_preserves_actual_bytes_and_requires_terminal_sources(tmp_path, terminal):
    # Synthetic transport fixtures validate the recorder, not the model or corpus.
    answer = b'{"answer":"Fact [1]","sources":[{"document_id":7}]}'
    sse = (
        f'event: token\ndata: {{"text":"Fact [1]"}}\n\n'
        f'event: {terminal}\ndata: [{{"document_id":7}}]\n\n'
    ).encode()

    def handle(request):
        assert request.headers["authorization"].startswith("Bearer ")
        return httpx.Response(200, content=sse if request.url.path.endswith("stream") else answer)

    output = tmp_path / "recording"
    with httpx.Client(transport=httpx.MockTransport(handle), base_url="http://test") as client:
        if terminal == "error":
            with pytest.raises(ValueError, match="did not end with sources"):
                capture(client, "company", "question", 5, output)
        else:
            capture(client, "company", "question", 5, output)
    assert (output / "ask.json").read_bytes() == answer
    assert (output / "ask.sse").read_bytes() == sse
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    assert request == {"company": "company", "question": "question", "top_k": 5}
    assert "Bearer" not in "".join(p.read_text(encoding="utf-8") for p in output.iterdir())
