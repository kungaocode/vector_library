import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import numpy as np
import pytest

from knowledge_vector.config import EmbeddingConfig
from knowledge_vector.embeddings import create_embedding_provider


class _EmbeddingHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(
            {
                "authorization": self.headers.get("Authorization"),
                "payload": payload,
            }
        )
        data = [
            {"index": index, "embedding": [float(len(text)), 1.0, 0.0]}
            for index, text in enumerate(payload["input"])
        ]
        response = json.dumps({"data": data}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.mark.parametrize("send_dimensions", [False, True])
def test_openai_compatible_embedding_uses_configured_endpoint(send_dimensions: bool) -> None:
    _EmbeddingHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _EmbeddingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        provider = create_embedding_provider(
            EmbeddingConfig(
                provider="openai-compatible",
                model="test-embedding",
                dimension=3,
                api_url=f"http://127.0.0.1:{server.server_port}/v1/embeddings",
                api_key="test-key",
                timeout_s=5,
                batch_size=2,
                send_dimensions=send_dimensions,
            )
        )
        vectors = provider.embed_documents(["alpha", "beta", "gamma"])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert vectors.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert len(_EmbeddingHandler.requests) == 2
    first_request, second_request = _EmbeddingHandler.requests
    assert first_request["authorization"] == "Bearer test-key"
    dimension_fields = {"dimensions": 3, "encoding_format": "float"} if send_dimensions else {}
    assert first_request["payload"] == {
        "model": "test-embedding",
        "input": ["alpha", "beta"],
        **dimension_fields,
    }
    assert second_request["payload"] == {
        "model": "test-embedding",
        "input": ["gamma"],
        **dimension_fields,
    }


def test_openai_compatible_provider_requires_api_url() -> None:
    with pytest.raises(ValueError, match="api_url"):
        create_embedding_provider(
            EmbeddingConfig(provider="openai-compatible", model="test-embedding")
        )
