from http.client import IncompleteRead
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from knowledge_vector.http_client import post_json


def test_truncated_cloud_response_is_retried(monkeypatch):
    calls = []
    delays = []

    class Response(BytesIO):
        def read(self, *args):
            if len(calls) == 1:
                raise IncompleteRead(b'{"data":')
            return super().read(*args)

    def open_request(request, timeout):
        calls.append(request)
        return Response(b'{"data": [1]}')

    monkeypatch.setattr("knowledge_vector.http_client.urlopen", open_request)
    monkeypatch.setattr("knowledge_vector.http_client.time.sleep", delays.append)
    assert post_json("https://example.test/embeddings", {"input": ["text"]}) == {"data": [1]}
    assert len(calls) == 2
    assert delays == [1]


def test_transport_retries_are_bounded_and_redact_key(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise URLError("connection lost secret-key")

    monkeypatch.setattr("knowledge_vector.http_client.urlopen", fail)
    monkeypatch.setattr("knowledge_vector.http_client.time.sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="after 3 attempts") as error:
        post_json("https://example.test/embeddings", {}, api_key="secret-key")
    assert len(calls) == 3
    assert "secret-key" not in str(error.value)
    assert "[redacted]" in str(error.value)


def test_http_rejection_is_not_retried(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise HTTPError("https://example.test", 401, "Unauthorized", {}, BytesIO(b"bad key"))

    monkeypatch.setattr("knowledge_vector.http_client.urlopen", fail)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        post_json("https://example.test/embeddings", {})
    assert len(calls) == 1
