from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from knowledge_vector.config import EmbeddingConfig

_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_ERROR_BYTES = 4096


class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedding:
    """Deterministic local embedding for pipeline smoke tests, not semantic search."""

    def __init__(self, dimension: int = 384, model: str = "hashing-v1") -> None:
        self.dimension = dimension
        self.name = model

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed_query(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = _tokens(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if value & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector


class SentenceTransformerEmbedding:
    def __init__(self, model: str, dimension: int | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -e '.[semantic]'"
            ) from exc
        self._model = SentenceTransformer(model)
        inferred_dimension = self._model.get_sentence_embedding_dimension()
        self.dimension = int(dimension or inferred_dimension or 0)
        self.name = model
        if self.dimension <= 0:
            raise RuntimeError(f"Could not determine embedding dimension for {model}")

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.asarray(
            self._model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class OpenAICompatibleEmbedding:
    """Embedding client for services exposing an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        dimension: int,
        timeout_s: float = 60.0,
        batch_size: int = 32,
    ) -> None:
        api_url = api_url.strip()
        model = model.strip()
        if not api_url:
            raise ValueError(
                "embedding.api_url must be configured when provider is openai-compatible"
            )
        parsed_url = urlparse(api_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("embedding.api_url must be an absolute http or https URL")
        if parsed_url.username or parsed_url.password:
            raise ValueError("embedding.api_url must not contain credentials")
        if not model:
            raise ValueError("embedding.model must be configured for openai-compatible provider")
        if dimension <= 0:
            raise ValueError("embedding.dimension must be positive")
        if timeout_s <= 0:
            raise ValueError("embedding.timeout_s must be positive")
        if batch_size <= 0:
            raise ValueError("embedding.batch_size must be positive")

        self.api_url = api_url
        self.api_key = api_key.strip()
        self.name = model
        self.dimension = dimension
        self.timeout_s = timeout_s
        self.batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches = [
            values[start : start + self.batch_size]
            for start in range(0, len(values), self.batch_size)
        ]
        return np.vstack([self._embed_batch(batch) for batch in batches])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: Sequence[str]) -> np.ndarray:
        payload = json.dumps(
            {"model": self.name, "input": list(texts)},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.api_url, data=payload, headers=headers, method="POST")

        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            detail = _read_error_detail(exc, secret=self.api_key)
            raise RuntimeError(
                f"Embedding API returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Embedding API request failed: {exc}") from exc

        if len(raw) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("Embedding API response exceeded the 16 MiB safety limit")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Embedding API returned invalid JSON") from exc
        return _parse_embedding_response(
            decoded,
            expected_count=len(texts),
            dimension=self.dimension,
        )


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider == "hashing":
        return HashingEmbedding(dimension=config.dimension, model=config.model)
    if config.provider == "sentence-transformers":
        return SentenceTransformerEmbedding(model=config.model, dimension=config.dimension)
    if config.provider in {"api", "openai-compatible"}:
        return OpenAICompatibleEmbedding(
            api_url=config.api_url,
            api_key=config.api_key,
            model=config.model,
            dimension=config.dimension,
            timeout_s=config.timeout_s,
            batch_size=config.batch_size,
        )
    raise ValueError(f"Unknown embedding provider: {config.provider}")


def _parse_embedding_response(payload: Any, *, expected_count: int, dimension: int) -> np.ndarray:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("Embedding API response must contain a data list")

    indexed_vectors: list[tuple[int, np.ndarray]] = []
    for position, item in enumerate(payload["data"]):
        if not isinstance(item, dict):
            raise RuntimeError("Embedding API data items must be objects")
        raw_index = item.get("index", position)
        if not isinstance(raw_index, int):
            raise RuntimeError("Embedding API data item index must be an integer")
        raw_vector = item.get("embedding")
        if not isinstance(raw_vector, list):
            raise RuntimeError("Embedding API data item must contain an embedding list")
        vector = np.asarray(raw_vector, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != dimension:
            actual_dimension = vector.shape[0] if vector.ndim == 1 else "invalid"
            raise RuntimeError(
                f"Embedding API returned dimension {actual_dimension}; expected {dimension}"
            )
        if not np.all(np.isfinite(vector)):
            raise RuntimeError("Embedding API returned a non-finite vector")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise RuntimeError("Embedding API returned a zero-length vector")
        indexed_vectors.append((raw_index, vector / norm))

    indices = [index for index, _ in indexed_vectors]
    if len(indexed_vectors) != expected_count or sorted(indices) != list(range(expected_count)):
        raise RuntimeError(
            f"Embedding API returned {len(indexed_vectors)} vectors for {expected_count} inputs"
        )
    indexed_vectors.sort(key=lambda item: item[0])
    return np.vstack([vector for _, vector in indexed_vectors]).astype(np.float32)


def _read_error_detail(exc: HTTPError, *, secret: str = "") -> str:
    try:
        raw = exc.read(_MAX_ERROR_BYTES)
    except OSError:
        return exc.reason or "request failed"
    text = raw.decode("utf-8", errors="replace").strip()
    detail = text or str(exc.reason or "request failed")
    return detail.replace(secret, "[redacted]") if secret else detail


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9_]+", lowered)
    cjk_runs = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", lowered)
    cjk: list[str] = []
    for run in cjk_runs:
        cjk.extend(run)
        cjk.extend(run[index : index + 2] for index in range(max(0, len(run) - 1)))
        cjk.extend(run[index : index + 3] for index in range(max(0, len(run) - 2)))
    return latin + cjk
