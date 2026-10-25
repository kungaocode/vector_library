from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from knowledge_vector.config import RetrievalConfig
from knowledge_vector.embeddings import EmbeddingProvider
from knowledge_vector.models import SearchResult
from knowledge_vector.store import SQLiteStore


class SearchEngine:
    def __init__(
        self,
        store: SQLiteStore,
        embedder: EmbeddingProvider | None,
        config: RetrievalConfig,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: str | None = None,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        selected_mode = mode or self.config.mode
        selected_top_k = top_k or self.config.top_k
        candidate_k = max(self.config.candidate_k, selected_top_k)

        lexical: list[dict[str, Any]] = []
        vector: list[dict[str, Any]] = []
        if selected_mode in {"hybrid", "lexical"}:
            lexical = self.store.lexical_candidates(query, candidate_k)
        if selected_mode in {"hybrid", "vector"}:
            if self.embedder is None:
                raise ValueError("An embedding provider is required for vector retrieval")
            vector = self.store.vector_candidates(
                self.embedder.embed_query(query),
                candidate_k,
                embedding_model=self.embedder.name,
            )

        if selected_mode == "lexical":
            ranked = _score_single_source(lexical, "lexical_score", "lexical")
        elif selected_mode == "vector":
            ranked = _score_single_source(vector, "vector_score", "vector")
        else:
            ranked = _fuse(lexical, vector, self.config.lexical_weight)

        top = ranked[:selected_top_k]
        chunk_map = self.store.get_chunks([item["chunk_id"] for item in top])
        results: list[SearchResult] = []
        for item in top:
            row = chunk_map.get(item["chunk_id"], {})
            if not row:
                continue
            metadata: dict[str, Any] = {}
            for key in ("metadata_json", "document_metadata_json"):
                raw = row.get(key)
                if isinstance(raw, str) and raw:
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        metadata.update(value)
            results.append(
                SearchResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    text=str(row["text"]),
                    title=str(row["title"]),
                    path=str(row["path"]),
                    source_type=str(row["source_type"]),
                    score=float(item["score"]),
                    vector_score=_optional_float(item.get("vector_score")),
                    lexical_score=_optional_float(item.get("lexical_score")),
                    retrieval_mode=str(item["retrieval_mode"]),
                    heading=str(row["heading"]) if row.get("heading") else None,
                    page_number=int(row["page_number"]) if row.get("page_number") else None,
                    metadata=metadata,
                )
            )
        return results


def _score_single_source(
    rows: Sequence[dict[str, Any]],
    score_key: str,
    mode: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    raw_scores = [float(row["score"]) for row in rows]
    normalized = _normalize(raw_scores)
    ranked = []
    for row, raw_score, score in zip(rows, raw_scores, normalized, strict=True):
        ranked.append(
            {
                "chunk_id": row["chunk_id"],
                score_key: raw_score,
                "score": score,
                "retrieval_mode": mode,
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _fuse(
    lexical: Sequence[dict[str, Any]],
    vector: Sequence[dict[str, Any]],
    lexical_weight: float,
) -> list[dict[str, Any]]:
    lexical_norm = _normalized_map(lexical)
    vector_norm = _normalized_map(vector)
    ids = set(lexical_norm) | set(vector_norm)
    ranked: list[dict[str, Any]] = []
    for chunk_id in ids:
        lexical_score = lexical_norm.get(chunk_id)
        vector_score = vector_norm.get(chunk_id)
        score = lexical_weight * (lexical_score or 0.0) + (1 - lexical_weight) * (
            vector_score or 0.0
        )
        ranked.append(
            {
                "chunk_id": chunk_id,
                "score": score,
                "lexical_score": lexical_score,
                "vector_score": vector_score,
                "retrieval_mode": "hybrid",
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _normalized_map(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    scores = [float(row["score"]) for row in rows]
    normalized = _normalize(scores)
    return {
        str(row["chunk_id"]): score
        for row, score in zip(rows, normalized, strict=True)
    }


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
