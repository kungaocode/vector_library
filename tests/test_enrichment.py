import json
from pathlib import Path

import pytest

from knowledge_vector.config import ChunkingConfig, EmbeddingConfig, EnrichmentConfig
from knowledge_vector.embeddings import create_embedding_provider
from knowledge_vector.enrichment import QwenEnricher
from knowledge_vector.pipeline import IngestionPipeline
from knowledge_vector.store import SQLiteStore


def test_cloud_annotation_preserves_source_and_skips_unchanged_file(tmp_path, monkeypatch):
    calls = []

    def response(url, payload, **kwargs):
        calls.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "summary": "数据库事务保证原子性", "keywords": ["事务", "原子性"],
        })}}]}

    monkeypatch.setattr("knowledge_vector.enrichment.post_json", response)
    config = EnrichmentConfig(
        provider="openai-compatible", api_url="https://example.test/chat/completions",
        api_key="test-key",
    )
    source = tmp_path / "note.txt"
    original = "事务中的所有写入要么一起完成，要么全部撤销。"
    source.write_text(original, encoding="utf-8")
    with SQLiteStore(tmp_path / "test.db") as store:
        pipeline = IngestionPipeline(
            store, create_embedding_provider(EmbeddingConfig()), ChunkingConfig(),
            QwenEnricher(config),
        )
        assert pipeline.ingest(source).indexed == 1
        assert pipeline.ingest(source).skipped == 1
        row = store.connection.execute("SELECT text,metadata_json FROM chunks").fetchone()
        assert row["text"] == original
        assert json.loads(row["metadata_json"])["_enrichment"]["keywords"] == ["事务", "原子性"]
        assert len(calls) == 1
        assert calls[0]["model"] == "qwen3.8-flash"
        assert calls[0]["response_format"] == {"type": "json_object"}


def test_invalid_cloud_response_does_not_destroy_existing_document(tmp_path, monkeypatch):
    source = tmp_path / "note.txt"
    source.write_text("Original source", encoding="utf-8")
    embedder = create_embedding_provider(EmbeddingConfig())
    with SQLiteStore(tmp_path / "test.db") as store:
        assert IngestionPipeline(store, embedder, ChunkingConfig()).ingest(source).indexed == 1
        before = store.document_hash(source)
        monkeypatch.setattr("knowledge_vector.enrichment.post_json", lambda *a, **k: {})
        enricher = QwenEnricher(EnrichmentConfig(
            provider="openai-compatible", api_url="https://example.test/chat/completions",
            api_key="test-key",
        ))
        result = IngestionPipeline(store, embedder, ChunkingConfig(), enricher).ingest(source)
        assert result.failed == 1
        assert store.document_hash(source) == before
        assert store.stats()["chunks"] == 1


def test_api_key_file_and_env_precedence(tmp_path: Path, monkeypatch):
    from knowledge_vector.config import resolve_api_key

    path = tmp_path / "key"
    path.write_text("file-key\n")
    config = EnrichmentConfig(api_key_file=path)
    monkeypatch.delenv(config.api_key_env, raising=False)
    assert resolve_api_key(config) == "file-key"
    monkeypatch.setenv(config.api_key_env, "env-key")
    assert resolve_api_key(config) == "env-key"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_truncated_cloud_response_is_rejected(finish_reason, monkeypatch):
    from knowledge_vector.models import Chunk

    monkeypatch.setattr("knowledge_vector.enrichment.post_json", lambda *a, **k: {
        "choices": [{"finish_reason": finish_reason, "message": {"content": "{}"}}],
    })
    enricher = QwenEnricher(EnrichmentConfig(
        provider="openai-compatible", api_url="https://example.test/chat/completions",
        api_key="test-key",
    ))
    with pytest.raises(RuntimeError, match="incomplete"):
        enricher.enrich([Chunk("c", "d", 0, "text", "hash")])
