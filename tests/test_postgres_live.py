"""Opt-in live integration tests. All tables are isolated and removed afterwards.

Set KVLIB_TEST_CONFIG for PostgreSQL and the desired embedding provider;
KVLIB_QWEN_TEST_CONFIG additionally enables Qwen enrichment. Live cloud tests
submit only the synthetic text below.
"""
from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest

from knowledge_vector.chunking import chunk_document
from knowledge_vector.config import AppConfig, load_config
from knowledge_vector.embeddings import create_embedding_provider
from knowledge_vector.enrichment import create_enricher
from knowledge_vector.loaders import load_document
from knowledge_vector.pipeline import IngestionPipeline
from knowledge_vector.retrieval import SearchEngine


def _live_config(variable: str) -> AppConfig:
    path = os.environ.get(variable)
    if not path:
        pytest.skip(f"Set {variable} to run this live integration test")
    pytest.importorskip("psycopg")
    return load_config(Path(path))


@contextmanager
def _isolated(config: AppConfig):
    from knowledge_vector.postgres_store import PostgresStore

    database = replace(config.database, namespace="test_" + uuid.uuid4().hex[:12])
    store = PostgresStore(database, config.embedding)
    try:
        yield store, database
    finally:
        for table in ("chunks", "documents", "settings"):
            store.connection.execute(store._query("DROP TABLE IF EXISTS {" + table + "}"))
        store.close()


def test_real_local_ingest_search_and_increment(tmp_path: Path) -> None:
    config = _live_config("KVLIB_TEST_CONFIG")
    embedder = create_embedding_provider(config.embedding)
    (tmp_path / "note.txt").write_text("数据库事务保证原子性和一致性。", encoding="utf-8")
    (tmp_path / "note.md").write_text(
        "# 向量检索\n通过余弦距离检索语义相关的文档。", encoding="utf-8",
    )
    (tmp_path / "note.html").write_text(
        "<html><p>索引可以加速数据查询。</p></html>", encoding="utf-8",
    )
    (tmp_path / "note.json").write_text('{"text":"JSON knowledge records"}')
    (tmp_path / "note.jsonl").write_text('{"text":"JSONL knowledge records"}\n')
    (tmp_path / "sample.py").write_text("def add(a, b):\n    return a + b\n")
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Attention mechanisms use queries, keys and values.")
        pdf.save(tmp_path / "paper.pdf")
    with _isolated(config) as (store, _):
        pipeline = IngestionPipeline(store, embedder, config.chunking)
        first = pipeline.ingest(tmp_path)
        assert first.indexed == 7, first.errors
        assert first.failed == 0
        assert pipeline.ingest(tmp_path).skipped == 7
        search = SearchEngine(store, embedder, config.retrieval)
        for mode in ("lexical", "vector", "hybrid"):
            hits = search.search("数据库事务保证原子性和一致性", mode=mode)
            assert hits and any("原子性" in hit.text for hit in hits)
        pdf_hits = search.search("Attention mechanisms", mode="lexical")
        assert pdf_hits and pdf_hits[0].page_number == 1
        dims = store.connection.execute(store._query(
            "SELECT DISTINCT vector_dims(embedding) AS dim FROM {chunks}"
        )).fetchall()
        assert dims == [{"dim": 1024}]
        source = tmp_path / "note.txt"
        source.write_text("事务隔离级别决定并发可见性。", encoding="utf-8")
        assert pipeline.ingest(source).indexed == 1
        assert store.stats()["documents"] == 7
        assert not store.lexical_candidates("原子性", 5)
        assert search.search("事务隔离级别", mode="lexical")


def test_pg_transaction_rolls_back_on_chunk_insert_failure(tmp_path: Path) -> None:
    config = _live_config("KVLIB_TEST_CONFIG")
    import psycopg

    embedder = create_embedding_provider(config.embedding)
    source = tmp_path / "rollback.txt"
    source.write_text("Transactions preserve the previous document on failure.")
    document = load_document(source)
    chunks = chunk_document(document, config.chunking)
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    with _isolated(config) as (store, _):
        assert store.index_document(document, chunks, vectors, embedder.name)
        duplicate = replace(chunks[0], ordinal=1)
        with pytest.raises(psycopg.errors.UniqueViolation):
            store.index_document(
                replace(document, content_hash="changed"), [chunks[0], duplicate],
                np.vstack([vectors[0], vectors[0]]), embedder.name,
            )
        assert store.document_hash(source) == document.content_hash
        assert store.stats()["chunks"] == len(chunks)


def test_pg_rejects_different_embedding_space() -> None:
    config = _live_config("KVLIB_TEST_CONFIG")
    from knowledge_vector.postgres_store import PostgresStore

    with _isolated(config) as (_, database):
        with pytest.raises(ValueError, match="differs"):
            PostgresStore(database, replace(config.embedding, dimension=384))
        with pytest.raises(ValueError, match="differs"):
            PostgresStore(database, replace(config.embedding, model="different-model"))


def test_real_qwen_enrichment_to_pgvector(tmp_path: Path) -> None:
    config = _live_config("KVLIB_QWEN_TEST_CONFIG")
    embedder = create_embedding_provider(config.embedding)
    enricher = create_enricher(config.enrichment)
    source = tmp_path / "cloud-test.txt"
    original = "PostgreSQL事务具有原子性。同一事务中的操作要么全部提交，要么全部回滚。"
    source.write_text(original, encoding="utf-8")
    with _isolated(config) as (store, _):
        pipeline = IngestionPipeline(store, embedder, config.chunking, enricher)
        first = pipeline.ingest(source)
        assert first.indexed == 1, first.errors
        assert pipeline.ingest(source).skipped == 1
        result = SearchEngine(store, embedder, config.retrieval).search("数据库如何保证事务原子性")
        assert result and result[0].text == original
        annotation = result[0].metadata["_enrichment"]
        assert annotation["model"] == "qwen3.8-flash"
        assert annotation["summary"]
        assert isinstance(annotation["keywords"], list)
        # The check validates the generated metadata without persisting API credentials.
        assert "api_key" not in json.dumps(result[0].metadata)


def test_real_multiple_sources_share_one_vector_store(tmp_path: Path) -> None:
    config = _live_config("KVLIB_MULTI_SOURCE_TEST_CONFIG")
    roots = [tmp_path / name for name in ("private", "shared-documents", "shared-code")]
    for root in roots:
        root.mkdir()
    (roots[0] / "note.txt").write_text("本机资料：PostgreSQL事务保证原子性。", encoding="utf-8")
    (roots[1] / "note.txt").write_text("共享资料：向量检索使用语义相似度。", encoding="utf-8")
    (roots[2] / "sample.py").write_text("def add(a, b):\n    return a + b\n")
    embedder = create_embedding_provider(config.embedding)
    with _isolated(config) as (store, _):
        pipeline = IngestionPipeline(
            store, embedder, config.chunking, create_enricher(config.enrichment),
        )
        first = pipeline.ingest_paths([*roots, roots[1]])
        assert first.indexed == 3 and first.discovered == 3, first.errors
        assert first.failed == 0
        assert pipeline.ingest_paths(roots).skipped == 3
        assert store.stats()["documents"] == 3
        engine = SearchEngine(store, embedder, config.retrieval)
        for query, expected_root in [("事务原子性", roots[0]), ("语义相似度", roots[1])]:
            hits = engine.search(query, mode="vector", top_k=3)
            assert hits and any(Path(hit.path).parent == expected_root for hit in hits)
        assert engine.search("return", mode="lexical")[0].source_type == "code"
