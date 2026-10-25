from pathlib import Path

import pymupdf as fitz

from knowledge_vector.config import ChunkingConfig, EmbeddingConfig, RetrievalConfig
from knowledge_vector.embeddings import create_embedding_provider
from knowledge_vector.pipeline import IngestionPipeline
from knowledge_vector.retrieval import SearchEngine
from knowledge_vector.store import SQLiteStore


def test_ingest_search_and_increment(tmp_path: Path) -> None:
    source = tmp_path / "attention.txt"
    source.write_text(
        "Transformer 使用自注意力机制。"
        "查询 Q、键 K 和值 V 共同决定上下文表示。",
        encoding="utf-8",
    )
    database = tmp_path / "knowledge.sqlite3"
    embedder = create_embedding_provider(EmbeddingConfig())

    with SQLiteStore(database) as store:
        pipeline = IngestionPipeline(
            store,
            embedder,
            ChunkingConfig(max_chars=200, overlap_chars=30),
        )
        first = pipeline.ingest(source)
        second = pipeline.ingest(source)
        results = SearchEngine(
            store,
            embedder,
            RetrievalConfig(top_k=3, candidate_k=10),
        ).search("自注意力机制")

    assert first.indexed == 1
    assert first.chunks == 1
    assert second.skipped == 1
    assert results
    assert "自注意力机制" in results[0].text
    assert Path(results[0].path) == source.resolve()


def test_changed_file_replaces_old_document(tmp_path: Path) -> None:
    source = tmp_path / "note.md"
    source.write_text("# 向量\n旧内容", encoding="utf-8")
    database = tmp_path / "knowledge.sqlite3"
    embedder = create_embedding_provider(EmbeddingConfig())

    with SQLiteStore(database) as store:
        pipeline = IngestionPipeline(
            store,
            embedder,
            ChunkingConfig(max_chars=100, overlap_chars=10),
        )
        pipeline.ingest(source)
        source.write_text("# 向量\n新内容：余弦相似度", encoding="utf-8")
        summary = pipeline.ingest(source)
        stats = store.stats()

    assert summary.indexed == 1
    assert stats["documents"] == 1
    assert stats["chunks"] == 2


def test_pdf_loader_preserves_page_number(tmp_path: Path) -> None:
    source = tmp_path / "retrieval.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Vector retrieval uses embedding similarity.")
    pdf.save(source)
    pdf.close()

    database = tmp_path / "knowledge.sqlite3"
    embedder = create_embedding_provider(EmbeddingConfig())
    with SQLiteStore(database) as store:
        summary = IngestionPipeline(
            store,
            embedder,
            ChunkingConfig(max_chars=200, overlap_chars=30),
        ).ingest(source)
        results = SearchEngine(
            store,
            embedder,
            RetrievalConfig(top_k=3, candidate_k=10),
        ).search("embedding similarity")

    assert summary.indexed == 1
    assert results
    assert results[0].page_number == 1
    assert results[0].source_type == "pdf"
