from __future__ import annotations

from pathlib import Path
from typing import Iterable

from knowledge_vector.chunking import chunk_document
from knowledge_vector.config import ChunkingConfig
from knowledge_vector.embeddings import EmbeddingProvider
from knowledge_vector.loaders import discover_files, load_document
from knowledge_vector.models import IngestSummary
from knowledge_vector.store import SQLiteStore


class IngestionPipeline:
    def __init__(
        self,
        store: SQLiteStore,
        embedder: EmbeddingProvider,
        chunking: ChunkingConfig,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunking = chunking

    def ingest(
        self,
        path: Path,
        *,
        source_type: str | None = None,
        recursive: bool = True,
        patterns: Iterable[str] | None = None,
    ) -> IngestSummary:
        files = discover_files(path, recursive=recursive, patterns=patterns)
        summary = IngestSummary(discovered=len(files))

        for file_path in files:
            try:
                document = load_document(file_path, source_type_override=source_type)
                if self.store.document_hash(document.path) == document.content_hash:
                    summary.skipped += 1
                    continue
                chunks = chunk_document(document, self.chunking)
                if not chunks:
                    summary.failed += 1
                    summary.errors.append((str(file_path), "No extractable text"))
                    continue
                embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
                changed = self.store.index_document(
                    document,
                    chunks,
                    embeddings,
                    embedding_model=self.embedder.name,
                )
                if changed:
                    summary.indexed += 1
                    summary.chunks += len(chunks)
                else:
                    summary.skipped += 1
            except Exception as exc:
                summary.failed += 1
                summary.errors.append((str(file_path), str(exc)))
        return summary
