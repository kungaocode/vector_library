from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path

from knowledge_vector.chunking import chunk_document
from knowledge_vector.config import ChunkingConfig
from knowledge_vector.embeddings import EmbeddingProvider
from knowledge_vector.enrichment import QwenEnricher
from knowledge_vector.loaders import discover_files, load_document
from knowledge_vector.models import IngestSummary
from knowledge_vector.store import Store


class IngestionPipeline:
    def __init__(
        self,
        store: Store,
        embedder: EmbeddingProvider,
        chunking: ChunkingConfig,
        enricher: QwenEnricher | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunking = chunking
        self.enricher = enricher

    def ingest(
        self,
        path: Path,
        *,
        source_type: str | None = None,
        recursive: bool = True,
        patterns: Iterable[str] | None = None,
    ) -> IngestSummary:
        return self.ingest_paths(
            [path], source_type=source_type, recursive=recursive, patterns=patterns,
        )

    def ingest_paths(
        self,
        paths: Iterable[Path],
        *,
        source_type: str | None = None,
        recursive: bool = True,
        patterns: Iterable[str] | None = None,
    ) -> IngestSummary:
        """Ingest multiple roots, counting each resolved file path only once."""
        selected_patterns = list(patterns) if patterns is not None else None
        files = list(dict.fromkeys(
            file.resolve()
            for root in paths
            for file in discover_files(root, recursive=recursive, patterns=selected_patterns)
        ))
        summary = IngestSummary(discovered=len(files))

        for file_path in files:
            try:
                document = load_document(file_path, source_type_override=source_type)
                # Rebuild when the model or chunking changes, even if the source is unchanged.
                identity = {
                    "version": 1,
                    "source_hash": document.content_hash,
                    "chunking": asdict(self.chunking),
                    "provider": type(self.embedder).__name__,
                    "model": self.embedder.name,
                    "dimension": self.embedder.dimension,
                    "enrichment": self.enricher.identity if self.enricher else None,
                }
                fingerprint = hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode("utf-8")
                ).hexdigest()
                document = replace(
                    document, content_hash=fingerprint,
                    metadata={**document.metadata, "_pipeline": identity},
                )
                if self.store.document_hash(document.path) == document.content_hash:
                    summary.skipped += 1
                    continue
                chunks = chunk_document(document, self.chunking)
                if not chunks:
                    summary.failed += 1
                    summary.errors.append((str(file_path), "No extractable text"))
                    continue
                if self.enricher:
                    chunks, embedding_texts = self.enricher.enrich(chunks)
                else:
                    embedding_texts = [chunk.text for chunk in chunks]
                embeddings = self.embedder.embed_documents(embedding_texts)
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
