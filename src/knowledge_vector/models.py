from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TextBlock:
    text: str
    heading: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedDocument:
    id: str
    path: Path
    source_type: str
    title: str
    text: str
    content_hash: str
    blocks: tuple[TextBlock, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    ordinal: int
    text: str
    content_hash: str
    heading: str | None = None
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    title: str
    path: str
    source_type: str
    score: float
    vector_score: float | None = None
    lexical_score: float | None = None
    retrieval_mode: str = "hybrid"
    heading: str | None = None
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestSummary:
    discovered: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
