from __future__ import annotations

import hashlib

from knowledge_vector.config import ChunkingConfig
from knowledge_vector.models import Chunk, LoadedDocument


def chunk_document(document: LoadedDocument, config: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0

    for block in document.blocks:
        for text, start, end in _split_text(block.text, config.max_chars, config.overlap_chars):
            chunk_id = _stable_id(
                "chunk",
                f"{document.id}:{ordinal}:{block.page_number}:{start}:{text}",
            )
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    ordinal=ordinal,
                    text=text,
                    content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    heading=block.heading,
                    page_number=block.page_number,
                    char_start=start,
                    char_end=end,
                    metadata=block.metadata,
                )
            )
            ordinal += 1

    return chunks


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[tuple[str, int, int]]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [(text, 0, len(text))]

    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        hard_end = min(start + max_chars, text_length)
        end = hard_end
        if hard_end < text_length:
            search_start = start + int(max_chars * 0.6)
            candidates = [
                text.rfind("\n\n", search_start, hard_end),
                text.rfind("\n", search_start, hard_end),
                text.rfind("。", search_start, hard_end),
                text.rfind(". ", search_start, hard_end),
                text.rfind(" ", search_start, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text[boundary : boundary + 2] == "\n\n" else 1)

        piece = text[start:end].strip()
        if piece:
            leading = len(text[start:end]) - len(text[start:end].lstrip())
            trailing = len(text[start:end]) - len(text[start:end].rstrip())
            chunks.append((piece, start + leading, end - trailing))

        if end >= text_length:
            break
        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"
