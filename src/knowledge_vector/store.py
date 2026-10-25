from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from knowledge_vector.models import Chunk, LoadedDocument


class SQLiteStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                heading TEXT,
                page_number INTEGER,
                char_start INTEGER,
                char_end INTEGER,
                metadata_json TEXT NOT NULL,
                embedding BLOB,
                embedding_dim INTEGER,
                embedding_model TEXT,
                UNIQUE(document_id, ordinal)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            """
        )
        try:
            self.connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    text,
                    heading,
                    tokenize = 'trigram'
                )
                """
            )
            self._fts_tokenizer = "trigram"
        except sqlite3.OperationalError:
            self.connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    text,
                    heading,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
                """
            )
            self._fts_tokenizer = "unicode61"
        self.connection.commit()

    def document_hash(self, path: Path) -> str | None:
        row = self.connection.execute(
            "SELECT content_hash FROM documents WHERE path = ?",
            (path.resolve().as_posix(),),
        ).fetchone()
        return str(row["content_hash"]) if row else None

    def index_document(
        self,
        document: LoadedDocument,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        embedding_model: str,
    ) -> bool:
        if len(chunks) != len(embeddings):
            raise ValueError("chunk count and embedding count must match")

        existing = self.document_hash(document.path)
        if existing == document.content_hash:
            return False

        with self.connection:
            self.connection.execute(
                "DELETE FROM chunks_fts WHERE document_id = ?",
                (document.id,),
            )
            self.connection.execute(
                "DELETE FROM documents WHERE path = ?",
                (document.path.as_posix(),),
            )
            self.connection.execute(
                """
                INSERT INTO documents (
                    id, path, source_type, title, content_hash, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.path.as_posix(),
                    document.source_type,
                    document.title,
                    document.content_hash,
                    _json(document.metadata),
                ),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                vector = np.asarray(embedding, dtype=np.float32)
                self.connection.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, ordinal, text, content_hash, heading,
                        page_number, char_start, char_end, metadata_json,
                        embedding, embedding_dim, embedding_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.text,
                        chunk.content_hash,
                        chunk.heading,
                        chunk.page_number,
                        chunk.char_start,
                        chunk.char_end,
                        _json(chunk.metadata),
                        vector.tobytes(),
                        int(vector.shape[0]),
                        embedding_model,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO chunks_fts (chunk_id, document_id, text, heading)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chunk.id, chunk.document_id, chunk.text, chunk.heading or ""),
                )
        return True

    def lexical_candidates(self, query: str, limit: int) -> list[dict[str, Any]]:
        fts_query = _fts_query(query)
        if fts_query:
            try:
                rows = self.connection.execute(
                    """
                    SELECT
                        c.id AS chunk_id,
                        c.document_id,
                        c.text,
                        c.heading,
                        c.page_number,
                        d.title,
                        d.path,
                        d.source_type,
                        -bm25(chunks_fts, 0.0, 0.0, 1.0, 0.4) AS score
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.chunk_id
                    JOIN documents d ON d.id = c.document_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY bm25(chunks_fts, 0.0, 0.0, 1.0, 0.4)
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
                if rows:
                    return [dict(row) for row in rows]
            except sqlite3.OperationalError:
                pass

        escaped = f"%{query.strip()}%"
        rows = self.connection.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.text,
                c.heading,
                c.page_number,
                d.title,
                d.path,
                d.source_type,
                CAST(
                    (LENGTH(c.text) - LENGTH(REPLACE(LOWER(c.text), LOWER(?), '')))
                    / MAX(LENGTH(?), 1) AS REAL
                ) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE LOWER(c.text) LIKE LOWER(?)
               OR LOWER(COALESCE(c.heading, '')) LIKE LOWER(?)
            ORDER BY score DESC
            LIMIT ?
            """,
            (query.strip(), query.strip(), escaped, escaped, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def vector_candidates(
        self,
        query_embedding: np.ndarray,
        limit: int,
        *,
        embedding_model: str,
    ) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.text,
                c.heading,
                c.page_number,
                c.embedding,
                d.title,
                d.path,
                d.source_type
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
              AND c.embedding_model = ?
            """,
            (embedding_model,),
        ).fetchall()
        if not rows:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.shape != query.shape:
                continue
            scored.append((float(np.dot(query, vector)), row))
        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, row in scored[:limit]:
            value = dict(row)
            value.pop("embedding", None)
            value["score"] = score
            results.append(value)
        return results

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.connection.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.document_id,
                c.text,
                c.heading,
                c.page_number,
                c.metadata_json,
                d.title,
                d.path,
                d.source_type,
                d.metadata_json AS document_metadata_json
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            tuple(chunk_ids),
        ).fetchall()
        return {str(row["chunk_id"]): dict(row) for row in rows}

    def stats(self) -> dict[str, Any]:
        document_count = self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        source_rows = self.connection.execute(
            """
            SELECT source_type, COUNT(*) AS count
            FROM documents
            GROUP BY source_type
            ORDER BY count DESC
            """
        ).fetchall()
        models = self.connection.execute(
            """
            SELECT DISTINCT embedding_model
            FROM chunks
            WHERE embedding_model IS NOT NULL
            ORDER BY embedding_model
            """
        ).fetchall()
        return {
            "documents": int(document_count),
            "chunks": int(chunk_count),
            "source_types": {str(row["source_type"]): int(row["count"]) for row in source_rows},
            "embedding_models": [str(row["embedding_model"]) for row in models],
            "fts_tokenizer": self._fts_tokenizer,
            "database_path": str(self.path),
        }

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT id, path, source_type, title, content_hash, updated_at
            FROM documents
            ORDER BY updated_at DESC, title
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def _fts_query(query: str) -> str:
    terms = [term for term in query.replace('"', " ").split() if len(term) >= 3]
    if not terms:
        return ""
    return " AND ".join(f'"{term}"' for term in terms)


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
