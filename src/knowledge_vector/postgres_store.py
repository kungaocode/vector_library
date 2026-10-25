from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from knowledge_vector.config import DatabaseConfig, EmbeddingConfig
from knowledge_vector.embeddings import _tokens
from knowledge_vector.models import Chunk, LoadedDocument

try:
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError as exc:
    raise RuntimeError("PostgreSQL support requires: pip install -e '.[postgres]'") from exc


class PostgresStore:
    """Transactional pgvector store, with one embedding space per namespace."""

    def __init__(self, database: DatabaseConfig, embedding: EmbeddingConfig) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", database.namespace):
            raise ValueError("database.namespace must match [a-z][a-z0-9_]{0,31}")
        if not 1 <= embedding.dimension <= 2000:
            raise ValueError("PostgreSQL HNSW vector dimension must be between 1 and 2000")
        self.namespace = database.namespace
        self.dimension = embedding.dimension
        self.model = embedding.model
        self.provider = embedding.provider
        self._names = {
            name: sql.Identifier("public", f"{self.namespace}_{name}")
            for name in ("settings", "documents", "chunks")
        }
        dsn = os.environ.get(database.dsn_env, "")
        options: dict[str, Any] = {}
        if not dsn:
            options = {
                "host": database.host, "port": database.port,
                "dbname": database.name, "user": database.user,
            }
            password = os.environ.get(database.password_env)
            if password is None and database.credentials_file:
                credentials = json.loads(
                    database.credentials_file.read_text(encoding="utf-8-sig")
                )
                account = credentials[database.credentials_account]
                if account["username"] != database.user:
                    raise ValueError("Credential file account does not match database.user")
                password = account["password"]
            if password is not None:
                options["password"] = password
        self.connection = psycopg.connect(
            dsn, **options, autocommit=True, row_factory=dict_row,
            connect_timeout=10, application_name="knowledge-vector",
        )
        try:
            extension = self.connection.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if not extension:
                raise RuntimeError("Ask the database administrator to CREATE EXTENSION vector")
            register_vector(self.connection)
            self._initialize()
        except Exception:
            self.connection.close()
            raise

    def _query(self, template: str) -> sql.Composed:
        return sql.SQL(template).format(**self._names)

    def _lock(self, key: str) -> None:
        lock_id = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big", signed=True)
        self.connection.execute("SELECT pg_advisory_xact_lock(%s)", (lock_id,))

    def _initialize(self) -> None:
        with self.connection.transaction():
            self._lock(f"knowledge-vector:{self.namespace}:schema")
            self.connection.execute(self._query("""
                CREATE TABLE IF NOT EXISTS {settings} (
                    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
                    schema_version integer NOT NULL,
                    provider text NOT NULL, model text NOT NULL, dimension integer NOT NULL
                )
            """))
            self.connection.execute(self._query("""
                INSERT INTO {settings} (schema_version, provider, model, dimension)
                VALUES (1, %s, %s, %s) ON CONFLICT DO NOTHING
            """), (self.provider, self.model, self.dimension))
            identity = self.connection.execute(
                self._query("SELECT * FROM {settings}")
            ).fetchone()
            expected = (1, self.provider, self.model, self.dimension)
            actual = tuple(identity[key] for key in (
                "schema_version", "provider", "model", "dimension",
            ))
            if actual != expected:
                raise ValueError(
                    "Stored embedding space differs from configuration; use a new "
                    "database.namespace to rebuild without mixing models or dimensions"
                )
            self.connection.execute(self._query("""
                CREATE TABLE IF NOT EXISTS {documents} (
                    id text PRIMARY KEY, path text NOT NULL UNIQUE,
                    source_type text NOT NULL, title text NOT NULL,
                    content_hash text NOT NULL, metadata_json jsonb NOT NULL,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
            """))
            self.connection.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {chunks} (
                    id text PRIMARY KEY,
                    document_id text NOT NULL REFERENCES {documents}(id) ON DELETE CASCADE,
                    ordinal integer NOT NULL, text text NOT NULL, content_hash text NOT NULL,
                    heading text, page_number integer, char_start integer, char_end integer,
                    metadata_json jsonb NOT NULL,
                    embedding vector({dimension}) NOT NULL,
                    embedding_model text NOT NULL,
                    search_text tsvector NOT NULL,
                    UNIQUE(document_id, ordinal)
                )
            """).format(**self._names, dimension=sql.Literal(self.dimension)))
            for suffix, expression in (
                ("hnsw", "USING hnsw (embedding vector_cosine_ops)"),
                ("fts", "USING gin (search_text)"),
            ):
                self.connection.execute(sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} ON {chunks} " + expression
                ).format(
                    index=sql.Identifier(f"{self.namespace}_chunks_{suffix}"),
                    chunks=self._names["chunks"],
                ))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> PostgresStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def document_hash(self, path: Path) -> str | None:
        row = self.connection.execute(self._query(
            "SELECT content_hash FROM {documents} WHERE path = %s"
        ), (path.resolve().as_posix(),)).fetchone()
        return row["content_hash"] if row else None

    def index_document(
        self, document: LoadedDocument, chunks: Sequence[Chunk],
        embeddings: np.ndarray, embedding_model: str,
    ) -> bool:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.shape != (len(chunks), self.dimension):
            raise ValueError("Embedding shape does not match chunk count and configured dimension")
        if embedding_model != self.model:
            raise ValueError("Embedding model does not match this database namespace")
        if not np.all(np.isfinite(vectors)) or np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("Embeddings must be finite, non-zero vectors")
        if any(chunk.document_id != document.id for chunk in chunks):
            raise ValueError("Chunks must belong to the document being indexed")
        with self.connection.transaction():
            self._lock(f"knowledge-vector:{self.namespace}:{document.id}")
            if self.document_hash(document.path) == document.content_hash:
                return False
            self.connection.execute(self._query("""
                INSERT INTO {documents} (id,path,source_type,title,content_hash,metadata_json)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (id) DO UPDATE SET
                    path=EXCLUDED.path, source_type=EXCLUDED.source_type, title=EXCLUDED.title,
                    content_hash=EXCLUDED.content_hash, metadata_json=EXCLUDED.metadata_json,
                    updated_at=now()
            """), (
                document.id, document.path.resolve().as_posix(), document.source_type,
                document.title, document.content_hash, Jsonb(document.metadata),
            ))
            self.connection.execute(self._query(
                "DELETE FROM {chunks} WHERE document_id = %s"
            ), (document.id,))
            records = [
                (
                    chunk.id, chunk.document_id, chunk.ordinal, chunk.text, chunk.content_hash,
                    chunk.heading, chunk.page_number, chunk.char_start, chunk.char_end,
                    Jsonb(chunk.metadata), vector, embedding_model,
                    " ".join(_lexical_terms(chunk.heading or "")),
                    " ".join(_lexical_terms(chunk.text)),
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
            with self.connection.cursor() as cursor:
                cursor.executemany(self._query("""
                    INSERT INTO {chunks} (
                        id,document_id,ordinal,text,content_hash,heading,page_number,
                        char_start,char_end,metadata_json,embedding,embedding_model,search_text
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        setweight(to_tsvector('simple', %s), 'A') ||
                        setweight(to_tsvector('simple', %s), 'B'))
                """), records)
        return True

    def lexical_candidates(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = list(dict.fromkeys(_lexical_terms(query)))[:128]
        if not terms:
            return []
        return self.connection.execute(self._query("""
            SELECT c.id AS chunk_id, ts_rank_cd(c.search_text, q.query) AS score
            FROM {chunks} c, to_tsquery('simple', %s) AS q(query)
            WHERE c.search_text @@ q.query
            ORDER BY score DESC, c.id LIMIT %s
        """), (" | ".join(terms), limit)).fetchall()

    def vector_candidates(
        self, query_embedding: np.ndarray, limit: int, *, embedding_model: str,
    ) -> list[dict[str, Any]]:
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.shape != (self.dimension,) or not np.all(np.isfinite(query)):
            raise ValueError("Query vector has an invalid dimension or values")
        if embedding_model != self.model:
            raise ValueError("Query model differs from the indexed model")
        if np.linalg.norm(query) == 0:
            return []
        return self.connection.execute(self._query("""
            SELECT id AS chunk_id, 1 - (embedding <=> %s) AS score
            FROM {chunks} ORDER BY embedding <=> %s LIMIT %s
        """), (query, query, limit)).fetchall()

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        rows = self.connection.execute(self._query("""
            SELECT c.id AS chunk_id,c.document_id,c.text,c.heading,c.page_number,
                   c.metadata_json::text AS metadata_json,d.title,d.path,d.source_type,
                   d.metadata_json::text AS document_metadata_json
            FROM {chunks} c JOIN {documents} d ON d.id=c.document_id
            WHERE c.id = ANY(%s)
        """), (list(chunk_ids),)).fetchall()
        return {row["chunk_id"]: row for row in rows}

    def stats(self) -> dict[str, Any]:
        count = self.connection.execute(self._query("""
            SELECT (SELECT count(*) FROM {documents}) AS documents,
                   (SELECT count(*) FROM {chunks}) AS chunks
        """)).fetchone()
        sources = self.connection.execute(self._query("""
            SELECT source_type,count(*) AS count FROM {documents} GROUP BY source_type
        """)).fetchall()
        info = self.connection.info
        return {
            **count, "source_types": {row["source_type"]: row["count"] for row in sources},
            "embedding_models": [self.model] if count["chunks"] else [],
            "embedding_dimension": self.dimension,
            "fts_tokenizer": "simple + CJK character ngrams (GIN)",
            "database_path": f"postgresql://{info.host}:{info.port}/{info.dbname}/{self.namespace}",
            "backend": "postgres", "namespace": self.namespace,
        }

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.connection.execute(self._query("""
            SELECT id,path,source_type,title,content_hash,updated_at FROM {documents}
            ORDER BY updated_at DESC,title LIMIT %s
        """), (limit,)).fetchall()


def _lexical_terms(text: str) -> list[str]:
    # Chinese is not segmented by PostgreSQL's built-in simple dictionary.
    return [
        token for token in _tokens(text)
        if len(token) > 1 or len(text.strip()) == 1 or token.isascii()
    ]
