from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 1200
    overlap_chars: int = 180

    def __post_init__(self) -> None:
        if self.max_chars < 100:
            raise ValueError("chunking.max_chars must be at least 100")
        if self.overlap_chars < 0:
            raise ValueError("chunking.overlap_chars cannot be negative")
        if self.overlap_chars >= self.max_chars:
            raise ValueError("chunking.overlap_chars must be smaller than max_chars")


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "hashing"
    model: str = "hashing-v1"
    dimension: int = 384
    api_url: str = ""
    api_key: str = field(default="", repr=False)
    api_key_env: str = "KVLIB_EMBEDDING_API_KEY"
    api_key_file: Path | None = None
    timeout_s: float = 60.0
    batch_size: int = 32
    send_dimensions: bool = False

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("embedding.dimension must be positive")
        if self.timeout_s <= 0:
            raise ValueError("embedding.timeout_s must be positive")
        if self.batch_size <= 0:
            raise ValueError("embedding.batch_size must be positive")


@dataclass(frozen=True)
class EnrichmentConfig:
    provider: str = "none"
    model: str = "qwen3.8-flash"
    api_url: str = ""
    api_key: str = field(default="", repr=False)
    api_key_env: str = "KVLIB_QWEN_API_KEY"
    api_key_file: Path | None = None
    timeout_s: float = 90.0
    max_tokens: int = 512

    def __post_init__(self) -> None:
        if self.provider not in {"none", "openai-compatible"}:
            raise ValueError("enrichment.provider must be none or openai-compatible")
        if self.timeout_s <= 0 or self.max_tokens <= 0:
            raise ValueError("enrichment timeout and max_tokens must be positive")


@dataclass(frozen=True)
class RetrievalConfig:
    mode: str = "hybrid"
    top_k: int = 8
    candidate_k: int = 40
    lexical_weight: float = 0.35

    def __post_init__(self) -> None:
        if self.mode not in {"hybrid", "vector", "lexical"}:
            raise ValueError("retrieval.mode must be hybrid, vector, or lexical")
        if self.top_k <= 0 or self.candidate_k <= 0:
            raise ValueError("retrieval result counts must be positive")
        if not 0 <= self.lexical_weight <= 1:
            raise ValueError("retrieval.lexical_weight must be between 0 and 1")


@dataclass(frozen=True)
class DatabaseConfig:
    backend: str = "sqlite"
    host: str = "127.0.0.1"
    port: int = 5432
    name: str = "vector_library"
    user: str = "windows_writer"
    namespace: str = "knowledge"
    dsn_env: str = "KVLIB_DATABASE_URL"
    password_env: str = "KVLIB_DB_PASSWORD"
    credentials_file: Path | None = None
    credentials_account: str = "windows"

    def __post_init__(self) -> None:
        if self.backend not in {"sqlite", "postgres"}:
            raise ValueError("database.backend must be sqlite or postgres")
        if not 1 <= self.port <= 65535:
            raise ValueError("database.port must be between 1 and 65535")


@dataclass(frozen=True)
class AppConfig:
    database_path: Path = Path("data/knowledge.sqlite3")
    sources_dir: Path = Path("sources")
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    database: DatabaseConfig = DatabaseConfig()
    enrichment: EnrichmentConfig = EnrichmentConfig()
    sources_dirs: tuple[Path, ...] = ()

    @property
    def ingest_roots(self) -> tuple[Path, ...]:
        return self.sources_dirs or (self.sources_dir,)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or Path(".knowledge-vector/config.toml")
    if not config_path.exists():
        return AppConfig()

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project = _section(raw, "project")
    database = _section(raw, "database")
    chunking = _section(raw, "chunking")
    embedding = _section(raw, "embedding")
    retrieval = _section(raw, "retrieval")
    enrichment = _section(raw, "enrichment")

    base_dir = (
        config_path.parent.parent
        if config_path.parent.name == ".knowledge-vector"
        else config_path.parent
    )
    database_path = Path(database.get("path", "data/knowledge.sqlite3"))
    sources_dir = Path(project.get("sources_dir", "sources"))
    if not database_path.is_absolute():
        database_path = base_dir / database_path
    if not sources_dir.is_absolute():
        sources_dir = base_dir / sources_dir
    sources_dirs: tuple[Path, ...] = ()
    if "sources_dirs" in project:
        values = project["sources_dirs"]
        if not isinstance(values, list) or not values or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError("project.sources_dirs must be a non-empty array of paths")
        roots = [Path(value).expanduser() for value in values]
        sources_dirs = tuple(root if root.is_absolute() else base_dir / root for root in roots)
    credentials_value = database.get("credentials_file")
    credentials_file = Path(credentials_value).expanduser() if credentials_value else None
    if credentials_file is not None and not credentials_file.is_absolute():
        credentials_file = base_dir / credentials_file

    def secret_path(value: Any) -> Path | None:
        if not value:
            return None
        result = Path(str(value)).expanduser()
        return result if result.is_absolute() else base_dir / result

    return AppConfig(
        database_path=database_path,
        sources_dir=sources_dir,
        sources_dirs=sources_dirs,
        database=DatabaseConfig(
            backend=str(database.get("backend", "sqlite")),
            host=str(database.get("host", "127.0.0.1")),
            port=int(database.get("port", 5432)),
            name=str(database.get("name", "vector_library")),
            user=str(database.get("user", "windows_writer")),
            namespace=str(database.get("namespace", "knowledge")),
            dsn_env=str(database.get("dsn_env", "KVLIB_DATABASE_URL")),
            password_env=str(database.get("password_env", "KVLIB_DB_PASSWORD")),
            credentials_file=credentials_file,
            credentials_account=str(database.get("credentials_account", "windows")),
        ),
        chunking=ChunkingConfig(
            max_chars=int(chunking.get("max_chars", 1200)),
            overlap_chars=int(chunking.get("overlap_chars", 180)),
        ),
        embedding=EmbeddingConfig(
            provider=str(embedding.get("provider", "hashing")),
            model=str(embedding.get("model", "hashing-v1")),
            dimension=int(embedding.get("dimension", 384)),
            api_url=str(embedding.get("api_url", "")).strip(),
            api_key=str(embedding.get("api_key", "")).strip(),
            api_key_env=str(embedding.get("api_key_env", "KVLIB_EMBEDDING_API_KEY")),
            api_key_file=secret_path(embedding.get("api_key_file")),
            timeout_s=float(embedding.get("timeout_s", 60.0)),
            batch_size=int(embedding.get("batch_size", 32)),
            send_dimensions=bool(embedding.get("send_dimensions", False)),
        ),
        retrieval=RetrievalConfig(
            mode=str(retrieval.get("mode", "hybrid")),
            top_k=int(retrieval.get("top_k", 8)),
            candidate_k=int(retrieval.get("candidate_k", 40)),
            lexical_weight=float(retrieval.get("lexical_weight", 0.35)),
        ),
        enrichment=EnrichmentConfig(
            provider=str(enrichment.get("provider", "none")),
            model=str(enrichment.get("model", "qwen3.8-flash")),
            api_url=str(enrichment.get("api_url", "")).strip(),
            api_key=str(enrichment.get("api_key", "")).strip(),
            api_key_env=str(enrichment.get("api_key_env", "KVLIB_QWEN_API_KEY")),
            api_key_file=secret_path(enrichment.get("api_key_file")),
            timeout_s=float(enrichment.get("timeout_s", 90)),
            max_tokens=int(enrichment.get("max_tokens", 512)),
        ),
    )


def resolve_api_key(config: EmbeddingConfig | EnrichmentConfig) -> str:
    value = os.environ.get(config.api_key_env, "").strip()
    if value:
        return value
    if config.api_key_file:
        return config.api_key_file.read_text(encoding="utf-8-sig").strip()
    return config.api_key


def write_default_config(path: Path, *, force: bool = False) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """[project]
data_dir = "data"
sources_dir = "sources"

[database]
path = "data/knowledge.sqlite3"

[chunking]
max_chars = 1200
overlap_chars = 180

[embedding]
provider = "hashing"
model = "hashing-v1"
dimension = 384
api_url = ""
api_key = ""
timeout_s = 60
batch_size = 32

[retrieval]
mode = "hybrid"
top_k = 8
candidate_k = 40
lexical_weight = 0.35
""",
        encoding="utf-8",
    )
    return True
