from __future__ import annotations

import tomllib
from dataclasses import dataclass
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
    api_key: str = ""
    timeout_s: float = 60.0
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("embedding.dimension must be positive")
        if self.timeout_s <= 0:
            raise ValueError("embedding.timeout_s must be positive")
        if self.batch_size <= 0:
            raise ValueError("embedding.batch_size must be positive")


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
class AppConfig:
    database_path: Path = Path("data/knowledge.sqlite3")
    sources_dir: Path = Path("sources")
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()


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

    base_dir = (
        config_path.parent.parent
        if config_path.parent.name == ".knowledge-vector"
        else Path(".")
    )
    database_path = Path(database.get("path", "data/knowledge.sqlite3"))
    sources_dir = Path(project.get("sources_dir", "sources"))
    if not database_path.is_absolute():
        database_path = base_dir / database_path
    if not sources_dir.is_absolute():
        sources_dir = base_dir / sources_dir

    return AppConfig(
        database_path=database_path,
        sources_dir=sources_dir,
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
            timeout_s=float(embedding.get("timeout_s", 60.0)),
            batch_size=int(embedding.get("batch_size", 32)),
        ),
        retrieval=RetrievalConfig(
            mode=str(retrieval.get("mode", "hybrid")),
            top_k=int(retrieval.get("top_k", 8)),
            candidate_k=int(retrieval.get("candidate_k", 40)),
            lexical_weight=float(retrieval.get("lexical_weight", 0.35)),
        ),
    )


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
