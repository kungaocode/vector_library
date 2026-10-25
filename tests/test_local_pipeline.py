from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from knowledge_vector.cli import app
from knowledge_vector.config import ChunkingConfig, EmbeddingConfig, load_config
from knowledge_vector.embeddings import create_embedding_provider
from knowledge_vector.loaders import discover_files, load_document
from knowledge_vector.pipeline import IngestionPipeline
from knowledge_vector.store import SQLiteStore


def test_model_and_chunking_change_rebuilds_unchanged_file(tmp_path: Path) -> None:
    source = tmp_path / "note.txt"
    source.write_text("向量数据库使用余弦距离检索知识。" * 30, encoding="utf-8")
    embedder = create_embedding_provider(EmbeddingConfig())
    with SQLiteStore(tmp_path / "test.db") as store:
        pipeline = IngestionPipeline(store, embedder, ChunkingConfig())
        assert pipeline.ingest(source).indexed == 1
        assert pipeline.ingest(source).skipped == 1
        changed_chunking = IngestionPipeline(store, embedder, ChunkingConfig(150, 20))
        assert changed_chunking.ingest(source).indexed == 1
        assert changed_chunking.ingest(source).skipped == 1
        changed_model = create_embedding_provider(replace(EmbeddingConfig(), model="hashing-v2"))
        changed = IngestionPipeline(store, changed_model, ChunkingConfig(150, 20))
        assert changed.ingest(source).indexed == 1
        assert store.stats()["embedding_models"] == ["hashing-v2"]


def test_code_keeps_indentation_and_avoids_runtime_directories(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    code = "def hello():\n    return 'hello'\n"
    source.write_text(code, encoding="utf-8")
    (tmp_path / "api_qwen.txt").write_text("credential fixture")
    assert not discover_files(tmp_path / "api_qwen.txt")
    for name in ("secrets", "node_modules", ".git", ".venv", "pgdata"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "private.txt").write_text("do not ingest", encoding="utf-8")
    assert discover_files(tmp_path) == [source]
    loaded = load_document(source)
    assert loaded.source_type == "code"
    assert "    return" in loaded.blocks[0].text
    source.write_text(code.replace("    ", "        "), encoding="utf-8")
    assert load_document(source).content_hash != loaded.content_hash


def test_config_paths_are_relative_to_config_file(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[project]\nsources_dir="inbox"\n[database]\npath="test.db"\n')
    config = load_config(path)
    assert config.sources_dir == tmp_path / "inbox"
    assert config.database_path == tmp_path / "test.db"


def test_cli_reports_failed_source_with_nonzero_exit(tmp_path: Path) -> None:
    runner = CliRunner()
    config = tmp_path / ".knowledge-vector" / "config.toml"
    assert runner.invoke(app, ["init", "--config", str(config)]).exit_code == 0
    source = tmp_path / "broken.json"
    source.write_text("{bad json", encoding="utf-8")
    result = runner.invoke(app, ["ingest", str(source), "--config", str(config)])
    assert result.exit_code == 1
    assert "failed=1" in result.output


def test_ingest_defaults_to_configured_inbox(tmp_path: Path) -> None:
    runner = CliRunner()
    config = tmp_path / ".knowledge-vector" / "config.toml"
    assert runner.invoke(app, ["init", "--config", str(config)]).exit_code == 0
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "hello.txt").write_text("Hello knowledge library.")
    result = runner.invoke(app, ["ingest", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "indexed=1" in result.output


def test_ingest_multiple_roots_deduplicates_paths_and_honors_explicit_path(tmp_path: Path) -> None:
    private = tmp_path / "private"
    shared = tmp_path / "shared"
    code = shared / "code"
    private.mkdir()
    code.mkdir(parents=True)
    (private / "note.txt").write_text("Private knowledge about database transactions.")
    (shared / "note.txt").write_text("Shared knowledge about semantic retrieval.")
    (code / "example.py").write_text("def add(a, b):\n    return a + b\n")
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        '[project]\nsources_dirs=["private", "shared", "shared/code", "private"]\n'
        '[database]\npath="test.db"\n',
    )
    config = load_config(config_path)
    assert config.ingest_roots[:2] == (private, shared)
    runner = CliRunner()
    first = runner.invoke(app, ["ingest", "--config", str(config_path)])
    assert first.exit_code == 0, first.output
    assert "discovered=3 indexed=3" in first.output
    repeated = runner.invoke(app, ["ingest", "--config", str(config_path)])
    assert repeated.exit_code == 0, repeated.output
    assert "discovered=3 indexed=0 skipped=3" in repeated.output
    specific = runner.invoke(app, ["ingest", str(code), "--config", str(config_path)])
    assert specific.exit_code == 0, specific.output
    assert "discovered=1 indexed=0 skipped=1" in specific.output
    with SQLiteStore(config.database_path) as store:
        assert store.stats()["documents"] == 3
        assert store.lexical_candidates("transactions", 5)
        assert store.lexical_candidates("semantic", 5)


def test_failed_file_does_not_stop_another_source(tmp_path: Path) -> None:
    private = tmp_path / "private"
    shared = tmp_path / "shared"
    private.mkdir()
    shared.mkdir()
    (private / "broken.json").write_text("{invalid")
    (shared / "valid.txt").write_text("A valid document in the other root.")
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        '[project]\nsources_dirs=["private", "shared"]\n[database]\npath="test.db"\n',
    )
    result = CliRunner().invoke(app, ["ingest", "--config", str(config_path)])
    assert result.exit_code == 1, result.output
    assert "discovered=2 indexed=1 skipped=0 failed=1" in result.output


@pytest.mark.parametrize("value", ['[]', '"private"', '[""]', '[42]'])
def test_invalid_source_list_is_rejected(tmp_path: Path, value: str) -> None:
    config = tmp_path / "settings.toml"
    config.write_text(f"[project]\nsources_dirs={value}\n")
    with pytest.raises(ValueError, match="sources_dirs"):
        load_config(config)
