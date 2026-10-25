from __future__ import annotations

from pathlib import Path

import typer

from knowledge_vector.config import load_config, write_default_config
from knowledge_vector.embeddings import create_embedding_provider
from knowledge_vector.pipeline import IngestionPipeline
from knowledge_vector.retrieval import SearchEngine
from knowledge_vector.store import SQLiteStore

app = typer.Typer(
    name="kvlib",
    help="Local-first personal knowledge vector library.",
    no_args_is_help=True,
)


@app.command()
def init(
    config_path: Path = typer.Option(
        Path(".knowledge-vector/config.toml"),
        "--config",
        help="Configuration file path.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing config."),
) -> None:
    """Initialize the local configuration and database."""
    created = write_default_config(config_path, force=force)
    config = load_config(config_path)
    with SQLiteStore(config.database_path) as store:
        stats = store.stats()
    state = "created" if created else "kept"
    typer.echo(f"Config {state}: {config_path}")
    typer.echo(f"Database ready: {stats['database_path']}")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest."),
    config_path: Path = typer.Option(
        Path(".knowledge-vector/config.toml"),
        "--config",
        help="Configuration file path.",
    ),
    source_type: str | None = typer.Option(
        None,
        "--source-type",
        help="Override source type, for example paper, wiki, book, or note.",
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    pattern: list[str] = typer.Option(
        None,
        "--pattern",
        help="Optional glob pattern. Can be supplied multiple times.",
    ),
) -> None:
    """Extract, chunk, embed, and index local source files."""
    config = load_config(config_path)
    embedder = create_embedding_provider(config.embedding)
    with SQLiteStore(config.database_path) as store:
        summary = IngestionPipeline(store, embedder, config.chunking).ingest(
            path,
            source_type=source_type,
            recursive=recursive,
            patterns=pattern,
        )
    typer.echo(
        "Ingestion complete: "
        f"discovered={summary.discovered} "
        f"indexed={summary.indexed} "
        f"skipped={summary.skipped} "
        f"failed={summary.failed} "
        f"chunks={summary.chunks}"
    )
    for file_path, error in summary.errors:
        typer.echo(f"[failed] {file_path}: {error}", err=True)


@app.command()
def search(
    query: str = typer.Argument(..., help="Question or concept to search for."),
    config_path: Path = typer.Option(
        Path(".knowledge-vector/config.toml"),
        "--config",
        help="Configuration file path.",
    ),
    top_k: int | None = typer.Option(None, "--top-k", min=1),
    mode: str | None = typer.Option(None, "--mode", help="hybrid, vector, or lexical."),
) -> None:
    """Search indexed chunks."""
    config = load_config(config_path)
    if mode and mode not in {"hybrid", "vector", "lexical"}:
        raise typer.BadParameter("mode must be hybrid, vector, or lexical")
    selected_mode = mode or config.retrieval.mode
    embedder = (
        None
        if selected_mode == "lexical"
        else create_embedding_provider(config.embedding)
    )
    with SQLiteStore(config.database_path) as store:
        results = SearchEngine(store, embedder, config.retrieval).search(
            query,
            top_k=top_k,
            mode=mode,
        )

    if not results:
        typer.echo("No results.")
        return
    for index, result in enumerate(results, start=1):
        location = f"page {result.page_number}" if result.page_number else "no page"
        typer.echo(f"\n[{index}] score={result.score:.4f} {result.title} ({location})")
        typer.echo(f"    {result.path}")
        if result.heading:
            typer.echo(f"    heading: {result.heading}")
        typer.echo(_preview(result.text))


@app.command()
def stats(
    config_path: Path = typer.Option(
        Path(".knowledge-vector/config.toml"),
        "--config",
        help="Configuration file path.",
    ),
) -> None:
    """Show database statistics."""
    config = load_config(config_path)
    with SQLiteStore(config.database_path) as store:
        value = store.stats()
    typer.echo(f"Database: {value['database_path']}")
    typer.echo(f"Documents: {value['documents']}")
    typer.echo(f"Chunks: {value['chunks']}")
    typer.echo(f"FTS tokenizer: {value['fts_tokenizer']}")
    typer.echo(f"Embedding models: {', '.join(value['embedding_models']) or 'none'}")
    if value["source_types"]:
        typer.echo("Source types:")
        for source_type, count in value["source_types"].items():
            typer.echo(f"  {source_type}: {count}")


@app.command()
def documents(
    config_path: Path = typer.Option(
        Path(".knowledge-vector/config.toml"),
        "--config",
        help="Configuration file path.",
    ),
    limit: int = typer.Option(100, "--limit", min=1),
) -> None:
    """List indexed documents."""
    config = load_config(config_path)
    with SQLiteStore(config.database_path) as store:
        rows = store.list_documents(limit=limit)
    if not rows:
        typer.echo("No documents.")
        return
    for row in rows:
        typer.echo(f"{row['source_type']:>10}  {row['title']}  {row['path']}")


def _preview(text: str, limit: int = 500) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return f"    {compact}"
    return f"    {compact[:limit].rstrip()}..."
