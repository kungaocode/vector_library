import json
from pathlib import Path

import pymupdf as fitz
from typer.testing import CliRunner

from knowledge_vector.cli import app


def test_cli_pipeline_supports_all_source_types(tmp_path: Path) -> None:
    config_path = tmp_path / ".knowledge-vector" / "config.toml"
    sources = tmp_path / "sources"
    sources.mkdir()
    runner = CliRunner()

    init_result = runner.invoke(app, ["init", "--config", str(config_path)])
    assert init_result.exit_code == 0, init_result.output

    (sources / "note.md").write_text(
        "# 注意力机制\nTransformer 使用自注意力。",
        encoding="utf-8",
    )
    (sources / "page.html").write_text(
        """
        <html>
          <head><title>向量检索</title></head>
          <body>
            <h1>余弦相似度</h1>
            <p>向量检索通过余弦相似度找回片段。</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    (sources / "plain.txt").write_text(
        "SQLite FTS5 提供本地关键词检索。",
        encoding="utf-8",
    )
    (sources / "data.json").write_text(
        json.dumps(
            {
                "title": "JSON 知识",
                "content": "JSON 文档也会进入同一个片段索引。",
                "tags": ["json"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (sources / "records.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"title": "JSONL 记录", "text": "逐行 JSON 记录支持批量导入。"},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "title": "混合检索",
                        "text": "关键词分数与向量分数可以融合。",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "PDF retrieval uses embedding similarity.")
    pdf.save(sources / "retrieval.pdf")
    pdf.close()

    ingest_result = runner.invoke(
        app,
        ["ingest", str(sources), "--config", str(config_path)],
    )
    assert ingest_result.exit_code == 0, ingest_result.output
    assert "discovered=6" in ingest_result.output
    assert "indexed=6" in ingest_result.output
    assert "failed=0" in ingest_result.output

    repeat_result = runner.invoke(
        app,
        ["ingest", str(sources), "--config", str(config_path)],
    )
    assert repeat_result.exit_code == 0, repeat_result.output
    assert "skipped=6" in repeat_result.output
    assert "indexed=0" in repeat_result.output

    lexical_result = runner.invoke(
        app,
        ["search", "余弦相似度", "--mode", "lexical", "--config", str(config_path)],
    )
    assert lexical_result.exit_code == 0, lexical_result.output
    assert "向量检索" in lexical_result.output

    vector_result = runner.invoke(
        app,
        ["search", "embedding similarity", "--mode", "vector", "--config", str(config_path)],
    )
    assert vector_result.exit_code == 0, vector_result.output
    assert "retrieval.pdf" in vector_result.output

    hybrid_result = runner.invoke(
        app,
        ["search", "Transformer 注意力", "--config", str(config_path)],
    )
    assert hybrid_result.exit_code == 0, hybrid_result.output
    assert "note.md" in hybrid_result.output

    stats_result = runner.invoke(app, ["stats", "--config", str(config_path)])
    assert stats_result.exit_code == 0, stats_result.output
    assert "Documents: 6" in stats_result.output
    assert "Embedding models: hashing-v1" in stats_result.output

    documents_result = runner.invoke(app, ["documents", "--config", str(config_path)])
    assert documents_result.exit_code == 0, documents_result.output
    assert "page.html" in documents_result.output

    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'provider = "hashing"',
            'provider = "openai-compatible"',
        ),
        encoding="utf-8",
    )
    lexical_without_api = runner.invoke(
        app,
        ["search", "余弦相似度", "--mode", "lexical", "--config", str(config_path)],
    )
    assert lexical_without_api.exit_code == 0, lexical_without_api.output
    assert "向量检索" in lexical_without_api.output
