from pathlib import Path

from knowledge_vector.chunking import chunk_document
from knowledge_vector.config import ChunkingConfig
from knowledge_vector.models import LoadedDocument, TextBlock


def test_long_block_is_split_with_overlap() -> None:
    text = "。".join(f"第{index}段知识内容" for index in range(80))
    document = LoadedDocument(
        id="doc_1",
        path=Path("/tmp/example.txt"),
        source_type="text",
        title="example",
        text=text,
        content_hash="hash",
        blocks=(TextBlock(text=text),),
    )

    chunks = chunk_document(document, ChunkingConfig(max_chars=200, overlap_chars=40))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert all(chunk.document_id == "doc_1" for chunk in chunks)
    assert chunks[0].ordinal == 0
    assert chunks[1].char_start is not None
    assert chunks[0].char_end is not None
    assert chunks[1].char_start < chunks[0].char_end
