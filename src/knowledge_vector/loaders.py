from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import fitz
from bs4 import BeautifulSoup

from knowledge_vector.models import LoadedDocument, TextBlock

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".txt",
    ".json",
    ".jsonl",
}

TYPE_BY_EXTENSION = {
    ".pdf": "pdf",
    ".html": "wiki",
    ".htm": "wiki",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".json": "wiki",
    ".jsonl": "wiki",
}


class UnsupportedSourceError(ValueError):
    pass


def discover_files(
    path: Path,
    *,
    recursive: bool = True,
    patterns: Iterable[str] | None = None,
) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if not path.is_dir():
        return []

    selected: list[Path] = []
    candidates = path.rglob("*") if recursive else path.glob("*")
    normalized_patterns = list(patterns or [])
    for candidate in sorted(candidates):
        if not candidate.is_file() or candidate.name.startswith("."):
            continue
        if candidate.name.endswith(".meta.json"):
            continue
        if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        relative = candidate.relative_to(path).as_posix()
        if normalized_patterns and not any(
            candidate.match(pattern) or Path(relative).match(pattern)
            for pattern in normalized_patterns
        ):
            continue
        selected.append(candidate)
    return selected


def load_document(
    path: Path,
    *,
    source_type_override: str | None = None,
) -> LoadedDocument:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedSourceError(f"Unsupported source format: {resolved.suffix}")

    sidecar = _read_sidecar(resolved)
    source_type = source_type_override or str(
        sidecar.get("source_type", TYPE_BY_EXTENSION[resolved.suffix.lower()])
    )
    suffix = resolved.suffix.lower()

    if suffix == ".pdf":
        title, blocks = _load_pdf(resolved)
    elif suffix in {".html", ".htm"}:
        title, blocks = _load_html(resolved)
    elif suffix in {".md", ".markdown"}:
        title, blocks = _load_markdown(resolved)
    elif suffix in {".json", ".jsonl"}:
        title, blocks = _load_json(resolved)
    else:
        title, blocks = _load_text(resolved)

    title = str(sidecar.get("title") or title or resolved.stem).strip()
    normalized_blocks_list: list[TextBlock] = []
    for block in blocks:
        normalized = _normalize_text(block.text)
        if normalized:
            normalized_blocks_list.append(
                TextBlock(
                    text=normalized,
                    heading=block.heading,
                    page_number=block.page_number,
                    metadata=block.metadata,
                )
            )
    normalized_blocks = tuple(normalized_blocks_list)
    full_text = _normalize_text("\n\n".join(block.text for block in normalized_blocks))
    document_id = _stable_id("document", resolved.as_posix())
    metadata = {key: value for key, value in sidecar.items() if key not in {"title", "source_type"}}
    metadata.setdefault("file_name", resolved.name)
    metadata.setdefault("extension", resolved.suffix.lower())
    fingerprint = json.dumps(
        {
            "text": full_text,
            "title": title,
            "source_type": source_type,
            "metadata": metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    content_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    return LoadedDocument(
        id=document_id,
        path=resolved,
        source_type=source_type,
        title=title,
        text=full_text,
        content_hash=content_hash,
        blocks=normalized_blocks,
        metadata=metadata,
    )


def _read_sidecar(path: Path) -> dict[str, Any]:
    sidecar = Path(f"{path}.meta.json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid sidecar metadata: {sidecar}") from exc
    return data if isinstance(data, dict) else {}


def _load_pdf(path: Path) -> tuple[str, list[TextBlock]]:
    with fitz.open(path) as document:
        pdf_title = (document.metadata or {}).get("title")
        blocks: list[TextBlock] = []
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text")
            if text.strip():
                blocks.append(TextBlock(text=text, page_number=page_index))
        title = str(pdf_title or "").strip()
    return title, blocks


def _load_html(path: Path) -> tuple[str, list[TextBlock]]:
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer", "form"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    container = soup.find("article") or soup.find("main") or soup.body or soup
    blocks: list[TextBlock] = []
    current_heading: str | None = None

    for element in container.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote"]
    ):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            current_heading = text
            blocks.append(TextBlock(text=text, heading=current_heading))
        else:
            blocks.append(TextBlock(text=text, heading=current_heading))

    if not blocks:
        text = container.get_text("\n", strip=True)
        if text:
            blocks.append(TextBlock(text=text))
    return title, blocks


def _load_markdown(path: Path) -> tuple[str, list[TextBlock]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    title = ""
    heading: str | None = None
    blocks: list[TextBlock] = []
    paragraph: list[str] = []
    in_code = False

    def flush() -> None:
        if paragraph:
            combined = "\n".join(paragraph).strip()
            if combined:
                blocks.append(TextBlock(text=combined, heading=heading))
            paragraph.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            paragraph.append(line)
            continue
        heading_match = None if in_code else re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if heading_match:
            flush()
            heading = heading_match.group(2).strip()
            if not title and len(heading_match.group(1)) == 1:
                title = heading
            blocks.append(TextBlock(text=heading, heading=heading))
        elif not stripped and not in_code:
            flush()
        else:
            paragraph.append(line)
    flush()
    return title, blocks


def _load_text(path: Path) -> tuple[str, list[TextBlock]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = re.split(r"\n\s*\n+", text)
    return "", [TextBlock(text=paragraph) for paragraph in paragraphs if paragraph.strip()]


def _load_json(path: Path) -> tuple[str, list[TextBlock]]:
    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                records.append(json.loads(line))
    else:
        records = [json.loads(path.read_text(encoding="utf-8"))]

    if len(records) == 1 and isinstance(records[0], list):
        records = records[0]

    blocks: list[TextBlock] = []
    title = ""
    for record in records:
        if isinstance(record, str):
            blocks.append(TextBlock(text=record))
            continue
        if not isinstance(record, dict):
            continue
        title = title or str(record.get("title") or record.get("name") or "")
        content = (
            record.get("text")
            or record.get("content")
            or record.get("body")
            or record.get("extract")
            or record.get("summary")
        )
        if content:
            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"title", "name", "text", "content", "body", "extract", "summary"}
            }
            blocks.append(
                TextBlock(
                    text=str(content),
                    heading=str(record.get("title") or "") or None,
                    metadata=metadata,
                )
            )
    if not blocks:
        blocks.append(TextBlock(text=json.dumps(records, ensure_ascii=False, indent=2)))
    return title, blocks


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"
