"""Corpus records. Chunks keep character offsets into their source document so every
retrieved passage can be traced back to exactly where it came from (auditability)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = (".md", ".txt")


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    start: int
    end: int
    title: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Chunk:
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            index=int(data["index"]),
            text=str(data["text"]),
            start=int(data["start"]),
            end=int(data["end"]),
            title=str(data["title"]),
        )


def make_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Content-addressed id: the same document text always yields the same ids, so an
    evaluation set can reference chunks and an index rebuild does not invalidate it."""
    digest = hashlib.sha1(f"{doc_id}\x00{index}\x00{text}".encode()).hexdigest()
    return f"{doc_id}#{index:03d}-{digest[:8]}"


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
        if stripped:
            break
    return fallback


def load_corpus(directory: Path | str) -> list[Document]:
    """Load every ``.md``/``.txt`` file in ``directory`` (sorted, so ordering is stable)."""
    root = Path(directory)
    if not root.is_dir():
        msg = f"corpus directory not found: {root}"
        raise FileNotFoundError(msg)
    docs: list[Document] = []
    for path in sorted(p for p in root.iterdir() if p.suffix in SUPPORTED_SUFFIXES):
        text = path.read_text(encoding="utf-8")
        docs.append(
            Document(
                doc_id=path.stem,
                title=_title_from_text(text, path.stem),
                text=text,
                metadata={"source": path.name},
            )
        )
    return docs


def corpus_fingerprint(documents: Iterable[Document]) -> str:
    """SHA-256 over (id, text) pairs, order-independent, for index/corpus consistency checks."""
    h = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.doc_id):
        h.update(doc.doc_id.encode())
        h.update(b"\x00")
        h.update(doc.text.encode())
        h.update(b"\x01")
    return h.hexdigest()


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_no}: invalid JSON ({exc.msg})"
                raise ValueError(msg) from exc
            if not isinstance(row, dict):
                msg = f"{path}:{line_no}: expected a JSON object"
                raise ValueError(msg)
            yield row
