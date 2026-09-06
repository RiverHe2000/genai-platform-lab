from __future__ import annotations

from pathlib import Path

import pytest

from ragpipe.documents import (
    Document,
    corpus_fingerprint,
    load_corpus,
    make_chunk_id,
    read_jsonl,
    write_jsonl,
)


def test_load_corpus_reads_supported_files_sorted_with_titles(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# Bravo Policy\n\nbody", encoding="utf-8")
    (tmp_path / "a.txt").write_text("plain text without heading", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].title == "a"
    assert docs[1].title == "Bravo Policy"
    assert docs[1].metadata["source"] == "b.md"
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


def test_fingerprint_is_order_independent_and_content_sensitive() -> None:
    a = Document("x", "X", "one")
    b = Document("y", "Y", "two")
    assert corpus_fingerprint([a, b]) == corpus_fingerprint([b, a])
    assert corpus_fingerprint([a, b]) != corpus_fingerprint([a, Document("y", "Y", "three")])


def test_make_chunk_id_stable() -> None:
    assert make_chunk_id("doc", 3, "text") == make_chunk_id("doc", 3, "text")
    assert make_chunk_id("doc", 3, "text") != make_chunk_id("doc", 4, "text")
    assert make_chunk_id("doc", 3, "text").startswith("doc#003-")


def test_jsonl_roundtrip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    assert write_jsonl(path, [{"a": 1}, {"b": "ü"}]) == 2
    assert list(read_jsonl(path)) == [{"a": 1}, {"b": "ü"}]
    path.write_text('{"a": 1}\n\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        list(read_jsonl(path))
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        list(read_jsonl(path))
