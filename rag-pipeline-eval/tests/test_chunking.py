from __future__ import annotations

from itertools import pairwise

import pytest

from ragpipe.chunking import FixedWindowChunker, RecursiveChunker, chunk_documents
from ragpipe.documents import Chunk, Document
from ragpipe.textproc import word_count


def _doc(text: str, doc_id: str = "d") -> Document:
    return Document(doc_id=doc_id, title="T", text=text)


def _assert_invariants(doc: Document, chunks: list[Chunk]) -> None:
    covered = [False] * len(doc.text)
    last_start = -1
    for i, c in enumerate(chunks):
        assert c.index == i
        assert c.doc_id == doc.doc_id
        assert c.text == doc.text[c.start : c.end]
        assert c.start > last_start
        last_start = c.start
        for pos in range(c.start, c.end):
            covered[pos] = True
    for pos, ch in enumerate(doc.text):
        if not ch.isspace():
            assert covered[pos], f"char {pos} ({ch!r}) not covered"


# ----- fixed window -------------------------------------------------------------------------


def test_fixed_window_counts_and_overlap() -> None:
    doc = _doc(" ".join(f"w{i}" for i in range(12)))
    chunks = FixedWindowChunker(window_words=5, overlap_words=2).chunk(doc)
    assert [c.text.split() for c in chunks] == [
        ["w0", "w1", "w2", "w3", "w4"],
        ["w3", "w4", "w5", "w6", "w7"],
        ["w6", "w7", "w8", "w9", "w10"],
        ["w9", "w10", "w11"],
    ]
    _assert_invariants(doc, chunks)
    for a, b in pairwise(chunks):
        assert len(set(a.text.split()) & set(b.text.split())) == 2 or b is chunks[-1]


def test_fixed_window_short_doc_single_chunk_and_empty() -> None:
    doc = _doc("only three words")
    chunks = FixedWindowChunker(window_words=10, overlap_words=3).chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "only three words"
    assert FixedWindowChunker().chunk(_doc("   ")) == []


@pytest.mark.parametrize(("window", "overlap"), [(0, 0), (5, 5), (5, 6), (5, -1)])
def test_fixed_window_validation(window: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        FixedWindowChunker(window_words=window, overlap_words=overlap)


# ----- recursive ----------------------------------------------------------------------------


def test_recursive_keeps_short_paragraphs_and_packs_them() -> None:
    doc = _doc("Para one has four words.\n\nPara two also short.\n\nThird paragraph here now.")
    chunks = RecursiveChunker(max_words=9).chunk(doc)
    assert [c.text for c in chunks] == [
        "Para one has four words.\n\nPara two also short.",
        "Third paragraph here now.",
    ]
    _assert_invariants(doc, chunks)


def test_recursive_splits_long_paragraph_by_sentence() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(6)]
    doc = _doc(" ".join(sentences))
    chunks = RecursiveChunker(max_words=10).chunk(doc)
    assert all(word_count(c.text) <= 10 for c in chunks)
    assert len(chunks) == 3
    assert chunks[0].text == "Sentence number 0 is here. Sentence number 1 is here."
    _assert_invariants(doc, chunks)


def test_recursive_splits_oversized_sentence_by_words() -> None:
    doc = _doc(" ".join(f"w{i}" for i in range(23)))
    chunks = RecursiveChunker(max_words=10).chunk(doc)
    assert [word_count(c.text) for c in chunks] == [10, 10, 3]
    _assert_invariants(doc, chunks)


def test_recursive_never_overlaps_and_covers() -> None:
    text = "\n\n".join(
        " ".join(f"tok{p}_{i}." if i % 7 == 6 else f"tok{p}_{i}" for i in range(p * 13 + 1))
        for p in range(1, 6)
    )
    doc = _doc(text)
    for max_words in (5, 12, 40, 500):
        chunks = RecursiveChunker(max_words=max_words).chunk(doc)
        assert all(word_count(c.text) <= max_words for c in chunks)
        for a, b in pairwise(chunks):
            assert a.end <= b.start
        _assert_invariants(doc, chunks)


def test_recursive_empty_doc_and_validation() -> None:
    assert RecursiveChunker().chunk(_doc("\n\n  \n")) == []
    with pytest.raises(ValueError):
        RecursiveChunker(max_words=0)


def test_chunk_ids_are_deterministic_and_unique() -> None:
    doc = _doc("Alpha beta gamma.\n\nDelta epsilon zeta.")
    a = RecursiveChunker(max_words=3).chunk(doc)
    b = RecursiveChunker(max_words=3).chunk(doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    assert a[0].chunk_id.startswith("d#000-")


def test_chunk_documents_resets_index_per_document() -> None:
    chunks = chunk_documents([_doc("one two", "a"), _doc("three four", "b")], RecursiveChunker())
    assert [(c.doc_id, c.index) for c in chunks] == [("a", 0), ("b", 0)]
    assert Chunk.from_dict(chunks[0].to_dict()) == chunks[0]
