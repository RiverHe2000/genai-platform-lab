"""Chunkers that return contiguous character spans of the source document.

Invariants (tested in ``tests/test_chunking.py``):

* every chunk's text equals ``doc.text[start:end]``;
* chunks are emitted in document order;
* the union of chunks covers every non-whitespace character of the document;
* ``RecursiveChunker`` never exceeds ``max_words`` and never overlaps;
* ``FixedWindowChunker`` windows have exactly ``overlap_words`` words in common.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragpipe.documents import Chunk, Document, make_chunk_id
from ragpipe.textproc import paragraph_spans, sentence_spans, word_count, word_spans


class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


def _make(doc: Document, index: int, start: int, end: int) -> Chunk:
    text = doc.text[start:end]
    return Chunk(
        chunk_id=make_chunk_id(doc.doc_id, index, text),
        doc_id=doc.doc_id,
        index=index,
        text=text,
        start=start,
        end=end,
        title=doc.title,
    )


@dataclass(frozen=True)
class FixedWindowChunker:
    """Sliding window over words with overlap — the classic baseline."""

    window_words: int = 180
    overlap_words: int = 40

    def __post_init__(self) -> None:
        if self.window_words <= 0:
            msg = "window_words must be positive"
            raise ValueError(msg)
        if not 0 <= self.overlap_words < self.window_words:
            msg = "overlap_words must satisfy 0 <= overlap < window"
            raise ValueError(msg)

    def chunk(self, doc: Document) -> list[Chunk]:
        words = word_spans(doc.text)
        if not words:
            return []
        step = self.window_words - self.overlap_words
        chunks: list[Chunk] = []
        i = 0
        while True:
            j = min(i + self.window_words, len(words))
            chunks.append(_make(doc, len(chunks), words[i][0], words[j - 1][1]))
            if j == len(words):
                return chunks
            i += step


@dataclass(frozen=True)
class RecursiveChunker:
    """Paragraph → sentence → word splitting, then greedy packing up to ``max_words``.

    Structure-aware chunks keep a policy clause together instead of cutting it mid-sentence,
    which matters for both retrieval precision and for the faithfulness judge (a half
    sentence cannot support a statement).
    """

    max_words: int = 180

    def __post_init__(self) -> None:
        if self.max_words <= 0:
            msg = "max_words must be positive"
            raise ValueError(msg)

    def _pieces(self, text: str) -> list[tuple[int, int]]:
        pieces: list[tuple[int, int]] = []
        for p_start, p_end in paragraph_spans(text):
            paragraph = text[p_start:p_end]
            if word_count(paragraph) <= self.max_words:
                pieces.append((p_start, p_end))
                continue
            for s_start, s_end in sentence_spans(paragraph):
                sentence = paragraph[s_start:s_end]
                abs_start = p_start + s_start
                if word_count(sentence) <= self.max_words:
                    pieces.append((abs_start, p_start + s_end))
                    continue
                words = word_spans(sentence)
                for k in range(0, len(words), self.max_words):
                    last = min(k + self.max_words, len(words)) - 1
                    pieces.append((abs_start + words[k][0], abs_start + words[last][1]))
        return pieces

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        cur: tuple[int, int, int] | None = None  # start, end, words
        for start, end in self._pieces(doc.text):
            n = word_count(doc.text[start:end])
            if cur is None:
                cur = (start, end, n)
            elif cur[2] + n <= self.max_words:
                cur = (cur[0], end, cur[2] + n)
            else:
                chunks.append(_make(doc, len(chunks), cur[0], cur[1]))
                cur = (start, end, n)
        if cur is not None:
            chunks.append(_make(doc, len(chunks), cur[0], cur[1]))
        return chunks


def chunk_documents(docs: list[Document], chunker: Chunker) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc))
    return chunks
