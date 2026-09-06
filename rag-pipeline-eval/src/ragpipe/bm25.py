"""Okapi BM25 with an inverted index, implemented from the formula so it can be tested
against hand-computed scores rather than trusted as a black box.

score(q, d) = Σ_{t ∈ q} idf(t) · tf(t, d)·(k1 + 1) / (tf(t, d) + k1·(1 - b + b·|d| / avgdl))
idf(t)      = ln((N - n_t + 0.5) / (n_t + 0.5) + 1)          (Lucene / rank_bm25 variant, ≥ 0)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75
    doc_ids: list[str] = field(default_factory=list)
    doc_lengths: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # ----- construction ---------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        doc_ids: Sequence[str],
        tokenized_docs: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        if len(doc_ids) != len(tokenized_docs):
            msg = "doc_ids and tokenized_docs must have the same length"
            raise ValueError(msg)
        if len(set(doc_ids)) != len(doc_ids):
            msg = "doc_ids must be unique"
            raise ValueError(msg)
        if k1 <= 0 or not 0.0 <= b <= 1.0:
            msg = "k1 must be > 0 and b in [0, 1]"
            raise ValueError(msg)
        postings: dict[str, list[tuple[int, int]]] = {}
        lengths = np.zeros(len(doc_ids), dtype=np.int64)
        for idx, tokens in enumerate(tokenized_docs):
            lengths[idx] = len(tokens)
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                postings.setdefault(t, []).append((idx, tf))
        return cls(k1=k1, b=b, doc_ids=list(doc_ids), doc_lengths=lengths, postings=postings)

    # ----- properties -----------------------------------------------------------------------

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def avgdl(self) -> float:
        return float(self.doc_lengths.mean()) if self.n_docs else 0.0

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, ()))

    def idf(self, term: str) -> float:
        n_t = self.document_frequency(term)
        return math.log((self.n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)

    # ----- scoring --------------------------------------------------------------------------

    def scores(self, query_tokens: Sequence[str]) -> NDArray[np.float64]:
        """BM25 score of every document for the query (duplicated query terms count twice,
        matching rank_bm25's behaviour)."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0:
            return scores
        avgdl = self.avgdl
        for term in query_tokens:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for idx, tf in plist:
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_lengths[idx] / avgdl)
                scores[idx] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return scores

    def search(self, query_tokens: Sequence[str], k: int) -> list[tuple[str, float]]:
        """Top-``k`` (doc_id, score) with score > 0, ties broken by insertion order."""
        if k <= 0:
            return []
        scores = self.scores(query_tokens)
        order = np.argsort(-scores, kind="stable")
        out: list[tuple[str, float]] = []
        for idx in order[:k]:
            s = float(scores[idx])
            if s <= 0.0:
                break
            out.append((self.doc_ids[idx], s))
        return out

    # ----- persistence ----------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": list(self.doc_ids),
            "doc_lengths": self.doc_lengths.tolist(),
            "postings": {t: [[i, tf] for i, tf in plist] for t, plist in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BM25Index:
        return cls(
            k1=float(data["k1"]),
            b=float(data["b"]),
            doc_ids=[str(d) for d in data["doc_ids"]],
            doc_lengths=np.asarray(data["doc_lengths"], dtype=np.int64),
            postings={
                str(t): [(int(i), int(tf)) for i, tf in plist]
                for t, plist in data["postings"].items()
            },
        )
