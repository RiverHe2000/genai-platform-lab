from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from ragpipe.pipeline import IndexBundle
from ragpipe.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    Hit,
    HybridRetriever,
    RerankingRetriever,
    convex_fuse,
    rrf_fuse,
)


class ScriptedRetriever:
    def __init__(self, name: str, ranking: Sequence[tuple[str, float]]) -> None:
        self._name = name
        self._ranking = list(ranking)
        self.requested_k: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, _query: str, k: int) -> list[Hit]:
        self.requested_k.append(k)
        return [
            Hit(chunk_id=c, score=s, rank=r, sources={self._name: s})
            for r, (c, s) in enumerate(self._ranking[:k], start=1)
        ]


def test_rrf_fuse_formula_and_tie_breaking() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 63)
    # a/b tie and c/d tie: ties resolve by first appearance in any ranking
    assert [d for d, _ in fused] == ["a", "b", "c", "d"]


def test_rrf_fuse_weights_and_validation() -> None:
    fused = dict(rrf_fuse([["a"], ["b"]], k=10, weights=[1.0, 3.0]))
    assert fused["b"] == pytest.approx(3 / 11)
    assert fused["a"] == pytest.approx(1 / 11)
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], k=0)
    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], weights=[1.0])


def test_convex_fuse_min_max_normalises_and_handles_absent_and_constant() -> None:
    fused = dict(convex_fuse([[("a", 10.0), ("b", 5.0)], [("b", 0.9), ("c", 0.1)]], [0.5, 0.5]))
    assert fused["a"] == pytest.approx(0.5)
    assert fused["b"] == pytest.approx(0.5)
    assert fused["c"] == pytest.approx(0.0)
    assert dict(convex_fuse([[("x", 2.0), ("y", 2.0)]], [1.0])) == {"x": 1.0, "y": 1.0}
    assert convex_fuse([[]], [1.0]) == []
    with pytest.raises(ValueError):
        convex_fuse([[("a", 1.0)]], [1.0, 2.0])


def test_bm25_and_dense_retrievers_return_ranked_hits(bundle: IndexBundle) -> None:
    bm25 = BM25Retriever(bundle.bm25)
    hits = bm25.retrieve("maximum LVR without lenders mortgage insurance", k=3)
    assert hits and hits[0].rank == 1
    assert bundle.chunk_map[hits[0].chunk_id].doc_id == "mortgages"
    assert "bm25" in hits[0].sources
    assert bm25.name == "bm25"

    dense = DenseRetriever(bundle.embedder, bundle.store)
    dhits = dense.retrieve("foreign exchange desk VaR limit", k=2)
    assert len(dhits) == 2
    assert bundle.chunk_map[dhits[0].chunk_id].doc_id == "var"
    assert dense.name == "dense[hash-256]"
    assert [h.rank for h in dhits] == [1, 2]


def test_hybrid_rrf_merges_sources_and_requests_candidates() -> None:
    r1 = ScriptedRetriever("lex", [("a", 3.0), ("b", 2.0), ("c", 1.0)])
    r2 = ScriptedRetriever("den", [("b", 0.9), ("d", 0.8)])
    hybrid = HybridRetriever([r1, r2], fusion="rrf", rrf_k=60, candidate_k=10)
    hits = hybrid.retrieve("q", k=3)
    assert [h.chunk_id for h in hits] == ["b", "a", "d"]
    assert hits[0].sources == {"lex": 2.0, "den": 0.9, "fused": pytest.approx(1 / 62 + 1 / 61)}
    assert hits[1].sources == {"lex": 3.0, "fused": pytest.approx(1 / 61)}
    assert r1.requested_k == [10]
    assert [h.rank for h in hits] == [1, 2, 3]
    assert hybrid.name == "hybrid[rrf](lex+den)"


def test_hybrid_convex_uses_weights() -> None:
    r1 = ScriptedRetriever("lex", [("a", 3.0), ("b", 1.0)])
    r2 = ScriptedRetriever("den", [("b", 1.0), ("a", 0.0)])
    hits = HybridRetriever([r1, r2], fusion="convex", weights=[0.2, 0.8]).retrieve("q", k=2)
    assert [h.chunk_id for h in hits] == ["b", "a"]
    assert hits[0].score == pytest.approx(0.8)
    with pytest.raises(ValueError):
        HybridRetriever([])
    with pytest.raises(ValueError):
        HybridRetriever([r1], weights=[1.0, 2.0])


class LengthReranker:
    name = "len"

    def score(self, _query: str, passages: Sequence[str]) -> list[float]:
        return [float(len(p)) for p in passages]


def test_reranking_retriever_reorders_and_keeps_sources() -> None:
    base = ScriptedRetriever(
        "lex", [("short", 3.0), ("a much longer passage", 2.0), ("mid len", 1.0)]
    )
    rr = RerankingRetriever(base, LengthReranker(), text_of=lambda cid: cid, candidate_k=10)
    hits = rr.retrieve("q", k=2)
    assert [h.chunk_id for h in hits] == ["a much longer passage", "mid len"]
    assert hits[0].sources == {"lex": 2.0, "rerank": float(len("a much longer passage"))}
    assert [h.rank for h in hits] == [1, 2]
    assert rr.name == "lex+rerank"
    assert base.requested_k == [10]
    empty = RerankingRetriever(ScriptedRetriever("e", []), LengthReranker(), text_of=lambda c: c)
    assert empty.retrieve("q", k=3) == []


class _FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, sentences: list[tuple[str, str]], **_kwargs: Any) -> list[float]:
        self.calls.append(sentences)
        return [float(i) for i in range(len(sentences))]


def test_cross_encoder_reranker_wraps_predict() -> None:
    ce = _FakeCrossEncoder()
    reranker = CrossEncoderReranker("m", model=ce)
    assert reranker.score("q", ["p1", "p2"]) == [0.0, 1.0]
    assert ce.calls == [[("q", "p1"), ("q", "p2")]]
    assert reranker.score("q", []) == []
    assert reranker.name == "cross-encoder[m]"
