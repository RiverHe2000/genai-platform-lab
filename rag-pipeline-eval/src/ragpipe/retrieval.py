"""Retrievers: lexical (BM25), dense, hybrid fusion (RRF or convex) and cross-encoder
reranking. Every hit carries the raw per-source scores so a retrieval trace can explain
*why* a passage ranked where it did."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ragpipe.bm25 import BM25Index
from ragpipe.embeddings import Embedder
from ragpipe.textproc import tokenize
from ragpipe.vectorstore import VectorStore


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: str
    score: float
    rank: int
    sources: Mapping[str, float]


class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(self, query: str, k: int) -> list[Hit]: ...


# ----- single-source retrievers -------------------------------------------------------------


class BM25Retriever:
    def __init__(self, index: BM25Index) -> None:
        self._index = index

    @property
    def name(self) -> str:
        return "bm25"

    def retrieve(self, query: str, k: int) -> list[Hit]:
        hits = self._index.search(tokenize(query), k)
        return [
            Hit(chunk_id=cid, score=s, rank=r, sources={"bm25": s})
            for r, (cid, s) in enumerate(hits, start=1)
        ]


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    @property
    def name(self) -> str:
        return f"dense[{self._embedder.name}]"

    def retrieve(self, query: str, k: int) -> list[Hit]:
        q = self._embedder.embed_queries([query])[0]
        return [
            Hit(chunk_id=h.id, score=h.score, rank=r, sources={"dense": h.score})
            for r, h in enumerate(self._store.search(q, k), start=1)
        ]


# ----- fusion -------------------------------------------------------------------------------


def rrf_fuse(
    rankings: Sequence[Sequence[str]], *, k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion (Cormack et al., 2009): score(d) = Σ_r w_r / (k + rank_r(d)).

    Rank-based, so it needs no score calibration between BM25 (unbounded) and cosine ([-1, 1]).
    """
    if k < 1:
        msg = "k must be >= 1"
        raise ValueError(msg)
    w = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(w) != len(rankings):
        msg = "weights must match rankings"
        raise ValueError(msg)
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking, weight in zip(rankings, w, strict=True):
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + weight / (k + rank)
            first_seen.setdefault(doc, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


def convex_fuse(
    score_lists: Sequence[Sequence[tuple[str, float]]], weights: Sequence[float]
) -> list[tuple[str, float]]:
    """Weighted sum of min-max normalised scores; a document absent from a list scores 0
    there. Sensitive to the score distributions, which is why RRF is the default."""
    if len(score_lists) != len(weights):
        msg = "weights must match score_lists"
        raise ValueError(msg)
    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for scores, weight in zip(score_lists, weights, strict=True):
        if not scores:
            continue
        values = [s for _, s in scores]
        lo, hi = min(values), max(values)
        span = hi - lo
        for doc, s in scores:
            norm = 1.0 if span == 0.0 else (s - lo) / span
            fused[doc] = fused.get(doc, 0.0) + weight * norm
            first_seen.setdefault(doc, len(first_seen))
    return sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


class HybridRetriever:
    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        fusion: Literal["rrf", "convex"] = "rrf",
        rrf_k: int = 60,
        weights: Sequence[float] | None = None,
        candidate_k: int = 20,
    ) -> None:
        if not retrievers:
            msg = "need at least one retriever"
            raise ValueError(msg)
        self._retrievers = list(retrievers)
        self._fusion = fusion
        self._rrf_k = rrf_k
        self._weights = list(weights) if weights is not None else [1.0] * len(retrievers)
        if len(self._weights) != len(self._retrievers):
            msg = "weights must match retrievers"
            raise ValueError(msg)
        self._candidate_k = candidate_k

    @property
    def name(self) -> str:
        parts = "+".join(r.name for r in self._retrievers)
        return f"hybrid[{self._fusion}]({parts})"

    def retrieve(self, query: str, k: int) -> list[Hit]:
        per_source = [r.retrieve(query, max(self._candidate_k, k)) for r in self._retrievers]
        if self._fusion == "rrf":
            fused = rrf_fuse(
                [[h.chunk_id for h in hits] for hits in per_source],
                k=self._rrf_k,
                weights=self._weights,
            )
        else:
            fused = convex_fuse(
                [[(h.chunk_id, h.score) for h in hits] for hits in per_source], self._weights
            )
        raw: dict[str, dict[str, float]] = {}
        for hits in per_source:
            for h in hits:
                raw.setdefault(h.chunk_id, {}).update(h.sources)
        return [
            Hit(chunk_id=cid, score=s, rank=r, sources={**raw.get(cid, {}), "fused": s})
            for r, (cid, s) in enumerate(fused[:k], start=1)
        ]


# ----- reranking ----------------------------------------------------------------------------


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderLike(Protocol):
    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 32,
        device: str | None = None,
        model: CrossEncoderLike | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model: CrossEncoderLike | None = model

    @property
    def name(self) -> str:
        return f"cross-encoder[{self._model_name}]"

    def _load(self) -> CrossEncoderLike:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        out = self._load().predict(
            [(query, p) for p in passages],
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(x) for x in out]


class RerankingRetriever:
    """Re-scores the top ``candidate_k`` hits of ``base`` with a cross-encoder."""

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        text_of: Callable[[str], str],
        *,
        candidate_k: int = 20,
    ) -> None:
        self._base = base
        self._reranker = reranker
        self._text_of = text_of
        self._candidate_k = candidate_k

    @property
    def name(self) -> str:
        return f"{self._base.name}+rerank"

    def retrieve(self, query: str, k: int) -> list[Hit]:
        candidates = self._base.retrieve(query, max(self._candidate_k, k))
        if not candidates:
            return []
        scores = self._reranker.score(query, [self._text_of(c.chunk_id) for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].rank))
        return [
            Hit(
                chunk_id=candidates[i].chunk_id,
                score=scores[i],
                rank=r,
                sources={**candidates[i].sources, "rerank": scores[i]},
            )
            for r, i in enumerate(order[:k], start=1)
        ]
