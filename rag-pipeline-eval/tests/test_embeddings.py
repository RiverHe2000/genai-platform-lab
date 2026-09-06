from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ragpipe.embeddings import (
    BGE_QUERY_PREFIX,
    CachedEmbedder,
    HashEmbedder,
    SentenceTransformerEmbedder,
    l2_normalize,
)


def test_l2_normalize_handles_zero_rows() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0.0, 0.0]])
    assert out.dtype == np.float32
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(3))


def test_hash_embedder_is_deterministic_unit_norm_and_sensible() -> None:
    a = HashEmbedder(dim=256)
    b = HashEmbedder(dim=256)
    texts = [
        "the LCR internal trigger is 110%",
        "liquidity coverage ratio trigger",
        "VaR desk limit",
    ]
    va = a.embed_documents(texts)
    vb = b.embed_documents(texts)
    np.testing.assert_array_equal(va, vb)
    np.testing.assert_allclose(np.linalg.norm(va, axis=1), 1.0, atol=1e-6)
    assert va.shape == (3, 256)
    assert va[0] @ va[1] > va[0] @ va[2]
    assert a.name == "hash-256"
    assert a.dim == 256
    assert a.embed_queries([]).shape == (0, 256)
    with pytest.raises(ValueError):
        HashEmbedder(dim=4)


def test_hash_embedder_seed_changes_vectors() -> None:
    v0 = HashEmbedder(dim=128, seed=0).embed_documents(["capital ratio"])
    v1 = HashEmbedder(dim=128, seed=1).embed_documents(["capital ratio"])
    assert not np.allclose(v0, v1)


class _FakeEncoder:
    def __init__(self, dim: int | None = 8) -> None:
        self.calls: list[list[str]] = []
        self._dim = dim

    def encode(self, sentences: list[str], **_kwargs: Any) -> NDArray[np.float32]:
        self.calls.append(list(sentences))
        rng = np.random.default_rng(len(sentences))
        return rng.normal(size=(len(sentences), 8)).astype(np.float32) * 3.0

    def get_sentence_embedding_dimension(self) -> int | None:
        return self._dim


def test_sentence_transformer_wrapper_prefixes_queries_only() -> None:
    enc = _FakeEncoder()
    emb = SentenceTransformerEmbedder("some/model", model=enc)
    docs = emb.embed_documents(["passage one", "passage two"])
    emb.embed_queries(["what is x"])
    assert enc.calls == [["passage one", "passage two"], [BGE_QUERY_PREFIX + "what is x"]]
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-6)
    assert emb.dim == 8
    assert emb.name == "st[some/model]"
    assert emb.embed_documents([]).shape == (0, 8)
    assert emb.embed_queries([]).shape == (0, 8)


def test_sentence_transformer_wrapper_probes_dimension_when_unknown() -> None:
    enc = _FakeEncoder(dim=None)
    emb = SentenceTransformerEmbedder("m", model=enc)
    assert emb.dim == 8
    assert enc.calls == [["probe"]]


class _CountingEmbedder:
    name = "counting"
    dim = 16

    def __init__(self) -> None:
        self.doc_calls: list[Sequence[str]] = []
        self.query_calls: list[Sequence[str]] = []
        self._inner = HashEmbedder(dim=16)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.doc_calls.append(texts)
        return self._inner.embed_documents(texts)

    def embed_queries(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.query_calls.append(texts)
        return -self._inner.embed_queries(texts)


def test_cached_embedder_hits_misses_order_and_modes(tmp_path: Path) -> None:
    inner = _CountingEmbedder()
    cache = CachedEmbedder(inner, tmp_path / "cache.sqlite")
    first = cache.embed_documents(["a", "b", "c"])
    assert (cache.hits, cache.misses) == (0, 3)
    second = cache.embed_documents(["c", "x", "a"])
    assert (cache.hits, cache.misses) == (2, 4)
    # misses are embedded in one batch (order is by cache key, not input order)
    assert [sorted(c) for c in inner.doc_calls] == [["a", "b", "c"], ["x"]]
    np.testing.assert_array_equal(second[0], first[2])
    np.testing.assert_array_equal(second[2], first[0])
    # queries are keyed separately from documents (different vectors for the same text)
    q = cache.embed_queries(["a"])
    assert inner.query_calls == [["a"]]
    np.testing.assert_array_equal(q[0], -first[0])
    assert cache.embed_documents([]).shape == (0, 16)
    assert cache.name == "counting"
    assert cache.dim == 16
    cache.close()

    # persistence: a fresh instance over the same file never calls the inner embedder
    inner2 = _CountingEmbedder()
    cache2 = CachedEmbedder(inner2, tmp_path / "cache.sqlite")
    again = cache2.embed_documents(["a", "b", "c", "x"])
    assert inner2.doc_calls == []
    np.testing.assert_array_equal(again[:3], first)
    cache2.close()


def test_cached_embedder_deduplicates_repeated_texts_in_one_batch() -> None:
    inner = _CountingEmbedder()
    cache = CachedEmbedder(inner)
    out = cache.embed_documents(["same", "same", "other"])
    assert inner.doc_calls == [["other", "same"]] or inner.doc_calls == [["same", "other"]]
    np.testing.assert_array_equal(out[0], out[1])
