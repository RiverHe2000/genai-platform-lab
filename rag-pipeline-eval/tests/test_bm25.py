from __future__ import annotations

import math

import numpy as np
import pytest

from ragpipe.bm25 import BM25Index

DOCS = [["a", "b", "c"], ["a", "d"], ["e", "f", "g", "h"]]
IDS = ["d0", "d1", "d2"]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.build(IDS, DOCS, k1=1.5, b=0.75)


def test_idf_matches_hand_computation(index: BM25Index) -> None:
    # N = 3, n_a = 2 -> ln((3 - 2 + 0.5) / (2 + 0.5) + 1) = ln(1.6)
    assert index.idf("a") == pytest.approx(math.log(1.6))
    # unseen term: ln((3 + 0.5) / 0.5 + 1) = ln(8)
    assert index.idf("zzz") == pytest.approx(math.log(8.0))
    assert index.avgdl == pytest.approx(3.0)


def test_scores_match_hand_computation(index: BM25Index) -> None:
    scores = index.scores(["a"])
    idf = math.log(1.6)
    # d0: tf=1, |d|=3=avgdl -> norm = 1.5 -> idf * 2.5 / 2.5
    assert scores[0] == pytest.approx(idf * 1.0)
    # d1: tf=1, |d|=2 -> norm = 1.5 * (0.25 + 0.75 * 2/3) = 1.125 -> idf * 2.5 / 2.125
    assert scores[1] == pytest.approx(idf * 2.5 / 2.125)
    assert scores[2] == 0.0
    assert scores[1] > scores[0], "shorter document wins under length normalisation"


def test_term_frequency_saturates() -> None:
    idx = BM25Index.build(["x", "y"], [["t"] * 1, ["t"] * 50], k1=1.2, b=0.0)
    s1, s50 = idx.scores(["t"])
    assert s50 > s1
    assert s50 < idx.idf("t") * (1.2 + 1.0)


def test_duplicate_query_terms_count_twice(index: BM25Index) -> None:
    assert index.scores(["a", "a"])[0] == pytest.approx(2 * index.scores(["a"])[0])


def test_search_orders_filters_zero_and_limits_k(index: BM25Index) -> None:
    hits = index.search(["a", "h"], k=10)
    # "h" is rarer (idf ln(8/3)) than "a" (idf ln(1.6)), so d2 wins; d1 beats d0 on length
    assert [h[0] for h in hits] == ["d2", "d1", "d0"]
    assert all(s > 0 for _, s in hits)
    assert len(index.search(["a"], k=1)) == 1
    assert index.search(["nope"], k=5) == []
    assert index.search(["a"], k=0) == []


def test_search_ties_resolve_by_insertion_order() -> None:
    idx = BM25Index.build(["p", "q", "r"], [["t"], ["t"], ["t"]])
    assert [h[0] for h in idx.search(["t"], k=3)] == ["p", "q", "r"]


def test_roundtrip_preserves_scores(index: BM25Index) -> None:
    restored = BM25Index.from_dict(index.to_dict())
    np.testing.assert_allclose(restored.scores(["a", "b", "h"]), index.scores(["a", "b", "h"]))
    assert restored.doc_ids == index.doc_ids


def test_empty_index_and_validation() -> None:
    empty = BM25Index.build([], [])
    assert empty.n_docs == 0
    assert empty.avgdl == 0.0
    assert empty.search(["a"], k=3) == []
    with pytest.raises(ValueError):
        BM25Index.build(["a"], [])
    with pytest.raises(ValueError):
        BM25Index.build(["a", "a"], [["x"], ["y"]])
    with pytest.raises(ValueError):
        BM25Index.build(["a"], [["x"]], k1=0)
    with pytest.raises(ValueError):
        BM25Index.build(["a"], [["x"]], b=1.5)
