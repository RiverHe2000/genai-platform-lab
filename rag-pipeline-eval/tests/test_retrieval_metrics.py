from __future__ import annotations

import math

import pytest

from ragpipe.evaluation.retrieval_metrics import (
    average_precision,
    dedupe,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    retrieval_scores,
)

RANKED = ["x", "a", "x", "b", "y", "c"]
GOLD = {"a", "b", "c"}


def test_dedupe_keeps_first_occurrence() -> None:
    assert dedupe(RANKED) == ["x", "a", "b", "y", "c"]


def test_hit_rate_recall_precision() -> None:
    assert hit_rate_at_k(RANKED, GOLD, 1) == 0.0
    assert hit_rate_at_k(RANKED, GOLD, 2) == 1.0
    assert recall_at_k(RANKED, GOLD, 3) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, GOLD, 10) == 1.0
    assert precision_at_k(RANKED, GOLD, 3) == pytest.approx(2 / 3)
    assert math.isnan(recall_at_k(RANKED, set(), 3))
    with pytest.raises(ValueError):
        precision_at_k(RANKED, GOLD, 0)


def test_mrr_and_average_precision() -> None:
    assert mrr(RANKED, GOLD) == pytest.approx(1 / 2)
    assert mrr(["q"], GOLD) == 0.0
    # hits at ranks 2, 3, 5 of the deduplicated list -> (1/2 + 2/3 + 3/5) / 3
    assert average_precision(RANKED, GOLD) == pytest.approx((1 / 2 + 2 / 3 + 3 / 5) / 3)
    assert math.isnan(average_precision(RANKED, set()))


def test_ndcg_hand_computed() -> None:
    # dedup: x a b y c ; k=3 -> rel at ranks 2, 3 -> DCG = 1/log2(3) + 1/log2(4)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, GOLD, 3) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["a", "b", "c"], GOLD, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["q"], GOLD, 3) == 0.0
    assert math.isnan(ndcg_at_k(RANKED, set(), 3))


def test_retrieval_scores_bundle_keys() -> None:
    scores = retrieval_scores(["a", "z"], {"a"}, 5)
    assert set(scores) == {
        "hit_rate@1",
        "hit_rate@5",
        "recall@5",
        "precision@5",
        "mrr",
        "map",
        "ndcg@5",
    }
    assert scores["hit_rate@1"] == 1.0
    assert scores["precision@5"] == pytest.approx(0.2)
    assert scores["map"] == 1.0
