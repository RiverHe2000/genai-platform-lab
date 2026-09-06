"""Rank-based retrieval metrics (binary relevance). Pure functions, tested against
hand-computed values in ``tests/test_retrieval_metrics.py``.

``retrieved`` is the ranked list of ids (duplicates removed, first occurrence kept) and
``relevant`` the gold set. Metrics that need at least one relevant item return ``nan`` when
the gold set is empty rather than silently reporting 0 or 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def dedupe(ids: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return seen


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(r in relevant for r in dedupe(retrieved)[:k]) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return math.nan
    top = dedupe(retrieved)[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        msg = "k must be positive"
        raise ValueError(msg)
    top = dedupe(retrieved)[:k]
    return sum(1 for r in top if r in relevant) / k


def mrr(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, r in enumerate(dedupe(retrieved), start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return math.nan
    hits = 0
    total = 0.0
    for rank, r in enumerate(dedupe(retrieved), start=1):
        if r in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return math.nan
    top = dedupe(retrieved)[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(top, start=1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def retrieval_scores(retrieved: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    return {
        "hit_rate@1": hit_rate_at_k(retrieved, relevant, 1),
        f"hit_rate@{k}": hit_rate_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        "mrr": mrr(retrieved, relevant),
        "map": average_precision(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }
