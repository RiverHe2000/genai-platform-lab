from __future__ import annotations

from ragpipe.evaluation.dataset import EvalSample
from ragpipe.evaluation.retrieval_bench import bench_retrievers, render_bench_markdown
from ragpipe.pipeline import IndexBundle
from ragpipe.retrieval import BM25Retriever, DenseRetriever


def test_bench_retrievers_scores_and_renders(bundle: IndexBundle) -> None:
    samples = [
        EvalSample(
            id="a",
            question="maximum LVR without lenders mortgage insurance",
            ground_truth="80%",
            gold_doc_ids=["mortgages"],
        ),
        EvalSample(
            id="b",
            question="foreign exchange desk VaR limit",
            ground_truth="2m",
            gold_doc_ids=["var"],
        ),
        EvalSample(id="c", question="unanswerable", ground_truth=None, gold_doc_ids=[]),
    ]
    rows = bench_retrievers(
        bundle,
        {
            "bm25": BM25Retriever(bundle.bm25),
            "dense": DenseRetriever(bundle.embedder, bundle.store),
        },
        samples,
        k=3,
        n_boot=20,
    )
    assert [r.name for r in rows] == ["bm25", "dense"]
    assert rows[0].n == 2
    assert rows[0].metrics["hit_rate@3"] == 1.0
    assert set(rows[0].per_sample) == {"a", "b"}
    assert rows[0].ci["mrr"][0] <= rows[0].metrics["mrr"] <= rows[0].ci["mrr"][1]
    md = render_bench_markdown(rows, k=3)
    assert "| `bm25` | 2 |" in md and "hit_rate@1" in md
    assert rows[0].to_dict()["name"] == "bm25"


def test_bench_with_no_usable_samples(bundle: IndexBundle) -> None:
    rows = bench_retrievers(bundle, {"bm25": BM25Retriever(bundle.bm25)}, [], k=3, n_boot=0)
    assert rows[0].n == 0 and rows[0].metrics == {}
    assert "n/a" in render_bench_markdown(rows, k=3)
