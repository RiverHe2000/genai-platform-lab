from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ragpipe.evaluation.dataset import EvalSample
from ragpipe.evaluation.judge import Judge
from ragpipe.evaluation.runner import (
    EvalConfig,
    bootstrap_ci,
    load_report,
    render_markdown,
    run_evaluation,
    save_report,
    summarise,
)
from ragpipe.llm import FakeLLM
from ragpipe.pipeline import IndexBundle, RAGPipeline
from ragpipe.retrieval import BM25Retriever

SAMPLES = [
    EvalSample(
        id="s1",
        question="What is the maximum LVR without lenders mortgage insurance?",
        ground_truth="The maximum LVR without LMI is 80%.",
        gold_doc_ids=["mortgages"],
        tags=["credit", "numeric"],
    ),
    EvalSample(
        id="s2",
        question="What is the foreign exchange desk VaR limit?",
        ground_truth="AUD 2 million.",
        gold_doc_ids=["var"],
        tags=["market"],
    ),
    EvalSample(id="s3", question="Who is the CRO?", ground_truth=None, tags=["unanswerable"]),
]

GENERATOR_RULES = [
    (r"lenders mortgage insurance", "The maximum LVR without LMI is 80% [1]."),
    (r"foreign exchange desk", "The FX desk limit is AUD 2 million [1]."),
]
JUDGE_RULES = [
    (r"Break the ANSWER", '{"statements": ["one full statement"]}'),
    (r"STATEMENTS TO CHECK", '{"verdicts": [{"id": 1, "verdict": 1, "reason": ""}]}'),
    (
        r"different questions",
        '{"questions": ["What is the maximum LVR without lenders mortgage insurance?"], "noncommittal": 0}',
    ),
    (r"CONTEXT PASSAGE", '{"useful": 1}'),
    (r"Split ONLY the REFERENCE", '{"items": [{"sentence": "x", "attributed": 1}]}'),
]


def _pipeline(bundle: IndexBundle) -> RAGPipeline:
    return RAGPipeline(
        BM25Retriever(bundle.bm25), bundle.chunk_map, FakeLLM(rules=GENERATOR_RULES), top_k=3
    )


def test_bootstrap_ci_properties() -> None:
    lo, hi = bootstrap_ci([0.0, 1.0, 1.0, 1.0, 0.0, 1.0], n_boot=500, seed=1)
    assert 0.0 <= lo <= 4 / 6 <= hi <= 1.0
    assert bootstrap_ci([0.7], n_boot=100) == (0.7, 0.7)
    assert bootstrap_ci([0.2, 0.4], n_boot=0) == pytest.approx((0.3, 0.3))
    assert all(math.isnan(x) for x in bootstrap_ci([]))
    assert bootstrap_ci([0.1, 0.9, 0.5], seed=3) == bootstrap_ci([0.1, 0.9, 0.5], seed=3)


def test_summarise_counts_missing() -> None:
    s = summarise("m", [1.0, None, 0.0, math.nan], EvalConfig(n_bootstrap=10))
    assert (s.n, s.n_missing) == (2, 2)
    assert s.mean == 0.5
    empty = summarise("m", [None], EvalConfig())
    assert empty.n == 0 and empty.mean is None


def test_run_evaluation_end_to_end_with_fakes(bundle: IndexBundle, tmp_path: Path) -> None:
    judge = Judge(FakeLLM(rules=JUDGE_RULES, default="{}"), max_retries=0)
    cfg = EvalConfig(k=3, n_bootstrap=50, seed=0, n_relevancy_questions=1)
    seen: list[str] = []
    report = run_evaluation(
        _pipeline(bundle),
        SAMPLES,
        cfg,
        judge=judge,
        embedder=bundle.embedder,
        generator_name="fake-gen",
        progress=lambda _i, _n, r: seen.append(r.id),
    )
    assert seen == ["s1", "s2", "s3"]
    assert report.n_samples == 3
    assert report.generator == "fake-gen"
    assert report.judge == "fake"
    assert report.retriever == "bm25"

    s1 = report.records[0]
    assert s1.metrics["retrieval/hit_rate@3"] == 1.0
    assert s1.metrics["retrieval/mrr"] == 1.0
    assert s1.metrics["faithfulness"] == 1.0
    assert s1.metrics["answer_relevancy"] == pytest.approx(1.0, abs=1e-5)
    assert s1.metrics["context_precision"] == 1.0
    assert s1.metrics["context_recall"] == 1.0
    assert s1.metrics["lexical/reference_coverage"] == 1.0
    assert s1.metrics["false_abstention"] == 0.0
    assert s1.cited_chunk_ids == [s1.retrieved_chunk_ids[0]]

    s3 = report.records[2]
    assert s3.abstained is True
    assert s3.metrics["correct_abstention"] == 1.0
    assert "faithfulness" not in s3.metrics
    assert "retrieval/mrr" not in s3.metrics

    assert report.summaries["faithfulness"].n == 2
    assert report.summaries["correct_abstention"].n == 1
    assert report.by_tag["credit"]["n"] == 1.0
    assert report.by_tag["unanswerable"]["abstained"] == 1.0
    assert report.judge_calls > 0
    assert report.judge_parse_failures == 0

    md = render_markdown(report)
    assert "| faithfulness |" in md
    assert "## By tag" in md
    assert "## Lowest faithfulness" in md

    json_path, md_path = save_report(report, tmp_path / "run")
    loaded = load_report(json_path)
    assert loaded.model_dump() == report.model_dump()
    assert md_path.read_text(encoding="utf-8") == md
    assert json.loads(json_path.read_text(encoding="utf-8"))["n_samples"] == 3


def test_run_evaluation_without_judge_skips_ragas(bundle: IndexBundle) -> None:
    report = run_evaluation(
        _pipeline(bundle), SAMPLES[:1], EvalConfig(metrics=["retrieval", "lexical"], n_bootstrap=0)
    )
    assert report.judge is None
    assert "faithfulness" not in report.summaries
    assert report.summaries["lexical/f1"].n == 1
    assert report.metric_values("lexical/f1") == {"s1": report.records[0].metrics["lexical/f1"]}
    assert "By tag" in render_markdown(report)


def test_judge_failures_become_missing_not_zero(bundle: IndexBundle) -> None:
    judge = Judge(FakeLLM(default="I refuse to answer in JSON"), max_retries=0)
    report = run_evaluation(
        _pipeline(bundle),
        SAMPLES[:1],
        EvalConfig(n_bootstrap=0),
        judge=judge,
        embedder=bundle.embedder,
    )
    assert report.records[0].metrics["faithfulness"] is None
    assert report.summaries["faithfulness"].n_missing == 1
    assert report.judge_parse_failures > 0
    assert "n/a" in render_markdown(report)


def test_report_with_all_missing_metric_round_trips_through_json(
    bundle: IndexBundle, tmp_path: Path
) -> None:
    judge = Judge(FakeLLM(default="never JSON"), max_retries=0)
    report = run_evaluation(
        _pipeline(bundle),
        SAMPLES[:2],
        EvalConfig(n_bootstrap=0),
        judge=judge,
        embedder=bundle.embedder,
    )
    assert report.summaries["context_recall"].mean is None
    json_path, _ = save_report(report, tmp_path / "nan_run")
    loaded = load_report(json_path)
    assert loaded.summaries["context_recall"].n == 0
    assert "null" in json_path.read_text(encoding="utf-8")
