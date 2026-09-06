"""Runs a pipeline over an evaluation set, computes per-sample metrics, and aggregates them
with percentile-bootstrap confidence intervals and per-tag slices."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ragpipe import __version__
from ragpipe.embeddings import Embedder
from ragpipe.evaluation.dataset import EvalSample
from ragpipe.evaluation.judge import Judge
from ragpipe.evaluation.ragas_metrics import (
    MetricResult,
    answer_relevancy,
    context_precision,
    context_recall,
    exact_match,
    faithfulness,
    reference_coverage,
    token_f1,
)
from ragpipe.evaluation.retrieval_metrics import retrieval_scores
from ragpipe.pipeline import RAGPipeline

MetricGroup = Literal["retrieval", "lexical", "ragas"]


def _default_metrics() -> list[MetricGroup]:
    return ["retrieval", "lexical", "ragas"]


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = Field(5, ge=1)
    metrics: list[MetricGroup] = Field(default_factory=_default_metrics)
    n_bootstrap: int = Field(1000, ge=0)
    seed: int = 0
    ci_level: float = Field(0.95, gt=0.0, lt=1.0)
    n_relevancy_questions: int = Field(3, ge=1)


class SampleRecord(BaseModel):
    id: str
    question: str
    ground_truth: str | None
    tags: list[str]
    answerable: bool
    answer: str
    abstained: bool
    model: str
    gold_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    retrieved_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    contexts: list[str]
    metrics: dict[str, float | None]
    details: dict[str, Any] = Field(default_factory=dict)
    timings_s: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    """``None`` means no sample produced a value (JSON has no NaN)."""

    name: str
    mean: float | None
    ci_low: float | None
    ci_high: float | None
    n: int
    n_missing: int


class EvalReport(BaseModel):
    created_at: str
    version: str
    retriever: str
    generator: str
    judge: str | None
    config: EvalConfig
    n_samples: int
    summaries: dict[str, MetricSummary]
    by_tag: dict[str, dict[str, float]]
    records: list[SampleRecord]
    judge_calls: int = 0
    judge_parse_failures: int = 0

    def metric_values(self, name: str) -> dict[str, float | None]:
        return {r.id: r.metrics.get(name) for r in self.records}


# ----- statistics ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (math.nan, math.nan)
    if arr.size == 1 or n_boot == 0:
        m = float(arr.mean())
        return (m, m)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def summarise(name: str, values: Sequence[float | None], cfg: EvalConfig) -> MetricSummary:
    present = [v for v in values if v is not None and not math.isnan(v)]
    n_missing = len(values) - len(present)
    if not present:
        return MetricSummary(
            name=name, mean=None, ci_low=None, ci_high=None, n=0, n_missing=n_missing
        )
    lo, hi = bootstrap_ci(present, n_boot=cfg.n_bootstrap, seed=cfg.seed, level=cfg.ci_level)
    return MetricSummary(
        name=name,
        mean=float(np.mean(present)),
        ci_low=lo,
        ci_high=hi,
        n=len(present),
        n_missing=n_missing,
    )


# ----- per-sample evaluation ----------------------------------------------------------------


def _put(metrics: dict[str, float | None], details: dict[str, Any], r: MetricResult) -> None:
    metrics[r.name] = r.value
    if r.detail:
        details[r.name] = r.detail


def evaluate_sample(
    pipeline: RAGPipeline,
    sample: EvalSample,
    cfg: EvalConfig,
    *,
    judge: Judge | None = None,
    embedder: Embedder | None = None,
) -> SampleRecord:
    result = pipeline.answer(sample.question)
    metrics: dict[str, float | None] = {}
    details: dict[str, Any] = {}

    if "retrieval" in cfg.metrics and sample.gold_doc_ids:
        gold = set(sample.gold_doc_ids)
        for k, v in retrieval_scores(result.retrieved_doc_ids, gold, cfg.k).items():
            metrics[f"retrieval/{k}"] = v

    metrics["abstained"] = float(result.abstained)
    if sample.answerable:
        metrics["false_abstention"] = float(result.abstained)
    else:
        metrics["correct_abstention"] = float(result.abstained)

    if "lexical" in cfg.metrics and sample.ground_truth is not None:
        metrics["lexical/f1"] = token_f1(result.answer, sample.ground_truth)
        metrics["lexical/exact_match"] = exact_match(result.answer, sample.ground_truth)
        metrics["lexical/reference_coverage"] = reference_coverage(
            result.answer, sample.ground_truth
        )

    if "ragas" in cfg.metrics and judge is not None:
        contexts = result.context_texts
        if not result.abstained:
            _put(metrics, details, faithfulness(judge, sample.question, result.answer, contexts))
        if sample.ground_truth is not None:
            if embedder is not None:
                _put(
                    metrics,
                    details,
                    answer_relevancy(
                        judge,
                        embedder,
                        sample.question,
                        result.answer,
                        n_questions=cfg.n_relevancy_questions,
                    ),
                )
            gt = sample.ground_truth
            _put(metrics, details, context_precision(judge, sample.question, gt, contexts))
            _put(metrics, details, context_recall(judge, sample.question, gt, contexts))

    return SampleRecord(
        id=sample.id,
        question=sample.question,
        ground_truth=sample.ground_truth,
        tags=list(sample.tags),
        answerable=sample.answerable,
        answer=result.answer,
        abstained=result.abstained,
        model=result.model,
        gold_doc_ids=list(sample.gold_doc_ids),
        retrieved_doc_ids=result.retrieved_doc_ids,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        cited_chunk_ids=list(result.cited_chunk_ids),
        contexts=result.context_texts,
        metrics=metrics,
        details=details,
        timings_s=result.timings_s,
        usage=result.usage,
    )


def aggregate(
    records: Sequence[SampleRecord], cfg: EvalConfig
) -> tuple[dict[str, MetricSummary], dict[str, dict[str, float]]]:
    names: list[str] = []
    for r in records:
        for n in r.metrics:
            if n not in names:
                names.append(n)
    summaries = {n: summarise(n, [r.metrics.get(n) for r in records], cfg) for n in names}
    by_tag: dict[str, dict[str, float]] = {}
    tags = sorted({t for r in records for t in r.tags})
    for tag in tags:
        subset = [r for r in records if tag in r.tags]
        by_tag[tag] = {"n": float(len(subset))}
        for n in names:
            vals = [v for r in subset if (v := r.metrics.get(n)) is not None and not math.isnan(v)]
            if vals:
                by_tag[tag][n] = float(np.mean(vals))
    return summaries, by_tag


def run_evaluation(
    pipeline: RAGPipeline,
    samples: Sequence[EvalSample],
    cfg: EvalConfig,
    *,
    judge: Judge | None = None,
    embedder: Embedder | None = None,
    generator_name: str = "",
    progress: Callable[[int, int, SampleRecord], None] | None = None,
) -> EvalReport:
    records: list[SampleRecord] = []
    for i, sample in enumerate(samples, start=1):
        record = evaluate_sample(pipeline, sample, cfg, judge=judge, embedder=embedder)
        records.append(record)
        if progress is not None:
            progress(i, len(samples), record)
    summaries, by_tag = aggregate(records, cfg)
    return EvalReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        version=__version__,
        retriever=pipeline.retriever_name,
        generator=generator_name or (records[0].model if records else ""),
        judge=judge.name if judge is not None else None,
        config=cfg,
        n_samples=len(records),
        summaries=summaries,
        by_tag=by_tag,
        records=records,
        judge_calls=len(judge.calls) if judge is not None else 0,
        judge_parse_failures=judge.parse_failures if judge is not None else 0,
    )


# ----- reporting ----------------------------------------------------------------------------


def _fmt(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.3f}"


def render_markdown(report: EvalReport) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- created: {report.created_at}  ",
        f"- retriever: `{report.retriever}`  ",
        f"- generator: `{report.generator}`  ",
        f"- judge: `{report.judge or 'none'}`  ",
        f"- samples: {report.n_samples}, k = {report.config.k}, "
        f"bootstrap = {report.config.n_bootstrap} resamples, CI level = {report.config.ci_level}  ",
        f"- judge calls: {report.judge_calls}, "
        f"unparseable after retry: {report.judge_parse_failures}",
        "",
        "## Metrics",
        "",
        "| Metric | Mean | 95% CI | n | missing |",
        "|---|---:|:---:|---:|---:|",
    ]
    for s in report.summaries.values():
        lines.append(
            f"| {s.name} | {_fmt(s.mean)} | [{_fmt(s.ci_low)}, {_fmt(s.ci_high)}] | "
            f"{s.n} | {s.n_missing} |"
        )
    if report.by_tag:
        names = list(report.summaries)
        lines += [
            "",
            "## By tag",
            "",
            "| Tag | n | " + " | ".join(names) + " |",
            "|---|---:|" + "---:|" * len(names),
        ]
        for tag, values in report.by_tag.items():
            cells = [_fmt(values[n]) if n in values else "n/a" for n in names]
            lines.append(f"| {tag} | {int(values['n'])} | " + " | ".join(cells) + " |")
    worst = sorted(
        (r for r in report.records if r.metrics.get("faithfulness") is not None),
        key=lambda r: r.metrics["faithfulness"] or 0.0,
    )[:5]
    if worst:
        lines += ["", "## Lowest faithfulness", ""]
        for r in worst:
            lines.append(
                f"- **{r.id}** ({_fmt(r.metrics['faithfulness'] or 0.0)}): "
                f"{r.answer[:160].replace(chr(10), ' ')}"
            )
    return "\n".join(lines) + "\n"


def save_report(report: EvalReport, out_dir: Path | str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def load_report(path: Path | str) -> EvalReport:
    return EvalReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
