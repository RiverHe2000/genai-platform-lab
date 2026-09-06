"""Runs a suite against a model through an OpenAI-compatible client and records per-case
scores, latencies and errors — the input to ``compare`` and ``promote``."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field

from llmgate import __version__
from llmgate.evaluation.suite import EvalCase, score_case


class CaseResult(BaseModel):
    id: str
    tags: list[str]
    kind: str
    score: float
    reason: str
    output: str
    latency_ms: float
    status: int
    completion_tokens: int = 0


class EvalRunReport(BaseModel):
    created_at: str
    version: str
    model: str
    target: str
    n: int
    n_error: int
    mean_score: float
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_tokens: int
    by_tag: dict[str, dict[str, float]] = Field(default_factory=dict)
    results: list[CaseResult]

    def scores(self) -> dict[str, float]:
        return {r.id: r.score for r in self.results}


async def _run_case(client: httpx.AsyncClient, case: EvalCase, model: str) -> CaseResult:
    body = case.to_request(model).model_dump(exclude_none=True, by_alias=True)
    started = time.perf_counter()
    try:
        resp = await client.post("/v1/chat/completions", json=body)
    except httpx.HTTPError as exc:
        return CaseResult(
            id=case.id,
            tags=case.tags,
            kind=case.kind,
            score=0.0,
            reason=f"transport error: {exc}",
            output="",
            latency_ms=(time.perf_counter() - started) * 1000,
            status=0,
        )
    latency = (time.perf_counter() - started) * 1000
    if resp.status_code != 200:
        return CaseResult(
            id=case.id,
            tags=case.tags,
            kind=case.kind,
            score=0.0,
            reason=f"HTTP {resp.status_code}: {resp.text[:120]}",
            output="",
            latency_ms=latency,
            status=resp.status_code,
        )
    data = resp.json()
    output = str(data["choices"][0]["message"]["content"]) if data.get("choices") else ""
    tokens = int(data.get("usage", {}).get("completion_tokens", 0))
    cs = score_case(case, output)
    return CaseResult(
        id=case.id,
        tags=case.tags,
        kind=case.kind,
        score=cs.score,
        reason=cs.reason,
        output=output[:500],
        latency_ms=latency,
        status=200,
        completion_tokens=tokens,
    )


async def run_suite(
    client: httpx.AsyncClient,
    cases: Sequence[EvalCase],
    *,
    model: str,
    target: str,
    concurrency: int = 4,
    progress: Callable[[CaseResult], None] | None = None,
) -> EvalRunReport:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(case: EvalCase) -> CaseResult:
        async with sem:
            result = await _run_case(client, case, model)
        if progress is not None:
            progress(result)
        return result

    results = list(await asyncio.gather(*(guarded(c) for c in cases)))
    ok = [r for r in results if r.status == 200]
    latencies = [r.latency_ms for r in ok]
    by_tag: dict[str, dict[str, float]] = {}
    for tag in sorted({t for r in results for t in r.tags}):
        subset = [r for r in results if tag in r.tags]
        by_tag[tag] = {
            "n": float(len(subset)),
            "mean_score": float(np.mean([r.score for r in subset])),
        }
    return EvalRunReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        version=__version__,
        model=model,
        target=target,
        n=len(results),
        n_error=len(results) - len(ok),
        mean_score=float(np.mean([r.score for r in results])) if results else 0.0,
        error_rate=(len(results) - len(ok)) / len(results) if results else 0.0,
        p50_latency_ms=float(np.percentile(latencies, 50)) if latencies else float("nan"),
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else float("nan"),
        total_tokens=sum(r.completion_tokens for r in results),
        by_tag=by_tag,
        results=results,
    )


def render_markdown(report: EvalRunReport) -> str:
    lines = [
        f"# Evaluation run — `{report.model}` via {report.target}",
        "",
        f"- created: {report.created_at}  ",
        f"- cases: {report.n}, errors: {report.n_error} ({report.error_rate:.1%})  ",
        f"- mean score: **{report.mean_score:.3f}**  ",
        f"- latency p50 / p95: {report.p50_latency_ms:.0f} / {report.p95_latency_ms:.0f} ms  ",
        f"- completion tokens: {report.total_tokens}",
        "",
        "| Case | Kind | Score | Reason | Output |",
        "|---|---|---:|---|---|",
    ]
    for r in report.results:
        out = r.output.replace("|", "\\|").replace("\n", " ")[:80]
        lines.append(
            f"| {r.id} | {r.kind} | {r.score:.1f} | {r.reason.replace('|', '/')[:60]} | {out} |"
        )
    if report.by_tag:
        lines += ["", "| Tag | n | Mean score |", "|---|---:|---:|"]
        for tag, v in report.by_tag.items():
            lines.append(f"| {tag} | {int(v['n'])} | {v['mean_score']:.3f} |")
    return "\n".join(lines) + "\n"


def save_report(report: EvalRunReport, out_dir: Path | str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def load_report(path: Path | str) -> EvalRunReport:
    return EvalRunReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def report_summary(report: EvalRunReport) -> dict[str, Any]:
    return {
        "model": report.model,
        "n": report.n,
        "mean_score": report.mean_score,
        "p95_latency_ms": report.p95_latency_ms,
    }
