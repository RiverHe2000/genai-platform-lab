"""The promotion decision: quality (non-inferior to the incumbent, above an absolute floor),
service levels (p95 latency, error rate) and sample size, rendered as a report with the
sections a model-validation function expects."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from llmgate.evaluation.compare import Comparison, compare_runs, render_comparison
from llmgate.evaluation.runner import EvalRunReport


class PromotionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_mean_score: float = Field(0.0, ge=0.0, le=1.0)
    non_inferiority_margin: float = Field(0.05, ge=0.0, le=1.0)
    require_improvement: bool = False
    max_p95_latency_ms: float | None = Field(default=None, gt=0)
    max_error_rate: float = Field(0.02, ge=0.0, le=1.0)
    min_cases: int = Field(20, ge=1)
    n_boot: int = Field(2000, ge=0)
    seed: int = 0

    @classmethod
    def load(cls, path: Path | str) -> PromotionPolicy:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
        return cls.model_validate(data or {})


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    passed: bool
    observed: str
    threshold: str


@dataclass(frozen=True, slots=True)
class Decision:
    promote: bool
    checks: list[Check]
    comparison: Comparison


def decide(candidate: EvalRunReport, baseline: EvalRunReport, policy: PromotionPolicy) -> Decision:
    comparison = compare_runs(
        candidate,
        baseline,
        n_boot=policy.n_boot,
        seed=policy.seed,
        non_inferiority_margin=policy.non_inferiority_margin,
    )
    checks = [
        Check(
            "sample size",
            comparison.n_pairs >= policy.min_cases,
            str(comparison.n_pairs),
            f">= {policy.min_cases}",
        ),
        Check(
            "absolute quality floor",
            candidate.mean_score >= policy.min_mean_score,
            f"{candidate.mean_score:.3f}",
            f">= {policy.min_mean_score:.3f}",
        ),
        Check(
            "quality vs baseline",
            comparison.verdict in ("better", "non-inferior")
            and (comparison.verdict == "better" or not policy.require_improvement),
            comparison.verdict,
            "better"
            if policy.require_improvement
            else f"non-inferior (margin {policy.non_inferiority_margin:.3f})",
        ),
        Check(
            "error rate",
            candidate.error_rate <= policy.max_error_rate,
            f"{candidate.error_rate:.1%}",
            f"<= {policy.max_error_rate:.1%}",
        ),
    ]
    if policy.max_p95_latency_ms is not None:
        p95 = candidate.p95_latency_ms
        checks.append(
            Check(
                "p95 latency",
                not math.isnan(p95) and p95 <= policy.max_p95_latency_ms,
                f"{p95:.0f} ms",
                f"<= {policy.max_p95_latency_ms:.0f} ms",
            )
        )
    return Decision(promote=all(c.passed for c in checks), checks=checks, comparison=comparison)


def _run_row(label: str, r: EvalRunReport) -> str:
    return (
        f"| {label} | {r.mean_score:.3f} | {r.n_error} | {r.p50_latency_ms:.0f} | "
        f"{r.p95_latency_ms:.0f} | {r.total_tokens} |"
    )


def render_report(
    decision: Decision, candidate: EvalRunReport, baseline: EvalRunReport, policy: PromotionPolicy
) -> str:
    c = decision.comparison
    lines = [
        "# Model promotion report",
        "",
        f"- generated: {datetime.now(UTC).isoformat(timespec='seconds')}  ",
        f"- candidate: `{candidate.model}` ({candidate.target})  ",
        f"- baseline: `{baseline.model}` ({baseline.target})  ",
        f"- decision: **{'PROMOTE' if decision.promote else 'HOLD'}**  ",
        f"- policy: floor {policy.min_mean_score:.2f}, margin {policy.non_inferiority_margin:.2f}, "
        f"max error rate {policy.max_error_rate:.1%}, min cases {policy.min_cases}",
        "",
        "## 1. Scope",
        "",
        "The candidate deployment is compared with the incumbent on the same evaluation cases "
        "through the same gateway path, so differences reflect the model, not the plumbing.",
        "",
        "## 2. Data",
        "",
        f"{c.n_pairs} paired cases; tags: {', '.join(sorted(candidate.by_tag)) or 'none'}.",
        "",
        "## 3. Results",
        "",
        "| Run | Mean score | Errors | p50 ms | p95 ms | Tokens |",
        "|---|---:|---:|---:|---:|---:|",
        _run_row("candidate", candidate),
        _run_row("baseline", baseline),
        "",
        "Per-tag mean score (candidate / baseline):",
        "",
        "| Tag | n | Candidate | Baseline |",
        "|---|---:|---:|---:|",
    ]
    for tag in sorted(set(candidate.by_tag) | set(baseline.by_tag)):
        cv = candidate.by_tag.get(tag, {}).get("mean_score", math.nan)
        bv = baseline.by_tag.get(tag, {}).get("mean_score", math.nan)
        n = int(candidate.by_tag.get(tag, baseline.by_tag.get(tag, {})).get("n", 0))
        lines.append(f"| {tag} | {n} | {cv:.3f} | {bv:.3f} |")
    lines += [
        "",
        "## 4. Statistical tests",
        "",
        render_comparison(c).rstrip(),
        "",
        "Paired percentile bootstrap on per-case score differences; exact two-sided McNemar test "
        "on discordant pairs. Verdicts: better (CI above 0), non-inferior (lower bound above "
        "-margin), worse (upper bound below -margin), inconclusive otherwise.",
        "",
        "## 5. Policy checks",
        "",
        "| Check | Observed | Threshold | Result |",
        "|---|---|---|:---:|",
    ]
    for ch in decision.checks:
        lines.append(
            f"| {ch.name} | {ch.observed} | {ch.threshold} | {'PASS' if ch.passed else 'FAIL'} |"
        )
    lines += [
        "",
        "## 6. Limitations",
        "",
        "Scores are deterministic string/schema checks on short answers; they measure task "
        "correctness on this suite, not general capability or safety. Latency figures depend on "
        "the serving hardware and concurrency at the time of the run.",
        "",
        "## 7. Decision",
        "",
        f"**{'PROMOTE' if decision.promote else 'HOLD'}** — "
        + (
            "all policy checks passed."
            if decision.promote
            else "failed checks: "
            + ", ".join(ch.name for ch in decision.checks if not ch.passed)
            + "."
        ),
        "",
    ]
    return "\n".join(lines)


def save_decision(decision: Decision, report_md: str, out_dir: Path | str) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = out / "promotion_report.md"
    js = out / "promotion.json"
    md.write_text(report_md, encoding="utf-8")
    js.write_text(
        json.dumps(
            {
                "promote": decision.promote,
                "checks": [asdict(ch) for ch in decision.checks],
                "comparison": asdict(decision.comparison),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return md, js
