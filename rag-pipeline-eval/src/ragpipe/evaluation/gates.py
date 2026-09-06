"""Quality gates (thresholds on a report) and paired comparison of two reports.

A gate can be evaluated on the point estimate or, conservatively, on the confidence bound
(``use_ci``): "the lower 95 % bound of faithfulness must be ≥ 0.8" is a much stronger
statement than "the mean is ≥ 0.8" on 40 samples.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragpipe.evaluation.runner import EvalReport


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    min: float | None = None
    max: float | None = None
    use_ci: bool = False
    min_n: int = Field(1, ge=1)

    @model_validator(mode="after")
    def _one_bound(self) -> Gate:
        if self.min is None and self.max is None:
            msg = f"gate {self.metric}: set min and/or max"
            raise ValueError(msg)
        return self


class GateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gates: list[Gate] = Field(min_length=1)

    @classmethod
    def load(cls, path: Path | str) -> GateSpec:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
        return cls.model_validate(data)


def _num(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{x:.3f}"


@dataclass(frozen=True, slots=True)
class GateOutcome:
    metric: str
    passed: bool
    observed: float | None
    reason: str


def evaluate_gates(report: EvalReport, spec: GateSpec) -> list[GateOutcome]:
    outcomes: list[GateOutcome] = []
    for gate in spec.gates:
        summary = report.summaries.get(gate.metric)
        if summary is None or summary.n == 0:
            outcomes.append(GateOutcome(gate.metric, False, None, "metric missing from report"))
            continue
        if summary.n < gate.min_n:
            outcomes.append(
                GateOutcome(gate.metric, False, summary.mean, f"n={summary.n} < min_n={gate.min_n}")
            )
            continue
        reasons: list[str] = []
        if gate.min is not None:
            observed = summary.ci_low if gate.use_ci else summary.mean
            label = "ci_low" if gate.use_ci else "mean"
            if observed is None or math.isnan(observed) or observed < gate.min:
                reasons.append(f"{label}={_num(observed)} < min={gate.min}")
        if gate.max is not None:
            observed = summary.ci_high if gate.use_ci else summary.mean
            label = "ci_high" if gate.use_ci else "mean"
            if observed is None or math.isnan(observed) or observed > gate.max:
                reasons.append(f"{label}={_num(observed)} > max={gate.max}")
        outcomes.append(
            GateOutcome(gate.metric, not reasons, summary.mean, "; ".join(reasons) or "ok")
        )
    return outcomes


def all_passed(outcomes: list[GateOutcome]) -> bool:
    return all(o.passed for o in outcomes)


def render_gates(outcomes: list[GateOutcome]) -> str:
    lines = ["| Metric | Observed mean | Result | Detail |", "|---|---:|:---:|---|"]
    for o in outcomes:
        obs = "n/a" if o.observed is None or math.isnan(o.observed) else f"{o.observed:.3f}"
        lines.append(f"| {o.metric} | {obs} | {'PASS' if o.passed else 'FAIL'} | {o.reason} |")
    return "\n".join(lines) + "\n"


# ----- paired comparison --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Comparison:
    metric: str
    n_pairs: int
    mean_candidate: float
    mean_baseline: float
    delta: float
    ci_low: float
    ci_high: float
    p_improve: float
    verdict: str


def compare_reports(
    candidate: EvalReport,
    baseline: EvalReport,
    metric: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    non_inferiority_margin: float = 0.0,
) -> Comparison:
    """Paired bootstrap on per-sample differences (same questions, same judge).

    Verdicts: ``better`` if the CI excludes 0 from above; ``worse`` if the whole CI lies below
    ``-margin``; ``non-inferior`` if the lower bound is above ``-margin``; else ``inconclusive``.
    """
    cand = candidate.metric_values(metric)
    base = baseline.metric_values(metric)
    pairs = [
        (cv, bv)
        for sid, cv in cand.items()
        if cv is not None
        and not math.isnan(cv)
        and (bv := base.get(sid)) is not None
        and not math.isnan(bv)
    ]
    if not pairs:
        nan = math.nan
        return Comparison(metric, 0, nan, nan, nan, nan, nan, nan, "no pairs")
    c = np.asarray([p[0] for p in pairs], dtype=np.float64)
    b = np.asarray([p[1] for p in pairs], dtype=np.float64)
    deltas = c - b
    if len(deltas) == 1 or n_boot == 0:
        lo = hi = float(deltas.mean())
        p_improve = float(deltas.mean() > 0)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
        means = deltas[idx].mean(axis=1)
        alpha = (1.0 - level) / 2.0
        lo, hi = (float(x) for x in np.quantile(means, [alpha, 1.0 - alpha]))
        p_improve = float((means > 0).mean())
    if lo > 0:
        verdict = "better"
    elif hi < -non_inferiority_margin:
        verdict = "worse"
    elif lo >= -non_inferiority_margin:
        verdict = "non-inferior"
    else:
        verdict = "inconclusive"
    return Comparison(
        metric=metric,
        n_pairs=len(pairs),
        mean_candidate=float(c.mean()),
        mean_baseline=float(b.mean()),
        delta=float(deltas.mean()),
        ci_low=lo,
        ci_high=hi,
        p_improve=p_improve,
        verdict=verdict,
    )


def render_comparison(rows: list[Comparison]) -> str:
    lines = [
        "| Metric | n | Candidate | Baseline | delta | 95% CI | P(delta>0) | Verdict |",
        "|---|---:|---:|---:|---:|:---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.metric} | {r.n_pairs} | {r.mean_candidate:.3f} | {r.mean_baseline:.3f} | "
            f"{r.delta:+.3f} | [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | "
            f"{r.p_improve:.2f} | {r.verdict} |"
        )
    return "\n".join(lines) + "\n"
