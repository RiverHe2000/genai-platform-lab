"""Paired comparison of two evaluation runs on the same cases: bootstrap interval for the
mean difference, an exact McNemar test for the binary win/loss pattern, and a verdict with
an explicit non-inferiority margin — the vocabulary a model-risk reviewer expects."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from llmgate.evaluation.runner import EvalRunReport


@dataclass(frozen=True, slots=True)
class Comparison:
    n_pairs: int
    mean_candidate: float
    mean_baseline: float
    delta: float
    ci_low: float
    ci_high: float
    p_improve: float
    wins: int
    losses: int
    mcnemar_p: float
    verdict: str


def mcnemar_exact(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (binomial, p = 0.5)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2.0 * tail))


def compare_runs(
    candidate: EvalRunReport,
    baseline: EvalRunReport,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
    non_inferiority_margin: float = 0.0,
) -> Comparison:
    cand = candidate.scores()
    base = baseline.scores()
    ids = [i for i in cand if i in base]
    if not ids:
        return Comparison(
            0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, 0, 0, 1.0, "no pairs"
        )
    c = np.asarray([cand[i] for i in ids], dtype=np.float64)
    b = np.asarray([base[i] for i in ids], dtype=np.float64)
    deltas = c - b
    wins = int((deltas > 0).sum())
    losses = int((deltas < 0).sum())
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
        n_pairs=len(ids),
        mean_candidate=float(c.mean()),
        mean_baseline=float(b.mean()),
        delta=float(deltas.mean()),
        ci_low=lo,
        ci_high=hi,
        p_improve=p_improve,
        wins=wins,
        losses=losses,
        mcnemar_p=mcnemar_exact(wins, losses),
        verdict=verdict,
    )


def render_comparison(c: Comparison) -> str:
    return (
        "| n | Candidate | Baseline | delta | 95% CI | P(delta>0) | wins / losses "
        "| McNemar p | Verdict |\n"
        "|---:|---:|---:|---:|:---:|---:|---:|---:|---|\n"
        f"| {c.n_pairs} | {c.mean_candidate:.3f} | {c.mean_baseline:.3f} | {c.delta:+.3f} | "
        f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}] | {c.p_improve:.2f} | {c.wins} / {c.losses} | "
        f"{c.mcnemar_p:.3f} | {c.verdict} |\n"
    )
