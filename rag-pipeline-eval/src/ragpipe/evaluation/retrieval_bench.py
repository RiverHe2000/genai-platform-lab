"""Compare retriever configurations on retrieval metrics only — no LLM involved, so the
benchmark is deterministic and runs in seconds. This is the experiment behind the retrieval
table in ``docs/RESULTS.md``."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ragpipe.evaluation.dataset import EvalSample
from ragpipe.evaluation.retrieval_metrics import retrieval_scores
from ragpipe.evaluation.runner import bootstrap_ci
from ragpipe.pipeline import IndexBundle
from ragpipe.retrieval import Retriever


@dataclass
class BenchRow:
    name: str
    n: int
    metrics: dict[str, float]
    ci: dict[str, tuple[float, float]]
    ms_per_query: float
    per_sample: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            "metrics": {k: (None if math.isnan(v) else v) for k, v in self.metrics.items()},
            "ci": {k: [None if math.isnan(x) else x for x in v] for k, v in self.ci.items()},
            "ms_per_query": self.ms_per_query,
        }


def bench_retrievers(
    bundle: IndexBundle,
    retrievers: Mapping[str, Retriever],
    samples: Sequence[EvalSample],
    *,
    k: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[BenchRow]:
    chunk_map = bundle.chunk_map
    usable = [s for s in samples if s.gold_doc_ids]
    rows: list[BenchRow] = []
    for name, retriever in retrievers.items():
        per_sample: dict[str, dict[str, float]] = {}
        started = time.perf_counter()
        for s in usable:
            hits = retriever.retrieve(s.question, k)
            doc_ids: list[str] = []
            for h in hits:
                d = chunk_map[h.chunk_id].doc_id
                if d not in doc_ids:
                    doc_ids.append(d)
            per_sample[s.id] = retrieval_scores(doc_ids, set(s.gold_doc_ids), k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(len(usable), 1)
        metric_names = list(next(iter(per_sample.values()))) if per_sample else []
        metrics: dict[str, float] = {}
        ci: dict[str, tuple[float, float]] = {}
        for m in metric_names:
            values = [v[m] for v in per_sample.values() if not math.isnan(v[m])]
            metrics[m] = float(np.mean(values)) if values else math.nan
            ci[m] = (
                bootstrap_ci(values, n_boot=n_boot, seed=seed) if values else (math.nan, math.nan)
            )
        rows.append(
            BenchRow(
                name=name,
                n=len(usable),
                metrics=metrics,
                ci=ci,
                ms_per_query=elapsed_ms,
                per_sample=per_sample,
            )
        )
    return rows


def render_bench_markdown(rows: Sequence[BenchRow], *, k: int) -> str:
    cols = ["hit_rate@1", f"hit_rate@{k}", f"recall@{k}", "mrr", f"ndcg@{k}"]
    lines = [
        "| Retriever | n | " + " | ".join(cols) + " | ms/query |",
        "|---|---:|" + "---:|" * len(cols) + "---:|",
    ]
    for r in rows:
        cells = []
        for c in cols:
            m = r.metrics.get(c, math.nan)
            lo, hi = r.ci.get(c, (math.nan, math.nan))
            cells.append("n/a" if math.isnan(m) else f"{m:.3f} [{lo:.2f}, {hi:.2f}]")
        lines.append(f"| `{r.name}` | {r.n} | " + " | ".join(cells) + f" | {r.ms_per_query:.1f} |")
    return "\n".join(lines) + "\n"
