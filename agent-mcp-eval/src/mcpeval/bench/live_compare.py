"""Compare two verified local model studies without counting repeats as new tasks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import Any

from mcpeval.bench.live import Protocol, file_digest, verify_receipt, write_json
from mcpeval.bench.runner import load_run
from mcpeval.metrics.stats import paired_bootstrap_diff


def compare_studies(baseline: Path, candidate: Path, out: Path) -> dict[str, Any]:
    """Two models, same frozen tasks and budgets; repeated tasks remain one sampling unit."""
    for path in (baseline, candidate):
        verify_receipt(path)
    receipts = [
        json.loads((p / "receipt.json").read_text(encoding="utf-8")) for p in (baseline, candidate)
    ]
    protocols = [Protocol.model_validate(r["protocol"]) for r in receipts]
    excluded = {"name", "model_id", "revision"}
    if protocols[0].model_dump(exclude=excluded) != protocols[1].model_dump(exclude=excluded):
        raise ValueError("model comparison requires identical task selection and budgets")
    for field in ("hardware", "environment"):
        if receipts[0][field] != receipts[1][field]:
            raise ValueError(f"model comparison requires identical {field}")
    if receipts[0]["source"]["source_sha256"] != receipts[1]["source"]["source_sha256"]:
        raise ValueError("model comparison requires identical source")
    if out.exists() and any(out.iterdir()):
        raise ValueError("comparison output directory must be fresh")
    comparisons = {}
    for architecture in ("single", "supervisor"):
        means = []
        for path in (baseline, candidate):
            repeats = [
                load_run(path / f"repeat-{repeat:02d}-{architecture}")[1] for repeat in (1, 2)
            ]
            means.append(
                [fmean([float(a.success), float(b.success)]) for a, b in zip(*repeats, strict=True)]
            )
        comparisons[architecture] = paired_bootstrap_diff(
            means[1], means[0], seed=protocols[0].seed
        ).model_dump(mode="json")
    result = {
        "baseline": protocols[0].model_id,
        "candidate": protocols[1].model_id,
        "receipt_sha256": [file_digest(p / "receipt.json") for p in (baseline, candidate)],
        "protocol_sha256": [r["protocol_sha256"] for r in receipts],
        "source_sha256": receipts[0]["source"]["source_sha256"],
        "independent_tasks": len(protocols[0].task_ids),
        "comparisons": comparisons,
        "decision": "EXPLORATORY_ONLY",
    }
    lines = [
        "# Same-budget local model comparison",
        "",
        f"Baseline: `{result['baseline']}`. Candidate: `{result['candidate']}`.",
        "",
        "Both models use the same tasks, tool permission policy, greedy decoding, "
        "20 model turns per task, and 512 new tokens per turn. Each architecture runs twice. "
        "This controls token/turn ceilings, not equal wall-clock or memory consumption.",
        "",
        "| Architecture | Baseline success | Candidate success | Difference | Paired 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for architecture, row in comparisons.items():
        lines.append(
            f"| {architecture} | {row['mean_b']:.3f} | {row['mean_a']:.3f} | "
            f"{row['diff']:+.3f} | [{row['ci']['low']:+.3f}, {row['ci']['high']:+.3f}] |"
        )
    lines += [
        "",
        f"**Exploratory only:** {result['independent_tasks']} fixed diagnostic tasks; "
        "two repeats are averaged per task before bootstrap. Two architecture-specific "
        "comparisons are descriptive, not a familywise significance claim. These are "
        "historically visible synthetic tasks, not a held-out production evaluation. "
        "Neither size nor a higher point score proves general model superiority. "
        "See each model's report for permission failures, attacks vs clean controls, "
        "repeat stability, backend errors and actual runtime/token cost.",
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "comparison.json", result)
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        compare_studies(args.baseline, args.candidate, args.out)
    except (OSError, ValueError) as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1
    print(f"Complete: {args.out / 'comparison.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
