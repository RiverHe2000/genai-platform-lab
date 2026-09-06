from __future__ import annotations

from pathlib import Path

import pytest

from ragpipe.evaluation.gates import (
    Gate,
    GateSpec,
    all_passed,
    compare_reports,
    evaluate_gates,
    render_comparison,
    render_gates,
)
from ragpipe.evaluation.runner import EvalConfig, EvalReport, MetricSummary, SampleRecord


def _record(sid: str, **metrics: float | None) -> SampleRecord:
    return SampleRecord(
        id=sid,
        question="q",
        ground_truth="g",
        tags=[],
        answerable=True,
        answer="a",
        abstained=False,
        model="m",
        gold_doc_ids=[],
        retrieved_doc_ids=[],
        retrieved_chunk_ids=[],
        cited_chunk_ids=[],
        contexts=[],
        metrics=dict(metrics),
    )


def _report(
    records: list[SampleRecord], summaries: dict[str, MetricSummary] | None = None
) -> EvalReport:
    return EvalReport(
        created_at="now",
        version="0",
        retriever="r",
        generator="g",
        judge=None,
        config=EvalConfig(),
        n_samples=len(records),
        summaries=summaries or {},
        by_tag={},
        records=records,
    )


def _summary(name: str, mean: float, lo: float, hi: float, n: int = 10) -> MetricSummary:
    return MetricSummary(name=name, mean=mean, ci_low=lo, ci_high=hi, n=n, n_missing=0)


def test_gate_validation_and_loading(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min and/or max"):
        Gate(metric="x")
    yaml_path = tmp_path / "g.yaml"
    yaml_path.write_text(
        "gates:\n  - metric: faithfulness\n    min: 0.8\n    use_ci: true\n", encoding="utf-8"
    )
    spec = GateSpec.load(yaml_path)
    assert spec.gates[0].use_ci is True
    json_path = tmp_path / "g.json"
    json_path.write_text('{"gates": [{"metric": "m", "max": 0.1}]}', encoding="utf-8")
    assert GateSpec.load(json_path).gates[0].max == 0.1


def test_evaluate_gates_outcomes() -> None:
    report = _report(
        [],
        {
            "faithfulness": _summary("faithfulness", 0.9, 0.82, 0.95),
            "false_abstention": _summary("false_abstention", 0.05, 0.0, 0.12),
            "tiny": _summary("tiny", 1.0, 1.0, 1.0, n=2),
        },
    )
    spec = GateSpec(
        gates=[
            Gate(metric="faithfulness", min=0.85),
            Gate(metric="faithfulness", min=0.85, use_ci=True),
            Gate(metric="false_abstention", max=0.10),
            Gate(metric="false_abstention", max=0.10, use_ci=True),
            Gate(metric="missing", min=0.5),
            Gate(metric="tiny", min=0.5, min_n=5),
        ]
    )
    outcomes = evaluate_gates(report, spec)
    assert [o.passed for o in outcomes] == [True, False, True, False, False, False]
    assert "ci_low=0.820 < min=0.85" in outcomes[1].reason
    assert "ci_high=0.120 > max=0.1" in outcomes[3].reason
    assert outcomes[4].reason == "metric missing from report"
    assert "min_n" in outcomes[5].reason
    assert all_passed(outcomes) is False
    assert all_passed(outcomes[:1]) is True
    table = render_gates(outcomes)
    assert "| PASS |" in table and "| FAIL |" in table and "n/a" in table


def test_compare_reports_paired_bootstrap_verdicts() -> None:
    base = _report([_record(f"s{i}", m=0.5) for i in range(20)])
    better = _report([_record(f"s{i}", m=0.8) for i in range(20)])
    same = _report([_record(f"s{i}", m=0.5) for i in range(20)])
    worse = _report([_record(f"s{i}", m=0.5 - (0.3 if i % 2 else 0.1)) for i in range(20)])

    c = compare_reports(better, base, "m", n_boot=200)
    assert c.verdict == "better"
    assert c.n_pairs == 20
    assert c.delta == pytest.approx(0.3)
    assert c.p_improve == 1.0

    c = compare_reports(same, base, "m", n_boot=200)
    assert c.verdict == "non-inferior"
    assert c.delta == 0.0

    c = compare_reports(worse, base, "m", n_boot=200, non_inferiority_margin=0.05)
    assert c.verdict == "worse"
    assert c.ci_high < 0

    noisy = _report([_record(f"s{i}", m=0.5 + (0.4 if i % 2 else -0.4)) for i in range(20)])
    c = compare_reports(noisy, base, "m", n_boot=200)
    assert c.verdict == "inconclusive"

    md = render_comparison([c])
    assert "| m |" in md and "inconclusive" in md


def test_compare_reports_handles_missing_and_partial_overlap() -> None:
    a = _report([_record("s1", m=0.9), _record("s2", m=None), _record("s3", m=0.4)])
    b = _report([_record("s1", m=0.8), _record("s2", m=0.5), _record("s9", m=0.1)])
    c = compare_reports(a, b, "m", n_boot=50)
    assert c.n_pairs == 1
    assert c.delta == pytest.approx(0.1)
    assert c.ci_low == c.ci_high
    none = compare_reports(a, b, "absent", n_boot=50)
    assert none.verdict == "no pairs"
