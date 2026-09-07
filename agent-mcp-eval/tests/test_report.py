"""Tests for :mod:`mcpeval.metrics.report`.

The byte-stability tests are the point of the file. A report that renders differently on
two runs over the same data cannot be committed, cannot be diffed, and quietly stops being
read; the assertions below are what keep it committable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mcpeval.metrics.report import (
    METRIC_KEYS,
    Aggregate,
    aggregate,
    compare,
    render_comparison_markdown,
    render_json,
    render_markdown,
)
from mcpeval.metrics.stats import Decision
from mcpeval.schemas import FailureClass, Grade, TaskFamily

N_BOOT = 200


def make_grade(task_id: str = "T-001", **overrides: Any) -> Grade:
    """A grade with plausible defaults; override only the field under test."""
    success = bool(overrides.pop("success", True))
    fields: dict[str, Any] = {
        "task_id": task_id,
        "family": TaskFamily.LOOKUP,
        "architecture": "supervisor",
        "model": "scripted",
        "success": success,
        "answer_score": float(success),
        "call_precision": 1.0,
        "call_recall": 1.0,
        "call_f1": 1.0,
        "step_efficiency": 1.0,
        "steps": 2,
        "optimal_steps": 2,
        "tokens": 1000,
        "wall_ms": 120.0,
        "failures": () if success else (FailureClass.MISSING_REQUIRED_CALL,),
    }
    fields.update(overrides)
    return Grade(**fields)


def run(successes: int, failures: int, *, prefix: str = "T", **overrides: Any) -> list[Grade]:
    """A run of graded tasks with stable ids, so two runs can be paired."""
    total = successes + failures
    return [
        make_grade(f"{prefix}-{i:03d}", success=i < successes, **overrides) for i in range(total)
    ]


def summarise(grades: Sequence[Grade], **overrides: Any) -> Aggregate:
    return aggregate(grades, n_boot=N_BOOT, **overrides)


# ----------------------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------------------


def test_aggregating_nothing_is_an_error_not_a_row_of_zeros() -> None:
    """A row of zeros in a report reads as a result. There is no summary of no data."""
    with pytest.raises(ValueError, match="empty run"):
        summarise([])


def test_overall_success_is_the_share_of_successful_tasks() -> None:
    agg = summarise(run(3, 1))
    assert agg.n == 4
    assert agg.overall.metrics["success"].point == pytest.approx(0.75)


def test_every_metric_is_reported_with_an_interval() -> None:
    agg = summarise(run(3, 1))
    assert tuple(agg.overall.metrics) == METRIC_KEYS
    for interval in agg.overall.metrics.values():
        assert interval.contains(interval.point)
        assert interval.n == 4


def test_family_counts_partition_the_run() -> None:
    grades = [
        make_grade("T-001", family=TaskFamily.LOOKUP),
        make_grade("T-002", family=TaskFamily.INJECTION, success=False),
        make_grade("T-003", family=TaskFamily.LOOKUP, success=False),
    ]
    agg = summarise(grades)
    assert sum(s.n for s in agg.by_family) == agg.n


def test_families_appear_in_enum_order_not_in_arrival_order() -> None:
    """Fixed order is what lets two reports be diffed."""
    grades = [
        make_grade("T-001", family=TaskFamily.UNANSWERABLE),
        make_grade("T-002", family=TaskFamily.LOOKUP),
        make_grade("T-003", family=TaskFamily.AGGREGATION),
    ]
    labels = [s.label for s in summarise(grades).by_family]
    assert labels == ["lookup", "aggregation", "unanswerable"]


def test_absent_families_are_not_reported_as_zeros() -> None:
    agg = summarise(run(2, 0))
    assert [s.label for s in agg.by_family] == ["lookup"]


def test_failure_classes_are_counted_and_given_a_rate() -> None:
    grades = [
        make_grade("T-001", failures=(FailureClass.LOOP,)),
        make_grade("T-002", failures=(FailureClass.LOOP, FailureClass.TOOL_ERROR)),
        make_grade("T-003", failures=(FailureClass.NONE,)),
        make_grade("T-004", failures=(FailureClass.NONE,)),
    ]
    rows = {row.failure: row for row in summarise(grades).failures}
    assert rows[FailureClass.LOOP].count == 2
    assert rows[FailureClass.LOOP].rate.point == pytest.approx(0.5)
    assert rows[FailureClass.TOOL_ERROR].count == 1
    assert list(rows) == [FailureClass.NONE, FailureClass.LOOP, FailureClass.TOOL_ERROR]


def test_failure_classes_that_never_occurred_are_omitted() -> None:
    grades = [make_grade("T-001", failures=(FailureClass.LOOP,))]
    assert [row.failure for row in summarise(grades).failures] == [FailureClass.LOOP]


def test_a_run_from_one_model_names_it_and_a_mixed_run_says_so() -> None:
    same = summarise(run(2, 0))
    assert (same.architecture, same.model) == ("supervisor", "scripted")
    mixed = summarise([make_grade("T-001"), make_grade("T-002", architecture="react")])
    assert mixed.architecture == "mixed"
    assert mixed.model == "scripted"


def test_an_interval_over_a_constant_sample_collapses_onto_the_point() -> None:
    """No resample of identical values can differ, so the bootstrap must say so."""
    interval = summarise(run(4, 0)).overall.metrics["success"]
    assert (interval.low, interval.point, interval.high) == (1.0, 1.0, 1.0)


@given(
    successes=st.integers(min_value=0, max_value=12),
    failures=st.integers(min_value=0, max_value=12),
)
def test_the_success_point_estimate_is_the_observed_rate(successes: int, failures: int) -> None:
    grades = run(successes, failures)
    if not grades:
        return
    agg = aggregate(grades, n_boot=50)
    assert agg.overall.metrics["success"].point == pytest.approx(successes / len(grades))
    assert agg.overall.metrics["success"].low <= agg.overall.metrics["success"].high


def test_the_bootstrap_settings_travel_with_the_numbers() -> None:
    agg = aggregate(run(2, 2), n_boot=64, alpha=0.1, seed=11, label="nightly")
    assert (agg.n_boot, agg.alpha, agg.seed, agg.label) == (64, 0.1, 11, "nightly")


# ----------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------


def test_markdown_is_byte_identical_for_the_same_run() -> None:
    first = render_markdown(summarise(run(3, 2)))
    second = render_markdown(summarise(run(3, 2)))
    assert first == second


def test_markdown_carries_no_timestamp_unless_one_is_injected() -> None:
    agg = summarise(run(2, 1))
    assert "Generated" not in render_markdown(agg)
    stamped = render_markdown(agg, generated_at="2026-09-07T00:00:00Z")
    assert "- Generated: 2026-09-07T00:00:00Z" in stamped


def test_markdown_reports_the_run_the_families_and_the_failures() -> None:
    grades = [
        make_grade("T-001", family=TaskFamily.LOOKUP),
        make_grade(
            "T-002", family=TaskFamily.INJECTION, success=False, failures=(FailureClass.LOOP,)
        ),
    ]
    text = render_markdown(summarise(grades, label="nightly"))
    assert "# Trajectory report: nightly" in text
    assert "- Tasks: 2" in text
    assert "## Overall" in text
    assert "| lookup | 1 |" in text
    assert "| injection | 1 |" in text
    assert "| loop | 1 |" in text
    assert text.endswith("\n")


def test_markdown_says_so_when_nothing_failed() -> None:
    text = render_markdown(summarise([make_grade("T-001", failures=())]))
    assert "| (none recorded) | 0 |" in text


def test_json_round_trips_back_into_the_same_aggregate() -> None:
    agg = summarise(run(3, 1))
    assert Aggregate.model_validate_json(render_json(agg)) == agg


def test_json_is_byte_identical_for_the_same_run() -> None:
    assert render_json(summarise(run(3, 1))) == render_json(summarise(run(3, 1)))


# ----------------------------------------------------------------------------------------
# Comparison and the release gate
# ----------------------------------------------------------------------------------------


def test_two_identical_runs_are_promoted_with_no_difference() -> None:
    result = compare(run(3, 1), run(3, 1), 0.05, n_boot=N_BOOT)
    assert result.n == 4
    assert result.paired["success"].diff == 0.0
    assert (result.only_a, result.only_b) == (0, 0)
    assert result.mcnemar_p == 1.0
    assert result.decision.decision is Decision.PROMOTE
    assert result.promote is True


def test_a_candidate_that_loses_every_task_is_rejected() -> None:
    result = compare(run(0, 4), run(4, 0), 0.05, n_boot=N_BOOT)
    assert result.paired["success"].diff == -1.0
    assert (result.only_a, result.only_b) == (0, 4)
    assert result.decision.decision is Decision.REJECT


def test_a_candidate_that_wins_every_task_is_promoted() -> None:
    result = compare(run(4, 0), run(0, 4), 0.05, n_boot=N_BOOT)
    assert (result.only_a, result.only_b) == (4, 0)
    assert result.decision.decision is Decision.PROMOTE


def test_the_decision_always_follows_the_interval_and_the_margin() -> None:
    """The three cases partition the possibilities; exactly one must apply."""
    margin = 0.5
    for successes in range(6):
        result = compare(run(successes, 5 - successes), run(5, 0), margin, n_boot=N_BOOT)
        interval = result.paired["success"].ci
        if interval.low > -margin:
            assert result.decision.decision is Decision.PROMOTE
        elif interval.high <= -margin:
            assert result.decision.decision is Decision.REJECT
        else:
            assert result.decision.decision is Decision.HOLD


def test_pairing_is_by_task_id_not_by_position() -> None:
    """A run that reordered its results must not shift every pair by one."""
    baseline = run(2, 2)
    shuffled = list(reversed(run(2, 2)))
    result = compare(run(2, 2), shuffled, 0.05, n_boot=N_BOOT)
    assert result.paired["success"].diff == 0.0
    assert compare(run(2, 2), baseline, 0.05, n_boot=N_BOOT).n == result.n


def test_only_the_tasks_both_runs_attempted_are_compared() -> None:
    candidate = [*run(2, 0), make_grade("T-900")]
    result = compare(candidate, run(2, 0), 0.05, n_boot=N_BOOT)
    assert result.n == 2


def test_grading_a_task_twice_makes_the_pairing_ambiguous() -> None:
    duplicated = [make_grade("T-001"), make_grade("T-001", success=False)]
    with pytest.raises(ValueError, match="more than once"):
        compare(duplicated, run(2, 0), 0.05, n_boot=N_BOOT)


def test_two_runs_with_no_tasks_in_common_cannot_be_paired() -> None:
    with pytest.raises(ValueError, match="no tasks in common"):
        compare(run(2, 0, prefix="A"), run(2, 0, prefix="B"), 0.05, n_boot=N_BOOT)


def test_the_discordant_counts_drive_the_exact_mcnemar_test() -> None:
    candidate = [make_grade("T-001"), make_grade("T-002"), make_grade("T-003", success=False)]
    baseline = [
        make_grade("T-001", success=False),
        make_grade("T-002", success=False),
        make_grade("T-003"),
    ]
    result = compare(candidate, baseline, 0.05, n_boot=N_BOOT)
    assert (result.only_a, result.only_b) == (2, 1)
    assert result.mcnemar_p == pytest.approx(1.0)


def test_every_metric_is_compared_not_only_success() -> None:
    result = compare(run(2, 2), run(2, 2), 0.05, n_boot=N_BOOT)
    assert tuple(result.paired) == METRIC_KEYS


def test_comparison_markdown_is_byte_identical_and_states_the_decision() -> None:
    first = compare(run(3, 1), run(2, 2), 0.05, n_boot=N_BOOT)
    second = compare(run(3, 1), run(2, 2), 0.05, n_boot=N_BOOT)
    text = render_comparison_markdown(first)
    assert text == render_comparison_markdown(second)
    assert "# candidate vs baseline" in text
    assert f"**{first.decision.decision.value}**" in text
    assert "Generated" not in text
    assert text.endswith("\n")


def test_comparison_markdown_takes_an_injected_timestamp() -> None:
    result = compare(run(2, 0), run(2, 0), 0.05, n_boot=N_BOOT, label_a="v2", label_b="v1")
    text = render_comparison_markdown(result, generated_at="2026-09-07T00:00:00Z")
    assert "# v2 vs v1" in text
    assert "- Generated: 2026-09-07T00:00:00Z" in text


def test_a_zero_margin_is_a_superiority_test_wearing_another_name() -> None:
    with pytest.raises(ValueError, match="margin must be strictly positive"):
        compare(run(2, 0), run(2, 0), 0.0, n_boot=N_BOOT)
