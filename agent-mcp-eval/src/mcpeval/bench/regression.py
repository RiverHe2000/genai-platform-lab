"""Fail CI when deterministic harness outcomes change, independently of research promotion.

The scripted agent intentionally fails many tasks. Its per-task outcomes are a regression
fixture, not an estimate of real model quality. Timing is excluded; task/policy identity,
permission outcomes, scoring and steps must still agree with the reviewed fixture.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from mcpeval.bench.runner import Manifest, load_run
from mcpeval.schemas import FailureClass, Grade


class Outcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    answer_score: float
    call_f1: float
    approval_ok: bool
    forbidden_violations: int
    steps: int
    failures: tuple[FailureClass, ...]

    @classmethod
    def from_grade(cls, grade: Grade) -> Outcome:
        return cls.model_validate(grade.model_dump(include=set(cls.model_fields)))


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    architecture: str
    task_set_digest: str
    policy_digest: str
    max_steps: int
    outcomes: dict[str, Outcome]


def regressions(manifest: Manifest, grades: Sequence[Grade], expected: Expectations) -> list[str]:
    """Report changes, including subsets or duplicate tasks that alter the denominator."""
    issues: list[str] = []
    if manifest.model != "scripted":
        issues.append("this check accepts scripted harness runs only")
    for name in ("architecture", "task_set_digest", "policy_digest", "max_steps"):
        if getattr(manifest, name) != getattr(expected, name):
            issues.append(f"{name} differs from the reviewed fixture")
    by_id = {grade.task_id: grade for grade in grades}
    if len(by_id) != len(grades):
        issues.append("duplicate task grades")
    if manifest.task_count != len(expected.outcomes) or set(by_id) != set(expected.outcomes):
        issues.append("task membership/count differs from the complete reviewed fixture")
    for task_id, outcome in expected.outcomes.items():
        grade = by_id.get(task_id)
        if grade is None:
            continue
        if grade.architecture != expected.architecture or grade.model != "scripted":
            issues.append(f"{task_id}: grade provenance differs")
        actual = Outcome.from_grade(grade)
        for name in Outcome.model_fields:
            if getattr(actual, name) != getattr(outcome, name):
                issues.append(
                    f"{task_id}: {name} expected {getattr(outcome, name)!r}, "
                    f"observed {getattr(actual, name)!r}"
                )
    return issues


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--expected", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        expected = Expectations.model_validate_json(args.expected.read_text(encoding="utf-8"))
        manifest, grades = load_run(args.run)
    except (OSError, ValueError) as exc:
        print(f"Cannot check harness: {exc}", file=sys.stderr)
        return 1
    issues = regressions(manifest, grades, expected)
    if issues:
        print("Harness regression:\n" + "\n".join(issues), file=sys.stderr)
        return 3
    print(f"PASS: {len(grades)} scripted {manifest.architecture} outcomes match the fixture")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
