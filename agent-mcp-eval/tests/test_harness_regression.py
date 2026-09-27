from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcpeval.bench.regression import Expectations, Outcome, main, regressions
from mcpeval.bench.runner import Manifest, load_run
from mcpeval.schemas import FailureClass, Grade

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reference() -> tuple[Manifest, list[Grade], Expectations]:
    manifest, grades = load_run(ROOT / "docs/experiments/scripted-single")
    expected = Expectations.model_validate_json(
        (ROOT / "gates/scripted-single.json").read_text(encoding="utf-8")
    )
    return manifest, grades, expected


def test_reviewed_outcomes_match_existing_scripted_evidence(
    reference: tuple[Manifest, list[Grade], Expectations],
) -> None:
    manifest, grades, expected = reference
    assert not regressions(manifest, grades, expected)
    # Wall time varies across operating systems and must not change a harness gate.
    changed = [g.model_copy(update={"wall_ms": g.wall_ms + 500}) for g in grades]
    assert not regressions(manifest, changed, expected)


@pytest.mark.parametrize(
    "change",
    [
        {"success": False},
        {"answer_score": 0.0},
        {"call_f1": 0.0},
        {"approval_ok": False},
        {"forbidden_violations": 1},
        {"steps": 500},
        {"failures": (FailureClass.RUN_ERROR,)},
        {"model": "unexpected"},
        {"architecture": "other"},
    ],
)
def test_changed_task_outcome_fails_even_when_aggregate_might_hide_it(
    change: dict[str, Any], reference: tuple[Manifest, list[Grade], Expectations]
) -> None:
    manifest, grades, expected = reference
    grades[0] = grades[0].model_copy(update=change)
    assert regressions(manifest, grades, expected)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "real-model"},
        {"architecture": "other"},
        {"task_set_digest": "new-tasks"},
        {"policy_digest": "new-policy"},
        {"max_steps": 100},
        {"task_count": 1},
    ],
)
def test_provenance_changes_fail(
    change: dict[str, Any], reference: tuple[Manifest, list[Grade], Expectations]
) -> None:
    manifest, grades, expected = reference
    assert regressions(manifest.model_copy(update=change), grades, expected)


def test_missing_or_duplicated_tasks_fail(
    reference: tuple[Manifest, list[Grade], Expectations],
) -> None:
    manifest, grades, expected = reference
    assert regressions(manifest, grades[:-1], expected)
    assert "duplicate task grades" in regressions(manifest, [*grades, grades[0]], expected)
    assert Outcome.from_grade(grades[0]).success


def test_command_exit_codes(
    tmp_path: Path,
    reference: tuple[Manifest, list[Grade], Expectations],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, grades, expected = reference
    path = tmp_path / "expected.json"
    path.write_text(expected.model_dump_json(), encoding="utf-8")
    args = [str(ROOT / "docs/experiments/scripted-single"), "--expected", str(path)]
    assert main(args) == 0
    assert "PASS: 72" in capsys.readouterr().out
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["outcomes"][grades[0].task_id]["success"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(args) == 3
    assert "Harness regression" in capsys.readouterr().err
    path.write_text("{}", encoding="utf-8")
    assert main(args) == 1
    assert main([str(tmp_path / "absent"), "--expected", str(tmp_path / "absent.json")]) == 1
