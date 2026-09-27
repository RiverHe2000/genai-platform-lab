from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mcpeval.agents.llm import ScriptedChatModel
from mcpeval.bench import live, live_v2
from mcpeval.bench.attention import ATTENTION_NAME
from mcpeval.bench.runner import scripted_benchmark_model


@pytest.fixture
def protocol() -> live_v2.ProtocolV2:
    return live_v2.ProtocolV2.model_validate_json(
        live_v2.DEFAULT_PROTOCOL.read_text(encoding="utf-8")
    )


@pytest.fixture
def setup_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(live, "local_snapshot", lambda *_args: snapshot)

    def fake_load(study: live.Protocol, path: Path) -> tuple[ScriptedChatModel, dict[str, Any]]:
        model = scripted_benchmark_model()
        model.model_name = study.model_id
        return model, {"attention_implementation": ATTENTION_NAME, "device": "fake"}

    monkeypatch.setattr(live_v2, "load_model", fake_load)
    monkeypatch.setattr(live_v2, "resource_peaks", lambda: {"peak_allocated_bytes": 123})


@pytest.fixture
def completed(tmp_path: Path, setup_backend: None, protocol: live_v2.ProtocolV2) -> Path:
    out = tmp_path / "study"
    live_v2.run_study(protocol, out)
    return out


def test_v2_keeps_all_frozen_v1_tasks_and_budgets(protocol: live_v2.ProtocolV2) -> None:
    for name in ("qwen15", "qwen4b"):
        old = live.Protocol.model_validate_json(
            (live.ROOT / f"protocols/local-{name}-v1.json").read_text(encoding="utf-8")
        )
        new = live_v2.ProtocolV2.model_validate_json(
            (live.ROOT / f"protocols/local-{name}-v2.json").read_text(encoding="utf-8")
        )
        assert old.model_dump(exclude={"name"}) == new.study.model_dump(exclude={"name"})
        assert len(live.select_tasks(new.study)) == 18
        assert new.attention_implementation == ATTENTION_NAME
    with pytest.raises(ValueError):
        live_v2.ProtocolV2.model_validate(
            {**protocol.model_dump(), "attention_implementation": "sdpa"}
        )


def test_complete_v2_receipt_regrades_all_four_arms(completed: Path) -> None:
    summary = live_v2.verify_receipt(completed)
    assert len(summary["runs"]) == 4
    assert summary["paired_task_mean_success"]["n"] == 18
    receipt = json.loads((completed / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert len(receipt["completed_runs"]) == 4
    assert receipt["resources"]["peak_allocated_bytes"] == 123
    assert "not statistically independent" in (completed / "report.md").read_text()
    assert live_v2.main(["--verify", "--out", str(completed)]) == 0


@pytest.mark.parametrize("fault", ["legacy", "hash", "backend", "embedded"])
def test_v2_rejects_version_or_backend_drift(completed: Path, fault: str) -> None:
    path = completed / "receipt.json"
    receipt = json.loads(path.read_text())
    if fault == "legacy":
        receipt.pop("protocol_v2")
    elif fault == "hash":
        receipt["protocol_v2_sha256"] = "wrong"
    elif fault == "backend":
        receipt["hardware"]["attention_implementation"] = "sdpa"
    else:
        receipt["protocol"]["name"] = "different"
    live.write_json(path, receipt)
    with pytest.raises(ValueError):
        live_v2.verify_receipt(completed)
    assert live_v2.main(["--verify", "--out", str(completed)]) == 1


def test_comparison_keeps_task_units_and_v2_identity(completed: Path, tmp_path: Path) -> None:
    out = tmp_path / "comparison"
    assert live_v2.main(["--compare", str(completed), str(completed), "--out", str(out)]) == 0
    comparison = json.loads((out / "comparison.json").read_text())
    assert comparison["resampling_task_units"] == 18
    assert "independent_tasks" not in comparison
    assert comparison["attention_implementation"] == ATTENTION_NAME
    assert comparison["comparisons"]["single"]["diff"] == 0


def test_runner_no_resume(completed: Path, protocol: live_v2.ProtocolV2) -> None:
    with pytest.raises(ValueError, match="not empty"):
        live_v2.run_study(protocol, completed)


@pytest.mark.parametrize("interrupted", [False, True])
def test_loading_failure_preserves_incomplete_receipt(
    tmp_path: Path,
    setup_backend: None,
    monkeypatch: pytest.MonkeyPatch,
    protocol: live_v2.ProtocolV2,
    interrupted: bool,
) -> None:
    def fail(*args: Any) -> Any:
        if interrupted:
            raise KeyboardInterrupt
        raise RuntimeError("synthetic backend failure")

    monkeypatch.setattr(live_v2, "load_model", fail)
    out = tmp_path / "failed"
    with pytest.raises((KeyboardInterrupt, RuntimeError)):
        live_v2.run_study(protocol, out)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == ("interrupted" if interrupted else "failed")
    assert receipt["completed_runs"] == []
    assert receipt["outputs"] == {}
    assert "finished_at" in receipt
    with pytest.raises(ValueError):
        live_v2.verify_receipt(out)


def test_backend_error_preserves_attempts_without_subsets(
    tmp_path: Path,
    setup_backend: None,
    monkeypatch: pytest.MonkeyPatch,
    protocol: live_v2.ProtocolV2,
) -> None:
    def empty(study: live.Protocol, path: Path) -> tuple[ScriptedChatModel, dict[str, Any]]:
        return ScriptedChatModel(replies=[], model_name=study.model_id), {
            "attention_implementation": ATTENTION_NAME,
        }

    monkeypatch.setattr(live_v2, "load_model", empty)
    out = tmp_path / "backend-failed"
    with pytest.raises(live.IncompleteStudyError):
        live_v2.run_study(protocol, out)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "incomplete"
    assert "repeat-01-single/model_calls.jsonl" in receipt["outputs"]
    assert not (out / "repeat-01-supervisor").exists()
