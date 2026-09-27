from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mcpeval.agents.llm import ScriptedChatModel
from mcpeval.bench import live, live_compare
from mcpeval.bench.runner import scripted_benchmark_model
from mcpeval.schemas import Message


@pytest.fixture
def protocol() -> live.Protocol:
    return live.Protocol.model_validate_json(live.DEFAULT_PROTOCOL.read_text(encoding="utf-8"))


@pytest.fixture
def completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: live.Protocol) -> Path:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(live, "local_snapshot", lambda *_args: snapshot)

    def fake_load(
        study: live.Protocol, path: Path, device: str
    ) -> tuple[ScriptedChatModel, dict[str, Any]]:
        model = scripted_benchmark_model()
        model.model_name = study.model_id
        return model, {"device": "cpu", "dtype": "fake", "gpu": None}

    monkeypatch.setattr(live, "load_model", fake_load)
    output = tmp_path / "study"
    live.run_study(protocol, output, device="cpu")
    return output


def test_frozen_protocols_share_tasks_and_cover_security_controls(protocol: live.Protocol) -> None:
    bigger = live.Protocol.model_validate_json(
        (live.ROOT / "protocols/local-qwen4b-v1.json").read_text(encoding="utf-8")
    )
    assert protocol.model_dump(exclude={"name", "model_id", "revision"}) == bigger.model_dump(
        exclude={"name", "model_id", "revision"}
    )
    tasks = live.select_tasks(protocol)
    assert len(tasks) == 18
    assert len({t.family for t in tasks}) == 8
    assert {t.required_calls[-1].tool for t in tasks if t.approval_expected} == {
        "note_append",
        "order_place",
    }
    assert protocol.schedule == ((1, "single"), (1, "supervisor"), (2, "supervisor"), (2, "single"))


def test_source_hash_is_stable_across_git_line_endings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    path = source / "example.py"
    path.write_bytes(b"x = 1\ny = 2\n")
    monkeypatch.setattr(live, "ROOT", tmp_path)
    first = live.source_identity()
    path.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert live.source_identity()["source_sha256"] == first["source_sha256"]
    path.write_bytes(b"x = 2\ny = 2\n")
    assert live.source_identity()["source_sha256"] != first["source_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        {"task_digest": "wrong"},
        {"policy_digest": "wrong"},
        {"world_seed": 8},
        {"task_ids": ("absent",)},
    ],
)
def test_protocol_drift_fails(protocol: live.Protocol, change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        live.select_tasks(protocol.model_copy(update=change))
    with pytest.raises(ValueError, match="duplicate"):
        live.Protocol.model_validate({**protocol.model_dump(), "task_ids": ["x", "x"]})


def test_audited_calls_include_real_usage_and_errors(
    tmp_path: Path, protocol: live.Protocol
) -> None:
    model = ScriptedChatModel(replies=['{"action":"final","answer":"ready"}'])
    audit = live.AuditedModel(model, tmp_path / "calls.jsonl", protocol)
    result = audit.complete([Message(role="user", content="hello")])
    assert audit.records[0]["usage"] == result.usage.model_dump()
    assert audit.records[0]["request_sha256"]
    with pytest.raises(RuntimeError):
        audit.complete([])
    assert "ran out" in audit.records[1]["error"]
    with pytest.raises(ValueError, match="decoding"):
        audit.complete([], temperature=0.1)
    audit.protocol = protocol.model_copy(update={"task_ids": ("one",), "max_steps": 1})
    with pytest.raises(RuntimeError, match="ceiling"):
        audit.complete([])
    assert len((tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_offline_end_to_end_receipt_and_repeat_unit(
    completed: Path, protocol: live.Protocol
) -> None:
    summary = live.verify_receipt(completed)
    assert summary["independent_tasks"] == 18
    assert summary["paired_task_mean_success"]["n"] == 18
    assert len(summary["runs"]) == 4
    assert summary["stability"]["single"]["identical_trajectories"] == 18
    assert all(row["executed_writes"] == 0 for row in summary["runs"].values())
    report = (completed / "report.md").read_text(encoding="utf-8")
    assert "Exploratory result" in report
    assert "True injection attacks" in report
    with pytest.raises(ValueError, match="not empty"):
        live.run_study(protocol, completed, device="cpu")
    assert live.main(["--out", str(completed), "--verify"]) == 0


@pytest.mark.parametrize(
    "relative",
    [
        "repeat-01-single/grades.jsonl",
        "repeat-01-supervisor/trajectories.jsonl",
        "repeat-02-single/model_calls.jsonl",
        "summary.json",
        "report.md",
    ],
)
def test_any_evidence_mutation_invalidates_receipt(completed: Path, relative: str) -> None:
    with (completed / relative).open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        live.verify_receipt(completed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "running"),
        ("protocol_sha256", "wrong"),
        ("outputs", {}),
    ],
)
def test_incomplete_or_mismatched_receipt_fails(completed: Path, field: str, value: Any) -> None:
    path = completed / "receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt[field] = value
    live.write_json(path, receipt)
    with pytest.raises(ValueError):
        live.verify_receipt(completed)


@pytest.mark.parametrize("fault", ["drop", "duplicate", "manifest", "grade", "tokens"])
def test_regrading_rejects_bad_evidence_even_if_checksums_recomputed(
    completed: Path, protocol: live.Protocol, fault: str
) -> None:
    directory = completed / "repeat-01-single"
    if fault in {"drop", "duplicate", "grade"}:
        path = directory / "grades.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if fault == "drop":
            rows.pop()
        elif fault == "duplicate":
            rows[-1] = rows[0]
        else:
            rows[0]["success"] = not rows[0]["success"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    elif fault == "manifest":
        path = directory / "manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["resumed"] = [protocol.task_ids[0]]
        live.write_json(path, payload)
    else:
        path = directory / "model_calls.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["usage"]["prompt_tokens"] += 1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        live.summarize(completed, protocol)


def test_cached_snapshot_requires_pinned_complete_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: live.Protocol
) -> None:
    snapshot = tmp_path / protocol.revision
    snapshot.mkdir()
    calls = []

    def cached(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(
        importlib, "import_module", lambda _name: SimpleNamespace(snapshot_download=cached)
    )
    with pytest.raises(ValueError, match="incomplete"):
        live.local_snapshot(protocol)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).write_text("{}", encoding="utf-8")
    assert live.local_snapshot(protocol) == snapshot
    assert calls[-1] == {
        "repo_id": protocol.model_id,
        "revision": protocol.revision,
        "local_files_only": True,
        "allow_patterns": ["*.json", "*.safetensors", "*.txt"],
    }
    with pytest.raises(ValueError, match="metadata"):
        live.local_snapshot(protocol, snapshot)
    metadata = snapshot / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    for path in snapshot.iterdir():
        if path.is_file():
            (metadata / f"{path.name}.metadata").write_text(
                protocol.revision + "\netag\n0", encoding="utf-8"
            )
    assert live.local_snapshot(protocol, snapshot) == snapshot
    live.write_json(
        snapshot / "model.safetensors.index.json", {"weight_map": {"one": "missing.safetensors"}}
    )
    with pytest.raises(ValueError, match="shards"):
        live.local_snapshot(protocol)


def test_failed_load_keeps_failed_receipt(
    completed: Path, protocol: live.Protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> None:
        raise RuntimeError("out of memory")

    monkeypatch.setattr(live, "load_model", fail)
    path = completed.parent / "failed"
    with pytest.raises(RuntimeError, match="out of memory"):
        live.run_study(protocol, path)
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        live.verify_receipt(path)


def test_backend_failure_cannot_become_a_complete_comparison(
    completed: Path, protocol: live.Protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = ScriptedChatModel(replies=[], model_name=protocol.model_id)
    monkeypatch.setattr(live, "load_model", lambda *_args: (broken, {"device": "cpu"}))
    path = completed.parent / "backend-failed"
    with pytest.raises(live.IncompleteStudyError, match="backend errors"):
        live.run_study(protocol, path)
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "incomplete"
    assert not receipt["completed_runs"]
    assert (path / "repeat-01-single/trajectories.jsonl").is_file()
    with pytest.raises(ValueError, match="incomplete"):
        live.verify_receipt(path)


def test_same_budget_comparison_and_mismatch_rejection(
    completed: Path, protocol: live.Protocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = completed.parent / "candidate"
    bigger = protocol.model_copy(update={"model_id": "other-model", "name": "candidate"})
    live.run_study(bigger, candidate, device="cpu")
    result = live_compare.compare_studies(completed, candidate, completed.parent / "comparison")
    assert result["independent_tasks"] == 18
    assert all(row["n"] == 18 and row["diff"] == 0 for row in result["comparisons"].values())
    with pytest.raises(ValueError, match="fresh"):
        live_compare.compare_studies(completed, candidate, completed.parent / "comparison")
    receipt_path = candidate / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["hardware"]["dtype"] = "different"
    live.write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="hardware"):
        live_compare.compare_studies(completed, candidate, completed.parent / "mismatch")
