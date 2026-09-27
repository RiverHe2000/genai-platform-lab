"""A frozen, local-only model study, separate from scripted regression and old results.

The receipt is a reproducibility record with file checksums, not a signed attestation.
No response cache or resumed trajectories are accepted. Repeats measure greedy-run
stability; they are not additional independent benchmark tasks.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcpeval.agents.llm import HFChatModel, cache_key
from mcpeval.bench.runner import (
    benchmark_policy,
    load_run,
    policy_digest,
    run_benchmark,
    task_set_digest,
)
from mcpeval.bench.tasks import build_tasks
from mcpeval.client.recorder import read_jsonl
from mcpeval.metrics.failures import graded
from mcpeval.metrics.stats import paired_bootstrap_diff
from mcpeval.schemas import ChatModel, Completion, Grade, Message, Task, Trajectory
from mcpeval.world.store import WorldLog, build_world

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "protocols/local-qwen15-v1.json"


class IncompleteStudyError(RuntimeError):
    """Backend failure makes a complete same-budget comparison unavailable."""


class Protocol(BaseModel):
    """All outcome-affecting study choices, saved before generating model responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    name: str
    model_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    task_ids: tuple[str, ...] = Field(min_length=1)
    task_digest: str
    policy_digest: str
    world_seed: int = 7
    repeats: Literal[2] = 2
    max_steps: Literal[20] = 20
    max_new_tokens: Literal[512] = 512
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    concurrency: Literal[1] = 1
    seed: int = 0
    dtype: Literal["bfloat16"] = "bfloat16"
    approval: Literal["deny_all"] = "deny_all"
    selection: str

    @model_validator(mode="after")
    def unique_tasks(self) -> Protocol:
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("duplicate protocol task ids")
        return self

    @property
    def schedule(self) -> tuple[tuple[int, str], ...]:
        return ((1, "single"), (1, "supervisor"), (2, "supervisor"), (2, "single"))


def digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def source_identity() -> dict[str, Any]:
    files = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(
            p.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return {
        "git_head": git.stdout.strip() or None,
        "source_sha256": digest(files),
        "normalization": "UTF-8 text with universal newlines normalized to LF",
        "files": files,
    }


def select_tasks(protocol: Protocol) -> tuple[Task, ...]:
    inventory = {task.id: task for task in build_tasks(build_world(protocol.world_seed))}
    missing = set(protocol.task_ids) - inventory.keys()
    if missing:
        raise ValueError(f"unknown protocol tasks: {sorted(missing)}")
    selected = tuple(inventory[task_id] for task_id in protocol.task_ids)
    if task_set_digest(selected) != protocol.task_digest:
        raise ValueError("task content differs from the frozen protocol")
    if policy_digest(benchmark_policy()) != protocol.policy_digest:
        raise ValueError("permission policy differs from the frozen protocol")
    return selected


def local_snapshot(protocol: Protocol, model_dir: Path | None = None) -> Path:
    """Resolve one immutable cached revision; never download or choose a moving ref."""
    if model_dir is None:
        hub = importlib.import_module("huggingface_hub")
        path = Path(
            hub.snapshot_download(
                repo_id=protocol.model_id,
                revision=protocol.revision,
                local_files_only=True,
                allow_patterns=["*.json", "*.safetensors", "*.txt"],
            )
        )
        if path.name != protocol.revision:
            raise ValueError("cached snapshot does not match the pinned revision")
    else:
        path = model_dir
        for item in path.iterdir():
            if item.is_file():
                metadata = path / ".cache/huggingface/download" / f"{item.name}.metadata"
                if (
                    not metadata.is_file()
                    or metadata.read_text(encoding="utf-8").splitlines()[0] != protocol.revision
                ):
                    raise ValueError(f"local file has no matching revision metadata: {item.name}")
    required = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    if any(not (path / name).is_file() for name in required):
        raise ValueError("cached model/tokenizer configuration is incomplete")
    if not list(path.glob("*.safetensors")):
        raise ValueError("cached model has no safetensors weights")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        if any(Path(shard).name != shard or not (path / shard).is_file() for shard in shards):
            raise ValueError("cached model shards are incomplete")
    return path


def snapshot_identity(path: Path) -> dict[str, Any]:
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": file_digest(p)}
        for p in sorted(path.iterdir())
        if p.is_file()
    }
    return {"files": files, "sha256": digest(files)}


class AuditedModel:
    """Record actual backend calls, including failures, outside the trajectory grader."""

    def __init__(self, model: ChatModel, path: Path, protocol: Protocol) -> None:
        self.model, self.path, self.protocol = model, path, protocol
        self.records: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self.model.name

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> Completion:
        if max_tokens != self.protocol.max_new_tokens or temperature != self.protocol.temperature:
            raise ValueError("agent decoding parameters differ from the frozen protocol")
        if len(self.records) >= len(self.protocol.task_ids) * self.protocol.max_steps:
            raise RuntimeError("study request ceiling exhausted")
        row: dict[str, Any] = {
            "index": len(self.records) + 1,
            "request_sha256": cache_key(
                self.name, messages, max_tokens=max_tokens, temperature=temperature, stop=stop
            ),
            "max_new_tokens": max_tokens,
            "temperature": temperature,
        }
        started = time.perf_counter()
        try:
            result = self.model.complete(
                messages, max_tokens=max_tokens, temperature=temperature, stop=stop
            )
            row.update(
                response_sha256=digest(result.text),
                usage=result.usage.model_dump(),
                finish_reason=result.finish_reason,
                error=None,
            )
            return result
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            row["wall_ms"] = (time.perf_counter() - started) * 1000
            self.records.append(row)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def trajectory_signature(trajectory: Trajectory) -> str:
    payload = trajectory.model_dump(mode="json", exclude={"wall_ms"})
    for call in payload["calls"]:
        call.pop("latency_ms")
    return digest(payload)


def _run_name(repeat: int, architecture: str) -> str:
    return f"repeat-{repeat:02d}-{architecture}"


def output_checksums(out: Path) -> dict[str, str]:
    return {
        p.relative_to(out).as_posix(): file_digest(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "receipt.json"
    }


def summarize(out: Path, protocol: Protocol) -> dict[str, Any]:
    """Regrade every complete trajectory; reject missing, duplicate or stale evidence."""
    tasks = select_tasks(protocol)
    task_ids = list(protocol.task_ids)
    runs: dict[str, Any] = {}
    all_grades: dict[str, list[Grade]] = {}
    signatures: dict[str, list[str]] = {}
    for repeat, architecture in protocol.schedule:
        name = _run_name(repeat, architecture)
        directory = out / name
        manifest, grades = load_run(directory)
        trajectories = read_jsonl(directory / "trajectories.jsonl")
        if [g.task_id for g in grades] != task_ids or [t.task_id for t in trajectories] != task_ids:
            raise ValueError(f"{name}: incomplete, duplicate or reordered tasks")
        if (
            manifest.task_set_digest != protocol.task_digest
            or manifest.policy_digest != protocol.policy_digest
            or manifest.model != protocol.model_id
            or manifest.architecture != architecture
            or manifest.max_steps != protocol.max_steps
            or manifest.concurrency != 1
            or manifest.world_seed != protocol.world_seed
            or manifest.resumed
        ):
            raise ValueError(f"{name}: incompatible run manifest")
        expected = [graded(t, task) for t, task in zip(trajectories, tasks, strict=True)]
        if expected != grades:
            raise ValueError(f"{name}: grades do not match raw trajectories")
        if any(
            t.model != protocol.model_id or t.architecture != architecture for t in trajectories
        ):
            raise ValueError(f"{name}: trajectory provenance mismatch")
        calls = [
            json.loads(line)
            for line in (directory / "model_calls.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        successful_calls = [row for row in calls if row["error"] is None]
        prompt_tokens = sum(row["usage"]["prompt_tokens"] for row in successful_calls)
        completion_tokens = sum(row["usage"]["completion_tokens"] for row in successful_calls)
        if prompt_tokens + completion_tokens != sum(g.tokens for g in grades):
            raise ValueError(f"{name}: model-call token totals disagree with trajectories")
        failures = Counter(f.value for g in grades for f in g.failures if f.value != "none")
        tool_calls = [call for t in trajectories for call in t.calls]
        groups = {
            family: {
                "n": sum(t.family.value == family for t in tasks),
                "successes": sum(g.success for g in grades if g.family.value == family),
            }
            for family in sorted({t.family.value for t in tasks})
        }
        runs[name] = {
            "repeat": repeat,
            "architecture": architecture,
            "n": len(grades),
            "successes": sum(g.success for g in grades),
            "families": groups,
            "failures": dict(sorted(failures.items())),
            "approval_failures": sum(not g.approval_ok for g in grades),
            "forbidden_violations": sum(g.forbidden_violations for g in grades),
            "refused_tool_calls": sum(not c.decision.allowed for c in tool_calls),
            "executed_writes": sum(
                c.executed and c.tool in {"note_append", "order_place"} for c in tool_calls
            ),
            "model_calls": len(calls),
            "model_errors": len(calls) - len(successful_calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model_wall_ms": sum(row["wall_ms"] for row in calls),
            "task_wall_ms": sum(g.wall_ms for g in grades),
            "attacks": {
                "n": sum(
                    task_id in task_ids
                    for task_id in ("injection-01-read-pol-0009", "injection-02-read-pol-0017")
                ),
                "successes": sum(
                    g.success
                    for g in grades
                    if g.task_id in {"injection-01-read-pol-0009", "injection-02-read-pol-0017"}
                ),
            },
            "injection_controls": {
                "n": sum(
                    task_id in task_ids
                    for task_id in ("injection-03-read-pol-0001", "injection-04-read-pol-0004")
                ),
                "successes": sum(
                    g.success
                    for g in grades
                    if g.task_id in {"injection-03-read-pol-0001", "injection-04-read-pol-0004"}
                ),
            },
        }
        all_grades[name] = grades
        signatures[name] = [trajectory_signature(t) for t in trajectories]
    stability = {}
    per_task = {}
    for architecture in ("single", "supervisor"):
        first, second = (_run_name(i, architecture) for i in (1, 2))
        stability[architecture] = {
            "tasks": len(tasks),
            "identical_trajectories": sum(
                a == b for a, b in zip(signatures[first], signatures[second], strict=True)
            ),
            "same_success": sum(
                a.success == b.success
                for a, b in zip(all_grades[first], all_grades[second], strict=True)
            ),
        }
        per_task[architecture] = [
            fmean([float(a.success), float(b.success)])
            for a, b in zip(all_grades[first], all_grades[second], strict=True)
        ]
    comparison = paired_bootstrap_diff(
        per_task["supervisor"], per_task["single"], seed=protocol.seed
    )
    return {
        "runs": runs,
        "stability": stability,
        "paired_task_mean_success": comparison.model_dump(mode="json"),
        "decision": "EXPLORATORY_ONLY",
        "independent_tasks": len(tasks),
        "api_spend_usd": 0,
        "token_measurement": "actual HF tokenizer; repeated prompts counted",
    }


def render_report(summary: dict[str, Any], receipt: dict[str, Any]) -> str:
    comparison = summary["paired_task_mean_success"]
    lines = [
        "# Local live model study",
        "",
        "**Exploratory result; no promotion claim.**",
        "",
        "Real model inference over synthetic tasks and an in-process MCP tool sandbox. "
        "No real customer records, external systems, or paid model APIs are used.",
        "",
        f"Model: `{receipt['protocol']['model_id']}` at `{receipt['protocol']['revision']}`.",
        f"Protocol SHA-256: `{receipt['protocol_sha256']}`.",
        f"Source SHA-256: `{receipt['source']['source_sha256']}`.",
        "",
        "| Run | Success | Model calls | Input/output tokens | Model seconds | "
        "Approval failures | Forbidden violations | Executed writes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["runs"].items():
        lines.append(
            f"| {name} | {row['successes']}/{row['n']} | {row['model_calls']} | "
            f"{row['prompt_tokens']}/{row['completion_tokens']} | "
            f"{row['model_wall_ms'] / 1000:.2f} | {row['approval_failures']} | "
            f"{row['forbidden_violations']} | {row['executed_writes']} |"
        )
    lines += [
        "",
        f"Supervisor minus single mean success: {comparison['diff']:+.3f}; "
        f"paired task bootstrap 95% interval [{comparison['ci']['low']:+.3f}, "
        f"{comparison['ci']['high']:+.3f}]. Repeats are averaged within each of "
        f"{summary['independent_tasks']} tasks before resampling; they do not double n.",
        "",
        "The fixed diagnostic slice is not a random sample of production work. "
        "The interval describes this slice, not general agent performance; a zero-containing "
        "interval does not establish equivalence. Greedy repeats test stability on this "
        "software/hardware stack, not stochastic robustness.",
        "",
        "## Repeat stability",
        "",
    ]
    for architecture, row in summary["stability"].items():
        lines.append(
            f"- {architecture}: {row['identical_trajectories']}/{row['tasks']} identical "
            f"trajectories after excluding timing; {row['same_success']}/{row['tasks']} "
            "identical success outcomes."
        )
    for name, row in summary["runs"].items():
        lines += [
            "",
            f"## {name}",
            "",
            f"True injection attacks: {row['attacks']['successes']}/{row['attacks']['n']}; "
            f"clean controls: {row['injection_controls']['successes']}/"
            f"{row['injection_controls']['n']}. "
            f"Refused tool calls: {row['refused_tool_calls']}; "
            f"backend errors: {row['model_errors']}.",
            "",
            "Family results:",
            "",
        ]
        lines += [
            f"- {family}: {values['successes']}/{values['n']}"
            for family, values in row["families"].items()
        ]
        lines += ["", "Failure labels (may overlap):", ""]
        lines += [f"- {label}: {count}" for label, count in row["failures"].items()] or ["- None"]
    lines += [
        "",
        "## Audit and cost boundaries",
        "",
        "`receipt.json` records pinned model file hashes, source hashes, environment, "
        "protocol, and every output checksum. It detects accidental mismatch; it is not "
        "an externally signed attestation. `model_calls.jsonl` records every backend call "
        "and `trajectories.jsonl` preserves messages, tool arguments, decisions and outputs.",
        "",
        "No response cache, resume, or best-run selection. Model timing excludes "
        "one-time model loading, which is recorded separately in the receipt. These are "
        "local wall-clock and token costs, not cloud prices or energy measurements. "
        "Zero executed writes reflects the deny-all approval policy; it does not imply "
        "the model never attempted an unauthorized action. Historical 72-task results "
        "and scripted regression fixtures remain separate.",
        "",
    ]
    return "\n".join(lines)


def verify_receipt(out: Path) -> dict[str, Any]:
    receipt: dict[str, Any] = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "complete":
        raise ValueError("study is incomplete")
    protocol = Protocol.model_validate(receipt["protocol"])
    if digest(protocol.model_dump(mode="json")) != receipt["protocol_sha256"]:
        raise ValueError("protocol checksum mismatch")
    if source_identity()["source_sha256"] != receipt["source"]["source_sha256"]:
        raise ValueError("source differs from the version that generated this receipt")
    if set(receipt["outputs"]) != set(output_checksums(out)):
        raise ValueError("receipt output inventory is incomplete or contains extra files")
    for relative, checksum in receipt["outputs"].items():
        path = (out / relative).resolve()
        if not path.is_relative_to(out.resolve()) or file_digest(path) != checksum:
            raise ValueError(f"output checksum mismatch: {relative}")
    summary = summarize(out, protocol)
    if summary != json.loads((out / "summary.json").read_text(encoding="utf-8")):
        raise ValueError("summary differs from regraded evidence")
    return summary


def load_model(
    protocol: Protocol, snapshot: Path, device: str
) -> tuple[HFChatModel, dict[str, Any]]:
    """Only this function imports torch or loads weights; preflight never invokes it."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose CPU explicitly if intended")
    torch.manual_seed(protocol.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    metadata = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": str(model.dtype),
        "cuda": torch.version.cuda,
    }
    return HFChatModel(
        protocol.model_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=protocol.dtype,
        seed=protocol.seed,
    ), metadata


def run_study(
    protocol: Protocol,
    out: Path,
    *,
    device: str = "cuda",
    model_dir: Path | None = None,
) -> None:
    tasks = select_tasks(protocol)
    snapshot = local_snapshot(protocol, model_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory is not empty; use a fresh directory (no resume)")
    out.mkdir(parents=True, exist_ok=True)
    packages: dict[str, str | None] = {}
    for package in ("torch", "transformers", "accelerate", "mcp", "langgraph", "pydantic"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    receipt: dict[str, Any] = {
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "protocol": protocol.model_dump(mode="json"),
        "protocol_sha256": digest(protocol.model_dump(mode="json")),
        "source": source_identity(),
        "model_files": snapshot_identity(snapshot),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        },
        "sandbox": "in-process synthetic MCP tools; deny_all writes; no external tool systems",
        "cache": False,
        "resume": False,
        "completed_runs": [],
    }
    write_json(out / "receipt.json", receipt)
    started = time.perf_counter()
    try:
        model, hardware = load_model(protocol, snapshot, device)
        receipt["hardware"] = hardware
        receipt["model_load_seconds"] = time.perf_counter() - started
        for repeat, architecture in protocol.schedule:
            name = _run_name(repeat, architecture)
            directory = out / name
            directory.mkdir()
            audited = AuditedModel(model, directory / "model_calls.jsonl", protocol)
            print(f"Running {name}: {len(tasks)} tasks", flush=True)
            asyncio.run(
                run_benchmark(
                    tasks,
                    architecture=architecture,
                    model=audited,
                    world=build_world(protocol.world_seed),
                    log=WorldLog(),
                    policy=benchmark_policy(),
                    max_steps=protocol.max_steps,
                    concurrency=1,
                    out_dir=directory,
                    resume=False,
                    seed=protocol.seed,
                    world_seed=protocol.world_seed,
                    created_at=datetime.now(UTC).isoformat(),
                )
            )
            if any(row["error"] is not None for row in audited.records):
                raise IncompleteStudyError(
                    f"{name} has backend errors; preserve all attempts, "
                    "do not compare or retry subsets"
                )
            receipt["completed_runs"].append(name)
            write_json(out / "receipt.json", receipt)
        summary = summarize(out, protocol)
        write_json(out / "summary.json", summary)
        (out / "report.md").write_text(
            render_report(summary, receipt), encoding="utf-8", newline="\n"
        )
        receipt["status"] = "complete"
        receipt["outputs"] = output_checksums(out)
    except BaseException as exc:
        receipt["status"] = (
            "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else "incomplete"
            if isinstance(exc, IncompleteStudyError)
            else "failed"
        )
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipt["elapsed_seconds_including_load"] = time.perf_counter() - started
        write_json(out / "receipt.json", receipt)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--model-dir", type=Path, help="existing HF local_dir with revision metadata"
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--preflight", action="store_true", help="check cache/protocol; no model load"
    )
    parser.add_argument(
        "--verify", action="store_true", help="verify completed evidence; no inference"
    )
    args = parser.parse_args(argv)
    try:
        protocol = Protocol.model_validate_json(args.protocol.read_text(encoding="utf-8"))
        tasks = select_tasks(protocol)
        if args.preflight:
            snapshot = local_snapshot(protocol, args.model_dir)
            print(
                json.dumps(
                    {
                        "model_id": protocol.model_id,
                        "revision": protocol.revision,
                        "snapshot": str(snapshot),
                        "tasks": len(tasks),
                        "attempts": 4 * len(tasks),
                        "request_ceiling": 4 * len(tasks) * protocol.max_steps,
                        "protocol_sha256": digest(protocol.model_dump(mode="json")),
                    },
                    indent=2,
                )
            )
        elif args.out is None:
            parser.error("--out is required unless --preflight is used")
        elif args.verify:
            verify_receipt(args.out)
            print("PASS: receipt, raw trajectories, grades and summary agree")
        else:
            run_study(protocol, args.out, device=args.device, model_dir=args.model_dir)
            verify_receipt(args.out)
            print(f"Complete: {args.out / 'report.md'}")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Live study failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
