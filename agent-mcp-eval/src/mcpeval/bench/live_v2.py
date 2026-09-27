"""Versioned local study requiring explicit KV expansion and efficient CUDA SDPA.

The v1 source and results stay unchanged. Both models must start fresh under v2;
the old math-kernel run is not a candidate baseline for this comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import platform
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from mcpeval.agents.llm import HFChatModel
from mcpeval.bench import live, live_compare
from mcpeval.bench.attention import ATTENTION_NAME, register_attention
from mcpeval.bench.runner import benchmark_policy, run_benchmark
from mcpeval.world.store import WorldLog, build_world

DEFAULT_PROTOCOL = live.ROOT / "protocols/local-qwen15-v2.json"
CAVEATS = """
## Version 2 interpretation

The 18 tasks are resampling units, not statistically independent users or real-world
cases. They share a synthetic world and template families; task bootstrap intervals
can understate wider uncertainty. The legacy summary key `independent_tasks` means
only 18 task units here. Repeats are averaged within task before resampling.

Qwen2.5-1.5B and Qwen3-4B differ in generation and training as well as parameter
count. Differences cannot be attributed to parameter count alone. Identical task,
output-token and turn ceilings do not equal total compute: inspect observed calls,
input/output tokens and latency. Zero API fees exclude hardware ownership and
electricity costs. Denied dangerous actions are not task successes; approval
failures, permission violations and executed writes are reported separately.

Both models use explicit repeated K/V heads and forced efficient SDPA, without
math fallback or quantization. Floating-point results are not bitwise identical
to v1. No v1 score is combined with v2; v1's 4B resource-feasibility interruption
was not an observed CUDA OOM or a completed model comparison.
"""


class ProtocolV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[2] = 2
    attention_implementation: Literal["mcpeval_sdpa_repeat_kv_efficient"] = (
        "mcpeval_sdpa_repeat_kv_efficient"
    )
    study: live.Protocol


def load_model(protocol: live.Protocol, snapshot: Path) -> tuple[HFChatModel, dict[str, Any]]:
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    if not torch.cuda.is_available():
        raise RuntimeError("v2 requires CUDA; no backend fallback is allowed")
    torch.manual_seed(protocol.seed)
    torch.cuda.reset_peak_memory_stats()
    name = register_attention()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation=name,
        )
        .to("cuda")
        .eval()
    )
    return HFChatModel(
        protocol.model_id,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        dtype=protocol.dtype,
        seed=protocol.seed,
    ), {
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(model.dtype),
        "cuda": torch.version.cuda,
        "attention_implementation": name,
    }


def resource_peaks() -> dict[str, int]:
    torch = importlib.import_module("torch")
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def verify_receipt(out: Path) -> dict[str, Any]:
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    if "protocol_v2" not in receipt:
        raise ValueError("v2 requires a fresh v2 receipt, not a v1 result")
    protocol = ProtocolV2.model_validate(receipt["protocol_v2"])
    if live.digest(protocol.model_dump(mode="json")) != receipt["protocol_v2_sha256"]:
        raise ValueError("v2 protocol checksum mismatch")
    if protocol.study.model_dump(mode="json") != receipt["protocol"]:
        raise ValueError("v2 study and embedded protocol disagree")
    if receipt.get("hardware", {}).get("attention_implementation") != ATTENTION_NAME:
        raise ValueError("v2 requires the frozen efficient attention implementation")
    return live.verify_receipt(out)


def run_study(protocol_v2: ProtocolV2, out: Path, *, model_dir: Path | None = None) -> None:
    protocol = protocol_v2.study
    tasks = live.select_tasks(protocol)
    snapshot = live.local_snapshot(protocol, model_dir)
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
        "protocol_sha256": live.digest(protocol.model_dump(mode="json")),
        "protocol_v2": protocol_v2.model_dump(mode="json"),
        "protocol_v2_sha256": live.digest(protocol_v2.model_dump(mode="json")),
        "source": live.source_identity(),
        "model_files": live.snapshot_identity(snapshot),
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
    live.write_json(out / "receipt.json", receipt)
    started = time.perf_counter()
    loaded = False
    try:
        model, receipt["hardware"] = load_model(protocol, snapshot)
        loaded = True
        receipt["model_load_seconds"] = time.perf_counter() - started
        live.write_json(out / "receipt.json", receipt)
        for repeat, architecture in protocol.schedule:
            name = f"repeat-{repeat:02d}-{architecture}"
            directory = out / name
            directory.mkdir()
            audited = live.AuditedModel(model, directory / "model_calls.jsonl", protocol)
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
                raise live.IncompleteStudyError(
                    f"{name} has backend errors; preserve attempts without subset retries"
                )
            receipt["completed_runs"].append(name)
            live.write_json(out / "receipt.json", receipt)
        summary = live.summarize(out, protocol)
        live.write_json(out / "summary.json", summary)
        (out / "report.md").write_text(
            live.render_report(summary, receipt) + CAVEATS, encoding="utf-8", newline="\n"
        )
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt["status"] = (
            "interrupted"
            if isinstance(exc, KeyboardInterrupt)
            else "incomplete"
            if isinstance(exc, live.IncompleteStudyError)
            else "failed"
        )
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Retain checksums for partial output too; an incomplete receipt cannot verify.
        receipt["outputs"] = live.output_checksums(out)
        if loaded:
            receipt["resources"] = resource_peaks()
        receipt["finished_at"] = datetime.now(UTC).isoformat()
        receipt["elapsed_seconds_including_load"] = time.perf_counter() - started
        live.write_json(out / "receipt.json", receipt)


def compare_studies(baseline: Path, candidate: Path, out: Path) -> dict[str, Any]:
    for path in (baseline, candidate):
        verify_receipt(path)
    result = live_compare.compare_studies(baseline, candidate, out)
    result["resampling_task_units"] = result.pop("independent_tasks")
    result["attention_implementation"] = ATTENTION_NAME
    result["protocol_v2_sha256"] = [
        json.loads((p / "receipt.json").read_text(encoding="utf-8"))["protocol_v2_sha256"]
        for p in (baseline, candidate)
    ]
    live.write_json(out / "comparison.json", result)
    report = out / "comparison.md"
    report.write_text(report.read_text(encoding="utf-8") + CAVEATS, encoding="utf-8", newline="\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BASELINE", "CANDIDATE"))
    args = parser.parse_args(argv)
    try:
        if args.compare:
            compare_studies(args.compare[0], args.compare[1], args.out)
        elif args.verify:
            verify_receipt(args.out)
        else:
            protocol = ProtocolV2.model_validate_json(args.protocol.read_text(encoding="utf-8"))
            run_study(protocol, args.out, model_dir=args.model_dir)
            verify_receipt(args.out)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"V2 study failed: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
