"""Closed-loop load generator against any OpenAI-compatible endpoint (the gateway, vLLM
directly, or the in-process ASGI app in tests). Reports latency percentiles, throughput,
error rate and, for streaming, time-to-first-token."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from llmgate.protocol import parse_sse_line


@dataclass(slots=True)
class Sample:
    ok: bool
    status: int
    latency_s: float
    ttft_s: float | None = None
    completion_tokens: int = 0


@dataclass(slots=True)
class LoadTestReport:
    created_at: str
    target: str
    model: str
    concurrency: int
    requests: int
    stream: bool
    wall_s: float
    n_ok: int
    n_error: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    throughput_rps: float
    tokens_per_s: float
    status_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


async def _one(
    client: httpx.AsyncClient, *, model: str, prompt: str, max_tokens: int, stream: bool
) -> Sample:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    started = time.perf_counter()
    try:
        if not stream:
            resp = await client.post("/v1/chat/completions", json=body)
            latency = time.perf_counter() - started
            tokens = 0
            if resp.status_code == 200:
                tokens = int(resp.json().get("usage", {}).get("completion_tokens", 0))
            return Sample(resp.status_code == 200, resp.status_code, latency, None, tokens)
        ttft: float | None = None
        tokens = 0
        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                await resp.aread()
                return Sample(False, resp.status_code, time.perf_counter() - started)
            async for line in resp.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None or chunk == "done":
                    continue
                if chunk.content:
                    tokens += 1
                    if ttft is None:
                        ttft = time.perf_counter() - started
        return Sample(True, 200, time.perf_counter() - started, ttft, tokens)
    except httpx.HTTPError:
        return Sample(False, 0, time.perf_counter() - started)


async def run_loadtest(
    client: httpx.AsyncClient,
    *,
    target: str,
    model: str,
    concurrency: int,
    requests: int,
    prompt: str = "In two sentences, what is a loan-to-value ratio?",
    max_tokens: int = 64,
    stream: bool = False,
) -> LoadTestReport:
    samples: list[Sample] = []
    remaining = {"n": requests}
    lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            async with lock:
                if remaining["n"] <= 0:
                    return
                remaining["n"] -= 1
            samples.append(
                await _one(client, model=model, prompt=prompt, max_tokens=max_tokens, stream=stream)
            )

    wall_started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    wall = time.perf_counter() - wall_started
    ok = [s for s in samples if s.ok]
    latencies = [s.latency_s * 1000 for s in ok]
    ttfts = [s.ttft_s * 1000 for s in ok if s.ttft_s is not None]
    counts: dict[str, int] = {}
    for s in samples:
        counts[str(s.status)] = counts.get(str(s.status), 0) + 1
    return LoadTestReport(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        target=target,
        model=model,
        concurrency=concurrency,
        requests=requests,
        stream=stream,
        wall_s=wall,
        n_ok=len(ok),
        n_error=len(samples) - len(ok),
        p50_ms=_pct(latencies, 50),
        p95_ms=_pct(latencies, 95),
        p99_ms=_pct(latencies, 99),
        mean_ms=float(np.mean(latencies)) if latencies else float("nan"),
        ttft_p50_ms=_pct(ttfts, 50) if ttfts else None,
        ttft_p95_ms=_pct(ttfts, 95) if ttfts else None,
        throughput_rps=len(ok) / wall if wall > 0 else 0.0,
        tokens_per_s=sum(s.completion_tokens for s in ok) / wall if wall > 0 else 0.0,
        status_counts=counts,
    )


def render_markdown(reports: Sequence[LoadTestReport]) -> str:
    lines = [
        "| Target | Model | Conc. | Requests | Stream | OK / Err | p50 ms | p95 ms | p99 ms "
        "| TTFT p50 ms | req/s | tok/s |",
        "|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in reports:
        ttft = f"{r.ttft_p50_ms:.0f}" if r.ttft_p50_ms is not None else "-"
        lines.append(
            f"| {r.target} | {r.model} | {r.concurrency} | {r.requests} | "
            f"{'yes' if r.stream else 'no'} | "
            f"{r.n_ok} / {r.n_error} | {r.p50_ms:.0f} | {r.p95_ms:.0f} | {r.p99_ms:.0f} | {ttft} | "
            f"{r.throughput_rps:.1f} | {r.tokens_per_s:.0f} |"
        )
    return "\n".join(lines) + "\n"


def save_reports(reports: Sequence[LoadTestReport], out: Path | str) -> tuple[Path, Path]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = path if path.suffix == ".md" else path.with_suffix(".md")
    js = md.with_suffix(".json")
    md.write_text(render_markdown(reports), encoding="utf-8")
    js.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
    return md, js
