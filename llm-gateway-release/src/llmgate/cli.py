"""``llmgate {serve,check-config,loadtest,eval,promote}``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from llmgate.backends.base import Backend
from llmgate.backends.fake import FakeBackend
from llmgate.backends.openai_compat import OpenAICompatBackend
from llmgate.backends.vllm import VLLMBackend
from llmgate.config import GatewaySettings, ProcessSettings
from llmgate.evaluation.promote import PromotionPolicy, decide, render_report, save_decision
from llmgate.evaluation.runner import CaseResult, load_report, run_suite, save_report
from llmgate.evaluation.suite import load_suite
from llmgate.gateway import Gateway
from llmgate.loadtest import LoadTestReport, run_loadtest, save_reports
from llmgate.loadtest import render_markdown as render_loadtest
from llmgate.observability import configure_logging, log_event

log = logging.getLogger("llmgate")


# ----- assembly -----------------------------------------------------------------------------


def build_backends(settings: GatewaySettings) -> dict[str, Backend]:
    backends: dict[str, Backend] = {}
    for b in settings.backends:
        if b.kind == "fake":
            backends[b.name] = FakeBackend(
                b.name,
                model=b.model or "fake-model",
                responses=b.fake_responses,
                latency_ms=b.fake_latency_ms,
                fail_every=b.fake_fail_every,
            )
        elif b.kind == "openai":
            backends[b.name] = OpenAICompatBackend(
                b.name, b.base_url, b.model, api_key_env=b.api_key_env, timeout_s=b.timeout_s
            )
        elif b.kind == "vllm":
            backends[b.name] = VLLMBackend(
                b.name, b.base_url, b.model, api_key_env=b.api_key_env, timeout_s=b.timeout_s
            )
        else:
            from llmgate.backends.hf_local import HFLocalBackend

            backends[b.name] = HFLocalBackend(b.name, b.model, device=b.device)
    return backends


def build_app(settings: GatewaySettings) -> Any:
    from llmgate.api import create_app

    gateway = Gateway(settings, build_backends(settings))
    return create_app(gateway, settings)


def _client_for(args: argparse.Namespace) -> tuple[httpx.AsyncClient, str]:
    if getattr(args, "base_url", None):
        base = str(args.base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return httpx.AsyncClient(base_url=base, timeout=args.timeout), base
    settings = GatewaySettings.load(args.config)
    app = build_app(settings)
    return (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway", timeout=args.timeout
        ),
        f"in-process:{Path(args.config).name}",
    )


# ----- commands -----------------------------------------------------------------------------


def cmd_check_config(args: argparse.Namespace) -> int:
    settings = GatewaySettings.load(args.config)
    sys.stdout.write(
        f"ok: {len(settings.backends)} backends {settings.backend_names}, "
        f"strategy={settings.routing.strategy}, primary={settings.routing.primary}, "
        f"virtual model={settings.routing.name}\n"
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = GatewaySettings.load(args.config)
    uvicorn.run(build_app(settings), host=args.host, port=args.port, log_config=None)
    return 0


def cmd_loadtest(args: argparse.Namespace) -> int:
    async def run() -> list[LoadTestReport]:
        client, target = _client_for(args)
        reports: list[LoadTestReport] = []
        try:
            for conc in args.concurrency:
                report = await run_loadtest(
                    client,
                    target=target,
                    model=args.model,
                    concurrency=conc,
                    requests=args.requests,
                    max_tokens=args.max_tokens,
                    stream=args.stream,
                    prompt=args.prompt,
                )
                reports.append(report)
                log_event(
                    log,
                    "loadtest_done",
                    concurrency=conc,
                    p95_ms=report.p95_ms,
                    rps=report.throughput_rps,
                )
        finally:
            await client.aclose()
        return reports

    reports = asyncio.run(run())
    sys.stdout.write(render_loadtest(reports))
    if args.out:
        md, js = save_reports(reports, args.out)
        sys.stdout.write(f"wrote {md} and {js}\n")
    return 0


def _progress(result: CaseResult) -> None:
    log_event(log, "case_done", id=result.id, score=result.score, status=result.status)


def cmd_eval(args: argparse.Namespace) -> int:
    cases = load_suite(args.suite)

    async def run() -> Any:
        client, target = _client_for(args)
        try:
            return await run_suite(
                client,
                cases,
                model=args.model,
                target=target,
                concurrency=args.concurrency,
                progress=_progress,
            )
        finally:
            await client.aclose()

    report = asyncio.run(run())
    json_path, md_path = save_report(report, args.out)
    sys.stdout.write(md_path.read_text(encoding="utf-8"))
    sys.stdout.write(f"\nwrote {json_path} and {md_path}\n")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    candidate = load_report(args.candidate)
    baseline = load_report(args.baseline)
    policy = PromotionPolicy.load(args.policy) if args.policy else PromotionPolicy()
    decision = decide(candidate, baseline, policy)
    report_md = render_report(decision, candidate, baseline, policy)
    sys.stdout.write(report_md)
    if args.out:
        md, js = save_decision(decision, report_md, args.out)
        sys.stdout.write(f"\nwrote {md} and {js}\n")
    return 0 if decision.promote else 1


# ----- parser -------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    defaults = ProcessSettings()
    parser = argparse.ArgumentParser(prog="llmgate", description=__doc__)
    parser.add_argument("--log-level", default=defaults.log_level)
    parser.add_argument("--plain-logs", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-config", help="validate a gateway config file")
    p.add_argument("--config", default=str(defaults.config_path))
    p.set_defaults(func=cmd_check_config)

    p = sub.add_parser("serve", help="run the gateway")
    p.add_argument("--config", default=str(defaults.config_path))
    p.add_argument("--host", default=defaults.host)
    p.add_argument("--port", type=int, default=defaults.port)
    p.set_defaults(func=cmd_serve)

    def target_args(q: argparse.ArgumentParser) -> None:
        q.add_argument(
            "--base-url",
            dest="base_url",
            default=None,
            help="running gateway/vLLM; omit to run in-process from --config",
        )
        q.add_argument("--config", default=str(defaults.config_path))
        q.add_argument("--model", required=True)
        q.add_argument("--timeout", type=float, default=120.0)

    p = sub.add_parser("loadtest", help="closed-loop load generator")
    target_args(p)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=64)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--prompt", default="In two sentences, what is a loan-to-value ratio?")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_loadtest)

    p = sub.add_parser("eval", help="run an evaluation suite through the gateway")
    target_args(p)
    p.add_argument("--suite", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("promote", help="promotion decision: candidate vs baseline (exit 1 = HOLD)")
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--policy", default="")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_promote)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_lines=not args.plain_logs)
    func: Any = args.func
    return int(func(args))
