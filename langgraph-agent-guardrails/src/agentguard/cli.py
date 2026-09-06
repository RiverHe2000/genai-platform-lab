"""``agentguard {chat,approve,eval,replay,serve}``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentguard.agent import build_agent, build_model
from agentguard.config import Settings
from agentguard.evaluation.runner import (
    ModelFactory,
    ScenarioOutcome,
    run_scenarios,
    save_report,
    scripted_factory,
)
from agentguard.evaluation.scenarios import Scenario, load_scenarios
from agentguard.llm import ChatModel
from agentguard.logging_utils import configure_logging, log_event

log = logging.getLogger("agentguard")


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    model: dict[str, Any] = {}
    for field_name, arg_name in (
        ("kind", "model"),
        ("model", "model_name"),
        ("base_url", "base_url"),
        ("max_tokens", "max_tokens"),
        ("device", "device"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            model[field_name] = value
    if model:
        overrides["model"] = model
    for key in ("max_steps", "checkpoint_path", "loanbook_path", "audit_path", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return Settings(**overrides)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=["fake", "openai", "hf"], default=None)
    parser.add_argument("--model-name", dest="model_name", default=None)
    parser.add_argument("--base-url", dest="base_url", default=None)
    parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)


def _add_state_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-path", dest="checkpoint_path", type=Path, default=None)
    parser.add_argument("--loanbook-path", dest="loanbook_path", type=Path, default=None)
    parser.add_argument("--audit-path", dest="audit_path", type=Path, default=None)


def _print_result(result: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")


def cmd_chat(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    agent = build_agent(settings)
    result = agent.run(args.thread, args.message)
    if result.status == "awaiting_approval" and args.approve is not None:
        result = agent.resume(args.thread, approved=args.approve, approver=args.approver)
    _print_result(result.to_dict())
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    if settings.checkpoint_path is None:
        log.error("--checkpoint-path is required to resume a thread from another process")
        return 2
    agent = build_agent(settings)
    try:
        result = agent.resume(
            args.thread, approved=args.approved, approver=args.approver, note=args.note
        )
    except RuntimeError as exc:
        log.error(str(exc))
        return 1
    _print_result(result.to_dict())
    return 0


def _progress(outcome: ScenarioOutcome) -> None:
    log_event(
        log,
        "scenario_done",
        id=outcome.id,
        passed=outcome.passed,
        status=outcome.status,
        failures=outcome.failures,
    )


def cmd_eval(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    scenarios: list[Scenario] = []
    for path in args.scenarios:
        scenarios.extend(load_scenarios(path))
    factory: ModelFactory
    if settings.model.kind == "fake":
        factory = scripted_factory
        model_name = "scripted"
    else:
        shared: ChatModel = build_model(settings)

        def _shared_factory(_s: Scenario) -> ChatModel:
            return shared

        factory = _shared_factory
        model_name = shared.name
    report = run_scenarios(
        scenarios, settings, model_factory=factory, model_name=model_name, progress=_progress
    )
    json_path, md_path = save_report(report, args.out)
    sys.stdout.write(md_path.read_text(encoding="utf-8"))
    sys.stdout.write(f"\nwrote {json_path} and {md_path}\n")
    if args.gate and not report.gate_passed:
        sys.stdout.write("GATE FAILED\n")
        return 1
    if args.gate:
        sys.stdout.write("GATE PASSED\n")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from agentguard.audit import AuditLog

    sys.stdout.write(AuditLog(args.audit_path).replay(args.thread))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from agentguard.api import create_app

    settings = settings_from_args(args)
    uvicorn.run(create_app(build_agent(settings)), host=args.host, port=args.port, log_config=None)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentguard", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--plain-logs", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chat", help="run one user turn")
    p.add_argument("message")
    p.add_argument("--thread", default="cli")
    p.add_argument(
        "--approve",
        type=lambda s: s.lower() in ("1", "true", "yes"),
        default=None,
        help="auto-decide a pending approval (true/false)",
    )
    p.add_argument("--approver", default="cli-user")
    _add_model_args(p)
    _add_state_args(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("approve", help="resume a thread waiting for approval")
    p.add_argument("--thread", required=True)
    p.add_argument("--approved", type=lambda s: s.lower() in ("1", "true", "yes"), required=True)
    p.add_argument("--approver", default="cli-user")
    p.add_argument("--note", default="")
    _add_model_args(p)
    _add_state_args(p)
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("eval", help="run scenario files and write report.json/report.md")
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--gate",
        action="store_true",
        help="exit 1 unless every adversarial case is caught and no benign case is blocked",
    )
    _add_model_args(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("replay", help="print the audit timeline of a thread")
    p.add_argument("--thread", required=True)
    p.add_argument("--audit-path", dest="audit_path", type=Path, required=True)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    _add_model_args(p)
    _add_state_args(p)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, json_lines=not args.plain_logs)
    func: Any = args.func
    return int(func(args))
