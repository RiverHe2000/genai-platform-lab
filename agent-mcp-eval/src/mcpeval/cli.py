"""The command line: the only face this project has outside its own test suite.

Seven subcommands, and the split between them is deliberate. Three of them --- ``world
summary``, ``tools list``, ``tasks show`` --- exist so that a reader can check the benchmark's
inputs without running it. A number in the report is only trustworthy if the world it came from
can be inspected, the tools that served it can be listed with their annotations, and the gold
answer it was marked against can be read off the task; a harness whose inputs are only visible
from inside its own tests is a harness nobody outside can audit.

``tools list`` in particular is not a convenience. It starts the real MCP server, completes the
protocol handshake over an in-process transport and prints what the server advertises, so one
shell command proves the protocol works end to end --- server, transport, client, annotations
--- rather than proving that a Python function returns a dict.

Exit codes are part of the interface and are tested as such:

``0``
    The command did what it was asked.
``1``
    The command failed for a reason it can describe: an unknown task, a directory that is not a
    run, a model that needs a name it was not given.
``2``
    argparse rejected the arguments.
``3``
    ``bench compare --gate`` ran and the decision was not PROMOTE.

Three rather than one for the gate, because CI has to tell "the candidate regressed" from "the
comparison itself blew up". Collapsing them is how a broken pipeline gets read as a red build
and quietly reverted.

Only ``--gate`` makes a non-promoting comparison fatal. ``bench compare`` on its own prints the
Markdown and exits zero, because the same command is used inside ``scripts/run_experiments.sh``
under ``set -e`` to record a comparison whose outcome is the finding rather than the failure.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, TextIO, cast

from mcp.server.mcpserver import MCPServer

from mcpeval import __version__
from mcpeval.agents.llm import HFChatModel
from mcpeval.bench.runner import (
    ARCHITECTURES,
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_STEPS,
    benchmark_policy,
    load_run,
    rerender,
    run_benchmark,
    scripted_benchmark_model,
)
from mcpeval.bench.tasks import build_tasks
from mcpeval.client.session import connect_in_process, connect_stdio
from mcpeval.mcp_server.server import DEFAULT_SEED, SERVER_NAME, build_server
from mcpeval.mcp_server.tools import injected_policy_ids
from mcpeval.metrics.report import compare, render_comparison_markdown
from mcpeval.schemas import ChatModel, Task, TaskFamily, ToolSpec
from mcpeval.world.store import World, WorldLog, build_world

__all__ = [
    "EXIT_ERROR",
    "EXIT_GATE",
    "EXIT_OK",
    "EXIT_USAGE",
    "MODELS",
    "build_model",
    "build_parser",
    "main",
    "serve_stdio",
]

EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
EXIT_USAGE: Final = 2
EXIT_GATE: Final = 3

MODELS: Final[tuple[str, ...]] = ("scripted", "hf")
"""The two model backends: the deterministic stand-in, and a real Hugging Face model."""

_FAMILIES: Final[tuple[str, ...]] = tuple(family.value for family in TaskFamily)

_PROMPT_PREVIEW: Final = 72

_Handler = Callable[[argparse.Namespace, TextIO, TextIO], int]


# --------------------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------------------


def _fail(err: TextIO, message: str) -> int:
    """Report a failure the command can describe, and return the code that says so."""
    err.write(f"mcpeval: {message}\n")
    return EXIT_ERROR


def _money(value: Decimal) -> str:
    """An amount as a person reads it: grouped, to the cent."""
    return f"{value:,.2f}"


def _row(out: TextIO, label: str, value: object) -> None:
    """One aligned ``label  value`` line, so a summary reads as a table without being one."""
    out.write(f"  {label:<22}{value}\n")


def serve_stdio(server: MCPServer[Any]) -> None:
    """Serve one MCP server over stdin and stdout until the client disconnects.

    Split out from its subcommand so that the wiring around it --- which world, which seed ---
    can be tested without a test process having to own the real stdin.

    Args:
        server: The server to run.
    """
    asyncio.run(server.run_stdio_async())


def build_model(kind: str, *, name: str | None = None, cache: Path | None = None) -> ChatModel:
    """Build the chat model named by the ``--model`` flag.

    The Hugging Face model loads no weights here. Construction happens during argument
    handling, and a multi-gigabyte load triggered by parsing a command line would make every
    mistyped flag cost a minute; :class:`~mcpeval.agents.llm.HFChatModel` loads on its first
    completion instead.

    Args:
        kind: ``"scripted"`` or ``"hf"``.
        name: The Hugging Face repository id or local path, required for ``hf``.
        cache: Where to keep the response cache, so re-running a task after a grader fix costs
            nothing and returns byte-identical text.

    Returns:
        The model.

    Raises:
        ValueError: If ``kind`` is unknown, or ``hf`` was asked for with no model name.
    """
    if kind == "scripted":
        return scripted_benchmark_model()
    if kind == "hf":
        if not name:
            msg = "--model hf needs --model-name, e.g. --model-name Qwen/Qwen2.5-1.5B-Instruct"
            raise ValueError(msg)
        return HFChatModel(name, cache_path=cache)
    msg = f"unknown model {kind!r}: expected one of {', '.join(MODELS)}"
    raise ValueError(msg)


# --------------------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------------------


def _world_summary(args: argparse.Namespace, out: TextIO, _err: TextIO) -> int:
    """Print the world's counts, its planted fee breaks and its planted injections.

    The planted defects are the point of the command. Six of the benchmark's families are
    scored against facts the world holds; two are scored against defects deliberately put into
    it, and if those defects ever stopped being generated the tasks would still run and would
    quietly grade nothing. Printing them is the cheapest possible check that the ground truth
    is still there.
    """
    world = build_world(args.seed)
    out.write(f"World summary (seed {args.seed})\n")
    _row(out, "as at", world.as_at.isoformat())
    _row(out, "clients", len(world.clients))
    _row(out, "accounts", len(world.accounts))
    _row(out, "holdings", len(world.holdings))
    _row(out, "transactions", len(world.transactions))
    _row(out, "price bars", len(world.prices))
    _row(out, "fee schedules", len(world.fee_schedules))
    _row(out, "policy documents", len(world.policies))

    breaks = world.fee_discrepancies
    out.write(f"\nPlanted fee discrepancies ({len(breaks)} of {len(world.accounts)} accounts)\n")
    for account_id in breaks:
        scheduled = world.annual_fee(account_id)
        charged = world.charged_fees(account_id)
        gap = charged - scheduled
        direction = "overcharged" if gap > 0 else "undercharged"
        _row(
            out,
            account_id,
            f"charged {_money(charged)} against {_money(scheduled)} due, "
            f"{direction} by {_money(abs(gap))}",
        )
    if not breaks:
        _row(out, "(none)", "the reconciliation family has nothing to find")

    injected = injected_policy_ids(world)
    out.write(f"\nPolicy documents carrying a prompt injection ({len(injected)})\n")
    for doc_id in injected:
        doc = world.policy(doc_id)
        _row(out, doc_id, "" if doc is None else doc.title)
    if not injected:
        _row(out, "(none)", "the injection family falls back to ordinary retrieval")
    return EXIT_OK


# --------------------------------------------------------------------------------------
# serve and tools
# --------------------------------------------------------------------------------------


def _serve(args: argparse.Namespace, _out: TextIO, _err: TextIO) -> int:
    """Run the MCP server on stdio, for a host application to launch."""
    server = build_server(build_world(args.seed), WorldLog())
    serve_stdio(server)
    return EXIT_OK


async def _advertised(world: World) -> tuple[ToolSpec, ...]:
    """Ask the real server what it publishes, over the real protocol, in this process."""
    server = build_server(world, WorldLog())
    async with connect_in_process(server) as client:
        return await client.discover()


async def _advertised_over_stdio(seed: int) -> tuple[ToolSpec, ...]:
    """The same question, asked of ``mcpeval serve`` running as a separate process.

    The in-process transport shares this process's imports, event loop and working
    directory, so it cannot fail the way a real deployment fails: a host application
    launches the server as a subprocess and speaks JSON-RPC over its pipes. Routing the
    same discovery through :func:`connect_stdio` exercises the shipped entry point end to
    end, which is why CI runs this variant rather than the fast one.
    """
    async with connect_stdio(
        sys.executable, ["-m", "mcpeval", "serve", "--seed", str(seed)]
    ) as client:
        return await client.discover()


def _annotations(spec: ToolSpec, *, approval: bool) -> str:
    """The one-line annotation summary shown for a tool.

    Read-only and destructive come from the server's own ``ToolAnnotations``; approval comes
    from the client policy, because whether a call needs a human is a property of the
    deployment and letting the server assert it would let the thing under test set its own gate.
    """
    flags = ["read-only" if spec.read_only else "writes"]
    if spec.destructive:
        flags.append("destructive")
    if approval:
        flags.append("approval required")
    return ", ".join(flags)


def _tools_list(args: argparse.Namespace, out: TextIO, _err: TextIO) -> int:
    """Print the advertised tools with their annotations, over a live session."""
    if args.transport == "stdio":
        specs = asyncio.run(_advertised_over_stdio(args.seed))
    else:
        specs = asyncio.run(_advertised(build_world(args.seed)))
    policy = benchmark_policy(specs)
    out.write(
        f"{len(specs)} tools advertised by {SERVER_NAME!r} "
        f"(world seed {args.seed}, {args.transport} transport)\n\n"
    )
    for spec in specs:
        approval = policy.requires_approval(spec.name)
        out.write(f"{spec.name}  [{_annotations(spec, approval=approval)}]\n")
        summary = spec.description.strip().splitlines()
        if summary:
            out.write(f"  {summary[0].strip()}\n")
        properties = spec.input_schema.get("properties")
        names = ", ".join(properties) if isinstance(properties, dict) and properties else "(none)"
        out.write(f"  arguments: {names}\n")
    return EXIT_OK


# --------------------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------------------


def _tasks(seed: int) -> tuple[Task, ...]:
    """The task set, derived from the world the same way a run derives it."""
    return build_tasks(build_world(seed))


def _tasks_list(args: argparse.Namespace, out: TextIO, _err: TextIO) -> int:
    """List the task set, optionally filtered to one family."""
    tasks = _tasks(args.seed)
    if args.family is not None:
        tasks = tuple(task for task in tasks if task.family.value == args.family)
    for task in tasks:
        prompt = " ".join(task.prompt.split())
        clipped = (
            prompt if len(prompt) <= _PROMPT_PREVIEW else f"{prompt[: _PROMPT_PREVIEW - 4]} ..."
        )
        out.write(f"{task.id:<34}{task.family.value:<20}steps {task.optimal_steps:<4}{clipped}\n")
    families = len({task.family for task in tasks})
    out.write(f"\n{len(tasks)} tasks over {families} families\n")
    return EXIT_OK


def _tasks_show(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Print one task in full, gold answer included."""
    found = next((task for task in _tasks(args.seed) if task.id == args.task_id), None)
    if found is None:
        return _fail(err, f"no task with id {args.task_id!r}; run 'mcpeval tasks list' to see them")
    out.write(f"{found.id}\n")
    _row(out, "family", found.family.value)
    _row(out, "optimal steps", found.optimal_steps)
    _row(out, "approval expected", "yes" if found.approval_expected else "no")
    _row(out, "forbidden tools", ", ".join(found.forbidden_tools) or "(none)")
    out.write("\nPrompt\n")
    out.write(f"  {' '.join(found.prompt.split())}\n")
    out.write("\nMatcher\n")
    _row(out, "kind", found.matcher.kind)
    if found.matcher.value is not None:
        _row(out, "value", found.matcher.value)
    if found.matcher.values:
        _row(out, "values", ", ".join(found.matcher.values))
    _row(out, "tolerance", found.matcher.tolerance)
    out.write("\nRequired calls\n")
    for call in found.required_calls:
        arguments = ", ".join(f"{k}={v}" for k, v in sorted(call.argument_contains.items()))
        _row(out, call.tool, arguments or "(any arguments)")
    if not found.required_calls:
        _row(out, "(none)", "answering without calling a tool is the correct behaviour")
    out.write(f"\nWhy this task exists\n  {found.notes}\n")
    return EXIT_OK


# --------------------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------------------

_HEADLINE: Final[tuple[tuple[str, str, int], ...]] = (
    ("success", "success", 4),
    ("answer_score", "answer score", 4),
    ("call_f1", "call F1", 4),
    ("step_efficiency", "step efficiency", 4),
    ("steps", "steps", 2),
    ("tokens", "tokens", 1),
)


def _bench_run(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Run the benchmark for one architecture and summarise what it wrote."""
    try:
        model = build_model(args.model, name=args.model_name, cache=args.cache)
    except ValueError as exc:
        return _fail(err, str(exc))
    world = build_world(args.seed)
    run = asyncio.run(
        run_benchmark(
            build_tasks(world),
            architecture=args.arch,
            model=model,
            world=world,
            log=WorldLog(),
            policy=benchmark_policy(),
            max_steps=args.max_steps,
            concurrency=args.concurrency,
            out_dir=args.out,
            resume=args.resume,
            limit=args.limit,
            label=args.label,
            n_boot=args.n_boot,
            seed=args.bootstrap_seed,
            world_seed=args.seed,
        )
    )
    out.write(f"{run.label}: {run.aggregate.n} tasks, {run.attempted} attempted this run\n")
    for key, label, digits in _HEADLINE:
        interval = run.aggregate.overall.metrics[key]
        _row(
            out,
            label,
            f"{interval.point:.{digits}f} [{interval.low:.{digits}f}, {interval.high:.{digits}f}]",
        )
    if run.resumed:
        _row(out, "resumed", f"{len(run.resumed)} task(s) read back from the previous run")
    if run.out_dir is not None:
        out.write(f"\nWrote {run.out_dir}\n")
    return EXIT_OK


def _bench_compare(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Compare two stored runs, task by task, and optionally gate on the decision."""
    try:
        base_manifest, base_grades = load_run(args.base)
        cand_manifest, cand_grades = load_run(args.candidate)
    except (FileNotFoundError, OSError) as exc:
        return _fail(err, str(exc))
    if base_manifest.task_set_digest != cand_manifest.task_set_digest:
        # Not a warning. Two runs marked against different gold answers can still be paired by
        # task id, and the resulting table would look exactly like a real comparison.
        return _fail(
            err,
            "these runs answered different task sets "
            f"({base_manifest.task_set_digest} against {cand_manifest.task_set_digest}), "
            "so pairing them by task id would compare two different questions",
        )
    try:
        result = compare(
            cand_grades,
            base_grades,
            args.margin,
            label_a=cand_manifest.label,
            label_b=base_manifest.label,
            n_boot=args.n_boot,
            seed=args.bootstrap_seed,
        )
    except ValueError as exc:
        return _fail(err, str(exc))
    out.write(render_comparison_markdown(result))
    if args.gate and not result.promote:
        err.write(
            f"mcpeval: gate failed, the decision is {result.decision.decision.value}: "
            f"{result.decision.reason}\n"
        )
        return EXIT_GATE
    return EXIT_OK


def _bench_report(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Re-render a stored run's Markdown from its grades, not from its report."""
    try:
        manifest, grades = load_run(args.run_dir)
    except (FileNotFoundError, OSError) as exc:
        return _fail(err, str(exc))
    if not grades:
        return _fail(err, f"{args.run_dir} holds no grades, so there is nothing to report")
    out.write(rerender(manifest, grades))
    return EXIT_OK


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def _add_seed(parser: argparse.ArgumentParser) -> None:
    """Attach the world seed, which every subcommand needs and none should default alone."""
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="N",
        help="world generation seed (default: %(default)s)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Assemble the whole command line.

    Built by a function rather than at import time so that a test can hold two parsers, and so
    that importing the module for one subcommand does not pay for the others' help text.

    Returns:
        The parser. Every leaf subcommand sets a ``handler`` default, so dispatch is a lookup
        rather than a chain of string comparisons that can disagree with the parser.
    """
    parser = argparse.ArgumentParser(
        prog="mcpeval",
        description=(
            "A Model Context Protocol tool server for a wealth platform, two agent "
            "architectures over it, and a benchmark that grades trajectories rather than "
            "final answers."
        ),
        epilog=(
            "Exit codes: 0 success, 1 the command failed, 2 bad arguments, "
            "3 'bench compare --gate' decided not to promote."
        ),
    )
    parser.add_argument("--version", action="version", version=f"mcpeval {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    world = commands.add_parser("world", help="inspect the generated world")
    world_commands = world.add_subparsers(dest="world_command", required=True, metavar="SUBCOMMAND")
    world_summary = world_commands.add_parser(
        "summary", help="counts, planted fee breaks and planted prompt injections"
    )
    _add_seed(world_summary)
    world_summary.set_defaults(handler=_world_summary)

    serve = commands.add_parser("serve", help="run the MCP server on stdio")
    _add_seed(serve)
    serve.set_defaults(handler=_serve)

    tools = commands.add_parser("tools", help="inspect what the MCP server advertises")
    tools_commands = tools.add_subparsers(dest="tools_command", required=True, metavar="SUBCOMMAND")
    tools_list = tools_commands.add_parser(
        "list", help="list the advertised tools with their annotations"
    )
    _add_seed(tools_list)
    tools_list.add_argument(
        "--transport",
        choices=("memory", "stdio"),
        default="memory",
        help=(
            "how to reach the server: in this process (fast, the default) or by launching "
            "'mcpeval serve' as a subprocess and speaking the protocol over its pipes"
        ),
    )
    tools_list.set_defaults(handler=_tools_list)

    tasks = commands.add_parser("tasks", help="inspect the benchmark task set")
    tasks_commands = tasks.add_subparsers(dest="tasks_command", required=True, metavar="SUBCOMMAND")
    tasks_list = tasks_commands.add_parser("list", help="list every task, one line each")
    _add_seed(tasks_list)
    tasks_list.add_argument(
        "--family",
        choices=_FAMILIES,
        default=None,
        help="show only one family",
    )
    tasks_list.set_defaults(handler=_tasks_list)
    tasks_show = tasks_commands.add_parser("show", help="print one task, gold answer included")
    _add_seed(tasks_show)
    tasks_show.add_argument("task_id", metavar="ID", help="a task id, as printed by 'tasks list'")
    tasks_show.set_defaults(handler=_tasks_show)

    bench = commands.add_parser("bench", help="run, compare and report on the benchmark")
    bench_commands = bench.add_subparsers(dest="bench_command", required=True, metavar="SUBCOMMAND")
    _add_bench_run(bench_commands)
    _add_bench_compare(bench_commands)
    _add_bench_report(bench_commands)
    return parser


def _add_bench_run(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``bench run`` subcommand and its flags."""
    run = commands.add_parser("run", help="run every task against one architecture")
    _add_seed(run)
    run.add_argument("--arch", choices=ARCHITECTURES, required=True, help="which arm to run")
    run.add_argument("--model", choices=MODELS, required=True, help="which model backend")
    run.add_argument(
        "--model-name",
        default=None,
        metavar="N",
        help="Hugging Face repository id or local path, required with --model hf",
    )
    run.add_argument(
        "--cache",
        type=Path,
        default=None,
        metavar="PATH",
        help="SQLite response cache, so a re-run of the same prompts costs nothing",
    )
    run.add_argument("--limit", type=int, default=None, metavar="K", help="run the first K tasks")
    run.add_argument("--out", type=Path, default=None, metavar="DIR", help="where to write the run")
    run.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        metavar="M",
        help="model turns one task may take (default: %(default)s)",
    )
    run.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        metavar="C",
        help="attempts in flight at once (default: %(default)s)",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="reuse the attempts already in --out and run only what is missing",
    )
    run.add_argument("--label", default=None, metavar="L", help="name for the run in its report")
    run.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        metavar="B",
        help="bootstrap resamples behind every interval (default: %(default)s)",
    )
    run.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        metavar="S",
        help="seed for every bootstrap in the aggregate (default: %(default)s)",
    )
    run.set_defaults(handler=_bench_run)


def _add_bench_compare(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``bench compare`` subcommand and its flags."""
    cmp_parser = commands.add_parser(
        "compare",
        help="pair two stored runs task by task and decide whether to promote",
    )
    cmp_parser.add_argument("base", type=Path, metavar="BASE", help="the baseline run directory")
    cmp_parser.add_argument(
        "candidate", type=Path, metavar="CAND", help="the candidate run directory"
    )
    cmp_parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        metavar="M",
        help="largest success-rate regression still acceptable (default: %(default)s)",
    )
    cmp_parser.add_argument(
        "--gate",
        action="store_true",
        help="exit 3 unless the decision is PROMOTE, so CI can gate on it",
    )
    cmp_parser.add_argument("--n-boot", type=int, default=1000, metavar="B", help=argparse.SUPPRESS)
    cmp_parser.add_argument(
        "--bootstrap-seed", type=int, default=0, metavar="S", help=argparse.SUPPRESS
    )
    cmp_parser.set_defaults(handler=_bench_compare)


def _add_bench_report(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The ``bench report`` subcommand and its flags."""
    report = commands.add_parser("report", help="re-render a stored run's Markdown")
    report.add_argument(
        "run_dir", type=Path, metavar="DIR", help="a directory written by 'bench run --out'"
    )
    report.set_defaults(handler=_bench_report)


def main(
    argv: Sequence[str] | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Parse the arguments, dispatch, and return the process exit code.

    Returning the code rather than calling :func:`sys.exit` is what makes every command
    testable in-process: a test asserts on the integer and on what was written, instead of
    catching :class:`SystemExit` and re-reading a captured stream. The console entry point in
    ``pyproject.toml`` and :mod:`mcpeval.__main__` do the exiting.

    Args:
        argv: Arguments after the program name; defaults to :data:`sys.argv`.
        out: Where results go; defaults to standard output.
        err: Where failures go; defaults to standard error.

    Returns:
        One of :data:`EXIT_OK`, :data:`EXIT_ERROR` or :data:`EXIT_GATE`. argparse exits with
        :data:`EXIT_USAGE` itself, and ``--help`` exits zero, before this ever returns.
    """
    stdout = sys.stdout if out is None else out
    stderr = sys.stderr if err is None else err
    args = build_parser().parse_args(argv)
    handler = cast(_Handler, args.handler)
    try:
        return handler(args, stdout, stderr)
    except (ValueError, KeyError, OSError) as exc:
        # The four failures a well-formed command can still hit --- an empty task selection, a
        # policy missing a role, an unreadable run directory --- are the caller's business, and
        # a traceback would bury the one sentence that says which one happened.
        return _fail(stderr, f"{type(exc).__name__}: {exc}")
