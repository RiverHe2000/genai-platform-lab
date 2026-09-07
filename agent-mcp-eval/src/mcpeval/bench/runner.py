"""Running the whole benchmark: every task, one server, one aggregate, one audit trail.

The runner is the only place where the world, the MCP server, the permission policy, an
architecture, a model and the grader meet, so three decisions live here that nothing else can
make.

**One server, many attempts.** The server is built once and every task speaks to it over the
same in-process transport. A benchmark that started a subprocess per task would spend most of
its wall-clock time in the operating system, and the wall-clock cost of an architecture is one
of the things being compared --- so the transport must not be part of the measurement. Each
attempt still gets its own recorder, its own guarded client and its own fresh budget, because
those *are* the measurement.

**A failed task is a data point, not the end of the run.** A model that raises on task forty
must not discard the thirty-nine attempts before it. Every attempt is therefore wrapped, and a
failure is recorded as a trajectory with ``stop_reason="error"`` carrying the exception, which
is exactly what the grader and the failure taxonomy already know how to score. The alternative
--- letting it propagate --- turns a measurable defect in the model into an unmeasurable defect
in the harness.

**Everything is written down as it happens.** A real run over seventy long-horizon tasks with a
1.5B model takes hours, and hours of work must survive a laptop lid. Trajectories are appended
to JSONL as each attempt finishes, so a crash costs one attempt rather than the run; ``resume``
picks the file up and re-runs only what is missing. The grades, the aggregate, the Markdown and
the manifest are derived at the end from that log, so any number in the report can be traced
back to the trajectory that produced it, and the manifest says which model, which architecture,
which task set, which policy and which code version produced them.

The scripted model in :func:`scripted_benchmark_model` deserves its own warning. It exists so
that the whole benchmark --- every task, both architectures, the real MCP protocol, the real
policy and the real grader --- runs in CI in seconds and deterministically. It is a harness
exerciser, not a baseline: its answers are echoes of tool output, so its scores measure nothing
about anything and must never be quoted as a result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from mcpeval import __version__
from mcpeval.agents.llm import ScriptedChatModel
from mcpeval.agents.protocol import (
    Action,
    CallToolAction,
    ClarifyAction,
    FinalAction,
    HandoffAction,
    render_action,
)
from mcpeval.agents.single import SingleAgent
from mcpeval.agents.state import Approver, deny_all
from mcpeval.agents.supervisor import ALL_ROLES, SupervisorAgent, shared_budgets
from mcpeval.bench.tasks import READ_TOOLS, WRITE_TOOLS
from mcpeval.client.policy import (
    ANALYST,
    RESEARCHER,
    SUPERVISOR,
    WRITER,
    BudgetState,
    PermissionPolicy,
    RolePolicy,
)
from mcpeval.client.recorder import Clock, TrajectoryRecorder, append_jsonl, read_jsonl, write_jsonl
from mcpeval.client.session import GuardedToolClient, connect_in_process
from mcpeval.mcp_server.server import build_server
from mcpeval.metrics.failures import graded
from mcpeval.metrics.report import Aggregate, aggregate, render_json, render_markdown
from mcpeval.schemas import ChatModel, Grade, Message, Task, ToolSpec, Trajectory
from mcpeval.world.store import World, WorldLog

__all__ = [
    "AGGREGATE_FILENAME",
    "ARCHITECTURES",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MAX_STEPS",
    "GRADES_FILENAME",
    "MANIFEST_FILENAME",
    "REPORT_FILENAME",
    "TRAJECTORIES_FILENAME",
    "BenchmarkRun",
    "Manifest",
    "RunPaths",
    "benchmark_policy",
    "build_agent",
    "load_run",
    "policy_digest",
    "read_grades",
    "rerender",
    "run_benchmark",
    "run_budget",
    "scripted_benchmark_model",
    "task_set_digest",
    "write_grades",
]

ARCHITECTURES: Final[tuple[str, ...]] = ("single", "supervisor")
"""The two arms of the experiment, in the order a report lists them."""

DEFAULT_MAX_STEPS: Final = 20
"""Model turns one attempt may take, and the same number for both architectures.

It has to clear what the *slower* architecture structurally needs, not what the faster one
does, or the ceiling becomes the thing being measured. The longest gold chain is six tool
calls. A single agent spends six turns on them plus one to answer. A supervisor spends a turn
routing to a specialist for each, a turn for the specialist, a turn for the writer and a turn
for the verifier -- roughly two and a half times as many for the same work -- so a ceiling
sized for the single agent truncates the supervisor on exactly the hardest tasks and reports
it as `max_steps`.

Twenty clears the supervisor's worst case with room for a wrong turn. It was 12, sized off
the gold chain alone, and that would have put a wiring artefact into the headline comparison
in the same way the missing loop guard did. The cost of orchestration is still measured --
steps and tokens are reported metrics — it is simply not enforced as a constraint that falls
on one arm only."""

DEFAULT_CONCURRENCY: Final = 4
"""Attempts in flight at once. Four keeps a GPU-bound run from queueing behind its own
tokenisation without making a stalled attempt hard to find in a log."""

TRAJECTORIES_FILENAME: Final = "trajectories.jsonl"
GRADES_FILENAME: Final = "grades.jsonl"
AGGREGATE_FILENAME: Final = "aggregate.json"
REPORT_FILENAME: Final = "report.md"
MANIFEST_FILENAME: Final = "manifest.json"

_Agent = SingleAgent | SupervisorAgent
"""Either arm. Both expose the same :meth:`run` signature, deliberately, so the runner
cannot treat them differently even by accident."""


# --------------------------------------------------------------------------------------
# Output layout
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunPaths:
    """The five files one run writes, named in one place.

    A dataclass rather than five string constants used at the call site, because ``resume``
    and ``bench report`` have to open exactly the files ``bench run`` wrote, and a
    disagreement between them would show up as an empty resume rather than as an error.
    """

    root: Path

    @property
    def trajectories(self) -> Path:
        """The append-only log of attempts: the source of truth everything else derives from."""
        return self.root / TRAJECTORIES_FILENAME

    @property
    def grades(self) -> Path:
        """Scores, rewritten whole at the end of a run because they are derived, not observed."""
        return self.root / GRADES_FILENAME

    @property
    def aggregate(self) -> Path:
        """The summarised run, as JSON."""
        return self.root / AGGREGATE_FILENAME

    @property
    def report(self) -> Path:
        """The summarised run, as Markdown."""
        return self.root / REPORT_FILENAME

    @property
    def manifest(self) -> Path:
        """What produced the numbers: model, architecture, task set, policy, code version."""
        return self.root / MANIFEST_FILENAME


def write_grades(path: Path, grades: Sequence[Grade]) -> int:
    """Write grades as JSONL, replacing whatever was there.

    Rewritten rather than appended because grades are a *derivation* of the trajectory log:
    a grader fix should change every row in this file on the next run, and an appended file
    would silently keep the old rows next to the new ones.

    Args:
        path: Destination file; parent directories are created.
        grades: What to write, in order.

    Returns:
        The number of rows written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in grades:
            handle.write(row.model_dump_json())
            handle.write("\n")
    return len(grades)


def read_grades(path: Path) -> list[Grade]:
    """Read a JSONL file of grades.

    Args:
        path: The file to read.

    Returns:
        The grades in file order. Blank lines are skipped; anything else that fails to parse
        raises, because a silently dropped grade would change a benchmark's denominator.
    """
    rows: list[Grade] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(Grade.model_validate(json.loads(line)))
    return rows


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def _digest(payload: str) -> str:
    """A short, stable fingerprint. Sixteen hex characters separate anything a run will see."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def task_set_digest(tasks: Sequence[Task]) -> str:
    """Fingerprint a task set, gold answers and all.

    Every field of every task goes into the digest, not just the identifiers, because the
    task set is derived from the world: regenerate the world with a different seed and the
    ids stay the same while every expected number moves. Two runs whose digests agree
    answered the same questions and were marked against the same answers, which is the only
    condition under which their scores may be compared.

    Args:
        tasks: The task set, in order.

    Returns:
        A sixteen-character hex digest.
    """
    return _digest("\n".join(task.model_dump_json() for task in tasks))


def policy_digest(policy: PermissionPolicy) -> str:
    """Fingerprint a permission policy.

    The payload is rebuilt with sorted keys and sorted tool lists rather than digesting
    ``model_dump_json`` directly: three of the policy's fields are ``frozenset``, whose
    iteration order depends on string hashing and therefore on ``PYTHONHASHSEED``. Digesting
    that would give the same policy a different fingerprint in a different process, which is
    the one thing a provenance record must never do.

    Args:
        policy: The policy that governed the run.

    Returns:
        A sixteen-character hex digest.
    """
    payload = {
        "roles": {
            name: {
                "allow": list(role.allow),
                "may_write": role.may_write,
                "max_steps": role.max_steps,
                "max_tokens": role.max_tokens,
            }
            for name, role in sorted(policy.roles.items())
        },
        "known_tools": sorted(policy.known_tools),
        "write_tools": sorted(policy.write_tools),
        "approval_required": sorted(policy.approval_required),
        "max_steps": policy.max_steps,
        "max_tool_calls": policy.max_tool_calls,
        "max_tokens": policy.max_tokens,
        "max_wall_ms": policy.max_wall_ms,
    }
    return _digest(json.dumps(payload, sort_keys=True, ensure_ascii=False))


class Manifest(BaseModel):
    """What produced a run's numbers, written next to them.

    A benchmark result is worthless six months later unless it says what it measured with.
    Every field here answers a question somebody will ask of a figure in the report: which
    model, which topology, which questions, marked against which answers, under which
    permission policy, with how much headroom, and from which version of this code.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    architecture: str
    model: str
    code_version: str
    task_count: int = Field(ge=0)
    task_set_digest: str
    policy_digest: str
    write_tools: tuple[str, ...]
    approval_required: tuple[str, ...]
    max_steps: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    n_boot: int = Field(ge=1)
    alpha: float = Field(gt=0.0, lt=1.0)
    bootstrap_seed: int
    world_seed: int | None = None
    limit: int | None = None
    resumed: tuple[str, ...] = ()
    created_at: str | None = None


class BenchmarkRun(BaseModel):
    """One architecture's attempt at one task set, with everything needed to re-derive it."""

    model_config = ConfigDict(frozen=True)

    manifest: Manifest
    trajectories: tuple[Trajectory, ...]
    grades: tuple[Grade, ...]
    aggregate: Aggregate
    out_dir: str | None = None

    @property
    def label(self) -> str:
        """The run's name, as it appears in the report heading."""
        return self.manifest.label

    @property
    def resumed(self) -> tuple[str, ...]:
        """Task ids that were read back from disk rather than attempted again."""
        return self.manifest.resumed

    @property
    def attempted(self) -> int:
        """How many tasks this invocation actually ran."""
        return len(self.trajectories) - len(self.manifest.resumed)


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def benchmark_policy(specs: Sequence[ToolSpec] | None = None) -> PermissionPolicy:
    """The permission policy the benchmark runs under, scoped to the live tool names.

    :func:`~mcpeval.client.policy.default_policy` ships scopes named after an earlier draft of
    the inventory (``get_client``, ``place_order``) while the server publishes ``client_lookup``
    and ``order_place``. Left alone its patterns match nothing, every specialist is refused
    every call, and the multi-agent arm would be measured with its hands tied --- a result about
    the wiring rather than about the topology. So the scopes are rebuilt here from the names in
    :mod:`mcpeval.bench.tasks`, which is also where the task set gets them, and a test asserts
    that list against the server's own advertisement.

    The division of labour is the one the architecture argues for: the researcher resolves
    identifiers and fetches documents but cannot compute; the analyst computes but cannot browse
    the client book; the writer sees only enough to cite. The verifier has no entry at all, so
    every call it makes is refused for having no scope --- which is the property wanted of
    something whose job is to check evidence rather than gather it. Only the supervisor may
    write, and only with a recorded approval.

    Args:
        specs: The server's live advertisement, if the caller has already discovered it. The
            runner refreshes the policy from it anyway; passing it here only saves a step.

    Returns:
        The policy, with ceilings wide enough that an attempt ends on the agent's own rules or
        on the runner's step limit rather than on a token count nobody chose.
    """
    roles = {
        SUPERVISOR: RolePolicy(
            name=SUPERVISOR, allow=("*",), may_write=True, max_steps=64, max_tokens=1_000_000
        ),
        RESEARCHER: RolePolicy(
            name=RESEARCHER,
            allow=(
                "client_lookup",
                "client_search",
                "account_holdings",
                "transactions_list",
                "policy_search",
                "policy_fetch",
            ),
            max_steps=64,
            max_tokens=1_000_000,
        ),
        ANALYST: RolePolicy(
            name=ANALYST,
            allow=(
                "portfolio_valuation",
                "fee_reconcile",
                "fee_schedule",
                "price_history",
                "calc_eval",
                "account_holdings",
                "transactions_list",
            ),
            max_steps=64,
            max_tokens=1_000_000,
        ),
        WRITER: RolePolicy(
            name=WRITER,
            allow=("client_lookup", "policy_fetch", "policy_search"),
            max_steps=64,
            max_tokens=1_000_000,
        ),
    }
    policy = PermissionPolicy(
        roles=roles,
        known_tools=frozenset(READ_TOOLS) | frozenset(WRITE_TOOLS),
        write_tools=frozenset(WRITE_TOOLS),
        approval_required=frozenset(WRITE_TOOLS),
        max_steps=64,
        max_tool_calls=64,
        max_tokens=1_000_000,
        max_wall_ms=600_000.0,
    )
    if specs is None:
        return policy
    return policy.with_tools(specs)


def run_budget(policy: PermissionPolicy, *, max_steps: int) -> BudgetState:
    """The tally one attempt starts with.

    The step ceiling comes from the runner rather than from the policy, because ``--max-steps``
    is the knob an experimenter turns and a run that stopped at the policy's twelve when it was
    asked for twenty would report ``budget`` where it should report ``max_steps`` --- and the
    two send you to different files to fix it. Every other ceiling is the policy's, which is
    where a deployment's limits belong.

    Args:
        policy: The policy in force; consulted for the supervisor role's ceilings.
        max_steps: Model turns this attempt may take.

    Returns:
        A fresh, zeroed tally.

    Raises:
        KeyError: If the policy has no supervisor role. Both architectures charge every model
            turn to that one role so their costs stay comparable, so its absence is a wiring
            fault rather than something an agent can cause.
    """
    return policy.new_budget(SUPERVISOR).model_copy(update={"max_steps": max_steps})


def build_agent(
    architecture: str, tools: Sequence[ToolSpec] = (), *, approver: Approver = deny_all
) -> _Agent:
    """Construct the arm named by ``architecture``.

    Args:
        architecture: ``"single"`` or ``"supervisor"``.
        tools: The server's advertisement, so the prompts carry real descriptions and schemas.
        approver: Consulted before a call that needs a human. Denies by default; see
            :func:`run_benchmark` for why the benchmark never approves.

    Returns:
        The agent, ready to run.

    Raises:
        ValueError: If ``architecture`` names no arm.
    """
    if architecture == "single":
        return SingleAgent(tools=tools, approver=approver)
    if architecture == "supervisor":
        return SupervisorAgent(tools=tools, approver=approver)
    msg = f"unknown architecture {architecture!r}: expected one of {', '.join(ARCHITECTURES)}"
    raise ValueError(msg)


# --------------------------------------------------------------------------------------
# The scripted stand-in model
# --------------------------------------------------------------------------------------

_ANSWER_CHARS: Final = 420
"""How much of a tool result the scripted model quotes back.

Not cosmetic: :class:`~mcpeval.agents.llm.ScriptedChatModel` honours ``max_tokens`` by
truncating, so a reply carrying a whole price series would be cut mid-JSON, fail to parse, and
turn every long result into a protocol failure that says nothing about the harness."""

_ROLE_RE: Final = re.compile(r"You are the (\w+) agent")
_ACCOUNT_RE: Final = re.compile(r"\bACC-[A-Z0-9]+")
_CLIENT_RE: Final = re.compile(r"\bCLI-[A-Z0-9]+")
_DOC_RE: Final = re.compile(r"\bPOL-[A-Z0-9]+")
_SCHEDULE_RE: Final = re.compile(r"\bFS-[A-Z0-9]+")
_TICKER_RE: Final = re.compile(r"\b[A-Z]{3,4}\b")
_DATE_RE: Final = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_WHITESPACE_RE: Final = re.compile(r"\s+")

_NOT_TICKERS: Final[frozenset[str]] = frozenset(
    {"ACC", "ACT", "AUD", "CLI", "ISO", "NSW", "NT", "POL", "QLD", "SA", "TAS", "VIC", "WA"}
)

_QUESTION_MARKER: Final = "The adviser asked: "

_ROUTING: Final[tuple[tuple[str, str], ...]] = (
    ("researcher", "look up the records this question needs and report what they say"),
    ("writer", "compose the answer from the findings, quoting no figure that is not in them"),
    ("verifier", "check every figure in the draft against the recorded tool results"),
)


def _first_user(messages: Sequence[Message]) -> str:
    """The turn that set the task: the adviser's question, or a specialist's brief."""
    return next((m.content for m in messages if m.role == "user"), "")


def _question(messages: Sequence[Message]) -> str:
    """The adviser's question, however it was framed to this role.

    A specialist is briefed with the question, its instruction and the findings so far, and
    the findings quote tool output full of identifiers that are not the question's. Taking
    only the line after the brief's marker keeps the scripted model planning against what was
    asked rather than against whatever the last tool happened to return.
    """
    first = _first_user(messages)
    if first.startswith(_QUESTION_MARKER):
        return first[len(_QUESTION_MARKER) :].split("\n", 1)[0]
    return first


def _last_tool_result(messages: Sequence[Message]) -> str:
    """The most recent tool observation, or "" when no tool has been called."""
    return next((m.content for m in reversed(messages) if m.role == "tool"), "")


def _plan_call(question: str) -> CallToolAction | None:
    """Choose one read tool from the identifiers in the question.

    Deliberately shallow: it looks for an identifier and calls the tool that owns it. That is
    enough to exercise the transport, the policy, the recorder and the grader on every task,
    and it is not enough to look like reasoning --- which is the point, since a scripted model
    that appeared to solve the benchmark would invite somebody to quote its score.
    """
    account = _ACCOUNT_RE.search(question)
    if account is not None:
        return CallToolAction(
            thought="The question names an account, so I will read what it holds.",
            tool="account_holdings",
            arguments={"account_id": account.group()},
        )
    client = _CLIENT_RE.search(question)
    if client is not None:
        return CallToolAction(
            thought="The question names a client, so I will pull their record.",
            tool="client_lookup",
            arguments={"client_id": client.group()},
        )
    doc = _DOC_RE.search(question)
    if doc is not None:
        return CallToolAction(
            thought="The question names a policy document, so I will read it.",
            tool="policy_fetch",
            arguments={"doc_id": doc.group()},
        )
    schedule = _SCHEDULE_RE.search(question)
    if schedule is not None:
        return CallToolAction(
            thought="The question names a fee schedule, so I will fetch it.",
            tool="fee_schedule",
            arguments={"schedule_id": schedule.group()},
        )
    return _plan_price_call(question)


def _plan_price_call(question: str) -> CallToolAction | None:
    """A one-day price window, when the question names a ticker and a date."""
    when = _DATE_RE.search(question)
    if when is None:
        return None
    tickers = [t for t in _TICKER_RE.findall(question) if t not in _NOT_TICKERS]
    if not tickers:
        return None
    return CallToolAction(
        thought="The question names an instrument and a date, so I will price that day.",
        tool="price_history",
        arguments={"ticker": tickers[0], "start": when.group(), "end": when.group()},
    )


def _clip(text: str) -> str:
    """One line, short enough to survive the decoding limit."""
    flat = _WHITESPACE_RE.sub(" ", text).strip()
    return flat if len(flat) <= _ANSWER_CHARS else f"{flat[:_ANSWER_CHARS]} ..."


def _answer_from(result: str) -> Action:
    """Turn the last tool observation into an answer.

    A result that reports ``found: false``, or an error, becomes a refusal with no figure in it
    --- which is the behaviour the unanswerable family is asking for, and the cheapest way to
    prove the refusal matcher is reachable from a real run rather than only from a unit test.
    """
    compact = _WHITESPACE_RE.sub("", result)
    if not result:
        return ClarifyAction(
            thought="No tool has returned anything, so answering would be a guess.",
            question="Which client or account do you mean? Please give the identifier.",
        )
    if result.startswith("ERROR:") or '"found":false' in compact:
        return FinalAction(
            thought="The platform holds nothing for that identifier.",
            answer=(
                "The platform holds no record matching that request, so I cannot answer it "
                "from the tools available."
            ),
        )
    return FinalAction(
        thought="The tool returned the figures, so I can answer from them.",
        answer=f"From the tool result: {_clip(result)}",
    )


def _route(messages: Sequence[Message]) -> Action:
    """The supervisor's decision: delegate to whoever has not reported yet, then answer.

    Reports arrive as ``[role] ...`` user turns, so what has already happened is legible from
    the supervisor's own conversation. Nothing else needs to be remembered, which is what keeps
    the scripted model a pure function of the transcript and therefore reproducible under
    concurrency.
    """
    reports = [
        line
        for message in messages
        if message.role == "user"
        for line in message.content.splitlines()
        if line.startswith("[")
    ]
    said = "\n".join(reports)
    for target, instruction in _ROUTING:
        if f"[{target}]" not in said:
            return HandoffAction(
                thought=f"The {target} owns this part of the work.",
                to=target,
                instruction=instruction,
            )
    return FinalAction(
        thought="Every specialist has reported, so the answer is ready.",
        answer=_clip(reports[-1].split("] ", 1)[-1]),
    )


def _draft(messages: Sequence[Message]) -> Action:
    """The writer's turn: compose from the findings it was briefed with, and nothing else."""
    findings = [line[2:] for line in _first_user(messages).splitlines() if line.startswith("- ")]
    if not findings:
        return FinalAction(
            thought="Nothing has been established, so there is nothing to write up.",
            answer="The team established nothing on this question, so I cannot answer it.",
        )
    return FinalAction(
        thought="The findings carry the figures, so the answer restates them.",
        answer=_clip(" ".join(findings)),
    )


def _scripted_reply(messages: Sequence[Message]) -> str:
    """Play one turn of the action protocol, deterministically, from the transcript alone.

    Statelessness is the requirement. Attempts run concurrently against one model object, so a
    reply that depended on a queue would depend on the interleaving; a reply that depends only
    on the conversation it is handed is reproducible whatever order the event loop chooses.
    """
    system = next((m.content for m in messages if m.role == "system"), "")
    role_match = _ROLE_RE.search(system)
    role = role_match.group(1) if role_match is not None else SUPERVISOR
    delegating = '"action": "handoff"' in system and "researcher" in system

    if role == SUPERVISOR and delegating:
        return render_action(_route(messages))
    if role == WRITER:
        return render_action(_draft(messages))
    if role == "verifier":
        return render_action(
            FinalAction(
                thought="Every figure in the draft appears in the recorded results.",
                answer="The draft is supported by the tool results recorded for this run.",
            )
        )
    if not any(m.role == "assistant" for m in messages):
        planned = _plan_call(_question(messages))
        if planned is not None:
            return render_action(planned)
    return render_action(_answer_from(_last_tool_result(messages)))


def scripted_benchmark_model() -> ScriptedChatModel:
    """A deterministic stand-in for a language model, good enough to drive the whole benchmark.

    It plays the action protocol properly --- one JSON object per turn, a tool call before an
    answer, a handoff chain in the multi-agent arm --- so a run against it exercises the MCP
    transport, the permission policy, the recorder, the grader and the report end to end in
    seconds. That is what makes it usable as a CI gate.

    What it is not is a model. Its answers quote tool output back, so any figure it gets right
    it got by copying, and its score is a property of this function rather than of anything
    under test. Quote it as evidence that the harness works, never as a baseline.

    Returns:
        A fresh model. Cheap to build, so a caller may make one per run rather than share one.
    """
    return ScriptedChatModel(responder=_scripted_reply, model_name="scripted")


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


def _select(tasks: Sequence[Task], limit: int | None) -> tuple[Task, ...]:
    """The tasks this invocation will cover, in order.

    Truncation is from the front rather than sampled. A ``--limit`` run is for smoke-testing a
    real model before committing hours to it, and it must cover the same items every time or
    two limited runs cannot be compared with each other at all.
    """
    if limit is None:
        return tuple(tasks)
    if limit < 0:
        msg = f"limit must not be negative, got {limit}"
        raise ValueError(msg)
    return tuple(tasks[:limit])


def _reusable(
    paths: RunPaths | None,
    tasks: Sequence[Task],
    *,
    architecture: str,
    model: str,
) -> dict[str, Trajectory]:
    """Load the attempts an interrupted run already finished.

    Three filters, and each one exists because ignoring it would produce a report that quietly
    mixes two runs: the trajectory must belong to a task in this selection, it must come from
    this architecture, and it must come from this model. The last write for a task wins, so
    re-running a task into the same directory replaces its earlier attempt rather than
    duplicating it.
    """
    if paths is None or not paths.trajectories.exists():
        return {}
    wanted = {task.id for task in tasks}
    found: dict[str, Trajectory] = {}
    for stored in read_jsonl(paths.trajectories):
        if (
            stored.task_id in wanted
            and stored.architecture == architecture
            and stored.model == model
        ):
            found[stored.task_id] = stored
    return found


async def _attempt(
    task: Task,
    *,
    agent: _Agent,
    client: GuardedToolClient,
    model: ChatModel,
    max_steps: int,
) -> Trajectory:
    """Run one task, turning any failure into a recorded trajectory.

    The recorder is built before the agent is touched, so even a model that raises on its first
    turn produces a trajectory with the task id, the architecture and the model on it --- a row
    in the report saying "this one failed", rather than a gap the denominator has to be adjusted
    for. Everything raisable is caught for the same reason; only cancellation is allowed
    through, because a cancelled run is the caller's decision rather than the agent's failure.
    """
    try:
        return await agent.run(task.prompt, client, model, task_id=task.id, max_steps=max_steps)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return client.recorder.finish(stop_reason="error", error=f"{type(exc).__name__}: {exc}")


async def run_benchmark(
    tasks: Sequence[Task],
    *,
    architecture: str,
    model: ChatModel,
    world: World,
    log: WorldLog,
    policy: PermissionPolicy,
    max_steps: int = DEFAULT_MAX_STEPS,
    concurrency: int = DEFAULT_CONCURRENCY,
    out_dir: Path | None = None,
    resume: bool = False,
    limit: int | None = None,
    label: str | None = None,
    approver: Approver = deny_all,
    clock: Clock = time.perf_counter,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    world_seed: int | None = None,
    created_at: str | None = None,
) -> BenchmarkRun:
    """Run every task against one architecture and return the graded, summarised run.

    One server is built for the whole run and every attempt speaks to it over the in-process
    transport, but each attempt gets its own recorder, its own guarded client and its own fresh
    budget: the transport is shared because it is not what is being measured, and the tallies
    are not because they are.

    The harness never approves a write. That is not an oversight --- the constrained-action
    family grades the agent on *seeking* approval, and a harness that granted it would be
    scoring how obedient the approver is instead. Pass ``approver`` only to study what changes
    when the gate opens.

    Args:
        tasks: The benchmark task set, in order.
        architecture: ``"single"`` or ``"supervisor"``.
        model: The chat model both arms share.
        world: The world the MCP server answers from.
        log: The append-only destination for the two write tools.
        policy: The permission policy; its inventory is refreshed from the server's live
            advertisement before the run starts.
        max_steps: Model turns one attempt may take.
        concurrency: Attempts in flight at once.
        out_dir: Where to write the trajectory log, the grades, the aggregate, the Markdown and
            the manifest. Nothing is written when this is omitted.
        resume: Reuse the attempts already in ``out_dir`` and run only what is missing.
        limit: Attempt only the first ``limit`` tasks.
        label: The run's name in the report; defaults to ``architecture/model``.
        approver: Consulted before a call that needs a human.
        clock: Monotonic seconds source, injected so a test can make wall times reproducible.
        n_boot: Bootstrap resamples behind every interval in the aggregate.
        alpha: One minus the nominal coverage of those intervals.
        seed: Seed for every bootstrap in the aggregate.
        world_seed: Recorded in the manifest, so a report names the world it was run against.
        created_at: Recorded in the manifest. Omitted by default, which keeps the run's output
            byte-stable; pass a timestamp when the output is going somewhere other than git.

    Returns:
        The finished run: every trajectory, every grade, and the aggregate.

    Raises:
        ValueError: If the architecture is unknown, the selection is empty, or ``concurrency``
            or ``max_steps`` is below one. All four are caller mistakes that would otherwise
            surface as an empty report.
    """
    if architecture not in ARCHITECTURES:
        msg = f"unknown architecture {architecture!r}: expected one of {', '.join(ARCHITECTURES)}"
        raise ValueError(msg)
    if concurrency < 1:
        msg = f"concurrency must be at least 1, got {concurrency}"
        raise ValueError(msg)
    if max_steps < 1:
        msg = f"max_steps must be at least 1, got {max_steps}"
        raise ValueError(msg)

    selected = _select(tasks, limit)
    if not selected:
        msg = "there are no tasks to run: the task set is empty or --limit excluded everything"
        raise ValueError(msg)

    paths = None if out_dir is None else RunPaths(Path(out_dir))
    reused = (
        _reusable(paths, selected, architecture=architecture, model=model.name) if resume else {}
    )
    pending = [task for task in selected if task.id not in reused]

    if paths is not None:
        paths.root.mkdir(parents=True, exist_ok=True)
        if not resume:
            # A fresh run into a used directory must not inherit the previous run's attempts:
            # they would be indistinguishable from this run's in the log every later step reads.
            write_jsonl(paths.trajectories, [])

    effective = policy
    attempts: dict[str, Trajectory] = dict(reused)
    if pending:
        server = build_server(world, log)
        async with connect_in_process(server) as tools_client:
            specs = await tools_client.discover()
            effective = policy.with_tools(specs)
            agent = build_agent(architecture, specs, approver=approver)
            gate = asyncio.Semaphore(concurrency)

            async def one(task: Task) -> Trajectory:
                async with gate:
                    recorder = TrajectoryRecorder(
                        task_id=task.id,
                        architecture=architecture,
                        model=model.name,
                        clock=clock,
                    )
                    guarded = GuardedToolClient(
                        tools_client,
                        effective,
                        recorder,
                        budgets=shared_budgets(
                            effective,
                            budget=run_budget(effective, max_steps=max_steps),
                            roles=ALL_ROLES,
                        ),
                    )
                    finished = await _attempt(
                        task, agent=agent, client=guarded, model=model, max_steps=max_steps
                    )
                    if paths is not None:
                        # Written here rather than after the gather: an interrupted run keeps
                        # every attempt that finished, which is the whole point of resume.
                        append_jsonl(paths.trajectories, [finished])
                    return finished

            for finished in await asyncio.gather(*(one(task) for task in pending)):
                attempts[finished.task_id] = finished

    ordered = tuple(attempts[task.id] for task in selected)
    grades = tuple(graded(traj, task) for task, traj in zip(selected, ordered, strict=True))
    name = label or f"{architecture}/{model.name}"
    summary = aggregate(list(grades), label=name, n_boot=n_boot, alpha=alpha, seed=seed)
    manifest = Manifest(
        label=name,
        architecture=architecture,
        model=model.name,
        code_version=__version__,
        task_count=len(selected),
        task_set_digest=task_set_digest(selected),
        policy_digest=policy_digest(effective),
        write_tools=tuple(sorted(effective.write_tools)),
        approval_required=tuple(sorted(effective.approval_required)),
        max_steps=max_steps,
        concurrency=concurrency,
        n_boot=n_boot,
        alpha=alpha,
        bootstrap_seed=seed,
        world_seed=world_seed,
        limit=limit,
        resumed=tuple(task.id for task in selected if task.id in reused),
        created_at=created_at,
    )
    if paths is not None:
        write_grades(paths.grades, grades)
        paths.aggregate.write_text(render_json(summary), encoding="utf-8", newline="\n")
        paths.report.write_text(
            render_markdown(summary, generated_at=created_at), encoding="utf-8", newline="\n"
        )
        paths.manifest.write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    return BenchmarkRun(
        manifest=manifest,
        trajectories=ordered,
        grades=grades,
        aggregate=summary,
        out_dir=None if paths is None else str(paths.root),
    )


def load_run(out_dir: Path) -> tuple[Manifest, list[Grade]]:
    """Read back what a run wrote, for re-rendering or comparing.

    Grades rather than the aggregate, because a report re-rendered from the aggregate could
    only ever reproduce the aggregate. Re-deriving it from the graded rows is what makes the
    claim in this module's docstring --- that every number traces back --- checkable rather
    than asserted.

    Args:
        out_dir: A directory written by :func:`run_benchmark`.

    Returns:
        The manifest and the grades, in file order.

    Raises:
        FileNotFoundError: If either file is missing, naming the one that is.
    """
    paths = RunPaths(Path(out_dir))
    for path in (paths.manifest, paths.grades):
        if not path.exists():
            msg = f"{path} does not exist: is {out_dir} a run directory?"
            raise FileNotFoundError(msg)
    manifest = Manifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))
    return manifest, read_grades(paths.grades)


def rerender(manifest: Manifest, grades: Sequence[Grade]) -> str:
    """Re-render a stored run's Markdown from its grades and its manifest.

    The bootstrap settings come from the manifest rather than from defaults, so the output is
    byte-identical to the report the run wrote. A re-render that quietly used a different
    resample count would produce a second, slightly different set of intervals for the same
    run, and there would be no way to tell which one a slide had quoted.

    Args:
        manifest: The stored manifest.
        grades: The stored grades.

    Returns:
        The Markdown document.
    """
    summary = aggregate(
        list(grades),
        label=manifest.label,
        n_boot=manifest.n_boot,
        alpha=manifest.alpha,
        seed=manifest.bootstrap_seed,
    )
    return render_markdown(summary, generated_at=manifest.created_at)
