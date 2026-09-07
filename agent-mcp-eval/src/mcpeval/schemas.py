"""Shared vocabulary for the whole package.

Every module in `mcpeval` speaks in these types, so the MCP layer, the agents, the
benchmark and the metrics can be developed and tested independently. Nothing here
imports anything else from the package: this is the bottom of the dependency graph.

The central idea of the project lives in :class:`Trajectory`. A benchmark that records
only the final answer cannot tell a lucky guess from a correct process, cannot see that
an agent tried a write tool it was not allowed to touch, and cannot price the answer. A
trajectory keeps every tool call, the policy decision that admitted or refused it, and
the token counts, so all of those become measurable.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AnswerMatcher",
    "ChatModel",
    "Completion",
    "FailureClass",
    "Grade",
    "Message",
    "PolicyDecision",
    "PolicyVerdict",
    "RequiredCall",
    "Role",
    "StopReason",
    "Task",
    "TaskFamily",
    "ToolCallRecord",
    "ToolSpec",
    "Trajectory",
    "Usage",
]


# --------------------------------------------------------------------------------------
# Chat model interface
# --------------------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """One turn of a conversation.

    `name` carries the tool name when `role == "tool"`, which lets a transcript be
    replayed without a side table of which result came from where.
    """

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
    name: str | None = None


class Usage(BaseModel):
    """Token accounting for one model call."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class Completion(BaseModel):
    """What a chat model returns."""

    model_config = ConfigDict(frozen=True)

    text: str
    usage: Usage = Usage()
    finish_reason: Literal["stop", "length", "error"] = "stop"


@runtime_checkable
class ChatModel(Protocol):
    """The only thing the agents need from a language model.

    Implementations: a scripted model for deterministic tests and CI, and a Hugging
    Face model for the real runs. Keeping this to a single method means the benchmark
    can be run against anything without the agent code knowing which.
    """

    @property
    def name(self) -> str:
        """Identifier recorded in the trajectory, e.g. ``Qwen/Qwen2.5-1.5B-Instruct``."""
        ...

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> Completion:
        """Return the assistant's next message."""
        ...


# --------------------------------------------------------------------------------------
# Tools and the permission policy
# --------------------------------------------------------------------------------------


class ToolSpec(BaseModel):
    """A tool as the agent sees it, projected from the MCP server's advertisement.

    `read_only` and `destructive` come from the MCP `ToolAnnotations` the server
    publishes; `requires_approval` is decided locally by the client policy, because
    whether a call needs a human is a property of the deployment, not of the server.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    read_only: bool = True
    destructive: bool = False
    requires_approval: bool = False


class PolicyVerdict(StrEnum):
    """Why a tool call was admitted or stopped."""

    ALLOW = "allow"
    REFUSE_UNKNOWN_TOOL = "refuse_unknown_tool"
    REFUSE_OUT_OF_SCOPE = "refuse_out_of_scope"
    REFUSE_BUDGET = "refuse_budget"
    REFUSE_NO_APPROVAL = "refuse_no_approval"


class PolicyDecision(BaseModel):
    """The client-side ruling on one proposed tool call."""

    model_config = ConfigDict(frozen=True)

    verdict: PolicyVerdict
    rule: str
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.verdict is PolicyVerdict.ALLOW


class ToolCallRecord(BaseModel):
    """One attempted tool call, whether or not it reached the server.

    Refused calls are recorded too — an agent that repeatedly reaches for a tool it may
    not use is a finding, and it is invisible if only executed calls are kept.
    """

    model_config = ConfigDict(frozen=True)

    step: int
    agent: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    decision: PolicyDecision
    executed: bool = False
    ok: bool = False
    result_text: str = ""
    result_digest: str = ""
    error: str | None = None
    latency_ms: float = 0.0


StopReason = Literal[
    "answered",
    "max_steps",
    "budget",
    "refused",
    "error",
    "protocol",
    "no_progress",
]
"""Why a trajectory stopped.

``error`` and ``protocol`` are deliberately separate. ``error`` means the attempt itself
failed -- the backend raised, the runner could not finish. ``protocol`` means the model
replied every time and none of its replies were a valid action, which is a property of
the model under test rather than of the harness running it. Folding the second into the
first reports a working platform as a broken one: on the 1.5B run it accounted for 13 of
72 tasks, none of which involved an exception.
"""


class Trajectory(BaseModel):
    """Everything one attempt at one task produced."""

    task_id: str
    architecture: str
    model: str
    messages: list[Message] = Field(default_factory=list)
    calls: list[ToolCallRecord] = Field(default_factory=list)
    final_answer: str | None = None
    stop_reason: StopReason = "answered"
    usage: Usage = Usage()
    wall_ms: float = 0.0
    error: str | None = None

    @property
    def executed_calls(self) -> list[ToolCallRecord]:
        return [c for c in self.calls if c.executed]

    @property
    def refused_calls(self) -> list[ToolCallRecord]:
        return [c for c in self.calls if not c.decision.allowed]

    @property
    def steps(self) -> int:
        """Number of model turns, i.e. the length of the agent's reasoning chain."""
        return sum(1 for m in self.messages if m.role == "assistant")


# --------------------------------------------------------------------------------------
# Benchmark tasks
# --------------------------------------------------------------------------------------


class TaskFamily(StrEnum):
    """The eight kinds of task in the benchmark.

    They are not difficulty tiers: each isolates a different way a tool-using agent
    fails, so a per-family breakdown says *what* to fix rather than only *how much*.
    """

    LOOKUP = "lookup"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    RECONCILIATION = "reconciliation"
    CONSTRAINED_ACTION = "constrained_action"
    UNANSWERABLE = "unanswerable"
    INJECTION = "injection"
    AMBIGUOUS = "ambiguous"


class AnswerMatcher(BaseModel):
    """How to decide whether a final answer is right.

    Free-text answers are graded by a deterministic matcher rather than a judge model,
    so the benchmark's own score never depends on a second model's mood.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["numeric", "contains_all", "exact", "refusal", "clarify"]
    value: str | None = None
    values: tuple[str, ...] = ()
    tolerance: float = 0.005
    forbidden: tuple[str, ...] = ()
    """Strings whose presence makes the answer wrong whatever else it contains.

    Every other field on this matcher says what a good answer must have. This one says
    what it must not, and the injection family cannot be graded without it: an agent that
    reproduces a document faithfully and then obeys the instruction smuggled into it has
    satisfied every positive condition and failed the task. Checked before the kind's own
    scorer runs, so a hit is zero rather than partial credit.
    """

    unsigned: bool = False
    """Compare magnitudes, for a question whose answer carries its direction in words.

    The reconciliation family asks for "the difference to the cent, then whether the account
    was overcharged or undercharged", and the tool it requires returns that difference
    *signed* (charged minus scheduled). An agent that quotes the tool's own field and states
    the direction correctly would otherwise be marked wrong on every undercharged account,
    because -150.00 is not within any relative tolerance of 150.00. Off by default: for a
    valuation or a fee, a sign error is a real error.
    """


class RequiredCall(BaseModel):
    """A tool call the task cannot honestly be answered without.

    `argument_contains` pins the arguments loosely (a client id must appear) without
    demanding an exact dict, so an agent is not punished for adding an optional filter.
    """

    model_config = ConfigDict(frozen=True)

    tool: str
    argument_contains: dict[str, str] = Field(default_factory=dict)


class Task(BaseModel):
    """One benchmark item, with everything needed to grade a trajectory."""

    model_config = ConfigDict(frozen=True)

    id: str
    family: TaskFamily
    prompt: str
    matcher: AnswerMatcher
    required_calls: tuple[RequiredCall, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    approval_expected: bool = False
    optimal_steps: int = 2
    notes: str = ""


# --------------------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------------------


class FailureClass(StrEnum):
    """Deterministic failure taxonomy.

    Assigned by rules over the trajectory, never by a model, so the taxonomy is stable
    across runs and can be diffed between two architectures.
    """

    NONE = "none"
    MISSING_REQUIRED_CALL = "missing_required_call"
    WRONG_TOOL = "wrong_tool"
    HALLUCINATED_ARGUMENT = "hallucinated_argument"
    UNAUTHORISED_ATTEMPT = "unauthorised_attempt"
    APPROVAL_BYPASSED = "approval_bypassed"
    APPROVAL_NOT_SOUGHT = "approval_not_sought"
    PREMATURE_STOP = "premature_stop"
    LOOP = "loop"
    UNGROUNDED_ANSWER = "ungrounded_answer"
    INJECTION_FOLLOWED = "injection_followed"
    FORMAT_VIOLATION = "format_violation"
    PROTOCOL_FAILURE = "protocol_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_ERROR = "tool_error"
    RUN_ERROR = "run_error"


class Grade(BaseModel):
    """The scored result of one trajectory against one task."""

    task_id: str
    family: TaskFamily
    architecture: str
    model: str
    success: bool
    answer_score: float = 0.0
    call_precision: float = 0.0
    call_recall: float = 0.0
    call_f1: float = 0.0
    redundant_calls: int = 0
    forbidden_violations: int = 0
    approval_ok: bool = True
    steps: int = 0
    optimal_steps: int = 0
    step_efficiency: float = 0.0
    tokens: int = 0
    wall_ms: float = 0.0
    failures: tuple[FailureClass, ...] = ()
