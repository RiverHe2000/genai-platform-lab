"""The typed blackboard the supervisor graph's nodes share.

LangGraph merges each node's returned partial dict into the graph state one channel at a
time, and a channel with no reducer is a *last-value* channel: the update replaces what was
there. That default is right for scalars --- the current role, the draft, the stop reason ---
and catastrophic for the accumulating ones.

Consider :data:`SupervisorState.findings` under the default. The researcher node returns
``{"findings": [<what it found>]}`` and the state holds one finding. The analyst node runs
next and returns ``{"findings": [<its own>]}``; the researcher's finding is now gone. Nothing
raises, because a shorter list is still a perfectly valid list of findings. The verifier then
checks the writer's draft against an evidence set missing most of the evidence, marks true
claims as unsupported, and sends the draft back --- and the run looks, from the outside,
exactly like a model that cannot cite its sources. The bug is invisible in the trajectory and
indistinguishable from the phenomenon the benchmark exists to measure. That is why every
accumulating channel here carries an explicit append reducer, and why the reducers are public,
named and tested rather than inline lambdas.

The state is also the reason the two architectures stay comparable. Everything that a run
spends or decides --- the step counter, the budget tally, the route history --- is on the
blackboard rather than hidden in a node's local variables, so a supervisor run can be
inspected, checkpointed and diffed against the single-agent control arm using the same
vocabulary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict

from mcpeval.client.policy import SUPERVISOR, BudgetState
from mcpeval.schemas import Message, StopReason

__all__ = [
    "Approver",
    "Finding",
    "SupervisorState",
    "add_steps",
    "append_findings",
    "append_messages",
    "append_routes",
    "deny_all",
    "new_state",
    "route_marker",
]

Approver = Callable[[str, str, Mapping[str, Any]], bool]
"""Decides whether a human has approved one proposed call: ``(role, tool, arguments)``.

A callable rather than a flag because approval is a fact about one call, not a mode a run is
left in. The benchmark passes a function that approves exactly the call a task authorises, so
an agent that reaches for a second write does not inherit the first one's permission.
"""


def deny_all(_role: str, _tool: str, _arguments: Mapping[str, Any]) -> bool:
    """The default approver: nothing is approved.

    Defaulting to denial means a harness that forgets to wire a human in produces refusals ---
    visible, recorded, gradeable --- rather than silent writes.
    """
    return False


class Finding(BaseModel):
    """One thing a specialist established, with the evidence it rests on.

    Findings are the only channel by which work crosses between specialists: a specialist
    reads the findings so far and writes at most one of its own. Keeping them separate from
    the transcript is what lets the writer be given a short, factual brief instead of four
    agents' worth of reasoning, and what lets the oscillation detector ask the one question
    that matters --- has anything new been learned since the last time we were here.

    Attributes:
        role: Which specialist recorded it.
        step: The run-level step counter when it was recorded.
        text: What the specialist reported, in its own words.
        evidence: Digests of the tool results the specialist saw while producing it, so a
            finding can be traced back to the recorded calls that support it.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    step: int
    text: str
    evidence: tuple[str, ...] = ()

    @property
    def cited(self) -> bool:
        """Whether any tool result backs this finding."""
        return bool(self.evidence)

    def render(self) -> str:
        """One line for a downstream agent's briefing."""
        return f"- {self.role}: {self.text}"


def _extend[T](left: Sequence[T] | None, right: Sequence[T] | None) -> list[T]:
    """Concatenate two possibly-absent sequences into a fresh list.

    A fresh list rather than an in-place extend because LangGraph may hand the same channel
    value to more than one node in a superstep, and mutating it would make the result depend
    on which node ran first.
    """
    return [*(left or ()), *(right or ())]


def append_messages(left: list[Message] | None, right: list[Message] | None) -> list[Message]:
    """Reducer for the supervisor's conversation: append, never replace."""
    return _extend(left, right)


def append_findings(left: list[Finding] | None, right: list[Finding] | None) -> list[Finding]:
    """Reducer for the shared findings: append, never replace.

    See the module docstring for what the replacing default costs.
    """
    return _extend(left, right)


def append_routes(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Reducer for the route history the oscillation detector reads."""
    return _extend(left, right)


def add_steps(left: int | None, right: int | None) -> int:
    """Reducer for the step counter: nodes contribute a *delta*, not a total.

    A delta makes "steps never decrease" structural rather than a convention every node has
    to remember, which matters because the step count is one of the two headline costs the
    benchmark compares between architectures.
    """
    return (left or 0) + (right or 0)


def route_marker(target: str, findings: int) -> str:
    """Identify one routing decision by where it went and what was known at the time.

    Findings only ever accumulate, so two markers are equal exactly when the supervisor sent
    work to the same specialist with the same evidence in hand. Counting equal markers is
    therefore a complete test for "we have been here before and learned nothing", which is
    the definition of the oscillation the supervisor must stop rather than pay for.

    Args:
        target: The specialist being routed to.
        findings: How many findings had been recorded when the decision was made.

    Returns:
        A short, stable, JSON-safe marker, e.g. ``"researcher@0"``.
    """
    return f"{target}@{findings}"


class SupervisorState(TypedDict):
    """The graph state: everything the supervisor and its specialists share.

    Two of these keys are deliberately redundant with objects the run already holds.
    ``budget`` is the very tally the guarded client spends, carried here so that a
    checkpointed state describes the run's remaining headroom without a side channel; and
    ``step`` mirrors the run's counter for the same reason. Both are read-only from a node's
    point of view --- nodes report a step delta and let the reducer do the arithmetic.

    Attributes:
        task_prompt: The user's question, verbatim, as every specialist is briefed with it.
        messages: The supervisor's own conversation. A specialist's *turns* are recorded on the
            trajectory but kept out of here; it contributes only a one-line report, so that
            delegating costs the supervisor a sentence rather than a sub-agent's context.
        findings: What the specialists have established, appended in order.
        routes: One :func:`route_marker` per routing decision, oldest first.
        role: Which node the supervisor last routed to; the router reads it.
        instruction: What the supervisor asked that role to do.
        draft: The writer's latest answer, the thing the verifier checks.
        revisions: How many times the verifier has sent the draft back. Capped at one, so a
            verifier and a writer that disagree cannot spend the whole budget arguing.
        budget: The live tally of steps, calls, tokens and wall time.
        step: Model turns taken so far by every role in the run.
        final_answer: The answer, once there is one.
        stop_reason: `None` while the run is live; set exactly once, by whichever node ends
            it. The router treats it as the only termination signal.
        error: Why the run failed, when it did.
    """

    task_prompt: str
    messages: Annotated[list[Message], append_messages]
    findings: Annotated[list[Finding], append_findings]
    routes: Annotated[list[str], append_routes]
    role: str
    instruction: str
    draft: str
    revisions: int
    budget: BudgetState
    step: Annotated[int, add_steps]
    final_answer: str | None
    stop_reason: StopReason | None
    error: str


def new_state(
    *,
    task_prompt: str,
    budget: BudgetState,
    messages: Sequence[Message] = (),
    role: str = SUPERVISOR,
) -> SupervisorState:
    """Build the state a run starts from, with every channel present.

    Every key is populated even where the value is empty. A `TypedDict` with missing keys
    forces every node into ``state.get(...)`` with a default, and one node choosing a
    different default from another is precisely the class of bug this module exists to
    prevent.

    Args:
        task_prompt: The user's question.
        budget: The tally the guarded client will spend; carried, not copied.
        messages: The supervisor's opening conversation, usually a system prompt and the
            task.
        role: The node to enter first.

    Returns:
        A fully populated state.
    """
    return SupervisorState(
        task_prompt=task_prompt,
        messages=list(messages),
        findings=[],
        routes=[],
        role=role,
        instruction="",
        draft="",
        revisions=0,
        budget=budget,
        step=0,
        final_answer=None,
        stop_reason=None,
        error="",
    )
