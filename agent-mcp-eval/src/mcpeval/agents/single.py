"""The single-agent control arm: one ReAct loop, one role, one context window.

This is the baseline the whole benchmark turns on. The headline comparison is between this
loop and the LangGraph supervisor in :mod:`mcpeval.agents.supervisor`, and a comparison is
only worth reporting if the two differ in *orchestration and nothing else*. So both
architectures run the same chat model, call the same MCP server over the same transport,
speak the same JSON action protocol from :mod:`mcpeval.agents.protocol`, are ruled on by the
same :class:`~mcpeval.client.policy.PermissionPolicy`, write to the same
:class:`~mcpeval.client.recorder.TrajectoryRecorder`, and --- this is the one that is easy to
get wrong --- charge their model turns and tokens to the *same* budget object. Give each
sub-agent its own budget and the supervisor architecture quietly acquires four times the
headroom, after which any result showing it is more thorough is a measurement of the wiring
rather than of the topology. Both classes therefore spend against the budget of a single
control role, and both expose the identical :meth:`SingleAgent.run` signature so the runner
cannot accidentally treat them differently.

What the control arm gives up is the point of the experiment. It has one role, so its scope
is the union of everything the task might need --- including, for a task that legitimately
ends in a write, the ability to reach a write tool. It has one context window, so every tool
result it has ever seen competes for space with the next decision. And it has no second
opinion: nothing checks its answer against what the tools actually returned. The supervisor
architecture buys separation of scope, a fresh context per specialist and a verifier, and
pays for them in steps and tokens. Whether that trade is worth making is the question the
benchmark answers, and it can only answer it against a control arm that is otherwise
identical.

Three behaviours here are not incidental and are measured rather than hidden:

* An unparseable reply is retried **once**, with the parse error quoted back, and then the
  run is abandoned with ``stop_reason="error"``. Retrying forever turns a model that cannot
  hold a format into a model that is merely slow, which is a much more flattering result than
  the truth.
* A call the agent has already made twice is not made a third time; the run stops with
  ``no_progress``. An agent stuck in a loop otherwise spends its whole budget rediscovering
  the same fact, and "ran out of budget" is a far less useful diagnosis than "looped".
* A refused call is fed back to the model as an observation, exactly like a successful one.
  The agent has to be able to notice it was refused and choose something else; hiding the
  refusal would measure the policy instead of the agent.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final

from mcpeval.agents.protocol import (
    CallToolAction,
    ClarifyAction,
    FinalAction,
    HandoffAction,
    parse_action,
    render_system_prompt,
)
from mcpeval.agents.state import Approver, deny_all
from mcpeval.client.policy import SUPERVISOR, BudgetState, PermissionPolicy
from mcpeval.client.session import GuardedToolClient
from mcpeval.schemas import ChatModel, Message, StopReason, ToolCallRecord, ToolSpec, Trajectory

__all__ = [
    "SINGLE_AGENT_GOAL",
    "SingleAgent",
    "call_signature",
    "observation",
    "repair_prompt",
    "tool_specs_for",
]

SINGLE_AGENT_GOAL: Final = (
    "answer the user's question about the wealth platform using the tools, and report only "
    "figures a tool has returned to you"
)

_UNDESCRIBED: Final = (
    "no description or argument schema was supplied to this agent; call it with the "
    "arguments the task implies"
)

_Outcome = tuple[StopReason, str | None, str | None]
"""How a turn ends: the stop reason, the answer if any, and the error if any."""


def tool_specs_for(client: GuardedToolClient, tools: Sequence[ToolSpec]) -> tuple[ToolSpec, ...]:
    """Decide which tools an agent's prompt advertises.

    Prefer what the caller passed, which is the server's own advertisement and carries the
    descriptions and JSON schemas the model needs to get its arguments right. Failing that,
    fall back to the names the policy already knows about, so a run started without the
    advertisement still produces a working agent rather than one that believes it has no
    tools --- a failure that looks, on the trajectory, exactly like a model refusing to act.

    The fallback is genuinely worse and is meant to be: a prompt listing names without schemas
    measurably raises the malformed-argument rate. Callers should pass
    :meth:`~mcpeval.client.session.ToolClient.discover`'s result.

    Whichever source is used, the approval flag is stamped on from the policy. The server does
    not get to say whether calling it needs a human --- that is a fact about the deployment ---
    and both architectures must state it identically, or one arm's model would be warned about
    a write that the other's was not.

    Args:
        client: The guarded client, consulted only for its policy's inventory.
        tools: The advertisement, if the caller has it.

    Returns:
        The specs to render into the system prompt, in a stable order.
    """
    policy: PermissionPolicy = client.policy
    if tools:
        return tuple(
            spec.model_copy(update={"requires_approval": policy.requires_approval(spec.name)})
            for spec in tools
        )
    return tuple(
        ToolSpec(
            name=name,
            description=_UNDESCRIBED,
            read_only=name not in policy.write_tools,
            destructive=name in policy.write_tools,
            requires_approval=policy.requires_approval(name),
        )
        for name in sorted(policy.known_tools)
    )


def call_signature(tool: str, arguments: Mapping[str, Any]) -> str:
    """Fingerprint a proposed call so two attempts at the same thing compare equal.

    The arguments are serialised with sorted keys, so a model that emits the same call with
    its fields in a different order --- which small models do constantly --- is correctly
    recognised as repeating itself rather than trying something new.

    Args:
        tool: The proposed tool name.
        arguments: The proposed arguments.

    Returns:
        A single string identifying the call.
    """
    return f"{tool}({json.dumps(arguments, sort_keys=True, default=str)})"


def observation(record: ToolCallRecord) -> str:
    """Render one attempted call back to the model as its observation.

    Failures and refusals are reported, not swallowed. An agent that is told nothing when the
    policy stops it will either repeat the call or answer from nothing; an agent told
    ``ERROR: scope.write_forbidden ...`` can choose a different route, and whether it does is
    exactly what the benchmark is measuring.

    Args:
        record: The stored record for the attempt.

    Returns:
        The text to append to the conversation as the tool's result.
    """
    if record.executed and record.ok:
        return record.result_text
    detail = record.error or record.result_text or "the call did not complete"
    return f"ERROR: {detail}"


def repair_prompt(error: str) -> str:
    """The nudge sent after a reply that did not yield a usable action.

    Public, and shared with the supervisor arm rather than duplicated there, because the two
    architectures have to be nudged in identical words. A model given a better-worded second
    chance in one arm would recover more often in that arm, and format compliance is one of the
    numbers the benchmark reports.

    Args:
        error: The parse or protocol failure, quoted back so the model can see what it broke.

    Returns:
        The user turn to append before asking again.
    """
    return (
        f"Your last reply could not be parsed as an action: {error}. Reply with exactly one "
        "JSON object and nothing else, in one of the shapes listed above."
    )


@dataclass(slots=True)
class SingleAgent:
    """A ReAct loop over the MCP tools: model, action, policy, tool, observation, repeat.

    Attributes:
        tools: The server's advertisement, as returned by
            :meth:`~mcpeval.client.session.ToolClient.discover`. Empty falls back to the
            policy's inventory; see :func:`tool_specs_for`.
        role: The policy role every call is made under. The control arm runs as the
            supervisor because that is the only role with the union scope a single agent
            needs --- narrowing it would hand the multi-agent arm a capability advantage and
            make the comparison meaningless.
        approver: Consulted before a call that needs a human. Denies everything by default.
        max_tokens: Decoding limit per turn.
        temperature: Decoding temperature; zero for a reproducible benchmark.
        parse_retries: How many times an unparseable reply is nudged before the run is
            abandoned.
        repeat_limit: How many times the same call may be *proposed* before the run stops
            with ``no_progress``. The last proposal is not executed.
    """

    tools: Sequence[ToolSpec] = ()
    role: str = SUPERVISOR
    approver: Approver = deny_all
    max_tokens: int = 512
    temperature: float = 0.0
    parse_retries: int = 1
    repeat_limit: int = 3

    architecture: ClassVar[str] = "single"
    """The value a recorder should be constructed with for a run of this class."""

    async def run(
        self,
        task_prompt: str,
        client: GuardedToolClient,
        model: ChatModel,
        *,
        task_id: str,
        max_steps: int,
    ) -> Trajectory:
        """Attempt one task and return the trajectory it produced.

        The trajectory is built from ``client.recorder``, not from a recorder of this
        agent's own, because the guarded client writes every attempted call there as it
        happens. An agent holding a second recorder would return a trajectory with a final
        answer and no evidence behind it, which is the one thing this benchmark must never
        produce.

        Args:
            task_prompt: The user's question.
            client: The guarded client; its policy, budgets and recorder are all in force.
            model: The chat model, called once per step.
            task_id: The task being attempted, checked against the recorder's.
            max_steps: Ceiling on model turns, applied alongside the budget's own.

        Returns:
            The finished trajectory, whatever the outcome. Model and transport failures are
            recorded as stop reasons rather than raised.

        Raises:
            ValueError: If ``task_id`` disagrees with the recorder's. That means the harness
                reused a client across tasks, which would mix one task's tool calls into
                another's evidence --- a corruption worth stopping the run for, unlike
                anything the agent itself can do.
        """
        recorder = client.recorder
        if recorder.task_id != task_id:
            msg = (
                f"the recorder is recording task {recorder.task_id!r} but this run is for "
                f"{task_id!r}; a recorder belongs to exactly one attempt"
            )
            raise ValueError(msg)

        specs = tool_specs_for(client, self.tools)
        budget = client.budget_for(self.role)
        messages: list[Message] = [
            recorder.say(
                "system",
                render_system_prompt(
                    specs,
                    role=self.role,
                    goal=SINGLE_AGENT_GOAL,
                    max_steps=max_steps,
                    allow_clarify=True,
                ),
            ),
            recorder.say("user", task_prompt),
        ]

        loop = _Loop(agent=self, client=client, model=model, messages=messages)
        stop, answer, error = await loop.run(max_steps=max_steps, budget=budget)
        return recorder.finish(final_answer=answer, stop_reason=stop, error=error)


@dataclass(slots=True)
class _Loop:
    """The mutable half of one run, kept off :class:`SingleAgent` so it stays reusable."""

    agent: SingleAgent
    client: GuardedToolClient
    model: ChatModel
    messages: list[Message]
    seen: Counter[str] = field(default_factory=Counter)
    retries: int = 0
    step: int = 0

    async def run(self, *, max_steps: int, budget: BudgetState) -> _Outcome:
        """Drive the loop to a stopping condition.

        The budget is read before every turn rather than after, so a run that starts with no
        headroom stops without calling the model at all. Charging a model call to a budget
        that was already spent would let every run overrun by exactly one turn.

        Args:
            max_steps: Ceiling on model turns.
            budget: The live tally.

        Returns:
            The stop reason, the final answer if there is one, and the error if there is one.
        """
        while self.step < max_steps:
            if budget.exhausted:
                return "budget", None, None
            self.step += 1
            outcome = await self._turn()
            if outcome is not None:
                return outcome
        return "max_steps", None, None

    async def _turn(self) -> _Outcome | None:
        """Take one model turn. Returns the run's outcome, or `None` to keep going.

        The model call is guarded because `run` promises that model failures come back as a
        stop reason rather than an exception, and it did not keep that promise: an out-of-
        memory error or a broken endpoint propagated out of the loop and took the whole
        trajectory with it, discarding every tool call the recorder had accumulated. The
        benchmark runner has its own catch-all, so this was invisible there; anyone using the
        documented API directly --- a notebook, a script, the single-task path --- got the
        exception and lost the evidence.
        """
        recorder = self.client.recorder
        try:
            completion = self.model.complete(
                self.messages,
                max_tokens=self.agent.max_tokens,
                temperature=self.agent.temperature,
            )
        except Exception as exc:
            return "error", None, f"{type(exc).__name__}: {exc}"
        recorder.add_usage(completion.usage)
        self.client.spend_step(self.agent.role)
        self.client.spend_tokens(self.agent.role, completion.usage.total_tokens)
        self.messages.append(recorder.say("assistant", completion.text))

        parsed = parse_action(completion.text)
        if parsed.action is None:
            return self._nudge(parsed.error)
        action = parsed.action
        if isinstance(action, HandoffAction):
            # Checked before the counter is cleared. A handoff parses perfectly and is still
            # unusable here, so clearing first would give a model that delegates every turn an
            # unlimited supply of retries and turn a protocol failure into a step-ceiling one.
            return self._nudge(f"there is no agent named {action.to!r} on this task")
        self.retries = 0
        if isinstance(action, FinalAction):
            return "answered", action.answer, None
        if isinstance(action, ClarifyAction):
            # A question back to the user is a legitimate ending: the ambiguous family of the
            # benchmark scores asking above guessing, and the grader has a matcher for it.
            return "answered", action.question, None
        return await self._call(action)

    def _nudge(self, error: str) -> _Outcome | None:
        """Quote a protocol failure back to the model, or give up if it has had its chance."""
        self.retries += 1
        if self.retries > self.agent.parse_retries:
            # ``protocol``, not ``error``: the model answered every time and none of its
            # answers were a valid action. Nothing on the platform failed, and a report
            # that says otherwise sends a reader to the wrong file.
            return "protocol", None, error
        self.messages.append(self.client.recorder.say("user", repair_prompt(error)))
        return None

    async def _call(self, action: CallToolAction) -> _Outcome | None:
        """Put one proposed call through the policy and feed the outcome back."""
        signature = call_signature(action.tool, action.arguments)
        self.seen[signature] += 1
        if self.seen[signature] >= self.agent.repeat_limit:
            # Stop before the call, not after: executing it would add a third identical
            # record to the trajectory and teach the grader nothing it does not already know.
            return "no_progress", None, None
        record = await self.client.call(
            self.agent.role,
            action.tool,
            action.arguments,
            step=self.step,
            approved=self.agent.approver(self.agent.role, action.tool, action.arguments),
        )
        self.messages.append(
            self.client.recorder.say("tool", observation(record), name=action.tool)
        )
        return None
