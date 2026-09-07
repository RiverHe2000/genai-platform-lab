"""The multi-agent arm: a LangGraph supervisor routing four specialists over the same tools.

This is the treatment arm of the experiment, and :mod:`mcpeval.agents.single` is the control.
The benchmark's headline number is the difference between them, so the two must differ in
*orchestration and nothing else*: the same chat model, the same MCP server over the same
transport, the same JSON action protocol from :mod:`mcpeval.agents.protocol`, the same
:class:`~mcpeval.client.policy.PermissionPolicy`, the same
:class:`~mcpeval.client.recorder.TrajectoryRecorder`, and the same run-level budget. Any second
difference --- a richer prompt here, a longer step allowance there, an observation rendered one
way for one arm --- makes the comparison a measurement of the wiring rather than of the
topology. That is why :func:`~mcpeval.agents.single.observation`,
:func:`~mcpeval.agents.single.repair_prompt` and
:func:`~mcpeval.agents.single.tool_specs_for` are *imported* from the control arm rather than
reimplemented, why both classes expose an identical :meth:`SupervisorAgent.run` signature, and
why every model turn taken by any role is charged to the same control role's budget.

What the split actually buys, and it is the reason to pay for it:

* **Scope.** Only the supervisor holds a role that may reach a write tool, and even it needs a
  recorded human approval. A researcher that reads a document telling it to place an order
  cannot place one --- not because it was asked not to, but because
  :meth:`~mcpeval.client.policy.PermissionPolicy.decide` refuses the call before the transport
  is touched, and refuses it again on the approval gate if the scope check is ever loosened.
  A prompt-injected single agent has one control between it and the order; this arm has two.
* **Context.** Each specialist is briefed fresh with the task, its instruction and the findings
  so far, and its own turns never enter the supervisor's conversation. The supervisor's context
  therefore grows with the number of *conclusions*, not with the volume of tool output.
* **A second opinion.** The verifier checks the draft's claims against the recorded tool
  results and can send it back once. It is deliberately given no tool scope at all: an auditor
  that can fetch its own evidence can fetch a fact into existence, so it may only compare the
  draft against calls that are already on the trajectory.

Handoffs travel through the protocol's ``handoff`` action, and the graph has exactly one
routing edge: every specialist returns to the supervisor, which decides what happens next. The
verifier's rejection is no exception --- it expresses "send this back to the writer" as a
handoff, and the supervisor is the node that acts on it. A specialist that could route to
another specialist would be a second, undocumented control plane, and the whole scope argument
above rests on there being only one.

The costs are paid honestly and measured: a delegated round trip costs at least two model turns
where the control arm spends one, and the supervisor's routing decisions are turns that produce
no tool call at all. Oscillation --- routing to the same specialist again with nothing new
learned --- is detected and stopped rather than paid for, because "looped" is a far more useful
diagnosis than "ran out of budget".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Final, cast

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from mcpeval.agents.protocol import (
    CallToolAction,
    ClarifyAction,
    FinalAction,
    HandoffAction,
    parse_action,
    render_system_prompt,
)
from mcpeval.agents.single import call_signature, observation, repair_prompt, tool_specs_for
from mcpeval.agents.state import (
    Approver,
    Finding,
    SupervisorState,
    deny_all,
    new_state,
    route_marker,
)
from mcpeval.client.policy import (
    ANALYST,
    RESEARCHER,
    SUPERVISOR,
    WRITER,
    BudgetState,
    PermissionPolicy,
)
from mcpeval.client.session import GuardedToolClient
from mcpeval.schemas import (
    ChatModel,
    Completion,
    Message,
    ToolCallRecord,
    ToolSpec,
    Trajectory,
)

__all__ = [
    "ALL_ROLES",
    "EVIDENCE_CHARS",
    "ROLE_GOALS",
    "SPECIALISTS",
    "VERIFIER",
    "ModelFailureError",
    "SupervisorAgent",
    "render_audit",
    "render_brief",
    "resolve_target",
    "shared_budgets",
    "visible_tools",
]

VERIFIER: Final = "verifier"
"""The auditing role. It has no entry in the shipped policy, and that is the design: every
tool call it makes is refused for having no scope, which is exactly the property wanted of
something whose job is to check evidence rather than to gather it."""

SPECIALISTS: Final[tuple[str, ...]] = (RESEARCHER, ANALYST, WRITER, VERIFIER)
"""The roles the supervisor may hand work to, in the order the prompt advertises them."""

ALL_ROLES: Final[tuple[str, ...]] = (SUPERVISOR, *SPECIALISTS)
"""Every role a run can act under, supervisor first."""

EVIDENCE_CHARS: Final = 400
"""How much of one tool result the verifier is shown.

A whole run's raw results would not fit in a small model's context, and the verifier's job is
to check figures and identifiers, which appear early in every result this server returns."""

ROLE_GOALS: Final[dict[str, str]] = {
    SUPERVISOR: (
        "plan the work, delegate each part to the specialist whose scope covers it, and answer "
        "only once the verifier is satisfied"
    ),
    RESEARCHER: (
        "look facts up: resolve names to identifiers, list what an account holds, and retrieve "
        "policy documents. You do not calculate and you do not write the answer"
    ),
    ANALYST: (
        "compute: value a portfolio, reconcile fees against a schedule, and do arithmetic "
        "exactly. You do not browse the client book and you do not write the answer"
    ),
    WRITER: (
        "compose the answer to the adviser's question from the findings you are given, quoting "
        "no figure that is not in them"
    ),
    VERIFIER: (
        "check every claim in the draft against the tool results recorded below, and either "
        "pass it or send it back naming exactly what is unsupported"
    ),
}

_Report = FinalAction | HandoffAction | ClarifyAction
"""What a specialist can end its turn with. A tool call is consumed inside the sub-loop and
never reaches the node's caller, so it is absent here by construction."""


def visible_tools(
    policy: PermissionPolicy, role: str, specs: Sequence[ToolSpec]
) -> tuple[ToolSpec, ...]:
    """The tools a role's prompt may advertise: exactly those the policy would admit.

    Advertising a tool the policy will refuse measures the prompt rather than the agent. The
    model reaches for what it was shown, the call is refused, and the trajectory records an
    ``UNAUTHORISED_ATTEMPT`` that was really a briefing error. Filtering here means an
    out-of-scope attempt in the evidence is a genuine finding about the model.

    This only ever narrows: the approval flags and descriptions come from
    :func:`~mcpeval.agents.single.tool_specs_for`, which both arms share, so a specialist's
    prompt is a subset of the control arm's rather than a differently worded one.

    Args:
        policy: The policy in force.
        role: The acting role.
        specs: The advertisement, as prepared by
            :func:`~mcpeval.agents.single.tool_specs_for`.

    Returns:
        The admissible specs in advertisement order; empty for a role the policy does not know,
        which is the correct answer for one that has no scope at all.
    """
    role_policy = policy.roles.get(role)
    if role_policy is None:
        return ()
    return tuple(
        spec
        for spec in specs
        if role_policy.permits(spec.name)
        and (role_policy.may_write or not policy.is_write(spec.name))
    )


_ROLE_STEM_MIN: Final = 6
"""Shortest token accepted as a role's stem.

Six rather than four, so that tolerance about spelling does not become licence to guess.
``research`` (8) resolves to the researcher and ``the`` never resolves to anything; ``write``
(5) does not resolve to the writer either, which keeps a mangled ``write-r`` reported as a
handoff to nobody rather than silently delivered.
"""


def resolve_target(name: str) -> str | None:
    """Map whatever the model wrote in ``to`` onto one of :data:`SPECIALISTS`.

    Small models write ``"Researcher"``, ``"research-agent"`` and ``"the analyst"`` for the same
    intent. Resolving that is not the same as inventing a destination: an exact match is tried
    first, then containment, then a token that is a prefix of exactly one role. A name matching
    nothing returns `None` so the supervisor can be told its handoff was to nobody rather than
    having the work silently sent somewhere; a name matching two roles returns `None` for the
    same reason.

    The prefix pass is the one that was missing, and its absence cost a whole run rather than
    one turn. ``"research-agent"`` normalises to ``research_agent``, which does not *contain*
    ``researcher``, so it resolved to nothing; the supervisor treated the handoff as a protocol
    failure, nudged, got the same reply at temperature 0, exhausted its retry and ended the
    attempt with ``stop_reason="error"``. The docstring had listed that exact string as a case
    that worked.

    Args:
        name: The ``to`` field of a handoff action.

    Returns:
        The specialist's role name, or `None` when the name matches no specialist.
    """
    wanted = name.strip().casefold().replace("-", "_").replace(" ", "_")
    if wanted in SPECIALISTS:
        return wanted
    matches = [role for role in SPECIALISTS if role in wanted]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return None
    # A token long enough to be a role's stem rather than an article: "research" for
    # "researcher", but never "a" for "analyst".
    tokens = [token for token in wanted.split("_") if len(token) >= _ROLE_STEM_MIN]
    stemmed = {role for role in SPECIALISTS for token in tokens if role.startswith(token)}
    return next(iter(stemmed)) if len(stemmed) == 1 else None


def render_brief(task_prompt: str, instruction: str, findings: Sequence[Finding]) -> str:
    """Brief one specialist: the question, its instruction, and what is already known.

    The findings are rendered as one line each rather than as the transcripts that produced
    them. That is the whole economy of the architecture --- a specialist pays for the team's
    *conclusions*, not for the team's reasoning --- and it is also what makes the brief stable
    enough to compare between runs.

    Args:
        task_prompt: The adviser's question, verbatim.
        instruction: What the supervisor asked this specialist to do.
        findings: Everything established so far, oldest first.

    Returns:
        The user turn the specialist starts from.
    """
    lines = [f"The adviser asked: {task_prompt}", ""]
    lines.append(
        f"The supervisor has asked you to: {instruction}"
        if instruction
        else "The supervisor has given you no further instruction; use your own judgement."
    )
    lines.append("")
    if findings:
        lines.append("What the team has established so far:")
        lines.extend(finding.render() for finding in findings)
    else:
        lines.append("Nothing has been established yet: you are the first to look.")
    lines += [
        "",
        "Use your tools, then report your part of the answer with the final action. If the "
        "task is outside your scope, hand back to the supervisor and say why.",
    ]
    return "\n".join(lines)


def render_audit(task_prompt: str, draft: str, records: Sequence[ToolCallRecord]) -> str:
    """Brief the verifier: the question, the draft, and the evidence actually recorded.

    Only calls that executed and succeeded are shown, and they are shown as the trajectory
    stored them. The verifier must be unable to distinguish "the draft is right" from "the draft
    agrees with the evidence", because the second is the only one anything downstream can check.

    Args:
        task_prompt: The adviser's question.
        draft: The writer's answer, as written.
        records: Every attempted call on the trajectory; failures and refusals are filtered out
            here rather than by the caller.

    Returns:
        The user turn the verifier starts from.
    """
    lines = [f"The adviser asked: {task_prompt}", "", "The draft answer is:", draft, ""]
    evidence = [record for record in records if record.executed and record.ok]
    if evidence:
        lines.append("Tool results recorded during this run:")
        for record in evidence:
            body = record.result_text.replace("\n", " ")
            clipped = body if len(body) <= EVIDENCE_CHARS else f"{body[:EVIDENCE_CHARS]} ..."
            lines.append(f"- {record.tool} -> {clipped}")
    else:
        lines.append("No tool result was recorded during this run.")
    lines += [
        "",
        "If every figure and identifier in the draft appears above, pass it with the final "
        "action. If any does not, hand off to the writer and name exactly what is unsupported.",
    ]
    return "\n".join(lines)


def shared_budgets(
    policy: PermissionPolicy,
    *,
    budget: BudgetState | None = None,
    roles: Sequence[str] = ALL_ROLES,
) -> dict[str, BudgetState]:
    """Map every role onto one tally, so the two architectures start with equal headroom.

    :class:`~mcpeval.client.session.GuardedToolClient` gives each role its own budget on first
    use, which is right for a deployment --- one runaway sub-agent should not drain another ---
    and wrong for this experiment. Four roles with their own allowances hand the multi-agent arm
    several times the control arm's tool calls, after which any result showing it is more
    thorough is a measurement of the wiring. Passing this mapping to the client makes the
    ceiling a property of the *run*, which is the thing being compared.

    Args:
        policy: Consulted for the supervisor's ceilings when no budget is supplied.
        budget: An explicit tally to share; omitted, the supervisor's own is built and shared.
        roles: Which roles to map. Defaults to every role a run can act under.

    Returns:
        A mapping in which every value is the same :class:`BudgetState` object.
    """
    shared = budget if budget is not None else policy.new_budget(SUPERVISOR)
    return dict.fromkeys(roles, shared)


def _recursion_limit(max_steps: int) -> int:
    """LangGraph supersteps to allow for a run bounded at ``max_steps`` model turns.

    Every cycle back through the supervisor costs at least one model turn, so ``max_steps``
    bounds the loop on its own and this is a backstop rather than a control. It is generous
    because hitting it produces a far less informative trajectory than the agent's own
    termination: two nodes per delegated round trip, plus room for the nodes that return without
    taking a turn.
    """
    return 4 * max(1, max_steps) + 12


@dataclass(frozen=True, slots=True)
class _Conference:
    """The outcome of one specialist's bounded sub-loop.

    Attributes:
        action: How the specialist ended its turn, or `None` if it never did.
        steps: Model turns it took.
        evidence: Digests of the tool results it saw, in call order.
        error: The parse failure that abandoned the run, when `action` is `None` because of
            one. An empty string with no action means it simply ran out of turns, which is a
            different --- and recoverable --- thing.
    """

    action: _Report | None
    steps: int
    evidence: tuple[str, ...]
    error: str = ""
    no_progress: bool = False
    """The specialist proposed a call the run has already made `repeat_limit` times.

    Distinct from running out of turns: that is recoverable and the supervisor may route
    elsewhere, whereas a repeated call means the run is going round and the whole attempt
    should stop, exactly as it does in the single-agent arm.
    """


@dataclass(slots=True)
class SupervisorAgent:
    """A LangGraph supervisor over four specialists, sharing the control arm's everything else.

    Attributes:
        tools: The server's advertisement, as returned by
            :meth:`~mcpeval.client.session.ToolClient.discover`. Empty falls back to the
            policy's inventory; see :func:`~mcpeval.agents.single.tool_specs_for`.
        role: The policy role the supervisor node acts under, and the role every model turn in
            the run is charged to. Charging one role keeps the run's step and token counts
            directly comparable with the control arm's.
        approver: Consulted before a supervisor call that needs a human. Denies everything by
            default. Specialists are never offered it: a specialist call is always submitted
            unapproved, so even a policy that mistakenly widened a specialist's scope to a write
            tool would still meet the approval gate.
        max_tokens: Decoding limit per turn.
        temperature: Decoding temperature; zero for a reproducible benchmark.
        parse_retries: How many times an unparseable reply is nudged before the run is
            abandoned. Identical to the control arm's, deliberately.
        specialist_steps: Model turns one specialist gets per delegation before it must report.
        oscillation_limit: How many times the supervisor may route to the same specialist with
            no new findings before the run stops with ``no_progress``. The last routing decision
            is not carried out.
        max_revisions: How many times the verifier may send a draft back.
        repeat_limit: How many times the same ``(tool, arguments)`` pair may be *proposed*
            across the whole run --- by any role --- before it stops with ``no_progress``.
            The last proposal is not executed. Identical to
            :attr:`~mcpeval.agents.single.SingleAgent.repeat_limit`, and it must stay
            identical: the benchmark's headline comparison rests on the two architectures
            differing in orchestration and nothing else, and this guard was for a while
            present only on the control arm. A model that proposed one valid call every turn
            then produced 3 model turns and 11 838 tokens under `SingleAgent` and 12 turns and
            64 530 tokens under `SupervisorAgent` --- a 5.5x cost gap attributable entirely to
            the missing guard, which any real-model comparison would have reported as the
            price of orchestration. The oscillation limit does not cover this case: it counts
            repeated *routing*, and a specialist looping on one tool never routes at all.
    """

    tools: Sequence[ToolSpec] = ()
    role: str = SUPERVISOR
    approver: Approver = deny_all
    max_tokens: int = 512
    temperature: float = 0.0
    parse_retries: int = 1
    specialist_steps: int = 4
    oscillation_limit: int = 3
    max_revisions: int = 1
    repeat_limit: int = 3

    architecture: ClassVar[str] = "supervisor"
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

        The signature is character-for-character the control arm's, so a runner cannot treat
        the two architectures differently even by accident, and the trajectory is built from
        ``client.recorder`` for the same reason it is there: the guarded client writes every
        attempted call to that recorder as it happens, and an agent holding a second one would
        return an answer with no evidence behind it.

        Args:
            task_prompt: The adviser's question.
            client: The guarded client; its policy, budgets and recorder are all in force.
            model: The chat model, shared by every role.
            task_id: The task being attempted, checked against the recorder's.
            max_steps: Ceiling on model turns across every role, applied alongside the budget's.

        Returns:
            The finished trajectory, whatever the outcome. Model and transport failures are
            recorded as stop reasons rather than raised.

        Raises:
            ValueError: If ``task_id`` disagrees with the recorder's, which means the harness
                reused a client across tasks and is about to mix one task's evidence into
                another's.
        """
        recorder = client.recorder
        if recorder.task_id != task_id:
            msg = (
                f"the recorder is recording task {recorder.task_id!r} but this run is for "
                f"{task_id!r}; a recorder belongs to exactly one attempt"
            )
            raise ValueError(msg)

        specs = tool_specs_for(client, self.tools)
        run = _Run(agent=self, client=client, model=model, specs=specs, max_steps=max_steps)
        system = render_system_prompt(
            visible_tools(client.policy, self.role, specs),
            role=self.role,
            goal=ROLE_GOALS[SUPERVISOR],
            peers=SPECIALISTS,
            max_steps=max_steps,
            allow_clarify=True,
        )
        opening = new_state(
            task_prompt=task_prompt,
            budget=client.budget_for(self.role),
            messages=[recorder.say("system", system), recorder.say("user", task_prompt)],
            role=self.role,
        )
        try:
            raw = await _build_graph(run).ainvoke(
                opening, config={"recursion_limit": _recursion_limit(max_steps)}
            )
        except GraphRecursionError:
            return recorder.finish(
                stop_reason="max_steps",
                error="the supervisor graph reached its recursion limit",
            )
        except ModelFailureError as failure:
            # `run` promises that a model failure comes back as a stop reason rather than an
            # exception, and it did not keep that promise: an out-of-memory error or a broken
            # endpoint propagated out of the graph and took the trajectory with it, discarding
            # every tool call the recorder had accumulated. The benchmark runner has its own
            # catch-all, so this was invisible there and not for anyone using the documented
            # API directly.
            return recorder.finish(stop_reason="error", error=str(failure))
        final = cast(SupervisorState, raw)
        return recorder.finish(
            final_answer=final["final_answer"],
            stop_reason=final["stop_reason"] or "max_steps",
            error=final["error"] or None,
        )


class ModelFailureError(RuntimeError):
    """The chat model raised inside a graph node.

    A dedicated type because it has to travel out through LangGraph's node machinery and be
    told apart at the top from a genuine harness bug, which must still propagate.
    """


@dataclass(slots=True)
class _Run:
    """One attempt's mutable wiring, bound into the graph's nodes as a closure.

    The graph is rebuilt per run rather than compiled once and parameterised, because the
    alternative is passing the client, the model and the recorder through the state --- which
    would put three unserialisable objects on a blackboard whose whole point is that it can be
    checkpointed and diffed.
    """

    agent: SupervisorAgent
    client: GuardedToolClient
    model: ChatModel
    specs: tuple[ToolSpec, ...]
    max_steps: int
    seen: Counter[str] = field(default_factory=Counter)
    """How many times each ``(tool, arguments)`` pair has been proposed, by any role.

    Counted across the whole run rather than per node, because a loop that alternates
    between the supervisor and a specialist is still a loop, and per-node counters would
    never see it.
    """

    def _repeated(self, tool: str, arguments: Mapping[str, Any]) -> bool:
        """Record a proposed call and say whether the run has now gone round.

        Mirrors `SingleAgent`'s guard exactly, including stopping *before* the call rather
        than after it: executing a third identical call adds a record the grader already has.
        """
        signature = call_signature(tool, arguments)
        self.seen[signature] += 1
        return self.seen[signature] >= self.agent.repeat_limit

    def _complete(self, convo: Sequence[Message]) -> Completion:
        """Take one model turn and charge it to the run's single control role.

        Raises:
            ModelFailureError: If the backend raised. Wrapped rather than left to propagate so
                that `SupervisorAgent.run` can turn it into a finished trajectory, which is
                what its docstring promises.
        """
        try:
            completion = self.model.complete(
                convo, max_tokens=self.agent.max_tokens, temperature=self.agent.temperature
            )
        except Exception as exc:
            raise ModelFailureError(f"{type(exc).__name__}: {exc}") from exc
        self.client.recorder.add_usage(completion.usage)
        self.client.spend_step(self.agent.role)
        self.client.spend_tokens(self.agent.role, completion.usage.total_tokens)
        return completion

    def _halt(self, state: SupervisorState) -> dict[str, Any] | None:
        """Stop the run before a node spends anything, or return `None` to proceed.

        Budget before steps, because a run that has exhausted its tokens and its steps at once
        should report the budget: raising the step ceiling would not have helped it, and a
        trajectory that says ``max_steps`` invites exactly that fix.
        """
        budget: BudgetState = state["budget"]
        if budget.exhausted:
            return {"stop_reason": "budget"}
        if state["step"] >= self.max_steps:
            return {"stop_reason": "max_steps"}
        return None

    def _room(self, state: SupervisorState, taken: int) -> bool:
        """Whether there is headroom for one more model turn, ``taken`` into a delegation."""
        budget: BudgetState = state["budget"]
        return not budget.exhausted and state["step"] + taken < self.max_steps

    async def _confer(
        self,
        role: str,
        brief: str,
        tools: Sequence[ToolSpec],
        state: SupervisorState,
    ) -> _Conference:
        """Run one role's bounded sub-loop in a fresh conversation.

        Fresh is the point: the specialist sees the task, its instruction and the findings, and
        none of the other specialists' reasoning. Its own turns are recorded on the trajectory,
        so nothing is hidden from the grader --- they simply do not enter the supervisor's
        context, which is the cost the architecture is trying to avoid.

        Args:
            role: The acting role; tool calls are made under it, so the policy scopes them.
            brief: The opening user turn.
            tools: What this role's prompt advertises.
            state: Read for the run-level step counter that stamps the call records.

        Returns:
            How the sub-loop ended.
        """
        recorder = self.client.recorder
        limit = self.agent.specialist_steps
        convo = [
            recorder.say(
                "system",
                render_system_prompt(
                    tools,
                    role=role,
                    goal=ROLE_GOALS[role],
                    peers=(SUPERVISOR,),
                    max_steps=limit,
                    allow_clarify=True,
                ),
            ),
            recorder.say("user", brief),
        ]
        evidence: list[str] = []
        taken = 0
        retries = 0
        # The run's ceilings bind inside a delegation, not only between them. Checking only on
        # entry would let a specialist overrun the budget by up to ``specialist_steps`` turns,
        # and a multi-agent arm that quietly buys itself extra turns is the exact wiring
        # artefact this module's docstring says the comparison must not contain.
        while taken < limit and self._room(state, taken):
            completion = self._complete(convo)
            taken += 1
            convo.append(recorder.say("assistant", completion.text))
            parsed = parse_action(completion.text)
            if parsed.action is None:
                retries += 1
                if retries > self.agent.parse_retries:
                    return _Conference(None, taken, tuple(evidence), parsed.error)
                convo.append(recorder.say("user", repair_prompt(parsed.error)))
                continue
            retries = 0
            action = parsed.action
            if isinstance(action, CallToolAction):
                if self._repeated(action.tool, action.arguments):
                    return _Conference(None, taken, tuple(evidence), no_progress=True)
                record = await self.client.call(
                    role,
                    action.tool,
                    action.arguments,
                    step=state["step"] + taken,
                    approved=False,
                )
                if record.result_digest:
                    evidence.append(record.result_digest)
                convo.append(recorder.say("tool", observation(record), name=action.tool))
                continue
            return _Conference(action, taken, tuple(evidence))
        return _Conference(None, taken, tuple(evidence))

    async def delegate(self, role: str, state: SupervisorState) -> dict[str, Any]:
        """The researcher, analyst and writer node: confer, then report back.

        Args:
            role: Which specialist this node is.
            state: The graph state.

        Returns:
            The state update. Control always returns to the supervisor: a specialist that could
            name its own successor would be a second router, and the permission argument for
            splitting the roles rests on there being only one.
        """
        halt = self._halt(state)
        if halt is not None:
            return halt
        conference = await self._confer(
            role,
            render_brief(state["task_prompt"], state["instruction"], state["findings"]),
            visible_tools(self.client.policy, role, self.specs),
            state,
        )
        return self._report(role, state, conference)

    def _report(self, role: str, state: SupervisorState, conference: _Conference) -> dict[str, Any]:
        """Turn a specialist's outcome into a state update.

        A finding is recorded for exactly one outcome: the specialist answered with ``final``.
        Handing back, asking a question and running out of turns all return a report line and
        no finding, which is what makes them visible to the oscillation detector --- routing
        again to a specialist that established nothing is precisely the loop worth stopping.
        """
        recorder = self.client.recorder
        update: dict[str, Any] = {"step": conference.steps, "role": SUPERVISOR}
        action = conference.action
        if conference.no_progress:
            update["stop_reason"] = "no_progress"
            return update
        if action is None:
            if conference.error:
                # A specialist that never produced a valid action is a protocol failure
                # of the model, not a failure of the run. See `StopReason`.
                update["stop_reason"] = "protocol"
                update["error"] = f"{role}: {conference.error}"
                return update
            summary = f"[{role}] stopped after {conference.steps} turn(s) with nothing to report."
            update["messages"] = [recorder.say("user", summary)]
            return update
        if isinstance(action, FinalAction):
            update["findings"] = [
                Finding(
                    role=role,
                    step=state["step"] + conference.steps,
                    text=action.answer,
                    evidence=conference.evidence,
                )
            ]
            update["messages"] = [recorder.say("user", f"[{role}] {action.answer}")]
            if role == WRITER:
                update["draft"] = action.answer
            return update
        if isinstance(action, ClarifyAction):
            update["messages"] = [recorder.say("user", f"[{role}] asks: {action.question}")]
            return update
        update["messages"] = [recorder.say("user", f"[{role}] hands back: {action.instruction}")]
        return update

    async def verify(self, state: SupervisorState) -> dict[str, Any]:
        """The verifier node: pass the draft, or send it back once.

        A pass ends the run with the *draft* as the answer, not with the verifier's own words.
        Letting the auditor rewrite what it approved would make it a second writer and remove
        the audit; its verdict is recorded as a finding instead.

        A second rejection ends the run too, with the draft still returned and a stop reason of
        ``no_progress``. The draft is returned because the grader must be able to score what the
        run actually produced, and the stop reason says why nothing better came: the writer and
        the verifier disagreed twice, and a third exchange would be the same argument at the
        benchmark's expense.
        """
        halt = self._halt(state)
        if halt is not None:
            return halt
        recorder = self.client.recorder
        draft = state["draft"]
        if not draft:
            return {
                "role": SUPERVISOR,
                "messages": [recorder.say("user", "[verifier] there is no draft to check yet.")],
            }
        evidence = tuple(
            record.result_digest
            for record in recorder.calls
            if record.executed and record.ok and record.result_digest
        )
        conference = await self._confer(
            VERIFIER,
            render_audit(state["task_prompt"], draft, recorder.calls),
            (),
            state,
        )
        update: dict[str, Any] = {"step": conference.steps, "role": SUPERVISOR}
        action = conference.action
        if conference.no_progress:
            update["stop_reason"] = "no_progress"
            return update
        if action is None:
            if conference.error:
                update["stop_reason"] = "protocol"
                update["error"] = f"{VERIFIER}: {conference.error}"
                return update
            update["messages"] = [
                recorder.say("user", "[verifier] stopped without reaching a verdict.")
            ]
            return update
        if isinstance(action, FinalAction):
            update["findings"] = [
                Finding(
                    role=VERIFIER,
                    step=state["step"] + conference.steps,
                    text=action.answer,
                    evidence=evidence,
                )
            ]
            update["messages"] = [recorder.say("user", f"[verifier] passed: {action.answer}")]
            update["final_answer"] = draft
            update["stop_reason"] = "answered"
            return update
        if isinstance(action, ClarifyAction):
            update["messages"] = [recorder.say("user", f"[verifier] asks: {action.question}")]
            return update
        if state["revisions"] >= self.agent.max_revisions:
            update["messages"] = [
                recorder.say("user", f"[verifier] rejected the draft again: {action.instruction}")
            ]
            update["final_answer"] = draft
            update["stop_reason"] = "no_progress"
            return update
        update["revisions"] = state["revisions"] + 1
        update["instruction"] = action.instruction
        update["messages"] = [
            recorder.say("user", f"[verifier] sent the draft back: {action.instruction}")
        ]
        return update

    async def supervise(self, state: SupervisorState) -> dict[str, Any]:
        """The supervisor node: one decision, with one nudge if the reply will not parse.

        A handoff to a name that resolves to no specialist is treated as a protocol failure
        rather than as a routing failure, and is nudged exactly like unparseable JSON. The two
        are the same defect from the benchmark's point of view --- the model did not produce a
        usable action --- and giving them different allowances would let a model that invents
        agent names loop for free.
        """
        halt = self._halt(state)
        if halt is not None:
            return halt
        recorder = self.client.recorder
        fresh: list[Message] = []
        taken = 0
        error = ""
        for _ in range(self.agent.parse_retries + 1):
            completion = self._complete([*state["messages"], *fresh])
            taken += 1
            fresh.append(recorder.say("assistant", completion.text))
            parsed = parse_action(completion.text)
            if parsed.action is None:
                error = parsed.error
                fresh.append(recorder.say("user", repair_prompt(error)))
                continue
            action = parsed.action
            if isinstance(action, HandoffAction):
                target = resolve_target(action.to)
                if target is None:
                    error = (
                        f"there is no agent named {action.to!r}; hand off to one of: "
                        f"{', '.join(SPECIALISTS)}"
                    )
                    fresh.append(recorder.say("user", repair_prompt(error)))
                    continue
                return self._route(state, action, target, fresh, taken)
            return await self._act(state, action, fresh, taken)
        return {
            "step": taken,
            "messages": fresh,
            "stop_reason": "protocol",
            "error": error,
        }

    def _route(
        self,
        state: SupervisorState,
        action: HandoffAction,
        target: str,
        fresh: list[Message],
        taken: int,
    ) -> dict[str, Any]:
        """Hand work to one specialist, unless that would be the third time round the loop.

        The marker pairs the destination with the number of findings held when the decision was
        made, so two markers are equal exactly when the supervisor sent work to the same
        specialist knowing the same things. The check fires *before* the handoff is carried out:
        a third identical delegation would produce a third identical result, and stopping after
        paying for it teaches the benchmark nothing it does not already know.
        """
        marker = route_marker(target, len(state["findings"]))
        update: dict[str, Any] = {
            "step": taken,
            "messages": fresh,
            "routes": [marker],
            "role": SUPERVISOR,
        }
        if state["routes"].count(marker) + 1 >= self.agent.oscillation_limit:
            update["stop_reason"] = "no_progress"
            return update
        update["role"] = target
        update["instruction"] = action.instruction
        return update

    async def _act(
        self,
        state: SupervisorState,
        action: FinalAction | ClarifyAction | CallToolAction,
        fresh: list[Message],
        taken: int,
    ) -> dict[str, Any]:
        """Carry out a supervisor action that is not a handoff.

        This is the only place in the architecture where a write tool is reachable, and it is
        reachable only with whatever the approver says about *this* call: approval is a fact
        about one proposed call, not a mode the run is left in.
        """
        update: dict[str, Any] = {"step": taken, "messages": fresh, "role": SUPERVISOR}
        if isinstance(action, FinalAction):
            update["final_answer"] = action.answer
            update["stop_reason"] = "answered"
            return update
        if isinstance(action, ClarifyAction):
            # Asking is a legitimate ending: the ambiguous family scores a question above a
            # confident guess, and the grader has a matcher for it.
            update["final_answer"] = action.question
            update["stop_reason"] = "answered"
            return update
        if self._repeated(action.tool, action.arguments):
            update["stop_reason"] = "no_progress"
            return update
        record = await self.client.call(
            self.agent.role,
            action.tool,
            action.arguments,
            step=state["step"] + taken,
            approved=self.agent.approver(self.agent.role, action.tool, action.arguments),
        )
        fresh.append(self.client.recorder.say("tool", observation(record), name=action.tool))
        return update


def _route_next(state: SupervisorState) -> str:
    """The graph's only routing rule: stop when a node set a stop reason, else go where told.

    Termination is a single signal on the blackboard rather than a condition each node's edge
    re-derives. Four nodes each deciding for themselves whether the run is over is four places
    for the answer to differ, and the one that matters --- "did anything actually finish this?"
    --- would then have no single owner.
    """
    return END if state["stop_reason"] is not None else state["role"]


def _specialist_node(run: _Run, role: str) -> Callable[[SupervisorState], Awaitable[Any]]:
    """Bind one specialist's role into a node coroutine."""

    async def node(state: SupervisorState) -> dict[str, Any]:
        return await run.delegate(role, state)

    return node


def _build_graph(run: _Run) -> Any:
    """Assemble the star topology: supervisor at the centre, four leaves, one way back."""
    builder = StateGraph(SupervisorState)
    builder.add_node(SUPERVISOR, run.supervise)
    for role in (RESEARCHER, ANALYST, WRITER):
        builder.add_node(role, _specialist_node(run, role))
    builder.add_node(VERIFIER, run.verify)
    builder.add_edge(START, SUPERVISOR)
    destinations: dict[str, str] = {name: name for name in ALL_ROLES}
    destinations[END] = END
    for name in ALL_ROLES:
        builder.add_conditional_edges(name, _route_next, destinations)
    return builder.compile()
