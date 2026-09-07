"""Tests for the benchmark runner: the thing that turns a task set into a defensible number.

The centrepiece is :func:`test_the_whole_benchmark_runs_in_both_architectures`, which runs every
task through both arms against the real MCP server, the real permission policy and the real
grader. That test is the project's CI gate, so it is written to be fast (a couple of seconds),
deterministic (a scripted model and an injected clock) and demanding: it asserts the aggregate
is internally consistent rather than merely non-empty. Everything else in this file exists to
pin one property of the runner that the gate alone would not notice --- that a model which
raises does not take the run down with it, that a resumed run and an uninterrupted one produce
the same grades, that concurrency changes the wall time and nothing else.

The runs are shared through module-scoped fixtures. A benchmark run is expensive enough that
paying for one per test would make the suite slow enough to skip, and every assertion below is a
question about the same run rather than about a run of its own.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mcpeval.agents.llm import ScriptedChatModel
from mcpeval.agents.protocol import parse_action
from mcpeval.agents.single import SingleAgent
from mcpeval.agents.supervisor import VERIFIER, SupervisorAgent
from mcpeval.bench.runner import (
    ARCHITECTURES,
    DEFAULT_MAX_STEPS,
    BenchmarkRun,
    Manifest,
    RunPaths,
    benchmark_policy,
    build_agent,
    load_run,
    policy_digest,
    read_grades,
    rerender,
    run_benchmark,
    run_budget,
    scripted_benchmark_model,
    task_set_digest,
    write_grades,
)
from mcpeval.bench.runner import _scripted_reply as scripted_reply
from mcpeval.bench.tasks import READ_TOOLS, WRITE_TOOLS, build_tasks
from mcpeval.client.policy import ANALYST, RESEARCHER, SUPERVISOR, WRITER, RolePolicy
from mcpeval.client.recorder import read_jsonl
from mcpeval.client.session import connect_in_process
from mcpeval.mcp_server.server import build_server
from mcpeval.metrics.report import Aggregate
from mcpeval.schemas import (
    Completion,
    FailureClass,
    Grade,
    Message,
    TaskFamily,
    Trajectory,
)
from mcpeval.world.store import World, WorldLog, build_world

# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------

_WORLD: Final[World] = build_world()
_TASKS: Final = build_tasks(_WORLD)

#: Small enough to keep a per-test run under a tenth of a second, wide enough to cross a
#: family boundary --- the first family is nine tasks long, so anything under ten would only
#: ever exercise ``lookup``.
SLICE: Final = 6

#: Resamples for a test's intervals. The gate needs the bootstrap to run, not to be precise:
#: 200 draws produce the same shape of interval as 2000 in a fraction of the time, and no
#: assertion here reads a bound to four decimal places.
BOOT: Final = 200


class FakeClock:
    """A counter standing in for the wall clock, so a stored run is byte-stable."""

    def __init__(self, step: float = 0.001) -> None:
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        now = self._now
        self._now += self._step
        return now


@dataclass
class ExplodingModel:
    """A model that raises on one task and behaves on the rest.

    The point of the benchmark surviving this is that the *other* tasks still count: a
    harness that let one exception unwind the run would lose seventy attempts to one, and the
    trajectory that recorded the failure is the only evidence that the failure happened.
    """

    trigger: str
    inner: ScriptedChatModel = field(default_factory=scripted_benchmark_model)
    raised: int = 0

    @property
    def name(self) -> str:
        return "exploding"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Sequence[str] | None = None,
    ) -> Completion:
        if any(self.trigger in message.content for message in messages):
            self.raised += 1
            msg = "the model fell over"
            raise RuntimeError(msg)
        return self.inner.complete(
            messages, max_tokens=max_tokens, temperature=temperature, stop=stop
        )


def silent_model() -> ScriptedChatModel:
    """A model that answers in prose, to exercise the protocol-failure path."""
    return ScriptedChatModel(default="I would rather not.", model_name="silent")


def projection(run: BenchmarkRun) -> list[tuple[str, bool, float, float, int]]:
    """Everything about a run's grades except the one column a rerun cannot reproduce."""
    return [(g.task_id, g.success, g.answer_score, g.call_f1, g.steps) for g in run.grades]


def bench(
    *,
    architecture: str = "single",
    limit: int | None = SLICE,
    out_dir: Path | None = None,
    concurrency: int = 4,
    resume: bool = False,
    model: object | None = None,
    max_steps: int = 12,
) -> BenchmarkRun:
    """Run the benchmark synchronously, with the defaults every test here wants.

    A fresh :class:`WorldLog` per run, because the two write tools append to it and a log
    shared between runs would make note and order identifiers depend on test ordering.
    """
    chat = scripted_benchmark_model() if model is None else model
    return asyncio.run(
        run_benchmark(
            _TASKS,
            architecture=architecture,
            model=chat,  # type: ignore[arg-type]
            world=_WORLD,
            log=WorldLog(),
            policy=benchmark_policy(),
            max_steps=max_steps,
            concurrency=concurrency,
            out_dir=out_dir,
            resume=resume,
            limit=limit,
            clock=FakeClock(),
            n_boot=BOOT,
            world_seed=7,
        )
    )


@pytest.fixture(scope="module")
def full_runs() -> dict[str, BenchmarkRun]:
    """Both architectures over the whole task set: the CI gate's shared subject."""
    return {arch: bench(architecture=arch, limit=None) for arch in ARCHITECTURES}


@pytest.fixture(scope="module")
def stored_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[BenchmarkRun, RunPaths]:
    """One small run written to disk, for the persistence and re-rendering tests."""
    out = tmp_path_factory.mktemp("stored")
    return bench(out_dir=out), RunPaths(out)


# --------------------------------------------------------------------------------------
# The gate: the whole benchmark, both arms
# --------------------------------------------------------------------------------------


def _assert_well_formed(agg: Aggregate, *, n: int) -> None:
    """Every internal consistency an aggregate claims, checked rather than assumed."""
    assert agg.n == n
    assert agg.overall.n == n
    assert sum(summary.n for summary in agg.by_family) == n
    for summary in (agg.overall, *agg.by_family):
        for key, interval in summary.metrics.items():
            assert interval.low <= interval.point <= interval.high, key
            assert interval.n == summary.n
    for row in agg.failures:
        assert 0 < row.count <= n
        assert row.rate.point == pytest.approx(row.count / n)


def test_the_whole_benchmark_runs_in_both_architectures(
    full_runs: dict[str, BenchmarkRun],
) -> None:
    """The CI gate: every task, both arms, one aggregate each, all of it well formed."""
    assert len(_TASKS) >= 60
    for arch, run in full_runs.items():
        assert run.manifest.architecture == arch
        assert len(run.trajectories) == len(_TASKS)
        assert [t.task_id for t in run.trajectories] == [task.id for task in _TASKS]
        _assert_well_formed(run.aggregate, n=len(_TASKS))


def test_every_family_is_represented_in_the_breakdown(full_runs: dict[str, BenchmarkRun]) -> None:
    """A family missing from the report is a family nobody is being scored on."""
    for run in full_runs.values():
        assert {summary.label for summary in run.aggregate.by_family} == {
            family.value for family in TaskFamily
        }


def test_every_trajectory_stops_for_a_declared_reason(full_runs: dict[str, BenchmarkRun]) -> None:
    """No attempt ends in an unrecorded state, in either arm."""
    allowed = {
        "answered",
        "max_steps",
        "budget",
        "refused",
        "error",
        "protocol",
        "no_progress",
    }
    for run in full_runs.values():
        assert {t.stop_reason for t in run.trajectories} <= allowed


def test_every_executed_call_was_admitted_by_the_policy(
    full_runs: dict[str, BenchmarkRun],
) -> None:
    """The guarded client's central invariant, checked over a whole benchmark rather than a unit.

    A call that reached the server without an allowing verdict would mean the enforcement point
    was consulted and then ignored --- the one thing that must be impossible, and the one thing
    a benchmark of permissioning cannot notice on its own.
    """
    for run in full_runs.values():
        for traj in run.trajectories:
            for call in traj.calls:
                assert call.executed <= call.decision.allowed
                assert not call.executed or call.tool in set(READ_TOOLS) | set(WRITE_TOOLS)


def test_the_supervisor_arm_costs_more_steps_than_the_control(
    full_runs: dict[str, BenchmarkRun],
) -> None:
    """The architecture's price, measured rather than asserted in a docstring.

    Delegating costs at least one routing turn per specialist, so the multi-agent arm cannot be
    cheaper in steps. If this ever flips, the supervisor stopped delegating.
    """
    single = full_runs["single"].aggregate.overall.metrics["steps"].point
    supervisor = full_runs["supervisor"].aggregate.overall.metrics["steps"].point
    assert supervisor > single


def test_the_benchmark_is_reproducible() -> None:
    """Same tasks, same model, same seed: the same grades, to the last decimal.

    Wall time is excluded because it is the one quantity a benchmark cannot reproduce; every
    other column of every row must match, or nothing downstream of this runner is comparable.
    """
    first = bench(limit=8)
    second = bench(limit=8)
    keys = [
        (g.task_id, g.success, g.answer_score, g.call_f1, g.steps, g.failures)
        for g in (*first.grades, *second.grades)
    ]
    assert keys[: len(first.grades)] == keys[len(first.grades) :]


def test_concurrency_changes_nothing_but_the_scheduling() -> None:
    """Running four at a time and one at a time must grade identically.

    Attempts share one MCP session and one model object, so this is the test that says the
    sharing is safe: a runner whose results depended on the interleaving would produce a
    different number every time CI happened to be busy.
    """
    serial = bench(limit=8, concurrency=1)
    parallel = bench(limit=8, concurrency=8)
    assert projection(serial) == projection(parallel)


# --------------------------------------------------------------------------------------
# Failure containment
# --------------------------------------------------------------------------------------


def test_a_model_that_raises_costs_one_task_not_the_run() -> None:
    """One exploding task is recorded as an error; every other task still gets a grade."""
    target = _TASKS[2]
    model = ExplodingModel(trigger=target.prompt)
    run = bench(model=model, limit=SLICE)

    assert model.raised >= 1
    assert len(run.grades) == SLICE
    failed = next(t for t in run.trajectories if t.task_id == target.id)
    assert failed.stop_reason == "error"
    assert failed.error is not None
    assert "the model fell over" in failed.error
    assert all(t.stop_reason == "answered" for t in run.trajectories if t.task_id != target.id)


def test_a_failed_attempt_is_graded_rather_than_dropped() -> None:
    """The failure has to reach the report, or the denominator quietly changes."""
    target = _TASKS[1]
    run = bench(model=ExplodingModel(trigger=target.prompt), limit=SLICE)
    grade = next(g for g in run.grades if g.task_id == target.id)
    assert not grade.success
    assert grade.answer_score == 0.0
    # A model that raised is a run error; nothing on the platform was ever asked.
    assert FailureClass.RUN_ERROR in grade.failures
    assert FailureClass.TOOL_ERROR not in grade.failures
    assert run.aggregate.n == SLICE


def test_a_model_that_will_not_follow_the_protocol_is_recorded_not_raised() -> None:
    """Prose instead of JSON is a measurable defect, not a crashed run.

    The taxonomy says so as well as the docstring: these are `protocol_failure`, and a reader
    of the failure table is not sent to look at the backend for them.
    """
    run = bench(model=silent_model(), limit=3)
    assert {t.stop_reason for t in run.trajectories} == {"protocol"}
    assert all(g.answer_score == 0.0 for g in run.grades)
    assert all(FailureClass.FORMAT_VIOLATION not in g.failures for g in run.grades)
    assert all(FailureClass.PROTOCOL_FAILURE in g.failures for g in run.grades)
    assert all(FailureClass.RUN_ERROR not in g.failures for g in run.grades)


# --------------------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"architecture": "swarm"}, "unknown architecture"),
        ({"limit": 0}, "no tasks to run"),
        ({"limit": -1}, "must not be negative"),
        ({"concurrency": 0}, "concurrency must be at least 1"),
        ({"max_steps": 0}, "max_steps must be at least 1"),
    ],
)
def test_bad_arguments_are_refused_with_a_reason(kwargs: dict[str, object], message: str) -> None:
    """Each of these would otherwise surface as an empty report rather than as a mistake."""
    with pytest.raises(ValueError, match=message):
        bench(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------


def test_a_run_writes_the_five_files_that_describe_it(
    stored_run: tuple[BenchmarkRun, RunPaths],
) -> None:
    """Trajectories, grades, aggregate, report, manifest: evidence, scores, summary, provenance."""
    _, paths = stored_run
    for path in (paths.trajectories, paths.grades, paths.aggregate, paths.report, paths.manifest):
        assert path.exists(), path
        assert path.read_text(encoding="utf-8").strip()


def test_the_trajectory_log_holds_every_attempt(
    stored_run: tuple[BenchmarkRun, RunPaths],
) -> None:
    """The log is the source of truth, so it must round-trip through JSONL without loss."""
    run, paths = stored_run
    stored = read_jsonl(paths.trajectories)
    assert {t.task_id for t in stored} == {t.task_id for t in run.trajectories}
    assert all(isinstance(t, Trajectory) for t in stored)


def test_the_stored_grades_are_the_run_s_grades(
    stored_run: tuple[BenchmarkRun, RunPaths],
) -> None:
    run, paths = stored_run
    assert read_grades(paths.grades) == list(run.grades)


def test_the_stored_report_is_reproducible_from_the_stored_grades(
    stored_run: tuple[BenchmarkRun, RunPaths],
) -> None:
    """The claim the manifest exists to support: every number traces back to the graded rows.

    Re-derived from ``grades.jsonl`` and the manifest's bootstrap settings, not copied from
    ``aggregate.json`` --- a re-render that read the summary could only ever reproduce itself.
    """
    _, paths = stored_run
    manifest, grades = load_run(paths.root)
    assert rerender(manifest, grades) == paths.report.read_text(encoding="utf-8")


def test_the_stored_aggregate_parses_back_into_an_aggregate(
    stored_run: tuple[BenchmarkRun, RunPaths],
) -> None:
    run, paths = stored_run
    parsed = Aggregate.model_validate_json(paths.aggregate.read_text(encoding="utf-8"))
    assert parsed == run.aggregate


def test_the_manifest_says_what_produced_the_numbers(
    stored_run: tuple[BenchmarkRun, RunPaths],
) -> None:
    """Model, topology, questions, gold answers, policy, code version, headroom."""
    run, paths = stored_run
    manifest = Manifest.model_validate_json(paths.manifest.read_text(encoding="utf-8"))
    assert manifest == run.manifest
    assert manifest.model == "scripted"
    assert manifest.architecture == "single"
    assert manifest.code_version
    assert manifest.task_count == SLICE
    assert manifest.task_set_digest == task_set_digest(_TASKS[:SLICE])
    assert manifest.policy_digest
    assert manifest.write_tools == tuple(sorted(WRITE_TOOLS))
    assert manifest.approval_required == tuple(sorted(WRITE_TOOLS))
    assert manifest.world_seed == 7


def test_a_fresh_run_does_not_inherit_the_previous_log(tmp_path: Path) -> None:
    """Re-running into a used directory must replace it, not silently double its denominator."""
    bench(limit=SLICE, out_dir=tmp_path)
    bench(limit=3, out_dir=tmp_path)
    assert len(read_jsonl(RunPaths(tmp_path).trajectories)) == 3


def test_load_run_names_the_file_that_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"manifest\.json"):
        load_run(tmp_path)


def test_write_and_read_grades_round_trip(tmp_path: Path) -> None:
    grades = [
        Grade(
            task_id=f"t-{index}",
            family=TaskFamily.LOOKUP,
            architecture="single",
            model="scripted",
            success=bool(index % 2),
            answer_score=index / 10,
        )
        for index in range(5)
    ]
    path = tmp_path / "nested" / "grades.jsonl"
    assert write_grades(path, grades) == 5
    assert read_grades(path) == grades


# --------------------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------------------


def test_resume_runs_only_what_is_missing(tmp_path: Path) -> None:
    """The property that makes an hours-long run survivable: finished work is not repeated."""
    first = bench(limit=3, out_dir=tmp_path)
    second = bench(limit=SLICE, out_dir=tmp_path, resume=True)

    assert first.attempted == 3
    assert second.attempted == SLICE - 3
    assert second.resumed == tuple(task.id for task in _TASKS[:3])
    assert second.aggregate.n == SLICE


def test_a_resumed_run_grades_the_same_as_an_uninterrupted_one(tmp_path: Path) -> None:
    """Interrupting a run must not change its result, only its wall time."""
    bench(limit=3, out_dir=tmp_path)
    resumed = bench(limit=SLICE, out_dir=tmp_path, resume=True)
    straight = bench(limit=SLICE)
    assert projection(resumed) == projection(straight)


def test_resume_ignores_another_architecture_s_attempts(tmp_path: Path) -> None:
    """Two arms in one directory would pair a supervisor's answer with a control's evidence."""
    bench(architecture="single", limit=SLICE, out_dir=tmp_path)
    crossed = bench(architecture="supervisor", limit=SLICE, out_dir=tmp_path, resume=True)
    assert crossed.resumed == ()
    assert crossed.attempted == SLICE
    assert all(t.architecture == "supervisor" for t in crossed.trajectories)


def test_resuming_a_complete_run_needs_no_server(tmp_path: Path) -> None:
    """Nothing pending means nothing started: a re-scored run costs a grade, not a run."""
    bench(limit=SLICE, out_dir=tmp_path)
    again = bench(limit=SLICE, out_dir=tmp_path, resume=True)
    assert again.attempted == 0
    assert len(again.resumed) == SLICE
    assert again.aggregate.n == SLICE


# --------------------------------------------------------------------------------------
# Provenance digests
# --------------------------------------------------------------------------------------


def test_the_task_set_digest_follows_the_gold_answers_not_the_ids() -> None:
    """Regenerate the world and the ids stay put while every expected number moves.

    A digest over ids alone would call two incomparable runs comparable, which is exactly the
    mistake the manifest exists to prevent.
    """
    other = build_tasks(build_world(11))
    # The lookup family's ids are the same strings under any seed, which is the point: they
    # identify the question, not the answer, and every gold answer behind them has moved.
    assert [t.id for t in other if t.family is TaskFamily.LOOKUP] == [
        t.id for t in _TASKS if t.family is TaskFamily.LOOKUP
    ]
    assert task_set_digest(other) != task_set_digest(_TASKS)
    assert task_set_digest(build_tasks(build_world())) == task_set_digest(_TASKS)


def test_the_task_set_digest_notices_a_single_edited_answer() -> None:
    edited = (*_TASKS[:-1], _TASKS[-1].model_copy(update={"optimal_steps": 99}))
    assert task_set_digest(edited) != task_set_digest(_TASKS)


def test_the_policy_digest_does_not_depend_on_set_iteration_order() -> None:
    """The frozensets in a policy iterate in hash order, which differs between processes.

    Digesting ``model_dump_json`` directly would give the same policy two fingerprints on two
    machines, which is the one failure a provenance record must not have.
    """
    forwards = benchmark_policy()
    backwards = forwards.model_copy(
        update={
            "known_tools": frozenset(sorted(forwards.known_tools, reverse=True)),
            "write_tools": frozenset(sorted(forwards.write_tools, reverse=True)),
        }
    )
    assert policy_digest(forwards) == policy_digest(backwards)


def test_the_policy_digest_notices_a_widened_scope() -> None:
    policy = benchmark_policy()
    widened = policy.model_copy(
        update={
            "roles": {
                **policy.roles,
                WRITER: RolePolicy(name=WRITER, allow=("*",), may_write=True),
            }
        }
    )
    assert policy_digest(widened) != policy_digest(policy)


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


async def test_the_benchmark_policy_matches_what_the_server_advertises() -> None:
    """The policy's inventory and the server's must be the same list.

    They are written in two places on purpose --- a policy that imported its tool names from the
    server could never disagree with it, and a permission layer that cannot disagree with the
    thing it governs is not governing anything. This is the test that keeps the two honest.
    """
    server = build_server(_WORLD, WorldLog())
    async with connect_in_process(server) as client:
        specs = await client.discover()
    live = {spec.name for spec in specs}
    policy = benchmark_policy(specs)
    assert live == set(READ_TOOLS) | set(WRITE_TOOLS)
    assert policy.known_tools == live
    assert policy.write_tools == set(WRITE_TOOLS)
    assert policy.approval_required == set(WRITE_TOOLS)
    assert policy_digest(policy) == policy_digest(benchmark_policy())


def test_only_the_supervisor_may_write() -> None:
    """The scope argument the multi-agent arm rests on, stated as a test."""
    policy = benchmark_policy()
    assert policy.roles[SUPERVISOR].may_write
    for role in (RESEARCHER, ANALYST, WRITER):
        assert not policy.roles[role].may_write
    assert VERIFIER not in policy.roles


def test_every_specialist_scope_is_a_subset_of_the_inventory() -> None:
    """A scope naming a tool the server does not publish is a scope that silently does nothing."""
    policy = benchmark_policy()
    for role in (RESEARCHER, ANALYST, WRITER):
        assert set(policy.roles[role].allow) <= set(READ_TOOLS)


def test_the_runner_step_ceiling_beats_the_policy_s() -> None:
    """``--max-steps`` is the knob an experimenter turns, so it has to be the one that binds."""
    budget = run_budget(benchmark_policy(), max_steps=5)
    assert budget.max_steps == 5
    assert budget.steps == 0
    assert not budget.exhausted


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [("single", SingleAgent), ("supervisor", SupervisorAgent)],
)
def test_build_agent_returns_the_named_arm(architecture: str, expected: type) -> None:
    agent = build_agent(architecture)
    assert isinstance(agent, expected)
    assert agent.architecture == architecture


def test_build_agent_refuses_an_unknown_arm() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        build_agent("swarm")


# --------------------------------------------------------------------------------------
# The scripted model
# --------------------------------------------------------------------------------------


@given(
    role=st.sampled_from(["supervisor", "researcher", "analyst", "writer", "verifier"]),
    text=st.text(max_size=200),
    seen=st.lists(st.sampled_from(["assistant", "tool", "user"]), max_size=4),
)
def test_the_scripted_model_always_emits_a_parseable_action(
    role: str, text: str, seen: list[str]
) -> None:
    """Whatever it is handed, the stand-in speaks the protocol.

    This is the property that makes it usable as a gate: a stand-in that could emit
    unparseable text would turn a harness bug into a format-compliance statistic and quietly
    corrupt exactly the number the benchmark reports.
    """
    conversation = [
        Message(role="system", content=f'You are the {role} agent. "action": "handoff" researcher'),
        Message(role="user", content=text),
        *[
            Message(role=turn, content=text, name="tool" if turn == "tool" else None)
            for turn in seen
        ],
    ]
    parsed = parse_action(scripted_reply(conversation))
    assert parsed.ok
    assert parsed.clean


def test_the_scripted_model_plans_a_read_before_it_answers() -> None:
    """One tool call, then an answer: the shape every arm's loop is written against."""
    model = scripted_benchmark_model()
    system = Message(role="system", content="You are the researcher agent.")
    opening = [system, Message(role="user", content="What does account ACC-0001 hold?")]
    first = parse_action(model.complete(opening).text).action
    assert first is not None
    assert getattr(first, "tool", "") == "account_holdings"
    assert getattr(first, "arguments", {}) == {"account_id": "ACC-0001"}

    followed = [
        *opening,
        Message(role="assistant", content="{}"),
        Message(role="tool", content='{"found": true, "cash_balance": "1000.00"}', name="x"),
    ]
    second = parse_action(model.complete(followed).text).action
    assert second is not None
    assert "1000.00" in getattr(second, "answer", "")


def test_the_scripted_model_refuses_when_the_platform_has_no_record() -> None:
    """A ``found: false`` result must not become a confident number."""
    model = scripted_benchmark_model()
    conversation = [
        Message(role="system", content="You are the supervisor agent."),
        Message(role="user", content="What is the balance of ACC-9999?"),
        Message(role="assistant", content="{}"),
        Message(role="tool", content='{"found": false, "reason": "no such account id"}', name="x"),
    ]
    action = parse_action(model.complete(conversation).text).action
    assert action is not None
    answer = getattr(action, "answer", "")
    assert "no record" in answer
    assert not any(character.isdigit() for character in answer)


def test_the_scripted_model_is_named_in_every_trajectory(
    stored_run: tuple[BenchmarkRun, RunPaths],
) -> None:
    """A trajectory has to say which model produced it, or it cannot be filtered on later."""
    run, _ = stored_run
    assert {t.model for t in run.trajectories} == {"scripted"}
    assert {g.model for g in run.grades} == {"scripted"}


def test_the_aggregate_json_is_valid_json(stored_run: tuple[BenchmarkRun, RunPaths]) -> None:
    """Dashboards read this file; a pydantic dump that only pydantic can read is not enough."""
    _, paths = stored_run
    payload = json.loads(paths.aggregate.read_text(encoding="utf-8"))
    assert payload["n"] == SLICE
    assert payload["overall"]["metrics"]["success"]["point"] >= 0.0


def test_read_grades_tolerates_a_blank_line(tmp_path: Path) -> None:
    """A file appended to by two processes can end up with a stray newline; it is not data loss."""
    path = tmp_path / "grades.jsonl"
    grade = Grade(
        task_id="t-1",
        family=TaskFamily.LOOKUP,
        architecture="single",
        model="scripted",
        success=True,
    )
    path.write_text(f"{grade.model_dump_json()}\n\n", encoding="utf-8", newline="\n")
    assert read_grades(path) == [grade]


def test_resume_without_an_output_directory_runs_everything() -> None:
    """There is nothing to resume from when nothing was written; that is not an error."""
    run = bench(limit=3, resume=True)
    assert run.resumed == ()
    assert run.attempted == 3


def test_cancellation_is_not_recorded_as_a_task_failure() -> None:
    """A cancelled run is the caller's decision, so it must not be filed as the agent's failure.

    Catching it alongside the model's own exceptions would turn ``Ctrl-C`` into a run of
    seventy error trajectories, which is a report of a benchmark that never happened.
    """

    def cancel(_messages: Sequence[Message]) -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        bench(model=ScriptedChatModel(responder=cancel, model_name="cancelling"), limit=2)


def test_the_scripted_supervisor_answers_once_everyone_has_reported() -> None:
    """The routing chain has an end: three reports in, and the supervisor stops delegating."""
    conversation = [
        Message(
            role="system", content='You are the supervisor agent. "action": "handoff" researcher'
        ),
        Message(role="user", content="What is the balance of ACC-0001?"),
        Message(role="user", content="[researcher] the account holds four lines"),
        Message(role="user", content="[writer] the balance is 1,000.00"),
        Message(role="user", content="[verifier] there is no draft to check yet."),
    ]
    action = parse_action(scripted_reply(conversation)).action
    assert action is not None
    assert getattr(action, "answer", "") == "there is no draft to check yet."


def test_the_step_ceiling_clears_what_the_supervisor_arm_structurally_needs() -> None:
    """The ceiling must not be the thing the comparison measures.

    A single agent spends one turn per tool call plus one to answer. A supervisor spends a
    routing turn and a specialist turn for each, then a writer turn and a verifier turn, so
    the same gold chain costs it roughly two and a half times as many model turns. A ceiling
    sized off the gold chain alone truncates the supervisor on exactly the hardest tasks and
    records it as `max_steps` -- a property of the harness reported as a property of the
    architecture.
    """
    longest_chain = max(task.optimal_steps for task in _TASKS)
    single_worst = longest_chain + 1
    supervisor_worst = 2 * longest_chain + 2  # a route and a turn per call, then write, verify
    assert DEFAULT_MAX_STEPS > supervisor_worst > single_worst
