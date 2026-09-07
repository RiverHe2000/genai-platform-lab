"""Tests for the trajectory recorder and its JSONL persistence.

The recorder is the only writer of the evidence a run is graded on, so the tests are
written against the properties a grader depends on rather than against sample output:
timing is exactly reproducible under an injected clock, the digest is invariant to the
things that differ between a Windows run and a Linux one, tokens are conserved under
accumulation, and a trajectory survives a trip through the file system unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from mcpeval.client.recorder import (
    DIGEST_LENGTH,
    Stopwatch,
    TrajectoryRecorder,
    append_jsonl,
    ensure_encodable,
    iter_jsonl,
    normalise_result_text,
    read_jsonl,
    result_digest,
    write_jsonl,
)
from mcpeval.schemas import (
    Message,
    PolicyDecision,
    PolicyVerdict,
    ToolCallRecord,
    Trajectory,
    Usage,
)

ALLOW = PolicyDecision(verdict=PolicyVerdict.ALLOW, rule="allow", reason="permitted")
REFUSED = PolicyDecision(
    verdict=PolicyVerdict.REFUSE_OUT_OF_SCOPE,
    rule="scope.write_forbidden",
    reason="read-only role",
)


class FakeClock:
    """A clock advancing a fixed step per read, so every latency is exactly predictable."""

    def __init__(self, step: float = 0.5) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


def recorder(clock: FakeClock | None = None) -> TrajectoryRecorder:
    """A recorder on a deterministic clock."""
    return TrajectoryRecorder(
        task_id="t-1",
        architecture="supervisor",
        model="scripted",
        clock=clock or FakeClock(),
    )


# --------------------------------------------------------------------------------------
# Normalisation and digests
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain", "plain"),
        ("windows\r\nlines", "windows\nlines"),
        ("old\rmac", "old\nmac"),
        ("  padded  ", "padded"),
        ("\n\nblank edges\n\n", "blank edges"),
        ("", ""),
    ],
)
def test_normalisation_removes_only_incidental_differences(raw: str, expected: str) -> None:
    assert normalise_result_text(raw) == expected


def test_normalisation_keeps_interior_whitespace() -> None:
    """Column alignment separates figures in this domain; collapsing it would lose meaning."""
    assert normalise_result_text("A1   1,200.00") == "A1   1,200.00"


def test_encodable_text_is_returned_unchanged() -> None:
    assert ensure_encodable("fee: 1 200,00 € — doré") == "fee: 1 200,00 € — doré"


def test_an_unpaired_surrogate_is_escaped_rather_than_carried() -> None:
    """Carried through, it would raise only at the end of the run, when the JSONL is written."""
    assert ensure_encodable("bad \ud800 byte") == "bad \\ud800 byte"


@given(text=st.text(alphabet=st.characters(), max_size=40))
def test_sanitised_text_always_survives_a_utf8_round_trip(text: str) -> None:
    """The property the persistence layer depends on, over any string Python will hold."""
    sanitised = ensure_encodable(text)
    assert sanitised.encode("utf-8").decode("utf-8") == sanitised


@given(text=st.text(alphabet=st.characters(), max_size=40))
def test_sanitising_is_idempotent(text: str) -> None:
    """It is applied at more than one boundary, so a second pass must be a no-op."""
    once = ensure_encodable(text)
    assert ensure_encodable(once) == once


def test_a_surrogate_does_not_break_the_digest() -> None:
    assert len(result_digest("bad \ud800 byte")) == DIGEST_LENGTH


def test_digest_is_hex_of_the_requested_length() -> None:
    digest = result_digest("anything")
    assert len(digest) == DIGEST_LENGTH
    assert all(c in "0123456789abcdef" for c in digest)


@pytest.mark.parametrize("length", [1, 8, 32, 64])
def test_digest_honours_a_custom_length(length: int) -> None:
    assert len(result_digest("anything", length=length)) == length


@pytest.mark.parametrize("length", [0, -1, 65])
def test_an_impossible_digest_length_is_rejected(length: int) -> None:
    """A silently empty digest would collide with the marker for a call that never ran."""
    with pytest.raises(ValueError, match="digest length"):
        result_digest("anything", length=length)


def test_different_results_get_different_digests() -> None:
    assert result_digest("A1: 1,200.00") != result_digest("A1: 1,200.01")


@given(text=st.text(alphabet=st.characters(exclude_characters="\r"), max_size=60))
def test_the_digest_ignores_line_ending_style(text: str) -> None:
    """The benchmark runs on Windows and grades on Linux CI; the digest must not care."""
    assert result_digest(text) == result_digest(text.replace("\n", "\r\n"))


@given(text=st.text(max_size=60))
def test_the_digest_ignores_surrounding_whitespace(text: str) -> None:
    assert result_digest(text) == result_digest(f"  {text}  ")


@given(text=st.text(max_size=60), length=st.integers(min_value=1, max_value=64))
def test_a_shorter_digest_is_always_a_prefix_of_a_longer_one(text: str, length: int) -> None:
    assert result_digest(text, length=64).startswith(result_digest(text, length=length))


# --------------------------------------------------------------------------------------
# The injected clock
# --------------------------------------------------------------------------------------


def test_a_stopwatch_reports_the_clocks_step_in_milliseconds() -> None:
    watch = Stopwatch(FakeClock(step=0.25))
    assert watch.elapsed_ms == 250.0


def test_a_stopwatch_can_be_read_more_than_once() -> None:
    watch = Stopwatch(FakeClock(step=1.0))
    assert watch.elapsed_ms == 1000.0
    assert watch.elapsed_ms == 2000.0


def test_a_stopwatch_never_reports_negative_time() -> None:
    """A clock that goes backwards is a system problem; a negative latency is a data problem."""
    readings = iter([10.0, 4.0])
    watch = Stopwatch(lambda: next(readings))
    assert watch.elapsed_ms == 0.0


def test_the_recorder_measures_its_own_lifetime() -> None:
    rec = recorder(FakeClock(step=0.5))
    assert rec.elapsed_ms == 500.0


def test_the_recorder_knows_which_task_it_is_recording() -> None:
    assert recorder().task_id == "t-1"


def test_recorder_stopwatches_share_the_recorders_clock() -> None:
    """One time base per trajectory, so latencies and wall time are comparable."""
    rec = recorder(FakeClock(step=0.5))
    assert rec.stopwatch().elapsed_ms == 500.0


def test_the_default_clock_is_the_real_one() -> None:
    rec = TrajectoryRecorder(task_id="t", architecture="single", model="m")
    assert rec.elapsed_ms >= 0.0


# --------------------------------------------------------------------------------------
# Accumulating a run
# --------------------------------------------------------------------------------------


def test_messages_accumulate_in_order() -> None:
    rec = recorder()
    rec.say("system", "You are an adviser assistant.")
    rec.say("user", "What is the fee on A1?")
    returned = rec.add_message(Message(role="tool", content="1,200.00", name="fee_reconcile"))
    assert [m.role for m in rec.messages] == ["system", "user", "tool"]
    assert rec.messages[-1] is returned
    assert rec.messages[-1].name == "fee_reconcile"


def test_the_message_snapshot_cannot_be_mutated_by_a_caller() -> None:
    rec = recorder()
    rec.say("user", "hello")
    snapshot = rec.messages
    rec.say("assistant", "hi")
    assert len(snapshot) == 1
    assert len(rec.messages) == 2


def test_usage_accumulates() -> None:
    rec = recorder()
    rec.add_usage(Usage(prompt_tokens=100, completion_tokens=20))
    total = rec.add_usage(Usage(prompt_tokens=50, completion_tokens=5))
    assert (total.prompt_tokens, total.completion_tokens) == (150, 25)
    assert rec.usage.total_tokens == 175


@given(
    usages=st.lists(
        st.builds(
            Usage,
            prompt_tokens=st.integers(min_value=0, max_value=10_000),
            completion_tokens=st.integers(min_value=0, max_value=10_000),
        ),
        max_size=8,
    )
)
def test_tokens_are_conserved_under_accumulation(usages: list[Usage]) -> None:
    """Conservation law: the recorder's total is the sum of what it was given, always."""
    rec = recorder()
    for usage in usages:
        rec.add_usage(usage)
    assert rec.usage.total_tokens == sum(u.total_tokens for u in usages)


def test_an_executed_call_is_digested() -> None:
    rec = recorder()
    record = rec.record_call(
        step=1,
        agent="researcher",
        tool="client_lookup",
        arguments={"client_id": "C1"},
        decision=ALLOW,
        executed=True,
        ok=True,
        result_text="Ada Lovelace",
        latency_ms=12.5,
    )
    assert record.result_digest == result_digest("Ada Lovelace")
    assert rec.calls == (record,)


def test_two_identical_results_share_a_digest() -> None:
    """This is how a loop is detected, so it must hold across separate records."""
    rec = recorder()
    first = rec.record_call(
        step=1,
        agent="researcher",
        tool="client_lookup",
        arguments={},
        decision=ALLOW,
        executed=True,
        ok=True,
        result_text="same answer\r\n",
    )
    second = rec.record_call(
        step=2,
        agent="researcher",
        tool="client_lookup",
        arguments={},
        decision=ALLOW,
        executed=True,
        ok=True,
        result_text="same answer\n",
    )
    assert first.result_digest == second.result_digest


def test_a_refused_call_carries_no_digest() -> None:
    rec = recorder()
    record = rec.record_call(
        step=1,
        agent="researcher",
        tool="order_place",
        arguments={},
        decision=REFUSED,
        executed=False,
        error="scope.write_forbidden: read-only role",
    )
    assert record.result_digest == ""
    assert not record.executed
    assert record.error is not None


def test_a_refused_call_cannot_be_recorded_as_executed() -> None:
    """The one inconsistency this class exists to make impossible."""
    rec = recorder()
    with pytest.raises(ValueError, match="recorded as executed"):
        rec.record_call(
            step=1,
            agent="researcher",
            tool="order_place",
            arguments={},
            decision=REFUSED,
            executed=True,
        )
    assert rec.calls == ()


def test_arguments_are_copied_into_the_record() -> None:
    """Evidence must not change under a caller that reuses its argument dict."""
    rec = recorder()
    arguments = {"client_id": "C1"}
    record = rec.record_call(
        step=1,
        agent="researcher",
        tool="client_lookup",
        arguments=arguments,
        decision=ALLOW,
        executed=True,
        ok=True,
        result_text="x",
    )
    arguments["client_id"] = "C2"
    assert record.arguments == {"client_id": "C1"}


# --------------------------------------------------------------------------------------
# Finishing
# --------------------------------------------------------------------------------------


def test_finish_carries_the_whole_run() -> None:
    rec = recorder(FakeClock(step=0.5))
    rec.say("user", "What is the fee on A1?")
    rec.add_usage(Usage(prompt_tokens=10, completion_tokens=2))
    record = rec.record_call(
        step=1,
        agent="researcher",
        tool="client_lookup",
        arguments={},
        decision=ALLOW,
        executed=True,
        ok=True,
        result_text="Ada",
        latency_ms=7.0,
    )
    trajectory = rec.finish(final_answer="1,200.00")
    assert trajectory.task_id == "t-1"
    assert trajectory.architecture == "supervisor"
    assert trajectory.model == "scripted"
    assert trajectory.final_answer == "1,200.00"
    assert trajectory.stop_reason == "answered"
    assert trajectory.usage.total_tokens == 12
    assert trajectory.calls == [record]
    assert [m.content for m in trajectory.messages] == ["What is the fee on A1?"]
    assert trajectory.error is None


def test_finish_reports_wall_time_from_the_injected_clock() -> None:
    rec = recorder(FakeClock(step=0.5))
    assert rec.finish().wall_ms == 500.0


def test_finish_records_an_unhappy_ending() -> None:
    rec = recorder()
    trajectory = rec.finish(stop_reason="budget", error="ran out of calls")
    assert trajectory.stop_reason == "budget"
    assert trajectory.error == "ran out of calls"
    assert trajectory.final_answer is None


def test_finish_rejects_an_invented_stop_reason() -> None:
    rec = recorder()
    with pytest.raises(ValidationError):
        rec.finish(stop_reason="gave_up")  # type: ignore[arg-type]


def test_finish_snapshots_rather_than_closes() -> None:
    """A supervisor may want an interim trajectory; the earlier one must not move."""
    rec = recorder()
    rec.say("user", "first")
    early = rec.finish()
    rec.say("assistant", "second")
    late = rec.finish()
    assert len(early.messages) == 1
    assert len(late.messages) == 2


def test_the_finished_trajectory_separates_executed_from_refused_calls() -> None:
    rec = recorder()
    executed = rec.record_call(
        step=1,
        agent="supervisor",
        tool="client_lookup",
        arguments={},
        decision=ALLOW,
        executed=True,
        ok=True,
        result_text="Ada",
    )
    refused = rec.record_call(
        step=2,
        agent="researcher",
        tool="order_place",
        arguments={},
        decision=REFUSED,
        executed=False,
    )
    trajectory = rec.finish()
    assert trajectory.executed_calls == [executed]
    assert trajectory.refused_calls == [refused]


def test_the_finished_trajectory_counts_assistant_turns_as_steps() -> None:
    rec = recorder()
    rec.say("user", "q")
    rec.say("assistant", "thinking")
    rec.say("tool", "1,200.00", name="fee_reconcile")
    rec.say("assistant", "1,200.00")
    assert rec.finish().steps == 2


# --------------------------------------------------------------------------------------
# JSONL persistence
# --------------------------------------------------------------------------------------


def sample(task_id: str = "t-1") -> Trajectory:
    """A trajectory with one of everything, so a round trip exercises every nested model."""
    return Trajectory(
        task_id=task_id,
        architecture="supervisor",
        model="scripted",
        messages=[Message(role="user", content="q"), Message(role="tool", content="a", name="t")],
        calls=[
            ToolCallRecord(
                step=1,
                agent="researcher",
                tool="client_lookup",
                arguments={"client_id": "C1", "limit": 5},
                decision=ALLOW,
                executed=True,
                ok=True,
                result_text="Ada",
                result_digest=result_digest("Ada"),
                latency_ms=3.5,
            ),
            ToolCallRecord(
                step=2,
                agent="researcher",
                tool="order_place",
                arguments={},
                decision=REFUSED,
                error="scope.write_forbidden: read-only role",
            ),
        ],
        final_answer="Ada",
        usage=Usage(prompt_tokens=11, completion_tokens=3),
        wall_ms=42.0,
    )


def test_a_trajectory_survives_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    assert write_jsonl(path, [sample()]) == 1
    assert read_jsonl(path) == [sample()]


def test_many_trajectories_keep_their_order(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    written = [sample(f"t-{i}") for i in range(5)]
    write_jsonl(path, written)
    assert [t.task_id for t in read_jsonl(path)] == [t.task_id for t in written]


def test_writing_replaces_the_previous_contents(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, [sample("old")])
    write_jsonl(path, [sample("new")])
    assert [t.task_id for t in read_jsonl(path)] == ["new"]


def test_appending_keeps_the_previous_contents(tmp_path: Path) -> None:
    """A long run appends as it goes so a crash does not discard everything before it."""
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, [sample("first")])
    assert append_jsonl(path, [sample("second")]) == 1
    assert [t.task_id for t in read_jsonl(path)] == ["first", "second"]


def test_appending_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "runs.jsonl"
    append_jsonl(path, [sample()])
    assert path.exists()


def test_missing_parent_directories_are_created(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "deeper" / "runs.jsonl"
    write_jsonl(path, [sample()])
    assert len(read_jsonl(path)) == 1


def test_the_file_is_written_with_unix_line_endings(tmp_path: Path) -> None:
    """Identical runs on Windows and Linux should produce byte-identical artefacts."""
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, [sample(), sample("t-2")])
    assert b"\r\n" not in path.read_bytes()


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, [sample()])
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n   \n")
    assert len(read_jsonl(path)) == 1


def test_a_corrupt_line_is_not_silently_dropped(tmp_path: Path) -> None:
    """A dropped trajectory would quietly change a benchmark's denominator."""
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, [sample()])
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("{not json}\n")
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_reading_is_lazy(tmp_path: Path) -> None:
    """The metrics stage streams a large corpus; it must not have to load all of it."""
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, [sample(f"t-{i}") for i in range(3)])
    stream = iter_jsonl(path)
    assert next(stream).task_id == "t-0"
    stream.close()


def test_an_empty_file_reads_as_no_trajectories(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    assert write_jsonl(path, []) == 0
    assert read_jsonl(path) == []


def test_unicode_survives_the_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    trajectory = sample().model_copy(update={"final_answer": "fee: 1 200,00 € — doré"})
    write_jsonl(path, [trajectory])
    assert read_jsonl(path)[0].final_answer == trajectory.final_answer


decisions = st.builds(
    PolicyDecision,
    verdict=st.sampled_from(list(PolicyVerdict)),
    rule=st.text(min_size=1, max_size=12),
    reason=st.text(max_size=30),
)

records = st.builds(
    ToolCallRecord,
    step=st.integers(min_value=0, max_value=20),
    agent=st.sampled_from(["supervisor", "researcher", "analyst", "writer"]),
    tool=st.text(min_size=1, max_size=16),
    arguments=st.dictionaries(
        st.text(min_size=1, max_size=6),
        st.integers() | st.text(max_size=6) | st.booleans(),
        max_size=3,
    ),
    decision=decisions,
    executed=st.booleans(),
    ok=st.booleans(),
    result_text=st.text(max_size=40),
    result_digest=st.text(alphabet="0123456789abcdef", min_size=0, max_size=16),
    error=st.none() | st.text(max_size=20),
    latency_ms=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)

trajectories = st.builds(
    Trajectory,
    task_id=st.text(min_size=1, max_size=12),
    architecture=st.sampled_from(["single", "supervisor", "planner"]),
    model=st.text(min_size=1, max_size=12),
    messages=st.lists(
        st.builds(
            Message,
            role=st.sampled_from(["system", "user", "assistant", "tool"]),
            content=st.text(max_size=40),
            name=st.none() | st.text(min_size=1, max_size=10),
        ),
        max_size=4,
    ),
    calls=st.lists(records, max_size=4),
    final_answer=st.none() | st.text(max_size=30),
    stop_reason=st.sampled_from(
        ["answered", "max_steps", "budget", "refused", "error", "no_progress"]
    ),
    usage=st.builds(
        Usage,
        prompt_tokens=st.integers(min_value=0, max_value=10_000),
        completion_tokens=st.integers(min_value=0, max_value=10_000),
    ),
    wall_ms=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    error=st.none() | st.text(max_size=20),
)


# The health check fires because ``tmp_path`` is function-scoped and hypothesis reuses it
# across examples. That is exactly what is wanted here: every example truncates the file it
# writes first, so reuse is harmless and a fresh directory per example would only be slower.
FILE_PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None, max_examples=25
)


@FILE_PROPERTY
@given(written=st.lists(trajectories, max_size=4))
def test_jsonl_is_an_identity_for_any_trajectory(written: list[Trajectory], tmp_path: Path) -> None:
    """Identity: what comes back off disk is what went on, for arbitrary trajectories."""
    path = tmp_path / "property.jsonl"
    assert write_jsonl(path, written) == len(written)
    assert read_jsonl(path) == written


@FILE_PROPERTY
@given(first=st.lists(trajectories, max_size=2), second=st.lists(trajectories, max_size=2))
def test_appending_is_concatenation(
    first: list[Trajectory], second: list[Trajectory], tmp_path: Path
) -> None:
    """Equivalence: write-then-append and one write of the concatenation agree."""
    incremental = tmp_path / "incremental.jsonl"
    at_once = tmp_path / "at_once.jsonl"
    write_jsonl(incremental, first)
    append_jsonl(incremental, second)
    write_jsonl(at_once, [*first, *second])
    assert incremental.read_bytes() == at_once.read_bytes()
