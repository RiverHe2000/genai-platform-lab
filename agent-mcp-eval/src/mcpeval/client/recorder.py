"""The trajectory recorder: the only writer of the evidence a run is graded on.

A benchmark that keeps just the final answer cannot tell a lucky guess from a sound
process. So every attempted tool call --- executed or refused --- every message and every
token is accumulated here, and the finished :class:`~mcpeval.schemas.Trajectory` is the
whole of what the grader sees. Making one object the sole writer means the invariants that
matter to grading are enforceable in one place: a refused call can never be marked
executed, and a record's latency always comes from the same clock as every other record's.

Two shaping decisions are worth stating up front.

*Time is injected.* The recorder never reads the wall clock directly; it calls a supplied
``clock``. Latency and ``wall_ms`` are otherwise the one part of a trajectory that changes
between two runs of the same deterministic task, which would make trajectories
unassertable in tests and undiffable between architectures. Tests pass a counter; the
benchmark passes :func:`time.perf_counter`.

*Results are digested, not just stored.* Two calls that returned the same thing are the
signature of an agent looping, and comparing whole result texts to spot that is both slow
and fragile. A short digest over the *normalised* text --- line endings folded, edges
trimmed --- makes the comparison cheap and, importantly, identical on Windows and on Linux
CI, where the same tool output can differ by a carriage return.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Generator, Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from mcpeval.schemas import (
    Message,
    PolicyDecision,
    Role,
    StopReason,
    ToolCallRecord,
    Trajectory,
    Usage,
)

__all__ = [
    "DIGEST_LENGTH",
    "Clock",
    "Stopwatch",
    "TrajectoryRecorder",
    "append_jsonl",
    "ensure_encodable",
    "iter_jsonl",
    "normalise_result_text",
    "read_jsonl",
    "result_digest",
    "write_jsonl",
]

#: A monotonic source of seconds. Seconds rather than milliseconds because that is what the
#: standard library offers; the conversion happens once, here, rather than at each caller.
Clock = Callable[[], float]

#: Hex characters kept from the sha256. Sixty-four bits is far more than enough to separate
#: the handful of distinct results one run produces, and the digest is carried on every
#: record, so keeping the full 64 characters would inflate a stored trajectory for no gain
#: in discriminating power.
DIGEST_LENGTH: Final = 16


def ensure_encodable(text: str) -> str:
    """Return text that is guaranteed to survive UTF-8 encoding.

    A JSON-RPC peer can hand back a string containing an unpaired surrogate. Python accepts
    it happily and then refuses to encode it, so the failure surfaces nowhere near its
    cause: the whole run's JSONL write raises at the end, after every attempt has been made
    and none can be saved. Escaping the offending code point keeps the evidence writable and
    keeps what actually arrived visible rather than replacing it with a question mark.

    The function is idempotent --- escaped text encodes cleanly and is returned unchanged
    --- so it is safe to apply at more than one boundary.

    Args:
        text: Any string, including one carrying unpaired surrogates.

    Returns:
        The same string, or one with the unencodable code points backslash-escaped.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "backslashreplace").decode("utf-8")
    return text


def normalise_result_text(text: str) -> str:
    """Fold the incidental differences out of a tool result before digesting it.

    Only differences that no grader would ever call meaningful are removed: line-ending
    style and leading or trailing whitespace. Interior whitespace is left alone, because in
    this domain it separates figures in a table and collapsing it could make two genuinely
    different results digest the same.

    Args:
        text: The result text exactly as the tool returned it.

    Returns:
        The text with CRLF and CR folded to LF, the edges trimmed, and any unpaired
        surrogate escaped so that digesting it cannot raise.
    """
    return ensure_encodable(text.replace("\r\n", "\n").replace("\r", "\n").strip())


def result_digest(text: str, *, length: int = DIGEST_LENGTH) -> str:
    """Return a short, stable fingerprint of a tool result.

    Args:
        text: The raw result text; it is normalised first, so a run on Windows and the same
            run on Linux CI produce the same digest.
        length: How many leading hex characters to keep.

    Returns:
        A lower-case hex string of exactly ``length`` characters.

    Raises:
        ValueError: If ``length`` is outside 1..64, which would silently produce an empty
            digest (colliding with the "not executed" marker) or a padded one.
    """
    if not 1 <= length <= 64:
        msg = f"digest length must be between 1 and 64 hex characters, got {length}"
        raise ValueError(msg)
    payload = normalise_result_text(text).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


class Stopwatch:
    """Elapsed time measured against an injected clock.

    It reads the clock lazily rather than storing an end time so that the same stopwatch can
    be read at several points of a call --- before and after a retry, say --- without the
    caller having to remember to stop it.
    """

    __slots__ = ("_clock", "_start")

    def __init__(self, clock: Clock) -> None:
        """Start the stopwatch.

        Args:
            clock: Returns monotonically non-decreasing seconds.
        """
        self._clock = clock
        self._start = clock()

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since construction, never negative."""
        return max(0.0, (self._clock() - self._start) * 1000.0)


class TrajectoryRecorder:
    """Accumulates one attempt at one task, and emits the finished trajectory.

    The recorder owns no policy and performs no I/O to the server: it is handed facts that
    have already happened. That keeps the enforcement point (the policy) and the evidence
    (this class) independent, so a bug in one cannot quietly rewrite the other --- a
    recorder that could veto a call would be a second, undocumented permission system.
    """

    def __init__(
        self,
        *,
        task_id: str,
        architecture: str,
        model: str,
        clock: Clock = time.perf_counter,
    ) -> None:
        """Begin recording.

        Args:
            task_id: Identifier of the benchmark task being attempted.
            architecture: The agent topology under test, e.g. ``supervisor``.
            model: The chat model's name, recorded so a trajectory is self-describing.
            clock: Monotonic seconds source; injected so tests are deterministic.
        """
        self._task_id = task_id
        self._architecture = architecture
        self._model = model
        self._clock = clock
        self._start = clock()
        self._messages: list[Message] = []
        self._calls: list[ToolCallRecord] = []
        self._usage = Usage()

    @property
    def task_id(self) -> str:
        """The task being attempted."""
        return self._task_id

    @property
    def messages(self) -> tuple[Message, ...]:
        """The transcript so far, as an immutable snapshot."""
        return tuple(self._messages)

    @property
    def calls(self) -> tuple[ToolCallRecord, ...]:
        """Every attempted call so far, refusals included, in attempt order."""
        return tuple(self._calls)

    @property
    def usage(self) -> Usage:
        """Tokens accumulated across every model call recorded so far."""
        return self._usage

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since the recorder was constructed."""
        return max(0.0, (self._clock() - self._start) * 1000.0)

    def stopwatch(self) -> Stopwatch:
        """Start a stopwatch on the recorder's clock.

        Callers time a tool call with this rather than with their own clock, so that every
        latency on a trajectory is measured on one time base and the sum of the latencies
        is comparable with ``wall_ms``.
        """
        return Stopwatch(self._clock)

    def add_message(self, message: Message) -> Message:
        """Append one turn to the transcript and return it unchanged."""
        self._messages.append(message)
        return message

    def say(self, role: Role, content: str, *, name: str | None = None) -> Message:
        """Append a turn without the caller constructing the :class:`Message`.

        Args:
            role: Who is speaking.
            content: The text of the turn.
            name: The tool name, when ``role`` is ``"tool"``.

        Returns:
            The appended message.
        """
        return self.add_message(Message(role=role, content=content, name=name))

    def add_usage(self, usage: Usage) -> Usage:
        """Add one model call's token counts to the running total and return the total."""
        self._usage = self._usage + usage
        return self._usage

    def record_call(
        self,
        *,
        step: int,
        agent: str,
        tool: str,
        arguments: Mapping[str, Any],
        decision: PolicyDecision,
        executed: bool,
        ok: bool = False,
        result_text: str = "",
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> ToolCallRecord:
        """Record one attempted tool call and return the stored record.

        The digest is computed only for calls that actually reached the server. A refused
        call has no result, and digesting its empty text would give every refusal in the
        corpus one shared fingerprint --- which a loop detector comparing digests would
        read as an agent repeating the same successful result.

        Args:
            step: The agent turn this attempt belongs to.
            agent: The calling role, e.g. ``researcher``.
            tool: The proposed tool name, exactly as the agent asked for it, even if no
                such tool exists.
            arguments: The proposed arguments; copied, so a caller reusing its dict cannot
                retroactively alter the evidence.
            decision: The policy ruling that admitted or refused this attempt.
            executed: Whether the call reached the transport.
            ok: Whether the server reported success.
            result_text: The normalised result text, or the error message on failure.
            error: A short failure description, or None when the call succeeded.
            latency_ms: Measured with :meth:`stopwatch`.

        Returns:
            The record as stored on the trajectory.

        Raises:
            ValueError: If a call is marked both refused and executed. That combination
                would mean the policy was consulted and then ignored, and it is the one
                inconsistency this class exists to make impossible.
        """
        if executed and not decision.allowed:
            msg = (
                f"call to {tool!r} was recorded as executed but the policy returned "
                f"{decision.verdict.value} ({decision.rule})"
            )
            raise ValueError(msg)
        record = ToolCallRecord(
            step=step,
            agent=agent,
            tool=tool,
            arguments=dict(arguments),
            decision=decision,
            executed=executed,
            ok=ok,
            result_text=result_text,
            result_digest=result_digest(result_text) if executed else "",
            error=error,
            latency_ms=latency_ms,
        )
        self._calls.append(record)
        return record

    def finish(
        self,
        *,
        final_answer: str | None = None,
        stop_reason: StopReason = "answered",
        error: str | None = None,
    ) -> Trajectory:
        """Snapshot everything recorded so far into a :class:`Trajectory`.

        This does not close the recorder. A run that ends in an exception wants a
        trajectory built from a partially filled recorder, and a supervisor loop may want
        an interim snapshot to grade a sub-agent; forbidding either would push callers into
        reaching for the private lists.

        Args:
            final_answer: The agent's answer, or None if it never produced one.
            stop_reason: Why the run ended.
            error: The failure that ended the run, if any.

        Returns:
            A trajectory holding copies of the accumulated messages and calls.
        """
        return Trajectory(
            task_id=self._task_id,
            architecture=self._architecture,
            model=self._model,
            messages=list(self._messages),
            calls=list(self._calls),
            final_answer=final_answer,
            stop_reason=stop_reason,
            usage=self._usage,
            wall_ms=self.elapsed_ms,
            error=error,
        )


def _dump_line(trajectory: Trajectory) -> str:
    """Serialise one trajectory to a single JSON line.

    ``model_dump_json`` is used rather than ``json.dumps`` over ``model_dump`` because
    pydantic already knows how to render the enums and nested models, and a hand-rolled
    encoder would be a second definition of the on-disk format.
    """
    return trajectory.model_dump_json()


def _write(path: Path, trajectories: Iterable[Trajectory], *, mode: str) -> int:
    """Write trajectories as JSONL, creating parent directories as needed.

    ``newline="\\n"`` is explicit because the default on Windows would translate every line
    ending to CRLF, which changes the file's checksum between platforms for identical runs
    and is exactly the kind of incidental difference the digests elsewhere in this module
    take pains to remove.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for trajectory in trajectories:
            handle.write(_dump_line(trajectory))
            handle.write("\n")
            written += 1
    return written


def write_jsonl(path: Path, trajectories: Iterable[Trajectory]) -> int:
    """Write trajectories to ``path``, replacing anything already there.

    Args:
        path: Destination file; parent directories are created.
        trajectories: What to write, in order.

    Returns:
        The number of trajectories written.
    """
    return _write(path, trajectories, mode="w")


def append_jsonl(path: Path, trajectories: Iterable[Trajectory]) -> int:
    """Append trajectories to ``path``, creating it if it does not exist.

    A long benchmark run appends each attempt as it finishes rather than holding every
    trajectory in memory to the end, so that a crash at attempt nine hundred does not
    discard the first eight hundred and ninety-nine.

    Args:
        path: Destination file; parent directories are created.
        trajectories: What to append, in order.

    Returns:
        The number of trajectories appended.
    """
    return _write(path, trajectories, mode="a")


def iter_jsonl(path: Path) -> Generator[Trajectory, None, None]:
    """Yield the trajectories in a JSONL file, one at a time.

    Blank lines are skipped so that a file written by one process and appended to by
    another --- or hand-edited --- still parses. Anything else that fails to parse raises,
    because a silently dropped trajectory would quietly change a benchmark's denominator.

    Typed as a generator rather than a plain iterator because it holds an open file: a
    caller that stops early should be able to ``close()`` it rather than wait for the
    garbage collector to get round to the handle.

    Args:
        path: The JSONL file to read.

    Yields:
        Each trajectory in file order.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield Trajectory.model_validate(json.loads(line))


def read_jsonl(path: Path) -> list[Trajectory]:
    """Read a whole JSONL file of trajectories into memory.

    Args:
        path: The JSONL file to read.

    Returns:
        The trajectories, in file order.
    """
    return list(iter_jsonl(path))
