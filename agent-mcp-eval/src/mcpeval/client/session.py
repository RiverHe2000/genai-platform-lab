"""The MCP client layer: the boundary every tool call crosses on its way out.

Three jobs live here, and they are deliberately three objects rather than one.

:class:`ToolClient` speaks MCP and nothing else. It discovers the server's inventory once,
projects it into :class:`~mcpeval.schemas.ToolSpec`, and calls a tool with a timeout. Its
single most important property is that it never raises for anything the server or the
transport does: a broken pipe, a validation error, an unknown tool and a timeout all come
back as the same :class:`ToolResult`. An agent that has to reason about exceptions will
reason about them badly --- the useful measurement is what the agent does with a failed
tool, and that measurement is destroyed if the failure unwinds the run instead.

:class:`GuardedToolClient` puts the permission policy in front of that transport. The
ordering is the whole point: the policy is asked first, and when it refuses, the transport
is not touched at all. A guard that calls the tool and then decides whether the agent was
allowed to has already performed the write it was meant to prevent.

The two ``connect_*`` helpers exist because the benchmark and the tests need different
transports for the same server. ``connect_in_process`` runs the server in this event loop
over in-memory streams: a full benchmark makes thousands of calls, and paying for a
subprocess round trip per call would make the run's wall time a measurement of the
operating system rather than of the agent. ``connect_stdio`` is the real, out-of-process
thing, kept so that the in-process path is never the only one that has ever worked.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Final, NamedTuple, Protocol

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError
from mcp.shared.memory import create_client_server_memory_streams
from mcp.types import (
    REQUEST_TIMEOUT,
    CallToolResult,
    ContentBlock,
    ListToolsResult,
    TextContent,
    Tool,
)

from mcpeval.client.policy import BudgetState, PermissionPolicy
from mcpeval.client.recorder import TrajectoryRecorder, ensure_encodable
from mcpeval.schemas import ToolCallRecord, ToolSpec

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from mcp.server.mcpserver import MCPServer

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "GuardedToolClient",
    "MCPToolSession",
    "ToolClient",
    "ToolResult",
    "connect_in_process",
    "connect_stdio",
    "project_tool",
]

#: Long enough for any tool in this server (they all read an in-memory world), short enough
#: that a wedged call cannot consume a benchmark run's wall-clock budget on its own.
DEFAULT_TIMEOUT_S: Final = 15.0


class ToolResult(NamedTuple):
    """The normalised outcome of one tool call.

    A tuple rather than a model because it is a return value, not a stored fact: what gets
    stored is the :class:`~mcpeval.schemas.ToolCallRecord` the recorder builds from it.
    ``ok`` is false for every kind of failure --- transport, timeout, tool error --- and
    ``text`` then carries the message the agent will see, so a caller never has to ask
    which of several failure shapes it is holding.
    """

    ok: bool
    text: str
    structured: dict[str, Any] | None


class MCPToolSession(Protocol):
    """The slice of :class:`mcp.ClientSession` this module actually uses.

    Narrowing the dependency to two methods lets the guard and the normalisation be tested
    against a stub that returns a specific failure on demand. Timeouts and broken
    transports are the paths most likely to be wrong and the hardest to provoke against a
    real server, so they are exactly the paths that must not need one.
    """

    async def list_tools(self) -> ListToolsResult:
        """Return the server's advertised inventory."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> object:
        """Invoke one tool. Typed as ``object`` because newer SDKs may answer with an
        elicitation rather than a result, and this client narrows at runtime."""
        ...


def project_tool(tool: Tool) -> ToolSpec:
    """Project one advertised MCP tool into the agent-facing :class:`ToolSpec`.

    The mapping of the annotations is the interesting part, and it follows the MCP
    specification's defaults rather than the friendlier reading. An absent ``read_only_hint``
    means *not* read-only, and an absent ``destructive_hint`` on a writing tool means
    destructive. Defaulting the other way would be a fail-open: the policy derives its write
    list from ``read_only`` (see :meth:`PermissionPolicy.with_tools`), so a server that
    forgot to annotate ``place_order`` would have it handed to every read-only role. The
    chosen defaults make that mistake fail loudly --- an unannotated read tool gets refused
    --- which is the direction an evaluation harness for permissioning should fail in.

    ``requires_approval`` is deliberately left at its default: whether a call needs a human
    is a property of the deployment, and letting the server assert it would let the thing
    being evaluated set its own gate.

    Args:
        tool: A tool as returned by ``session.list_tools()``.

    Returns:
        The projected spec.
    """
    annotations = tool.annotations
    read_only = bool(annotations is not None and annotations.read_only_hint)
    if read_only:
        destructive = False
    else:
        hint = None if annotations is None else annotations.destructive_hint
        destructive = True if hint is None else bool(hint)
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        input_schema=dict(tool.input_schema),
        read_only=read_only,
        destructive=destructive,
    )


def _render_content(blocks: Sequence[ContentBlock]) -> str:
    """Flatten MCP content blocks into the text an agent will read.

    Non-text blocks are reduced to a bracketed type marker rather than dropped. An agent
    that received an image and was shown nothing would be graded as having ignored evidence
    it was never given, and a marker keeps the transcript honest about what arrived.

    This is where text from the server first enters the process, so it is also where it is
    made UTF-8 encodable. Sanitising here rather than in the recorder covers every copy that
    follows --- the prompt the agent sees, the message appended to the transcript, and the
    recorded result --- instead of only the last of them.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, TextContent):
            parts.append(block.text)
        else:
            parts.append(f"[{block.type}]")
    return ensure_encodable("\n".join(parts))


class ToolClient:
    """An async wrapper over an MCP session that never raises at the agent.

    Discovery is cached after the first call. The inventory of a benchmark server does not
    change mid-run, and re-listing on every step would add a round trip per step to every
    architecture equally --- which sounds harmless until the thing being compared is how
    many round trips each architecture needs.
    """

    def __init__(self, session: MCPToolSession, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Wrap a live, initialised session.

        Args:
            session: An MCP client session on which ``initialize()`` has already run.
            timeout_s: Per-call ceiling, passed to the SDK so that a timeout also sends the
                server a cancellation rather than merely abandoning the request here.

        Raises:
            ValueError: If the timeout is not positive; a zero or negative ceiling would
                make every call time out and look like a broken server.
        """
        if timeout_s <= 0:
            msg = f"timeout_s must be positive, got {timeout_s}"
            raise ValueError(msg)
        self._session = session
        self._timeout_s = timeout_s
        self._specs: tuple[ToolSpec, ...] | None = None

    @property
    def timeout_s(self) -> float:
        """The per-call ceiling in seconds."""
        return self._timeout_s

    async def discover(self, *, refresh: bool = False) -> tuple[ToolSpec, ...]:
        """List the server's tools, projecting each into a :class:`ToolSpec`.

        Unlike :meth:`call`, this is allowed to raise. A failed handshake is a wiring
        fault, not agent behaviour: there is no trajectory worth recording for a run whose
        agent was never told which tools exist, and swallowing it would produce a run in
        which every call is refused as an unknown tool.

        Args:
            refresh: Discard the cache and ask the server again.

        Returns:
            The advertised tools, in the order the server listed them.
        """
        if self._specs is None or refresh:
            listing = await self._session.list_tools()
            self._specs = tuple(project_tool(tool) for tool in listing.tools)
        return self._specs

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """The cached inventory, empty until :meth:`discover` has run."""
        return self._specs or ()

    def spec(self, name: str) -> ToolSpec | None:
        """The cached spec for one tool, or None if it was never advertised."""
        return next((s for s in self.specs if s.name == name), None)

    async def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        """Call one tool and normalise everything that can go wrong into a result.

        Four failure shapes are collapsed here: a tool that reported an error, a timeout, a
        transport or protocol error, and an unexpected response type. All four come back as
        ``ok=False`` with a human-readable ``text``, so the agent's prompt template has one
        case to handle and the recorder has one thing to digest.

        The unexpected-response case is not defensive padding: this SDK's ``call_tool`` can
        answer with an elicitation asking a human for more input, and this client has no
        human attached. Treating that as a failure is both honest and safe; treating it as
        a result would hand the agent an empty answer it might report as a fact.

        Args:
            tool: The tool name, exactly as the agent asked for it.
            arguments: The arguments to send; None is sent as an empty object.

        Returns:
            The normalised outcome. Never raises for server-side or transport failure.
        """
        payload = dict(arguments or {})
        try:
            raw = await self._session.call_tool(tool, payload, read_timeout_seconds=self._timeout_s)
        except MCPError as exc:
            if exc.code == REQUEST_TIMEOUT:
                return ToolResult(
                    False, f"Tool {tool!r} timed out after {self._timeout_s:g}s", None
                )
            return ToolResult(False, f"Tool {tool!r} failed: {exc}", None)
        except TimeoutError:
            # The transport's own bounded write, not the request timeout above; the SDK
            # propagates it raw, and to an agent it is the same event.
            return ToolResult(False, f"Tool {tool!r} timed out after {self._timeout_s:g}s", None)
        except Exception as exc:
            # Broad by design: naming the transport's exception types here would make this
            # class fail open the first time the SDK grew a new one.
            return ToolResult(False, f"Tool {tool!r} failed: {type(exc).__name__}: {exc}", None)

        if not isinstance(raw, CallToolResult):
            return ToolResult(
                False,
                f"Tool {tool!r} returned an unsupported response ({type(raw).__name__})",
                None,
            )
        text = _render_content(raw.content)
        structured = raw.structured_content if isinstance(raw.structured_content, dict) else None
        if raw.is_error:
            return ToolResult(False, text or f"Tool {tool!r} reported an error", structured)
        return ToolResult(True, text, structured)


class GuardedToolClient:
    """A :class:`ToolClient` with the permission policy in front and the recorder behind.

    This is the object an agent is given. It cannot be talked past: the policy rules on
    every attempt before the transport is reachable, and every attempt --- admitted or
    refused --- lands on the trajectory. Refusals matter as much as successes here. An
    agent that repeatedly reaches for a write tool it may not use has revealed something
    about its prompt or its planner, and that finding is invisible in a log that records
    only the calls that went out.

    Budgets are per role and are spent only by calls that actually executed, which keeps
    the policy's contract intact: a refused call costs nothing, so an agent cannot exhaust
    another role's headroom by proposing things it is not allowed to do.
    """

    def __init__(
        self,
        client: ToolClient,
        policy: PermissionPolicy,
        recorder: TrajectoryRecorder,
        *,
        budgets: Mapping[str, BudgetState] | None = None,
    ) -> None:
        """Compose the three layers.

        Args:
            client: The transport wrapper.
            policy: The permission policy; ask :meth:`PermissionPolicy.with_tools` to
                refresh it from :meth:`ToolClient.discover` before starting a run.
            recorder: Where every attempt is written.
            budgets: Pre-built tallies keyed by role, for a run that resumes or that shares
                one budget across sub-agents. Omitted, each role gets a fresh budget on
                first use.
        """
        self._client = client
        self._policy = policy
        self._recorder = recorder
        self._budgets: dict[str, BudgetState] = dict(budgets or {})

    @property
    def policy(self) -> PermissionPolicy:
        """The policy in force."""
        return self._policy

    @property
    def recorder(self) -> TrajectoryRecorder:
        """The recorder every attempt is written to."""
        return self._recorder

    def budget_for(self, role: str) -> BudgetState:
        """The live tally for one role, created on first use.

        A role with no policy entry gets a zero-headroom tally rather than an error.
        :meth:`PermissionPolicy.decide` refuses an unknown role before it ever reads the
        budget, so the value is only reachable if that ordering is ever changed --- and if
        it is, a budget of zero refuses, which is the safe direction to be wrong in.

        Args:
            role: The calling agent's role name.

        Returns:
            The mutable tally, the same object on every call for that role.
        """
        existing = self._budgets.get(role)
        if existing is not None:
            return existing
        try:
            fresh = self._policy.new_budget(role)
        except KeyError:
            fresh = BudgetState(max_steps=0, max_calls=0, max_tokens=0, max_wall_ms=0.0)
        self._budgets[role] = fresh
        return fresh

    def spend_step(self, role: str, n: int = 1) -> BudgetState:
        """Charge ``n`` model turns to a role's budget and return it."""
        budget = self.budget_for(role)
        budget.spend_step(n)
        return budget

    def spend_tokens(self, role: str, n: int) -> BudgetState:
        """Charge ``n`` tokens to a role's budget and return it."""
        budget = self.budget_for(role)
        budget.spend_tokens(n)
        return budget

    async def call(
        self,
        role: str,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        step: int,
        approved: bool = False,
    ) -> ToolCallRecord:
        """Rule on a proposed call, execute it if it is admitted, and record the attempt.

        The record is returned rather than the raw result because the record is what the
        run is graded on: it carries the verdict that admitted the call alongside the text
        it produced. Structured content is intentionally not returned. Only the text is
        stored on the trajectory, and an agent allowed to answer from JSON that the
        trajectory does not contain would be graded on evidence no one can audit.

        Args:
            role: The calling agent's role name.
            tool: The proposed tool name, recorded verbatim even when no such tool exists.
            arguments: The proposed arguments.
            step: The agent turn this attempt belongs to.
            approved: Whether a human approval has been recorded for this call. Explicit,
                because a default of "approved" is how approval gates quietly stop working.

        Returns:
            The stored record for this attempt.
        """
        payload = dict(arguments or {})
        budget = self.budget_for(role)
        decision = self._policy.decide(role, tool, payload, budget=budget, approved=approved)
        watch = self._recorder.stopwatch()

        if not decision.allowed:
            return self._recorder.record_call(
                step=step,
                agent=role,
                tool=tool,
                arguments=payload,
                decision=decision,
                executed=False,
                ok=False,
                result_text="",
                error=f"{decision.rule}: {decision.reason}",
                latency_ms=watch.elapsed_ms,
            )

        result = await self._client.call(tool, payload)
        latency_ms = watch.elapsed_ms
        budget.spend_call()
        budget.spend_wall_ms(latency_ms)
        return self._recorder.record_call(
            step=step,
            agent=role,
            tool=tool,
            arguments=payload,
            decision=decision,
            executed=True,
            ok=result.ok,
            result_text=result.text,
            error=None if result.ok else result.text,
            latency_ms=latency_ms,
        )


def _sole_exception(group: BaseExceptionGroup[BaseException]) -> BaseException | None:
    """Return the single exception a group wraps, or None if it wraps more than one.

    ``ClientSession`` runs its reader in an anyio task group, and ``stdio_client`` adds a
    second one around it, so an exception raised in the *caller's* ``async with`` body comes
    back out wrapped in a ``BaseExceptionGroup`` --- twice, over stdio. That turns
    ``except TimeoutError`` in a benchmark runner into a silent non-match, which is a nasty
    way to lose an error.

    The descent stops the moment a level holds anything but one exception. A group with
    several members is genuine concurrent failure, and choosing one of them to re-raise
    would throw the others away, so such a group is handed back whole.

    Args:
        group: The group an ``async with`` body's failure arrived in.

    Returns:
        The one exception nested inside, or None if the group should propagate as it is.
    """
    inner: BaseException = group
    while isinstance(inner, BaseExceptionGroup):
        if len(inner.exceptions) != 1:
            return None
        inner = inner.exceptions[0]
    return inner


@asynccontextmanager
async def connect_in_process(
    server: MCPServer,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> AsyncIterator[ToolClient]:
    """Run an MCP server in this event loop and yield a client wired to it.

    The server task is cancelled on exit and its cancellation is absorbed, because by then
    the client session has already closed its half of the streams: whatever the server task
    raises while noticing that is a teardown artefact, not a test result, and letting it
    escape would fail runs for a reason unrelated to what they were measuring.

    Args:
        server: The ``MCPServer`` to run. Its private low-level server is used because that
            is the object exposing the stream-level ``run`` loop this transport needs.
        timeout_s: Per-call ceiling for the yielded client.

    Yields:
        A client whose session has completed the MCP handshake.

    Raises:
        BaseExceptionGroup: If teardown fails in more than one way at once; a single
            failure is unwrapped and re-raised as itself (see :func:`_sole_exception`).
    """
    lowlevel = server._lowlevel_server
    init_opts = lowlevel.create_initialization_options()
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        task = asyncio.create_task(
            lowlevel.run(server_read, server_write, init_opts, raise_exceptions=False)
        )
        try:
            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield ToolClient(session, timeout_s=timeout_s)
        except BaseExceptionGroup as group:
            sole = _sole_exception(group)
            if sole is None:
                raise
            raise sole from None
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


@asynccontextmanager
async def connect_stdio(
    command: str,
    args: Sequence[str] = (),
    *,
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> AsyncIterator[ToolClient]:
    """Launch an MCP server as a subprocess over stdio and yield a client wired to it.

    Kept alongside the in-process transport because they can diverge: an in-process server
    shares this process's imports, working directory and event loop, so a tool that depends
    on any of those can pass in-process and fail the moment the server is a real
    subprocess. The benchmark uses the fast path; this one proves the fast path is not the
    only one that works.

    Args:
        command: The executable, typically ``sys.executable``.
        args: Its arguments, e.g. ``["-m", "mcpeval.mcp_server.server"]``.
        env: Environment for the child; None inherits the SDK's default.
        cwd: Working directory for the child.
        timeout_s: Per-call ceiling for the yielded client.

    Yields:
        A client whose session has completed the MCP handshake.

    Raises:
        BaseExceptionGroup: If teardown fails in more than one way at once; a single
            failure is unwrapped and re-raised as itself (see :func:`_sole_exception`).
    """
    params = StdioServerParameters(
        command=command,
        args=list(args),
        env=dict(env) if env is not None else None,
        cwd=cwd,
    )
    try:
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            yield ToolClient(session, timeout_s=timeout_s)
    except BaseExceptionGroup as group:
        sole = _sole_exception(group)
        if sole is None:
            raise
        raise sole from None
