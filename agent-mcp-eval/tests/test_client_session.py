"""Tests for the MCP client layer.

Two properties carry most of the weight here and are asserted repeatedly rather than once.

*Nothing reaches the transport that the policy refused.* It is not enough to check the
returned record: a guard could refuse and still have called the tool. So the stub session
counts invocations, and the refusal tests assert that count is zero --- the only assertion
that can tell a real gate from a decorative one.

*No failure escapes as an exception.* Every way a call can go wrong is provoked (tool
error, bad arguments, unknown tool, timeout, dead transport, unsupported response) and each
is asserted to arrive as an ordinary falsy result, because an agent that has to catch
exceptions cannot be measured on how it recovers from tool failure.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import Mapping
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INTERNAL_ERROR,
    REQUEST_TIMEOUT,
    CallToolResult,
    ImageContent,
    ListToolsResult,
    Result,
    TextContent,
    Tool,
    ToolAnnotations,
)

from mcpeval.client.policy import (
    RESEARCHER,
    SUPERVISOR,
    WRITER,
    BudgetState,
    default_policy,
)
from mcpeval.client.recorder import TrajectoryRecorder
from mcpeval.client.session import (
    DEFAULT_TIMEOUT_S,
    GuardedToolClient,
    MCPToolSession,
    ToolClient,
    ToolResult,
    _sole_exception,
    connect_in_process,
    connect_stdio,
    project_tool,
)
from mcpeval.schemas import PolicyVerdict

# --------------------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------------------


class FakeClock:
    """A clock that advances by a fixed step on every read.

    Injected everywhere a latency is asserted, so a test can state the exact millisecond
    figure it expects instead of asserting the empty tautology ``latency >= 0``.
    """

    def __init__(self, step: float = 0.25) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


OK_RESULT = CallToolResult(content=[TextContent(type="text", text="ok")], is_error=False)


class FakeSession:
    """A stub standing in for :class:`mcp.ClientSession`.

    It records what it was asked for, which is how the refusal tests prove the transport
    was never touched, and it can be told to raise, which is how the timeout and
    broken-transport paths are exercised without a real broken server.
    """

    def __init__(
        self,
        tools: tuple[Tool, ...] = (),
        *,
        outcome: object = OK_RESULT,
        outcomes: Mapping[str, object] | None = None,
        list_error: Exception | None = None,
    ) -> None:
        self.tools = list(tools)
        self.outcome = outcome
        self.outcomes = dict(outcomes or {})
        self.list_error = list_error
        self.list_count = 0
        self.calls: list[tuple[str, dict[str, Any], float | None]] = []

    async def list_tools(self) -> ListToolsResult:
        self.list_count += 1
        if self.list_error is not None:
            raise self.list_error
        return ListToolsResult(tools=list(self.tools))

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> object:
        self.calls.append((name, dict(arguments or {}), read_timeout_seconds))
        outcome = self.outcomes.get(name, self.outcome)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_tool(
    name: str = "client_lookup",
    *,
    description: str | None = "Fetch a client.",
    read_only: bool | None = True,
    destructive: bool | None = None,
    annotated: bool = True,
) -> Tool:
    """Build an advertised tool with exactly the annotation combination under test."""
    annotations = (
        ToolAnnotations(read_only_hint=read_only, destructive_hint=destructive)
        if annotated
        else None
    )
    return Tool(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {"client_id": {"type": "string"}}},
        annotations=annotations,
    )


def guarded(
    session: FakeSession,
    *,
    clock: FakeClock | None = None,
    budgets: Mapping[str, BudgetState] | None = None,
) -> tuple[GuardedToolClient, TrajectoryRecorder]:
    """Wire the three layers over a stub session, on a deterministic clock."""
    recorder = TrajectoryRecorder(
        task_id="t-1",
        architecture="supervisor",
        model="scripted",
        clock=clock or FakeClock(step=0.5),
    )
    client = ToolClient(session)
    return GuardedToolClient(client, default_policy(), recorder, budgets=budgets), recorder


@pytest.fixture
def echo_server() -> MCPServer:
    """A tiny real MCP server, used to prove the wrapper against the actual SDK.

    The stub session is faster and can provoke failures the real server cannot, but only a
    real server can catch a wrong assumption about the SDK's wire behaviour --- which is
    precisely the class of bug a hand-written stub reproduces faithfully and wrongly.
    """
    server = MCPServer(name="echo-test", instructions="A server for the client tests.")

    def echo(text: str) -> str:
        """Return the text unchanged."""
        return text

    def explode() -> str:
        """Always fail."""
        msg = "tool blew up"
        raise RuntimeError(msg)

    async def hang(seconds: float) -> str:
        """Sleep far longer than any test timeout."""
        await asyncio.sleep(seconds)
        return "never"

    server.add_tool(echo, name="echo", annotations=ToolAnnotations(read_only_hint=True))
    server.add_tool(
        explode,
        name="explode",
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
    )
    server.add_tool(hang, name="hang", annotations=ToolAnnotations(read_only_hint=True))
    return server


# --------------------------------------------------------------------------------------
# Projecting the server's advertisement
# --------------------------------------------------------------------------------------


def test_project_tool_reads_the_read_only_hint() -> None:
    spec = project_tool(make_tool(read_only=True))
    assert spec.name == "client_lookup"
    assert spec.description == "Fetch a client."
    assert spec.read_only
    assert not spec.destructive


def test_project_tool_reads_the_destructive_hint() -> None:
    spec = project_tool(make_tool("order_place", read_only=False, destructive=True))
    assert not spec.read_only
    assert spec.destructive


def test_project_tool_treats_an_unannotated_tool_as_a_write() -> None:
    """The MCP default for ``readOnlyHint`` is false, and failing closed is the safe way."""
    spec = project_tool(make_tool("mystery", annotated=False))
    assert not spec.read_only
    assert spec.destructive


def test_project_tool_honours_a_non_destructive_write() -> None:
    spec = project_tool(make_tool("note_append", read_only=False, destructive=False))
    assert not spec.read_only
    assert not spec.destructive


def test_project_tool_defaults_destructive_when_only_read_only_is_declared() -> None:
    spec = project_tool(make_tool("note_append", read_only=False, destructive=None))
    assert spec.destructive


def test_project_tool_never_asserts_approval() -> None:
    """Approval is the deployment's call, not the server's."""
    assert not project_tool(make_tool("order_place", read_only=False)).requires_approval


def test_project_tool_tolerates_a_missing_description() -> None:
    assert project_tool(make_tool(description=None)).description == ""


def test_project_tool_copies_the_input_schema() -> None:
    tool = make_tool()
    spec = project_tool(tool)
    tool.input_schema["properties"] = {}
    assert spec.input_schema["properties"] == {"client_id": {"type": "string"}}


@given(
    read_only=st.sampled_from([True, False, None]),
    destructive=st.sampled_from([True, False, None]),
    annotated=st.booleans(),
)
def test_project_tool_never_calls_a_read_only_tool_destructive(
    read_only: bool | None, destructive: bool | None, annotated: bool
) -> None:
    """The invariant the policy relies on: read-only and destructive are never both true."""
    spec = project_tool(
        make_tool(read_only=read_only, destructive=destructive, annotated=annotated)
    )
    assert not (spec.read_only and spec.destructive)


# --------------------------------------------------------------------------------------
# ToolClient: discovery
# --------------------------------------------------------------------------------------


def test_fake_session_satisfies_the_declared_protocol() -> None:
    """If the stub drifts from the protocol, every test using it is testing fiction."""
    session: MCPToolSession = FakeSession()
    assert session is not None


async def test_discover_projects_every_advertised_tool() -> None:
    session = FakeSession((make_tool("a"), make_tool("b", read_only=False)))
    specs = await ToolClient(session).discover()
    assert [s.name for s in specs] == ["a", "b"]
    assert [s.read_only for s in specs] == [True, False]


async def test_discover_is_cached() -> None:
    session = FakeSession((make_tool(),))
    client = ToolClient(session)
    first = await client.discover()
    second = await client.discover()
    assert first == second
    assert session.list_count == 1


async def test_discover_can_be_refreshed() -> None:
    session = FakeSession((make_tool(),))
    client = ToolClient(session)
    await client.discover()
    session.tools = [make_tool("b")]
    assert [s.name for s in await client.discover(refresh=True)] == ["b"]
    assert session.list_count == 2


async def test_specs_are_empty_before_discovery() -> None:
    client = ToolClient(FakeSession((make_tool(),)))
    assert client.specs == ()
    assert client.spec("client_lookup") is None
    await client.discover()
    spec = client.spec("client_lookup")
    assert spec is not None
    assert spec.name == "client_lookup"


async def test_spec_returns_none_for_an_unknown_name() -> None:
    client = ToolClient(FakeSession((make_tool(),)))
    await client.discover()
    assert client.spec("nope") is None


async def test_discovery_failure_is_raised_not_swallowed() -> None:
    """A failed handshake is a wiring fault; there is no run worth recording."""
    session = FakeSession(list_error=MCPError(code=INTERNAL_ERROR, message="down"))
    with pytest.raises(MCPError):
        await ToolClient(session).discover()


# --------------------------------------------------------------------------------------
# ToolClient: normalising results
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("timeout", [0.0, -1.0])
def test_a_non_positive_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        ToolClient(FakeSession(), timeout_s=timeout)


def test_the_default_timeout_is_used_when_none_is_given() -> None:
    assert ToolClient(FakeSession()).timeout_s == DEFAULT_TIMEOUT_S


async def test_a_successful_call_returns_text_and_structured_content() -> None:
    session = FakeSession(
        outcome=CallToolResult(
            content=[TextContent(type="text", text="42")],
            structured_content={"result": 42},
            is_error=False,
        )
    )
    result = await ToolClient(session).call("portfolio_valuation", {"account_id": "A1"})
    assert result == ToolResult(True, "42", {"result": 42})


async def test_the_configured_timeout_is_passed_to_the_transport() -> None:
    session = FakeSession()
    await ToolClient(session, timeout_s=2.5).call("echo")
    assert session.calls == [("echo", {}, 2.5)]


async def test_absent_arguments_are_sent_as_an_empty_object() -> None:
    session = FakeSession()
    await ToolClient(session).call("echo")
    assert session.calls[0][1] == {}


async def test_arguments_are_copied_before_being_sent() -> None:
    session = FakeSession()
    payload = {"client_id": "C1"}
    await ToolClient(session).call("client_lookup", payload)
    payload["client_id"] = "C2"
    assert session.calls[0][1] == {"client_id": "C1"}


async def test_an_error_result_is_normalised_not_raised() -> None:
    session = FakeSession(
        outcome=CallToolResult(content=[TextContent(type="text", text="boom")], is_error=True)
    )
    ok, text, structured = await ToolClient(session).call("explode")
    assert not ok
    assert text == "boom"
    assert structured is None


async def test_an_error_result_with_no_content_still_says_something() -> None:
    session = FakeSession(outcome=CallToolResult(content=[], is_error=True))
    ok, text, _ = await ToolClient(session).call("explode")
    assert not ok
    assert "reported an error" in text


async def test_multiple_content_blocks_are_joined() -> None:
    session = FakeSession(
        outcome=CallToolResult(
            content=[TextContent(type="text", text="one"), TextContent(type="text", text="two")],
            is_error=False,
        )
    )
    assert (await ToolClient(session).call("echo")).text == "one\ntwo"


async def test_non_text_content_is_marked_rather_than_dropped() -> None:
    session = FakeSession(
        outcome=CallToolResult(
            content=[
                TextContent(type="text", text="chart"),
                ImageContent(type="image", data="aGk=", mime_type="image/png"),
            ],
            is_error=False,
        )
    )
    assert (await ToolClient(session).call("echo")).text == "chart\n[image]"


async def test_unencodable_text_is_escaped_at_the_boundary() -> None:
    """Otherwise the surrogate reaches the transcript and kills the run's JSONL write."""
    session = FakeSession(
        outcome=CallToolResult(
            content=[TextContent(type="text", text="risk \ud800 profile")], is_error=False
        )
    )
    result = await ToolClient(session).call("client_lookup")
    assert result.text == "risk \\ud800 profile"
    assert result.text.encode("utf-8")


async def test_non_dict_structured_content_is_discarded() -> None:
    """The field is typed loosely on the wire; the agent contract says dict or nothing."""
    session = FakeSession(
        outcome=CallToolResult(
            content=[TextContent(type="text", text="x")],
            structured_content=[1, 2, 3],
            is_error=False,
        )
    )
    assert (await ToolClient(session).call("echo")).structured is None


async def test_a_timeout_is_reported_as_a_timeout() -> None:
    session = FakeSession(outcome=MCPError(code=REQUEST_TIMEOUT, message="Request timed out"))
    ok, text, _ = await ToolClient(session, timeout_s=3).call("hang")
    assert not ok
    assert text == "Tool 'hang' timed out after 3s"


async def test_a_bare_timeout_error_is_reported_as_a_timeout() -> None:
    """The SDK propagates the transport's own bounded write raw; to an agent it is the same."""
    session = FakeSession(outcome=TimeoutError("write timed out"))
    ok, text, _ = await ToolClient(session, timeout_s=1).call("hang")
    assert not ok
    assert "timed out" in text


async def test_a_protocol_error_is_reported_without_the_exception_escaping() -> None:
    session = FakeSession(outcome=MCPError(code=INTERNAL_ERROR, message="server exploded"))
    ok, text, _ = await ToolClient(session).call("echo")
    assert not ok
    assert "server exploded" in text


async def test_a_dead_transport_is_reported_without_the_exception_escaping() -> None:
    session = FakeSession(outcome=BrokenPipeError("stream closed"))
    ok, text, _ = await ToolClient(session).call("echo")
    assert not ok
    assert "BrokenPipeError" in text
    assert "stream closed" in text


async def test_an_unsupported_response_type_is_a_failure_not_an_answer() -> None:
    """An elicitation is the SDK asking for a human this harness does not have."""
    session = FakeSession(outcome=Result())
    ok, text, _ = await ToolClient(session).call("echo")
    assert not ok
    assert "unsupported response" in text


@given(message=st.text(min_size=1, max_size=40))
async def test_no_transport_failure_ever_escapes(message: str) -> None:
    session = FakeSession(outcome=RuntimeError(message))
    result = await ToolClient(session).call("echo")
    assert not result.ok
    assert result.structured is None


# --------------------------------------------------------------------------------------
# The in-process transport, against a real server
# --------------------------------------------------------------------------------------


async def test_in_process_discovery_matches_the_servers_annotations(echo_server: MCPServer) -> None:
    async with connect_in_process(echo_server) as client:
        specs = {s.name: s for s in await client.discover()}
    assert set(specs) == {"echo", "explode", "hang"}
    assert specs["echo"].read_only
    assert not specs["explode"].read_only
    assert specs["explode"].destructive
    assert specs["echo"].input_schema["type"] == "object"


async def test_in_process_call_round_trips(echo_server: MCPServer) -> None:
    async with connect_in_process(echo_server) as client:
        result = await client.call("echo", {"text": "hello"})
    assert result.ok
    assert result.text == "hello"
    assert result.structured == {"result": "hello"}


async def test_in_process_unknown_tool_is_a_normalised_failure(echo_server: MCPServer) -> None:
    async with connect_in_process(echo_server) as client:
        result = await client.call("no_such_tool", {})
    assert not result.ok
    assert "Unknown tool" in result.text


async def test_in_process_bad_arguments_are_a_normalised_failure(echo_server: MCPServer) -> None:
    async with connect_in_process(echo_server) as client:
        result = await client.call("echo", {"text": 5, "unexpected": True})
    assert not result.ok
    assert "echo" in result.text


async def test_in_process_tool_exception_is_a_normalised_failure(echo_server: MCPServer) -> None:
    async with connect_in_process(echo_server) as client:
        result = await client.call("explode", {})
    assert not result.ok
    assert "explode" in result.text


async def test_in_process_timeout_is_a_normalised_failure(echo_server: MCPServer) -> None:
    """The real SDK timeout path: the server is still sleeping when the client gives up."""
    async with connect_in_process(echo_server, timeout_s=0.05) as client:
        result = await client.call("hang", {"seconds": 30})
    assert not result.ok
    assert "timed out" in result.text


async def raise_inside_connection(server: MCPServer) -> None:
    """Fail inside the connection body, the way a benchmark task would."""
    async with connect_in_process(server) as client:
        await client.discover()
        msg = "inside the body"
        raise RuntimeError(msg)


async def test_a_failure_in_the_body_arrives_as_itself(echo_server: MCPServer) -> None:
    """The session's task group re-wraps it; a runner catching RuntimeError must still win."""
    with pytest.raises(RuntimeError, match="inside the body"):
        await raise_inside_connection(echo_server)


def test_a_multi_error_group_is_not_unwrapped() -> None:
    """Two concurrent failures are real news; picking one would discard the other."""
    group = ExceptionGroup("both", [RuntimeError("a"), ValueError("b")])
    assert _sole_exception(group) is None


def test_a_nested_single_error_group_is_unwrapped() -> None:
    """Over stdio the body's failure is wrapped twice, once per task group."""
    error = RuntimeError("only")
    inner: ExceptionGroup[Exception] = ExceptionGroup("inner", [error])
    assert _sole_exception(ExceptionGroup("outer", [inner])) is error


def test_a_group_nesting_several_errors_is_not_unwrapped() -> None:
    inner: ExceptionGroup[Exception] = ExceptionGroup("inner", [RuntimeError("a"), ValueError("b")])
    assert _sole_exception(ExceptionGroup("outer", [inner])) is None


def test_a_single_error_group_is_unwrapped() -> None:
    error = RuntimeError("only")
    assert _sole_exception(ExceptionGroup("one", [error])) is error


# --------------------------------------------------------------------------------------
# The stdio transport, against a real subprocess
# --------------------------------------------------------------------------------------

INLINE_SERVER = """
import asyncio
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

server = MCPServer(name="inline", instructions="stdio transport test")


def echo(text: str) -> str:
    "Return the text unchanged."
    return text


server.add_tool(echo, name="echo", annotations=ToolAnnotations(read_only_hint=True))
asyncio.run(server.run_stdio_async())
"""


@pytest.mark.slow
async def test_stdio_transport_round_trips() -> None:
    """Exercise the out-of-process path with a server this file owns.

    The benchmark uses the in-process transport, so without this the stdio code path would
    only ever be typechecked. A subprocess costs about a second, hence the ``slow`` mark.
    """
    async with connect_stdio(sys.executable, ["-c", INLINE_SERVER]) as client:
        specs = await client.discover()
        result = await client.call("echo", {"text": "over stdio"})
    assert [s.name for s in specs] == ["echo"]
    assert result.ok
    assert result.text == "over stdio"


async def raise_inside_stdio_connection() -> None:
    """Fail inside the stdio connection body, with a live subprocess to tear down."""
    async with connect_stdio(sys.executable, ["-c", INLINE_SERVER]):
        msg = "inside the stdio body"
        raise RuntimeError(msg)


@pytest.mark.slow
async def test_a_failure_in_the_stdio_body_arrives_as_itself() -> None:
    """Same unwrapping as the in-process path, and the subprocess is still reaped."""
    with pytest.raises(RuntimeError, match="inside the stdio body"):
        await raise_inside_stdio_connection()


@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("mcpeval.mcp_server.server") is None,
    reason="the benchmark's MCP server module is not present in this checkout",
)
async def test_stdio_against_the_benchmark_server() -> None:
    """The deployment shape the project actually ships: the real server, out of process."""
    async with connect_stdio(sys.executable, ["-m", "mcpeval.mcp_server.server"]) as client:
        specs = await client.discover()
    assert specs
    assert all(spec.description for spec in specs)


# --------------------------------------------------------------------------------------
# GuardedToolClient
# --------------------------------------------------------------------------------------


async def test_an_allowed_call_reaches_the_transport_and_is_recorded() -> None:
    session = FakeSession()
    guard, recorder = guarded(session)
    record = await guard.call(RESEARCHER, "client_lookup", {"client_id": "C1"}, step=1)
    assert record.executed
    assert record.ok
    assert record.decision.verdict is PolicyVerdict.ALLOW
    assert record.result_text == "ok"
    assert record.result_digest
    assert record.error is None
    assert session.calls == [("client_lookup", {"client_id": "C1"}, DEFAULT_TIMEOUT_S)]
    assert recorder.calls == (record,)


async def test_an_unknown_tool_never_reaches_the_transport() -> None:
    session = FakeSession()
    guard, recorder = guarded(session)
    record = await guard.call(RESEARCHER, "drop_database", {}, step=1)
    assert not record.executed
    assert not record.ok
    assert record.decision.verdict is PolicyVerdict.REFUSE_UNKNOWN_TOOL
    assert session.calls == []
    assert len(recorder.calls) == 1


async def test_a_write_by_a_read_only_role_never_reaches_the_transport() -> None:
    """The control the whole project exists to measure: refusal precedes the order."""
    session = FakeSession()
    guard, _ = guarded(session)
    record = await guard.call(RESEARCHER, "order_place", {"ticker": "BHP"}, step=2)
    assert record.decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE
    assert not record.executed
    assert session.calls == []


async def test_an_out_of_scope_read_never_reaches_the_transport() -> None:
    session = FakeSession()
    guard, _ = guarded(session)
    record = await guard.call(WRITER, "transactions_list", {"account_id": "A1"}, step=1)
    assert record.decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE
    assert session.calls == []


async def test_a_write_without_approval_never_reaches_the_transport() -> None:
    session = FakeSession()
    guard, _ = guarded(session)
    record = await guard.call(SUPERVISOR, "order_place", {"ticker": "BHP"}, step=3)
    assert record.decision.verdict is PolicyVerdict.REFUSE_NO_APPROVAL
    assert session.calls == []


async def test_an_approved_write_does_reach_the_transport() -> None:
    session = FakeSession()
    guard, _ = guarded(session)
    record = await guard.call(SUPERVISOR, "order_place", {"ticker": "BHP"}, step=3, approved=True)
    assert record.executed
    assert record.decision.verdict is PolicyVerdict.ALLOW
    assert len(session.calls) == 1


async def test_a_refusal_records_the_rule_that_fired() -> None:
    guard, _ = guarded(FakeSession())
    record = await guard.call(RESEARCHER, "order_place", {}, step=1)
    assert record.error is not None
    assert record.error.startswith(record.decision.rule)
    assert record.decision.reason in record.error


async def test_a_refusal_carries_no_digest() -> None:
    """A shared digest across every refusal would read as a repeated identical result."""
    guard, _ = guarded(FakeSession())
    record = await guard.call(RESEARCHER, "order_place", {}, step=1)
    assert record.result_digest == ""
    assert record.result_text == ""


async def test_a_failed_executed_call_is_recorded_as_executed() -> None:
    session = FakeSession(
        outcome=CallToolResult(
            content=[TextContent(type="text", text="no such client")], is_error=True
        )
    )
    guard, _ = guarded(session)
    record = await guard.call(RESEARCHER, "client_lookup", {"client_id": "C999"}, step=1)
    assert record.executed
    assert not record.ok
    assert record.error == "no such client"
    assert record.result_digest


async def test_arguments_are_recorded_verbatim_even_when_refused() -> None:
    guard, _ = guarded(FakeSession())
    record = await guard.call(
        RESEARCHER, "order_place", {"ticker": "BHP", "amount": "5000"}, step=1
    )
    assert record.arguments == {"ticker": "BHP", "amount": "5000"}
    assert record.agent == RESEARCHER
    assert record.step == 1


async def test_latency_comes_from_the_recorders_clock() -> None:
    guard, _ = guarded(FakeSession(), clock=FakeClock(step=0.5))
    record = await guard.call(RESEARCHER, "client_lookup", {}, step=1)
    assert record.latency_ms == 500.0


async def test_a_refused_call_spends_nothing() -> None:
    guard, _ = guarded(FakeSession())
    before = guard.budget_for(RESEARCHER).model_copy()
    await guard.call(RESEARCHER, "order_place", {}, step=1)
    after = guard.budget_for(RESEARCHER)
    assert (after.calls, after.elapsed_ms) == (before.calls, before.elapsed_ms)


async def test_an_executed_call_spends_a_call_and_its_wall_time() -> None:
    guard, _ = guarded(FakeSession(), clock=FakeClock(step=0.5))
    await guard.call(RESEARCHER, "client_lookup", {}, step=1)
    budget = guard.budget_for(RESEARCHER)
    assert budget.calls == 1
    assert budget.elapsed_ms == 500.0


async def test_an_exhausted_budget_refuses_before_the_transport() -> None:
    session = FakeSession()
    budgets = {RESEARCHER: BudgetState(max_calls=1)}
    guard, _ = guarded(session, budgets=budgets)
    first = await guard.call(RESEARCHER, "client_lookup", {}, step=1)
    second = await guard.call(RESEARCHER, "client_lookup", {}, step=2)
    assert first.executed
    assert second.decision.verdict is PolicyVerdict.REFUSE_BUDGET
    assert len(session.calls) == 1


async def test_supplied_budgets_are_used_rather_than_rebuilt() -> None:
    supplied = BudgetState(max_calls=9, calls=4)
    guard, _ = guarded(FakeSession(), budgets={RESEARCHER: supplied})
    assert guard.budget_for(RESEARCHER).calls == 4


def test_a_budget_is_created_once_per_role() -> None:
    guard, _ = guarded(FakeSession())
    assert guard.budget_for(RESEARCHER) is guard.budget_for(RESEARCHER)
    assert guard.budget_for(RESEARCHER) is not guard.budget_for(SUPERVISOR)


def test_an_unknown_role_gets_a_zero_headroom_budget() -> None:
    """The fallback must fail closed if the policy's check order is ever reordered."""
    guard, _ = guarded(FakeSession())
    budget = guard.budget_for("intern")
    assert budget.exhausted


async def test_an_unknown_role_is_refused_out_of_scope() -> None:
    session = FakeSession()
    guard, _ = guarded(session)
    record = await guard.call("intern", "client_lookup", {}, step=1)
    assert record.decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE
    assert session.calls == []


def test_steps_and_tokens_can_be_charged_to_a_role() -> None:
    guard, _ = guarded(FakeSession())
    guard.spend_step(RESEARCHER, 2)
    budget = guard.spend_tokens(RESEARCHER, 1_200)
    assert (budget.steps, budget.tokens) == (2, 1_200)


def test_the_guard_exposes_the_policy_and_the_recorder() -> None:
    guard, recorder = guarded(FakeSession())
    assert guard.recorder is recorder
    assert guard.policy.is_write("order_place")


@given(names=st.lists(st.sampled_from(["client_lookup", "order_place", "nonesuch"]), max_size=8))
async def test_every_attempt_is_recorded_and_only_allowed_ones_execute(names: list[str]) -> None:
    """Conservation law: attempts equal records, and executions equal transport calls."""
    session = FakeSession()
    guard, recorder = guarded(session)
    for step, name in enumerate(names):
        await guard.call(RESEARCHER, name, {}, step=step)
    executed = [c for c in recorder.calls if c.executed]
    assert len(recorder.calls) == len(names)
    assert len(executed) == len(session.calls)
    assert all(c.decision.allowed for c in executed)
    assert all(not c.decision.allowed for c in recorder.calls if not c.executed)


async def test_the_guard_works_over_a_real_in_process_server(echo_server: MCPServer) -> None:
    """End to end: live inventory, refreshed policy, one allowed call and one refusal."""
    async with connect_in_process(echo_server) as client:
        specs = await client.discover()
        policy = default_policy().with_tools(specs)
        recorder = TrajectoryRecorder(
            task_id="t-e2e", architecture="single", model="scripted", clock=FakeClock()
        )
        guard = GuardedToolClient(client, policy, recorder)
        allowed = await guard.call(SUPERVISOR, "echo", {"text": "hi"}, step=1)
        refused = await guard.call(RESEARCHER, "explode", {}, step=2)
    assert allowed.executed
    assert allowed.result_text == "hi"
    assert not refused.executed
    assert refused.decision.verdict is PolicyVerdict.REFUSE_OUT_OF_SCOPE
    assert recorder.finish(final_answer="hi").executed_calls == [allowed]


async def test_one_server_can_back_two_successive_connections(echo_server: MCPServer) -> None:
    """The benchmark reuses a server object across tasks, so teardown must be complete."""
    async with connect_in_process(echo_server) as first:
        assert (await first.call("echo", {"text": "one"})).text == "one"
    async with connect_in_process(echo_server) as second:
        assert (await second.call("echo", {"text": "two"})).text == "two"
