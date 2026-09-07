"""Tests for the action protocol, its tolerant parser and the system prompt.

Two properties carry most of the weight here:

* anything the agents can emit must survive a JSON round trip with **no** repairs, so
  that a repair count is a real signal about the model rather than noise from our own
  serialiser;
* the parser must never raise, whatever a small model produces, because a malformed
  turn is a data point the benchmark classifies, not a crash that ends the run.

The rest are the specific malformations observed from small instruct models, each
pinned to the repair tag it must report.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from mcpeval.agents.protocol import (
    ACTION_NAMES,
    Action,
    CallToolAction,
    ClarifyAction,
    FinalAction,
    HandoffAction,
    ParsedAction,
    parse_action,
    render_action,
    render_system_prompt,
)
from mcpeval.schemas import ToolSpec

FENCE = "`" * 3
"""A Markdown code fence, built rather than written so this file stays fence-free."""

TEXT = st.text(alphabet=st.characters(codec="utf-8"), max_size=60)
SCALARS = st.one_of(
    st.none(), st.booleans(), st.integers(min_value=-(10**9), max_value=10**9), TEXT
)
ARGUMENTS = st.dictionaries(TEXT, st.one_of(SCALARS, st.lists(SCALARS, max_size=3)), max_size=4)

ACTIONS: st.SearchStrategy[Action] = st.one_of(
    st.builds(CallToolAction, thought=TEXT, tool=TEXT, arguments=ARGUMENTS),
    st.builds(FinalAction, thought=TEXT, answer=TEXT),
    st.builds(HandoffAction, thought=TEXT, to=TEXT, instruction=TEXT),
    st.builds(ClarifyAction, thought=TEXT, question=TEXT),
)

TOOLS = (
    ToolSpec(
        name="client_get",
        description="Look up one client by id.",
        input_schema={
            "type": "object",
            "properties": {"client_id": {"type": "string"}},
            "required": ["client_id"],
        },
    ),
    ToolSpec(
        name="order_place",
        description="Place a buy or sell order on an account.",
        input_schema={
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "units": {"type": "integer"},
                "dry_run": {"type": "boolean"},
            },
            "required": ["account_id", "units"],
        },
        read_only=False,
        destructive=True,
        requires_approval=True,
    ),
)


def _ok(text: str) -> ParsedAction:
    """Parse `text`, asserting that something was recovered."""
    parsed = parse_action(text)
    assert parsed.ok, parsed.error
    return parsed


# --------------------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------------------


@given(ACTIONS)
def test_a_rendered_action_parses_back_unchanged_and_unrepaired(action: Action) -> None:
    """The identity the whole format metric rests on."""
    parsed = parse_action(render_action(action))
    assert parsed.action == action
    assert parsed.repairs == ()
    assert parsed.clean is True
    assert parsed.error == ""
    assert parsed.raw == render_action(action)


@given(ACTIONS)
def test_parsing_is_idempotent(action: Action) -> None:
    once = parse_action(render_action(action)).action
    assert once is not None
    assert parse_action(render_action(once)).action == once


@given(st.text(max_size=300))
def test_the_parser_never_raises_whatever_the_model_emits(text: str) -> None:
    parsed = parse_action(text)
    assert isinstance(parsed, ParsedAction)
    assert parsed.ok is (parsed.action is not None)
    assert (parsed.error == "") is parsed.ok
    assert parsed.raw == text


@given(ACTIONS)
def test_a_fenced_action_is_recovered_with_exactly_one_repair(action: Action) -> None:
    rendered = render_action(action)
    # An action whose own text contains a code fence is excluded, and the exclusion is
    # honest rather than convenient: with a fence inside the payload there is no single
    # correct answer to "where does the fence end", so which repair the parser reports is
    # genuinely ambiguous. Recovery is unaffected -- the test below covers those cases and
    # still requires the action to round-trip. Found by hypothesis on
    # `FinalAction(answer="```")`.
    assume(FENCE not in rendered)
    parsed = parse_action(f"{FENCE}json\n{rendered}\n{FENCE}")
    assert parsed.action == action
    assert parsed.repairs == ("code_fence",)


@given(ACTIONS)
def test_a_fenced_action_round_trips_even_when_it_contains_a_fence(action: Action) -> None:
    """The case above excludes: the repair label is ambiguous, the recovery is not."""
    parsed = parse_action(f"{FENCE}json\n{render_action(action)}\n{FENCE}")
    assert parsed.action == action
    assert parsed.repairs


@given(ACTIONS)
def test_prose_around_the_action_is_recovered(action: Action) -> None:
    parsed = parse_action(f"Sure, here is my next step.\n{render_action(action)}\nLet me know.")
    assert parsed.action == action
    assert "surrounding_prose" in parsed.repairs


@pytest.mark.parametrize("name", ACTION_NAMES)
def test_every_action_name_has_a_model(name: str) -> None:
    assert any(name == json.loads(render_action(a))["action"] for a in _one_of_each())


def _one_of_each() -> tuple[Action, ...]:
    return (
        CallToolAction(thought="t", tool="client_get", arguments={"client_id": "CLI-0001"}),
        FinalAction(thought="t", answer="42"),
        HandoffAction(thought="t", to="fees", instruction="reconcile"),
        ClarifyAction(thought="t", question="which client?"),
    )


# --------------------------------------------------------------------------------------
# Repairs, one malformation at a time
# --------------------------------------------------------------------------------------


def test_a_bare_fence_without_a_language_tag() -> None:
    parsed = _ok('```\n{"thought": "t", "action": "final", "answer": "42"}\n```')
    assert parsed.repairs == ("code_fence",)


def test_a_fence_the_model_never_closed() -> None:
    parsed = _ok('```json\n{"thought": "t", "action": "final", "answer": "42"}')
    assert parsed.action == FinalAction(thought="t", answer="42")
    assert "code_fence" in parsed.repairs


def test_prose_before_the_object_only() -> None:
    parsed = _ok('Let me look that up. {"action": "clarify", "question": "who?", "thought": "t"}')
    assert parsed.repairs == ("surrounding_prose",)


def test_single_quotes() -> None:
    parsed = _ok("{'thought': 't', 'action': 'final', 'answer': 'forty two'}")
    assert parsed.repairs == ("single_quotes",)
    assert parsed.action == FinalAction(thought="t", answer="forty two")


def test_a_trailing_comma() -> None:
    parsed = _ok('{"thought": "t", "action": "final", "answer": "42",}')
    assert parsed.repairs == ("trailing_comma",)


def test_python_keywords_instead_of_json_ones() -> None:
    parsed = _ok(
        "{'thought': 't', 'action': 'call_tool', 'tool': 'order_place', "
        "'arguments': {'dry_run': true}}"
    )
    assert parsed.repairs == ("single_quotes", "python_keywords")
    assert parsed.action == CallToolAction(
        thought="t", tool="order_place", arguments={"dry_run": True}
    )


def test_a_double_quoted_python_literal_is_still_recognised() -> None:
    parsed = _ok('{"thought": "t", "action": "final", "answer": "42", "extra": (1, 2)}')
    assert "python_literal" in parsed.repairs
    assert "dropped_keys:extra" in parsed.repairs


def test_a_missing_thought_is_recorded_not_hidden() -> None:
    parsed = _ok('{"action": "final", "answer": "42"}')
    assert parsed.repairs == ("missing_thought",)
    assert parsed.action == FinalAction(thought="", answer="42")


def test_a_missing_arguments_object_is_recorded() -> None:
    parsed = _ok('{"thought": "t", "action": "call_tool", "tool": "client_list"}')
    assert parsed.repairs == ("missing_arguments",)


def test_arguments_delivered_as_a_json_string() -> None:
    parsed = _ok(
        '{"thought": "t", "action": "call_tool", "tool": "client_get", '
        '"arguments": "{\\"client_id\\": \\"CLI-0001\\"}"}'
    )
    assert parsed.repairs == ("arguments_json_string",)
    assert parsed.action == CallToolAction(
        thought="t", tool="client_get", arguments={"client_id": "CLI-0001"}
    )


def test_null_arguments_become_an_empty_object() -> None:
    parsed = _ok(
        '{"thought": "t", "action": "call_tool", "tool": "client_list", "arguments": null}'
    )
    assert parsed.repairs == ("null_arguments",)
    assert isinstance(parsed.action, CallToolAction)
    assert parsed.action.arguments == {}


@pytest.mark.parametrize(
    ("emitted", "expected"),
    [
        ("tool_name", "alias:tool_name->tool"),
        ("name", "alias:name->tool"),
        ("function", "alias:function->tool"),
    ],
)
def test_the_tool_name_goes_by_several_names(emitted: str, expected: str) -> None:
    parsed = _ok(
        json.dumps({"thought": "t", "action": "call_tool", emitted: "client_get", "arguments": {}})
    )
    assert expected in parsed.repairs
    assert isinstance(parsed.action, CallToolAction)
    assert parsed.action.tool == "client_get"


@pytest.mark.parametrize("emitted", ["args", "params", "parameters", "input", "tool_input"])
def test_the_arguments_go_by_several_names(emitted: str) -> None:
    parsed = _ok(
        json.dumps({"thought": "t", "action": "call_tool", "tool": "client_get", emitted: {"a": 1}})
    )
    assert f"alias:{emitted}->arguments" in parsed.repairs
    assert isinstance(parsed.action, CallToolAction)
    assert parsed.action.arguments == {"a": 1}


def test_aliases_are_scoped_to_the_action() -> None:
    """`name` means the tool for a call, but nothing at all for a handoff."""
    parsed = _ok('{"thought": "t", "action": "handoff", "agent": "fees", "task": "check fees"}')
    assert parsed.action == HandoffAction(thought="t", to="fees", instruction="check fees")
    assert set(parsed.repairs) == {"alias:agent->to", "alias:task->instruction"}


def test_the_thought_goes_by_several_names() -> None:
    parsed = _ok('{"reasoning": "because", "action": "final", "answer": "42"}')
    assert parsed.repairs == ("alias:reasoning->thought",)
    assert isinstance(parsed.action, FinalAction)
    assert parsed.action.thought == "because"


def test_an_alias_never_clobbers_the_real_field() -> None:
    parsed = _ok('{"thought": "t", "action": "final", "answer": "real", "response": "alias"}')
    assert isinstance(parsed.action, FinalAction)
    assert parsed.action.answer == "real"
    assert parsed.repairs == ("dropped_keys:response",)


def test_unknown_keys_are_dropped_and_named() -> None:
    parsed = _ok(
        '{"thought": "t", "action": "final", "answer": "42", "confidence": 0.9, "step": 3}'
    )
    assert parsed.repairs == ("dropped_keys:confidence,step",)


def test_an_object_wrapped_in_another_object() -> None:
    parsed = _ok('{"response": {"thought": "t", "action": "final", "answer": "42"}}')
    assert parsed.repairs == ("unwrapped:response",)


def test_the_action_key_holding_the_whole_object() -> None:
    parsed = _ok('{"action": {"thought": "t", "action": "final", "answer": "42"}}')
    assert parsed.repairs == ("unwrapped:action",)


def test_an_object_nested_as_deep_as_the_parser_will_go() -> None:
    payload = (
        '{"action": {"action": {"action": {"thought": "t", "action": "final", "answer": "42"}}}}'
    )
    parsed = _ok(payload)
    assert parsed.action == FinalAction(thought="t", answer="42")
    assert parsed.repairs == ("unwrapped:action",)


def test_nesting_beyond_the_limit_is_reported_rather_than_chased() -> None:
    """The unwrap depth is bounded on purpose; the failure has to say so honestly."""
    inner = '{"thought": "t", "action": "final", "answer": "42"}'
    payload = '{"action": {"action": {"action": {"action": ' + inner + "}}}}"
    parsed = parse_action(payload)
    assert not parsed.ok
    assert parsed.error == "the 'action' field is dict, not a string"


def test_two_plausible_wrappers_are_not_guessed_between() -> None:
    text = '{"a": {"action": "final", "answer": "1"}, "b": {"action": "final", "answer": "2"}}'
    parsed = parse_action(text)
    assert not parsed.ok
    assert "action" in parsed.error


def test_a_one_element_list() -> None:
    parsed = _ok('[{"thought": "t", "action": "final", "answer": "42"}]')
    assert parsed.repairs == ("single_element_list",)


@pytest.mark.parametrize(
    ("emitted", "expected"),
    [
        ("tool_call", "call_tool"),
        ("use_tool", "call_tool"),
        ("ANSWER", "final"),
        ("Final Answer", "final"),
        ("delegate", "handoff"),
        ("ask_user", "clarify"),
    ],
)
def test_action_verbs_go_by_several_names(emitted: str, expected: str) -> None:
    payload = {
        "thought": "t",
        "action": emitted,
        "tool": "client_get",
        "arguments": {},
        "answer": "42",
        "to": "fees",
        "instruction": "go",
        "question": "who?",
    }
    parsed = _ok(json.dumps(payload))
    assert parsed.action is not None
    assert parsed.action.action == expected
    assert f"action_alias:{emitted}" in parsed.repairs


def test_non_string_fields_are_coerced_and_flagged() -> None:
    parsed = _ok('{"thought": 7, "action": "final", "answer": 42}')
    assert parsed.action == FinalAction(thought="7", answer="42")
    assert set(parsed.repairs) == {"coerced:thought", "coerced:answer"}


def test_a_structured_answer_is_serialised_rather_than_lost() -> None:
    parsed = _ok('{"thought": "t", "action": "final", "answer": {"value": 42}}')
    assert isinstance(parsed.action, FinalAction)
    assert json.loads(parsed.action.answer) == {"value": 42}
    assert parsed.repairs == ("coerced:answer",)


def test_repairs_stack_when_the_model_gets_everything_wrong() -> None:
    text = (
        "Here you go:\n```json\n"
        "{'action': 'tool_call', 'tool_name': 'client_get', 'args': {'client_id': 'CLI-0001',},}\n"
        "```\nHope that helps."
    )
    parsed = _ok(text)
    assert parsed.action == CallToolAction(
        thought="", tool="client_get", arguments={"client_id": "CLI-0001"}
    )
    assert parsed.repairs == (
        "code_fence",
        "single_quotes",
        "action_alias:tool_call",
        "alias:tool_name->tool",
        "alias:args->arguments",
        "missing_thought",
    )


@given(ACTIONS)
def test_repair_tags_are_never_duplicated(action: Action) -> None:
    text = f"Thinking...\n```json\n{render_action(action)}\n```\ndone"
    repairs = parse_action(text).repairs
    assert len(set(repairs)) == len(repairs)


# --------------------------------------------------------------------------------------
# Structured failures
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_an_empty_turn_fails_with_a_reason(text: str) -> None:
    parsed = parse_action(text)
    assert not parsed.ok
    assert parsed.error == "the model returned no text"


def test_prose_with_no_json_at_all() -> None:
    parsed = parse_action("I am sorry, I cannot help with that request.")
    assert not parsed.ok
    assert "no JSON object found" in parsed.error


def test_an_object_the_model_never_closed() -> None:
    parsed = parse_action('{"thought": "t", "action": "call_tool", "tool": "client_get"')
    assert not parsed.ok
    assert "unterminated" in parsed.error


def test_a_json_value_that_is_not_an_object() -> None:
    parsed = parse_action('"just a string"')
    assert not parsed.ok
    assert parsed.error == "expected a JSON object, found str"


def test_a_list_of_several_actions_is_refused() -> None:
    parsed = parse_action(
        '[{"action": "final", "answer": "1"}, {"action": "final", "answer": "2"}]'
    )
    assert not parsed.ok
    assert "found list" in parsed.error


def test_a_fenced_json_value_that_is_not_an_object() -> None:
    parsed = parse_action("```json\n[1, 2]\n```")
    assert not parsed.ok
    assert parsed.error == "expected a JSON object, found list"


def test_an_object_with_no_action_field() -> None:
    parsed = parse_action('{"thought": "t", "answer": "42"}')
    assert not parsed.ok
    assert parsed.error == "no 'action' field in the JSON object"


def test_an_unknown_action_verb_is_quoted_back() -> None:
    parsed = parse_action('{"thought": "t", "action": "search_web", "query": "x"}')
    assert not parsed.ok
    assert parsed.error == "unknown action 'search_web'"


def test_an_action_field_that_is_not_a_string() -> None:
    parsed = parse_action('{"thought": "t", "action": 3, "answer": "42"}')
    assert not parsed.ok
    assert parsed.error == "the 'action' field is int, not a string"


def test_a_missing_required_field_is_a_schema_violation() -> None:
    parsed = parse_action('{"thought": "t", "action": "final"}')
    assert not parsed.ok
    assert parsed.error.startswith("schema violation:")
    assert "answer" in parsed.error


def test_an_explicit_null_answer_is_dropped_and_then_refused() -> None:
    parsed = parse_action('{"thought": "t", "action": "final", "answer": null}')
    assert not parsed.ok
    assert parsed.repairs == ("dropped_null:answer",)
    assert "answer" in parsed.error


def test_arguments_that_are_a_string_but_not_json() -> None:
    parsed = parse_action(
        '{"thought": "t", "action": "call_tool", "tool": "client_get", "arguments": "CLI-0001"}'
    )
    assert not parsed.ok
    assert parsed.error == "the 'arguments' field is a string that is not a JSON object"


def test_arguments_that_are_a_list() -> None:
    parsed = parse_action(
        '{"thought": "t", "action": "call_tool", "tool": "client_get", "arguments": ["CLI-0001"]}'
    )
    assert not parsed.ok
    assert parsed.error == "the 'arguments' field is list, not an object"


def test_a_failure_still_reports_the_repairs_it_managed() -> None:
    parsed = parse_action('```json\n{"action": "final"}\n```')
    assert not parsed.ok
    assert parsed.repairs == ("code_fence", "missing_thought")


# --------------------------------------------------------------------------------------
# The system prompt
# --------------------------------------------------------------------------------------

_FENCED = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


def test_the_prompt_lists_every_tool_with_its_schema_and_annotations() -> None:
    prompt = render_system_prompt(TOOLS, "supervisor")
    for tool in TOOLS:
        assert tool.name in prompt
        assert tool.description in prompt
    assert '"client_id": {"type": "string"}' in prompt
    assert "(read-only)" in prompt
    assert "(writes data, destructive, requires human approval)" in prompt


def test_every_example_in_the_prompt_parses_with_no_repairs() -> None:
    """The examples are the format specification; a broken one teaches the wrong thing."""
    prompt = render_system_prompt(TOOLS, "supervisor", peers=["fees"])
    examples = _FENCED.findall(prompt)
    assert len(examples) >= 2
    for example in examples:
        parsed = parse_action(example)
        assert parsed.clean, f"{example} -> {parsed.error} {parsed.repairs}"


def test_the_tool_example_uses_a_real_tool_and_its_required_arguments() -> None:
    prompt = render_system_prompt(TOOLS, "supervisor")
    first = parse_action(_FENCED.findall(prompt)[0]).action
    assert isinstance(first, CallToolAction)
    assert first.tool == "client_get"
    assert first.arguments == {"client_id": "<client_id>"}


def test_the_example_arguments_follow_the_schema_types() -> None:
    prompt = render_system_prompt([TOOLS[1]], "trader")
    first = parse_action(_FENCED.findall(prompt)[0]).action
    assert isinstance(first, CallToolAction)
    assert first.arguments == {"account_id": "<account_id>", "units": 1}


def test_a_schema_with_no_required_list_falls_back_to_the_first_properties() -> None:
    tool = ToolSpec(
        name="search",
        description="Search.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "ignored": {"type": "boolean"},
            },
        },
    )
    first = parse_action(_FENCED.findall(render_system_prompt([tool], "r"))[0]).action
    assert isinstance(first, CallToolAction)
    assert first.arguments == {"query": "<query>", "limit": 1}


def test_a_schema_example_wins_over_the_placeholder() -> None:
    tool = ToolSpec(
        name="client_get",
        description="Look up a client.",
        input_schema={
            "type": "object",
            "properties": {"client_id": {"type": "string", "examples": ["CLI-0007"]}},
            "required": ["client_id"],
        },
    )
    first = parse_action(_FENCED.findall(render_system_prompt([tool], "r"))[0]).action
    assert isinstance(first, CallToolAction)
    assert first.arguments == {"client_id": "CLI-0007"}


@pytest.mark.parametrize(
    "schema",
    [{}, {"type": "object"}, {"type": "object", "properties": {}}, {"properties": "broken"}],
)
def test_a_tool_with_no_usable_schema_still_gets_an_example(schema: dict[str, Any]) -> None:
    tool = ToolSpec(name="ping", description="Ping.", input_schema=schema)
    first = parse_action(_FENCED.findall(render_system_prompt([tool], "r"))[0]).action
    assert isinstance(first, CallToolAction)
    assert first.arguments == {}


def test_an_unusual_property_type_falls_back_to_a_placeholder() -> None:
    tool = ToolSpec(
        name="odd",
        description="Odd.",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "null"}, "b": "not-a-schema"},
            "required": ["a", "b"],
        },
    )
    first = parse_action(_FENCED.findall(render_system_prompt([tool], "r"))[0]).action
    assert isinstance(first, CallToolAction)
    assert first.arguments == {"a": "<a>", "b": "<b>"}


def test_handoff_is_offered_only_when_there_are_peers() -> None:
    alone = render_system_prompt(TOOLS, "solo")
    supervised = render_system_prompt(TOOLS, "supervisor", peers=["fees", "positions"])
    assert '"action": "handoff"' not in alone
    assert '"action": "handoff"' in supervised
    assert "fees, positions" in supervised
    assert any(
        isinstance(parse_action(e).action, HandoffAction) for e in _FENCED.findall(supervised)
    )


def test_clarify_can_be_switched_off() -> None:
    prompt = render_system_prompt(TOOLS, "worker", allow_clarify=False)
    assert '"action": "clarify"' not in prompt
    assert all(
        not isinstance(parse_action(e).action, ClarifyAction) for e in _FENCED.findall(prompt)
    )


def test_the_clarify_example_appears_when_there_are_no_peers() -> None:
    prompt = render_system_prompt(TOOLS, "worker")
    assert any(isinstance(parse_action(e).action, ClarifyAction) for e in _FENCED.findall(prompt))


def test_an_agent_with_no_tools_is_told_so() -> None:
    prompt = render_system_prompt([], "summariser")
    assert "(none:" in prompt
    examples = _FENCED.findall(prompt)
    assert examples and all(parse_action(e).clean for e in examples)
    assert not any(isinstance(parse_action(e).action, CallToolAction) for e in examples)


def test_the_goal_and_the_step_budget_are_stated_when_given() -> None:
    prompt = render_system_prompt(
        TOOLS, "supervisor", goal="Answer adviser questions.", max_steps=8
    )
    assert "Answer adviser questions." in prompt
    assert "at most 8 steps" in prompt


def test_the_step_budget_is_left_out_when_there_is_none() -> None:
    assert "at most" not in render_system_prompt(TOOLS, "supervisor")


def test_examples_can_be_replaced_wholesale() -> None:
    custom = [FinalAction(thought="mine", answer="mine")]
    prompt = render_system_prompt(TOOLS, "supervisor", examples=custom)
    assert _FENCED.findall(prompt) == [render_action(custom[0])]


def test_examples_can_be_removed_entirely() -> None:
    prompt = render_system_prompt(TOOLS, "supervisor", examples=[])
    assert _FENCED.findall(prompt) == []
    assert "Examples" not in prompt


def test_the_prompt_is_deterministic() -> None:
    """Two runs of the same configuration must give byte-identical prompts, or a
    cached completion from an earlier run would silently answer a different prompt."""
    first = render_system_prompt(TOOLS, "supervisor", peers=["fees"], max_steps=6)
    second = render_system_prompt(TOOLS, "supervisor", peers=["fees"], max_steps=6)
    assert first == second
