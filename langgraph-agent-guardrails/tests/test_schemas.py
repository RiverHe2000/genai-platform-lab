from __future__ import annotations

import pytest

from agentguard.schemas import ActionParseError, FinalAnswer, ToolCall, parse_action


def test_parse_tool_call_and_final() -> None:
    p = parse_action('{"type": "tool", "tool": "calculate", "args": {"expression": "1+1"}}')
    assert isinstance(p.action, ToolCall)
    assert p.action.tool == "calculate"
    assert p.action.args == {"expression": "1+1"}
    assert p.fallback is False
    f = parse_action('{"type": "final", "answer": "done"}')
    assert isinstance(f.action, FinalAnswer)
    assert f.action.answer == "done"


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"type": "final", "answer": "fenced"}\n```',
        'Sure, here is my action: {"type": "final", "answer": "fenced"} — hope that helps',
        '{"unrelated": 1} then {"type": "final", "answer": "fenced"}',
    ],
)
def test_parse_finds_json_in_noise(text: str) -> None:
    p = parse_action(text)
    assert isinstance(p.action, FinalAnswer)
    assert p.action.answer == "fenced"


def test_plain_text_is_a_fallback_final_answer() -> None:
    p = parse_action("The maximum LVR is 80%.")
    assert isinstance(p.action, FinalAnswer)
    assert p.fallback is True
    assert p.action.answer == "The maximum LVR is 80%."


@pytest.mark.parametrize(
    "text",
    [
        '{"type": "tool"}',
        '{"type": "final"}',
        '{"type": "tool", "tool": "", "args": {}}',
        '{"type": "final", "answer": "x", "extra": 1}',
        "",
        "   ",
    ],
)
def test_invalid_json_actions_raise(text: str) -> None:
    with pytest.raises(ActionParseError):
        parse_action(text)


@pytest.mark.parametrize(
    "text",
    [
        '{"type": "calculate", "tool": "calculate", "args": {"expression": "1+1"}}',
        '{"type": "calculate", "args": {"expression": "1+1"}}',
        '{"tool": "calculate", "args": {"expression": "1+1"}}',
        '{"type": "action", "tool": "calculate", "args": {"expression": "1+1"}}',
    ],
)
def test_envelope_mistakes_are_normalised(text: str) -> None:
    p = parse_action(text)
    assert isinstance(p.action, ToolCall)
    assert p.action.tool == "calculate" and p.action.args == {"expression": "1+1"}


def test_missing_type_with_answer_is_final_and_unknown_shapes_still_fail() -> None:
    p = parse_action('{"answer": "done"}')
    assert isinstance(p.action, FinalAnswer) and p.action.answer == "done"
    with pytest.raises(ActionParseError):
        parse_action('{"type": "error", "message": "Invalid input format."}')


def test_args_default_to_empty_dict() -> None:
    p = parse_action('{"type": "tool", "tool": "describe_loanbook"}')
    assert isinstance(p.action, ToolCall)
    assert p.action.args == {}
