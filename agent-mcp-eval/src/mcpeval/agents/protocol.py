"""The JSON action protocol the agents speak, its tolerant parser, and the system prompt.

An agent turn is exactly one JSON object: a thought plus one of four actions - call a
tool, answer, hand off to another agent, or ask the user a question. Modelling the turn
as a discriminated union rather than free text is what makes the benchmark gradeable: a
tool call is a structured record with a name and arguments, so a policy can rule on it
before it is executed and the grader can compare it against the calls a task required.

Small open-weight models do not emit clean JSON. They fence it, they narrate around it,
they use single quotes, they drop the thought, they wrap the object in another object.
:func:`parse_action` repairs all of that, but every repair it applies is listed in
:attr:`ParsedAction.repairs`. A silently repaired output looks like a model that
followed the format, which flatters the model and hides a real, measurable defect - the
same mistake the JSON floor caught in the ops-loop project. Format compliance is a
headline metric here, and it can only be measured if repairs are counted.

Repair tags, all lower-case and stable enough to aggregate over a run:

``code_fence``
    The object was inside a Markdown fence.
``surrounding_prose``
    The object was extracted from a longer blob of text.
``single_element_list``
    The object arrived wrapped in a one-element list.
``trailing_comma``
    A comma before ``}`` or ``]`` had to be removed.
``single_quotes`` / ``python_literal`` / ``python_keywords``
    The payload was a Python literal rather than JSON.
``unwrapped:<key>``
    The action object was nested under ``<key>``.
``alias:<from>-><to>``
    A field went by another name.
``action_alias:<raw>``
    The action verb went by another name, or another case.
``missing_thought`` / ``missing_arguments``
    A required-by-convention field was absent and was defaulted.
``arguments_json_string`` / ``null_arguments``
    ``arguments`` was a JSON string, or null, rather than an object.
``coerced:<field>``
    A field that should be a string was a number, a bool or a structure.
``dropped_keys:<a,b>`` / ``dropped_null:<field>``
    Keys the schema does not define, or explicit nulls, were removed.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from mcpeval.schemas import ToolSpec

__all__ = [
    "ACTION_ADAPTER",
    "ACTION_NAMES",
    "Action",
    "CallToolAction",
    "ClarifyAction",
    "FinalAction",
    "HandoffAction",
    "ParsedAction",
    "parse_action",
    "render_action",
    "render_system_prompt",
]


class _BaseAction(BaseModel):
    """Fields shared by every action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    thought: str = ""
    """One line of reasoning. Defaulted so an action can be built in code without one;
    a *parsed* action that lacked it is flagged with the ``missing_thought`` repair."""


class CallToolAction(_BaseAction):
    """Call one MCP tool and wait for its result."""

    action: Literal["call_tool"] = "call_tool"
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class FinalAction(_BaseAction):
    """Stop and answer the user."""

    action: Literal["final"] = "final"
    answer: str


class HandoffAction(_BaseAction):
    """Delegate the rest of the task to another agent."""

    action: Literal["handoff"] = "handoff"
    to: str
    instruction: str


class ClarifyAction(_BaseAction):
    """Ask the user a question, for tasks that are genuinely ambiguous.

    A benchmark that has no way to say "this question cannot be answered as asked"
    rewards a confident guess, so asking has to be a first-class action.
    """

    action: Literal["clarify"] = "clarify"
    question: str


Action = Annotated[
    CallToolAction | FinalAction | HandoffAction | ClarifyAction,
    Field(discriminator="action"),
]
"""One agent turn, discriminated on the ``action`` field."""

ACTION_ADAPTER: TypeAdapter[Action] = TypeAdapter(Action)
"""Validator for :data:`Action`, reused rather than rebuilt on every parse."""

ACTION_NAMES: Final[tuple[str, ...]] = ("call_tool", "final", "handoff", "clarify")


class ParsedAction(BaseModel):
    """The outcome of parsing one model turn.

    Never a raised exception: a model that emits rubbish is a data point the benchmark
    must record and classify, not an error that ends the run.

    Attributes:
        action: The recovered action, or `None` if nothing could be recovered.
        repairs: Every repair applied, in the order applied, deduplicated.
        error: Why parsing failed; empty when it succeeded.
        raw: The text as the model emitted it.
    """

    model_config = ConfigDict(frozen=True)

    action: Action | None = None
    repairs: tuple[str, ...] = ()
    error: str = ""
    raw: str = ""

    @property
    def ok(self) -> bool:
        """Whether an action was recovered."""
        return self.action is not None

    @property
    def clean(self) -> bool:
        """Whether an action was recovered with no repairs at all."""
        return self.action is not None and not self.repairs


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?\s*(.*?)(?:```|\Z)", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_KEYWORD_RE = re.compile(r"\b(true|false|null)\b")
_KEYWORDS: Final[dict[str, str]] = {"true": "True", "false": "False", "null": "None"}

_ACTION_ALIASES: Final[dict[str, str]] = {
    "tool": "call_tool",
    "tool_call": "call_tool",
    "toolcall": "call_tool",
    "use_tool": "call_tool",
    "call": "call_tool",
    "function_call": "call_tool",
    "answer": "final",
    "final_answer": "final",
    "finish": "final",
    "respond": "final",
    "done": "final",
    "delegate": "handoff",
    "transfer": "handoff",
    "hand_off": "handoff",
    "ask": "clarify",
    "ask_user": "clarify",
    "question": "clarify",
    "clarification": "clarify",
}

_COMMON_ALIASES: Final[dict[str, str]] = {
    "reasoning": "thought",
    "rationale": "thought",
    "reason": "thought",
    "thinking": "thought",
    "scratchpad": "thought",
}

_FIELD_ALIASES: Final[dict[str, dict[str, str]]] = {
    "call_tool": {
        "tool_name": "tool",
        "toolname": "tool",
        "name": "tool",
        "function": "tool",
        "args": "arguments",
        "arg": "arguments",
        "input": "arguments",
        "inputs": "arguments",
        "parameters": "arguments",
        "params": "arguments",
        "tool_input": "arguments",
        "tool_args": "arguments",
        "arguments_json": "arguments",
    },
    "final": {
        "final_answer": "answer",
        "response": "answer",
        "output": "answer",
        "result": "answer",
        "text": "answer",
        "content": "answer",
    },
    "handoff": {
        "agent": "to",
        "target": "to",
        "to_agent": "to",
        "recipient": "to",
        "task": "instruction",
        "message": "instruction",
        "request": "instruction",
    },
    "clarify": {
        "ask": "question",
        "query": "question",
        "clarification": "question",
    },
}

_ACTION_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "call_tool": ("thought", "action", "tool", "arguments"),
    "final": ("thought", "action", "answer"),
    "handoff": ("thought", "action", "to", "instruction"),
    "clarify": ("thought", "action", "question"),
}

_STRING_FIELDS: Final[frozenset[str]] = frozenset(
    {"thought", "tool", "answer", "to", "instruction", "question"}
)

_MAX_UNWRAP_DEPTH: Final = 3
_UNPARSED: Final = object()


def render_action(action: Action) -> str:
    """Serialise an action to the canonical one-line JSON the protocol expects.

    Args:
        action: Any action.

    Returns:
        Compact JSON with the fields in protocol order, which :func:`parse_action`
        round-trips with no repairs.
    """
    return action.model_dump_json()


def _unfence(text: str) -> str | None:
    """Return the contents of the first Markdown fence, or `None` if there is none.

    An unterminated fence counts: a model that hits its token limit mid-answer leaves
    the closing backticks off, and the JSON inside is usually still complete.
    """
    match = _FENCE_RE.search(text)
    if match is None:
        return None
    inner = match.group(1).strip()
    return inner or None


def _find_json_span(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` span, or `None`.

    Brace counting is string-aware (both quote styles, with escapes) so that prose or
    JSON containing braces inside a string value does not throw the depth off.
    """
    start = -1
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            break
    if start < 0:
        return None
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in "\"'":
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _load_json(text: str) -> tuple[Any, tuple[str, ...]]:
    """Parse `text` as JSON, repairing the syntax small models actually get wrong.

    Returns:
        A ``(payload, repairs)`` pair; `payload` is the module sentinel `_UNPARSED`
        when nothing worked.
    """
    try:
        return json.loads(text), ()
    except ValueError:
        pass
    # The trailing-comma regex can in principle fire inside a string literal, but it is
    # only reached once strict JSON has already failed, so a slightly wrong repair is
    # better than no answer - and it is recorded either way.
    without_commas = _TRAILING_COMMA_RE.sub(r"\1", text)
    if without_commas != text:
        try:
            return json.loads(without_commas), ("trailing_comma",)
        except ValueError:
            pass
    tags: tuple[str, ...] = ("single_quotes",) if "'" in text else ("python_literal",)
    try:
        return ast.literal_eval(text), tags
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    swapped = _KEYWORD_RE.sub(lambda m: _KEYWORDS[m.group()], text)
    if swapped != text:
        try:
            return ast.literal_eval(swapped), (*tags, "python_keywords")
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return _UNPARSED, ()


def _payload_candidates(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """Progressively more aggressive views of the model's output, cheapest first.

    The untouched text comes first so that a well-formed action is parsed with no
    repairs recorded at all; only when that fails do the fence and prose strippers run.
    """
    candidates: list[tuple[str, tuple[str, ...]]] = [(text, ())]
    inner = _unfence(text)
    if inner is not None and inner != text:
        candidates.append((inner, ("code_fence",)))
    for base, tags in list(candidates):
        span = _find_json_span(base)
        if span is not None and span != base:
            candidates.append((span, (*tags, "surrounding_prose")))
    return candidates


def _is_object_like(payload: Any) -> bool:
    """Whether a payload can be coerced into a single action object."""
    if isinstance(payload, dict):
        return True
    return isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict)


def _unwrap(payload: Any, repairs: list[str]) -> dict[str, Any]:
    """Strip the wrappers models put around the action object.

    Handles ``[{...}]``, ``{"response": {...}}`` and ``{"action": {"action": ...}}``.
    The payload has already passed :func:`_is_object_like`, so it is a dict or a
    one-element list holding one.
    """
    obj: dict[str, Any] = payload[0] if isinstance(payload, list) else payload
    if isinstance(payload, list):
        repairs.append("single_element_list")
    for _ in range(_MAX_UNWRAP_DEPTH):
        inner = obj.get("action")
        if isinstance(inner, dict):
            repairs.append("unwrapped:action")
            obj = inner
            continue
        if "action" in obj:
            return obj
        nested = [
            (key, value)
            for key, value in obj.items()
            if isinstance(value, dict) and "action" in value
        ]
        if len(nested) != 1:
            return obj
        key, value = nested[0]
        repairs.append(f"unwrapped:{key}")
        obj = value
    return obj


def _normalise_verb(raw: Any, repairs: list[str]) -> tuple[str, str]:
    """Map the emitted action name onto one of :data:`ACTION_NAMES`.

    Returns:
        A ``(verb, error)`` pair; `verb` is empty when the error is set.
    """
    if not isinstance(raw, str):
        return "", f"the 'action' field is {type(raw).__name__}, not a string"
    verb = raw.strip().lower().replace("-", "_").replace(" ", "_")
    verb = _ACTION_ALIASES.get(verb, verb)
    if verb not in ACTION_NAMES:
        return "", f"unknown action {raw!r}"
    if verb != raw:
        repairs.append(f"action_alias:{raw}")
    return verb, ""


def _stringify(value: Any) -> str:
    """Render a non-string field value as the string the schema wants."""
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _coerce_arguments(value: Any, repairs: list[str]) -> tuple[dict[str, Any] | None, str]:
    """Turn whatever landed in ``arguments`` into a dict."""
    if isinstance(value, dict):
        return value, ""
    if value is None:
        repairs.append("null_arguments")
        return {}, ""
    if isinstance(value, str):
        payload, _ = _load_json(value.strip())
        if isinstance(payload, dict):
            repairs.append("arguments_json_string")
            return payload, ""
        return None, "the 'arguments' field is a string that is not a JSON object"
    return None, f"the 'arguments' field is {type(value).__name__}, not an object"


def _normalise_fields(obj: Mapping[str, Any], repairs: list[str]) -> tuple[dict[str, Any], str]:
    """Rename, default and coerce the fields of one action object.

    Returns:
        A ``(payload, error)`` pair ready for validation; `error` is set when the
        object cannot be made to fit any action.
    """
    verb, error = _normalise_verb(obj.get("action"), repairs)
    if error:
        return {}, error

    aliases = _FIELD_ALIASES[verb]
    allowed = _ACTION_FIELDS[verb]
    out: dict[str, Any] = {"action": verb}
    dropped: list[str] = []
    for key, value in obj.items():
        if key == "action":
            continue
        canonical = aliases.get(key) or _COMMON_ALIASES.get(key) or key
        if canonical not in allowed or (canonical in out and canonical != key):
            dropped.append(key)
            continue
        if canonical != key:
            repairs.append(f"alias:{key}->{canonical}")
        out[canonical] = value
    if dropped:
        repairs.append(f"dropped_keys:{','.join(sorted(dropped))}")

    for field_name in list(out):
        if out[field_name] is None and field_name != "arguments":
            del out[field_name]
            repairs.append(f"dropped_null:{field_name}")

    if "arguments" in out:
        arguments, error = _coerce_arguments(out["arguments"], repairs)
        if arguments is None:
            return {}, error
        out["arguments"] = arguments
    elif verb == "call_tool":
        repairs.append("missing_arguments")
        out["arguments"] = {}

    for field_name in allowed:
        if field_name in _STRING_FIELDS and field_name in out:
            value = out[field_name]
            if not isinstance(value, str):
                out[field_name] = _stringify(value)
                repairs.append(f"coerced:{field_name}")

    if "thought" not in out:
        repairs.append("missing_thought")
        out["thought"] = ""
    return out, ""


def _describe(exc: ValidationError) -> str:
    """One-line summary of the first validation error."""
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"] if part != "tagged-union")
    return f"{location}: {first['msg']}" if location else str(first["msg"])


def parse_action(text: str) -> ParsedAction:
    """Recover one action from a model's raw output.

    The repairs are applied cheapest-first, and the untouched text is tried before any
    of them, so well-formed output is reported as well-formed. Everything that had to
    be fixed is listed in :attr:`ParsedAction.repairs`, because format compliance is a
    metric the benchmark reports and a repair that is not counted is a defect that is
    not measured.

    Args:
        text: Exactly what the model emitted, fences, prose and all.

    Returns:
        A :class:`ParsedAction`. On failure `action` is `None` and `error` says why;
        this function never raises.
    """
    stripped = text.strip()
    if not stripped:
        return ParsedAction(error="the model returned no text", raw=text)

    repairs: list[str] = []
    payload: Any = _UNPARSED
    payload_tags: tuple[str, ...] = ()
    fallback: Any = _UNPARSED
    for candidate, tags in _payload_candidates(stripped):
        parsed, syntax_tags = _load_json(candidate)
        if parsed is _UNPARSED:
            continue
        if _is_object_like(parsed):
            payload = parsed
            payload_tags = (*tags, *syntax_tags)
            break
        if fallback is _UNPARSED:
            fallback = parsed
    if payload is _UNPARSED:
        if fallback is not _UNPARSED:
            found = type(fallback).__name__
            return ParsedAction(error=f"expected a JSON object, found {found}", raw=text)
        reason = "unterminated JSON object" if "{" in stripped else "no JSON object found"
        return ParsedAction(error=f"{reason} in the model output", raw=text)

    repairs.extend(payload_tags)
    obj = _unwrap(payload, repairs)
    if "action" not in obj:
        return ParsedAction(
            error="no 'action' field in the JSON object",
            repairs=tuple(dict.fromkeys(repairs)),
            raw=text,
        )
    fields, error = _normalise_fields(obj, repairs)
    if error:
        return ParsedAction(error=error, repairs=tuple(dict.fromkeys(repairs)), raw=text)
    try:
        action = ACTION_ADAPTER.validate_python(fields)
    except ValidationError as exc:
        return ParsedAction(
            error=f"schema violation: {_describe(exc)}",
            repairs=tuple(dict.fromkeys(repairs)),
            raw=text,
        )
    return ParsedAction(action=action, repairs=tuple(dict.fromkeys(repairs)), raw=text)


# --------------------------------------------------------------------------------------
# The system prompt
# --------------------------------------------------------------------------------------


def _example_value(name: str, spec: Any) -> Any:
    """A placeholder argument value for the few-shot example, from a JSON schema node."""
    if not isinstance(spec, dict):
        return f"<{name}>"
    examples = spec.get("examples")
    if isinstance(examples, list) and examples:
        return examples[0]
    defaults: dict[str, Any] = {
        "string": f"<{name}>",
        "integer": 1,
        "number": 1.0,
        "boolean": True,
        "array": [],
        "object": {},
    }
    return defaults.get(str(spec.get("type", "string")), f"<{name}>")


def _example_arguments(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Placeholder arguments for a tool, taken from its input schema."""
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return {}
    required = [key for key in schema.get("required", []) if key in properties]
    names = required or list(properties)[:2]
    return {name: _example_value(name, properties[name]) for name in names}


def _tool_flags(tool: ToolSpec) -> str:
    """The one-line annotation summary shown after a tool's schema."""
    flags = ["read-only" if tool.read_only else "writes data"]
    if tool.destructive:
        flags.append("destructive")
    if tool.requires_approval:
        flags.append("requires human approval")
    return ", ".join(flags)


def _default_examples(
    tools: Sequence[ToolSpec], peers: Sequence[str], allow_clarify: bool
) -> list[Action]:
    """Two or three examples covering a tool call, an answer, and delegation or asking."""
    examples: list[Action] = []
    if tools:
        tool = tools[0]
        examples.append(
            CallToolAction(
                thought="I need this tool's data before I can say anything about it.",
                tool=tool.name,
                arguments=_example_arguments(tool.input_schema),
            )
        )
    examples.append(
        FinalAction(
            thought="The tool returned the figure, so I can answer from it.",
            answer="The account was valued at 412,300.50 AUD as at 30 June 2026.",
        )
    )
    if peers:
        examples.append(
            HandoffAction(
                thought="Fees are not my remit; the fees agent has the schedules.",
                to=peers[0],
                instruction="Reconcile the charged fees for ACC-0007 against its schedule.",
            )
        )
    elif allow_clarify:
        examples.append(
            ClarifyAction(
                thought="Two clients share that surname, so answering would be a guess.",
                question="Do you mean Jordan Blake (CLI-0012) or Robin Blake (CLI-0031)?",
            )
        )
    return examples


def render_system_prompt(
    tools: Sequence[ToolSpec],
    role: str = "assistant",
    *,
    goal: str = "",
    peers: Sequence[str] = (),
    max_steps: int | None = None,
    allow_clarify: bool = True,
    examples: Sequence[Action] | None = None,
) -> str:
    """Build the system prompt: the tool catalogue, the action schema, and examples.

    The few-shot examples are not decoration. Empirically, a 1.5B instruct model given
    only a prose description of the JSON format emits prose; given two or three
    concrete examples of the exact object it is asked for, it emits the object. The
    examples are rendered with :func:`render_action`, the same serialiser the parser
    round-trips, so what the model is shown is exactly what it is scored against.

    Only the actions that are actually available are described: offering ``handoff``
    to an agent with no peers, or ``clarify`` on a benchmark run that cannot answer a
    question, invites a failure mode the deployment does not have.

    Args:
        tools: The tools this agent may call, in the order to advertise them.
        role: The agent's name, e.g. ``supervisor`` or ``fees``.
        goal: One line describing what this agent is for.
        peers: Agents that may be handed off to; empty disables the handoff action.
        max_steps: Step budget to state, if there is one.
        allow_clarify: Whether the clarify action is offered.
        examples: Replaces the default few-shot examples.

    Returns:
        The prompt text, deterministic for a given set of arguments.
    """
    lines: list[str] = [
        f"You are the {role} agent in a wealth-platform assistant.",
        "You work by emitting exactly one JSON object per turn, and nothing else.",
    ]
    if goal:
        lines += ["", f"Your goal: {goal}"]

    lines += ["", "Tools you may call:"]
    if tools:
        for tool in tools:
            schema = json.dumps(tool.input_schema, sort_keys=True, ensure_ascii=False)
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  arguments schema: {schema}")
            lines.append(f"  ({_tool_flags(tool)})")
    else:
        lines.append("- (none: answer from the conversation or say that you cannot)")

    lines += ["", "Your reply must be one JSON object in one of these shapes:"]
    lines.append(
        '  {"thought": "<one sentence>", "action": "call_tool", '
        '"tool": "<tool name>", "arguments": {<matching the schema above>}}'
    )
    lines.append('  {"thought": "<one sentence>", "action": "final", "answer": "<your answer>"}')
    if peers:
        lines.append(
            '  {"thought": "<one sentence>", "action": "handoff", '
            f'"to": "<one of: {", ".join(peers)}>", "instruction": "<what they should do>"}}'
        )
    if allow_clarify:
        lines.append(
            '  {"thought": "<one sentence>", "action": "clarify", "question": "<what you need>"}'
        )

    lines += ["", "Rules:"]
    lines.append("- Emit the JSON object alone: no prose around it, no code fence, one line.")
    lines.append("- Call one tool per turn and wait for its result before choosing the next.")
    lines.append("- Use only tool names listed above, with arguments that match the schema.")
    lines.append("- Never state a number or a name that a tool has not returned to you.")
    lines.append("- If the tools cannot answer the question, say so with the final action.")
    if max_steps is not None:
        lines.append(f"- You have at most {max_steps} steps; leave one for the final answer.")

    chosen = (
        list(examples) if examples is not None else _default_examples(tools, peers, allow_clarify)
    )
    if chosen:
        lines += ["", "Examples of well-formed replies:"]
        for example in chosen:
            lines.append("```json")
            lines.append(render_action(example))
            lines.append("```")
    return "\n".join(lines)
