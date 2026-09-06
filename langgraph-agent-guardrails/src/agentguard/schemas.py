"""The action contract between the model and the graph, the typed graph state, and the
event records that end up in the audit trail.

The model must answer with exactly one JSON object::

    {"type": "tool", "tool": "<name>", "args": {...}}   or   {"type": "final", "answer": "..."}

Plain text without JSON is accepted as a final answer (``fallback=True``) so a small model
that forgets the envelope still terminates; JSON that *is* present but does not validate is
an ``ActionParseError`` and triggers a repair turn.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

# ----- actions ------------------------------------------------------------------------------


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["tool"]
    tool: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["final"]
    answer: str


Action = Annotated[ToolCall | FinalAnswer, Field(discriminator="type")]
_ACTION_ADAPTER: TypeAdapter[ToolCall | FinalAnswer] = TypeAdapter(Action)


class ActionParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedAction:
    action: ToolCall | FinalAnswer
    raw: str
    fallback: bool = False


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        nl = stripped.find("\n")
        stripped = stripped[nl + 1 :] if nl != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _candidate_objects(text: str, limit: int = 8) -> list[Any]:
    """Every JSON object that starts at some ``{`` in the text (first ``limit`` starts)."""
    decoder = json.JSONDecoder()
    found: list[Any] = []
    start = text.find("{")
    tries = 0
    while start != -1 and tries < limit:
        tries += 1
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(value, dict):
                found.append(value)
        start = text.find("{", start + 1)
    return found


def normalise_envelope(obj: dict[str, Any]) -> dict[str, Any]:
    """Repair the two envelope mistakes small models make constantly: ``"type"`` set to the
    tool name instead of ``"tool"``, and a missing ``"type"`` altogether. Arguments are still
    validated strictly by the tool registry, so leniency here costs nothing."""
    kind = obj.get("type")
    if kind in ("tool", "final"):
        return obj
    fixed = dict(obj)
    if "tool" in fixed:
        fixed["type"] = "tool"
    elif "answer" in fixed:
        fixed["type"] = "final"
    elif isinstance(kind, str) and kind and "args" in fixed:
        fixed["tool"] = kind
        fixed["type"] = "tool"
    return fixed


def parse_action(text: str) -> ParsedAction:
    body = _strip_fences(text)
    candidates = [normalise_envelope(c) for c in _candidate_objects(body)]
    if not candidates:
        if not body:
            msg = "empty model output"
            raise ActionParseError(msg)
        if body.startswith("{") or '"type"' in body:
            msg = "output looks like a truncated or malformed JSON action"
            raise ActionParseError(msg)
        return ParsedAction(FinalAnswer(type="final", answer=body), raw=text, fallback=True)
    errors: list[str] = []
    for obj in candidates:
        try:
            return ParsedAction(_ACTION_ADAPTER.validate_python(obj), raw=text)
        except ValidationError as exc:
            errors.append(str(exc).splitlines()[0])
    msg = f"JSON present but not a valid action: {errors[0]}"
    raise ActionParseError(msg)


# ----- records ------------------------------------------------------------------------------

Stage = Literal["input", "tool_args", "tool_output", "output"]
RailAction = Literal["allow", "redact", "block", "flag", "modify"]


class GuardrailEvent(BaseModel):
    rail: str
    stage: Stage
    action: RailAction
    score: float = 0.0
    detail: str = ""


class ToolRecord(BaseModel):
    step: int
    tool: str
    args: dict[str, Any]
    ok: bool
    output: str
    risk: Literal["low", "high"]
    approved: bool | None = None
    latency_s: float = 0.0


Status = Literal["ok", "blocked", "awaiting_approval", "rejected", "max_steps", "error"]


# ----- graph state --------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """LangGraph state. Lists with ``operator.add`` accumulate across nodes and turns."""

    thread_id: str
    turn: int
    user_input: str
    sanitized_input: str
    messages: Annotated[list[dict[str, str]], operator.add]
    steps: int
    parse_failures: int
    tool_records: Annotated[list[dict[str, Any]], operator.add]
    guardrail_events: Annotated[list[dict[str, Any]], operator.add]
    pending_action: dict[str, Any] | None
    approval: dict[str, Any] | None
    final_answer: str | None
    status: Status
