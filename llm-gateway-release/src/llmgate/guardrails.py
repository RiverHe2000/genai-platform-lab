"""Request and response rails at the gateway boundary.

Request: prompt-size and max-token caps, blocked terms, prompt-injection heuristics, PII
redaction of user turns. Response: PII redaction/blocking, JSON-schema enforcement for
structured output (with a repair round-trip handled by the gateway), and a streaming
redactor that holds back a small tail window so PII split across chunks never reaches the
client.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jsonschema

from llmgate.config import GuardrailSettings
from llmgate.protocol import ChatCompletionRequest, ChatMessage

# ----- PII ----------------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_AU_PHONE_RE = re.compile(
    r"(?<![\w-])(?:\+61[ -]?|0)(?:4\d{2}[ -]?\d{3}[ -]?\d{3}|[2378][ -]?\d{4}[ -]?\d{4})(?![\w-])"
)
_TFN_RE = re.compile(r"(?<![\w-])\d{3}[ -]?\d{3}[ -]?\d{2,3}(?![\w-])")
_CARD_RE = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_TFN_WEIGHTS_9 = (1, 4, 3, 7, 5, 8, 6, 9, 10)
_TFN_WEIGHTS_8 = (10, 7, 8, 4, 6, 3, 5, 1)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def tfn_valid(number: str) -> bool:
    digits = _digits(number)
    weights: tuple[int, ...]
    if len(digits) == 9:
        weights = _TFN_WEIGHTS_9
    elif len(digits) == 8:
        weights = _TFN_WEIGHTS_8
    else:
        return False
    return sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11 == 0


def luhn_valid(number: str) -> bool:
    digits = _digits(number)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


@dataclass(frozen=True, slots=True)
class PIIMatch:
    kind: str
    start: int
    end: int


def detect_pii(text: str) -> list[PIIMatch]:
    raw: list[PIIMatch] = []
    raw += [
        PIIMatch("CARD", m.start(), m.end())
        for m in _CARD_RE.finditer(text)
        if luhn_valid(m.group())
    ]
    raw += [
        PIIMatch("TFN", m.start(), m.end()) for m in _TFN_RE.finditer(text) if tfn_valid(m.group())
    ]
    raw += [PIIMatch("PHONE", m.start(), m.end()) for m in _AU_PHONE_RE.finditer(text)]
    raw += [PIIMatch("EMAIL", m.start(), m.end()) for m in _EMAIL_RE.finditer(text)]
    raw.sort(key=lambda p: (p.start, -(p.end - p.start)))
    kept: list[PIIMatch] = []
    for p in raw:
        if kept and p.start < kept[-1].end:
            continue
        kept.append(p)
    return kept


def redact_pii(text: str) -> tuple[str, list[str]]:
    matches = detect_pii(text)
    if not matches:
        return text, []
    out: list[str] = []
    cursor = 0
    for m in matches:
        out.append(text[cursor : m.start])
        out.append(f"[{m.kind}]")
        cursor = m.end
    out.append(text[cursor:])
    return "".join(out), sorted({m.kind for m in matches})


# ----- prompt injection ---------------------------------------------------------------------

_INJECTION_PATTERNS: tuple[tuple[str, float], ...] = (
    (
        r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|all|your)\b.{0,20}\b(instructions?|rules?|prompt)",
        0.9,
    ),
    (r"\b(reveal|print|show|repeat)\b.{0,30}\b(system prompt|hidden instructions?)\b", 0.8),
    (r"\b(developer mode|jailbreak|do anything now|no restrictions)\b", 0.8),
    (r"\byou are now\b|\bpretend (to be|you are)\b", 0.5),
    (r"(^|\n)\s*(system|assistant)\s*:", 0.4),
    (r"<\s*/?\s*(system|instructions?)\s*>", 0.6),
)
_INJECTION = [(re.compile(p, re.IGNORECASE | re.DOTALL), w) for p, w in _INJECTION_PATTERNS]


def injection_score(text: str) -> float:
    survive = 1.0
    for regex, weight in _INJECTION:
        if regex.search(text):
            survive *= 1.0 - weight
    return round(1.0 - survive, 4)


# ----- events -------------------------------------------------------------------------------

Action = Literal["block", "redact", "flag", "modify"]


@dataclass(frozen=True, slots=True)
class GuardEvent:
    rail: str
    action: Action
    detail: str = ""


class GuardrailBlockedError(Exception):
    def __init__(self, message: str, *, rail: str) -> None:
        super().__init__(message)
        self.rail = rail


# ----- request ------------------------------------------------------------------------------


class RequestGuard:
    def __init__(self, settings: GuardrailSettings) -> None:
        self._s = settings

    def check(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionRequest, list[GuardEvent]]:
        s = self._s
        events: list[GuardEvent] = []
        if request.prompt_chars() > s.max_prompt_chars:
            msg = f"prompt exceeds {s.max_prompt_chars} characters"
            raise GuardrailBlockedError(msg, rail="length")
        user_text = "\n".join(m.content for m in request.messages if m.role == "user")
        lowered = user_text.lower()
        for term in s.blocked_terms:
            if term.lower() in lowered:
                msg = "request contains a blocked term"
                raise GuardrailBlockedError(msg, rail="blocked_terms")
        score = injection_score(user_text)
        if score >= s.injection_threshold:
            msg = f"request looks like a prompt-injection attempt (score {score:.2f})"
            raise GuardrailBlockedError(msg, rail="injection")
        updates: dict[str, Any] = {}
        if request.max_tokens is not None and request.max_tokens > s.max_tokens_cap:
            updates["max_tokens"] = s.max_tokens_cap
            events.append(GuardEvent("max_tokens", "modify", f"capped to {s.max_tokens_cap}"))
        if s.redact_input_pii:
            new_messages: list[ChatMessage] = []
            kinds: set[str] = set()
            for m in request.messages:
                if m.role == "user":
                    text, found = redact_pii(m.content)
                    kinds.update(found)
                    new_messages.append(ChatMessage(role=m.role, content=text))
                else:
                    new_messages.append(m)
            if kinds:
                updates["messages"] = new_messages
                events.append(GuardEvent("pii", "redact", ", ".join(sorted(kinds))))
        if updates:
            request = request.model_copy(update=updates)
        return request, events


# ----- response -----------------------------------------------------------------------------


@dataclass(slots=True)
class ResponseVerdict:
    text: str
    events: list[GuardEvent] = field(default_factory=list)
    blocked: bool = False
    json_invalid: bool = False
    json_error: str = ""


def extract_json_object(text: str) -> Any:
    body = text.strip()
    if body.startswith("```"):
        nl = body.find("\n")
        body = body[nl + 1 :] if nl != -1 else body[3:]
        if body.rstrip().endswith("```"):
            body = body.rstrip()[:-3]
    decoder = json.JSONDecoder()
    start = body.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(body[start:])
        except json.JSONDecodeError:
            start = body.find("{", start + 1)
            continue
        return value
    msg = "no JSON object found"
    raise ValueError(msg)


class ResponseGuard:
    def __init__(self, settings: GuardrailSettings) -> None:
        self._s = settings

    def check(self, text: str, request: ChatCompletionRequest) -> ResponseVerdict:
        s = self._s
        verdict = ResponseVerdict(text=text)
        redacted, kinds = redact_pii(text)
        if kinds:
            if s.block_output_pii:
                verdict.blocked = True
                verdict.text = ""
                verdict.events.append(GuardEvent("pii", "block", ", ".join(kinds)))
                return verdict
            if s.redact_output_pii:
                verdict.text = redacted
                verdict.events.append(GuardEvent("pii", "redact", ", ".join(kinds)))
        fmt = request.response_format
        if s.enforce_json_schema and fmt is not None and fmt.type != "text":
            try:
                value = extract_json_object(verdict.text)
                schema = fmt.schema_dict
                if schema is not None:
                    jsonschema.validate(value, schema)
            except (ValueError, jsonschema.ValidationError) as exc:
                verdict.json_invalid = True
                verdict.json_error = str(exc).splitlines()[0][:200]
                verdict.events.append(GuardEvent("json_schema", "flag", verdict.json_error))
            else:
                verdict.text = json.dumps(value, ensure_ascii=False)
        return verdict


class StreamRedactor:
    """Redacts PII in a token stream by holding back a tail window (``hold`` characters)
    and only emitting text up to a whitespace boundary that does not split a digit group."""

    def __init__(self, hold: int = 48) -> None:
        self._hold = hold
        self._buffer = ""
        self.kinds: set[str] = set()

    def _emit(self, text: str) -> str:
        redacted, kinds = redact_pii(text)
        self.kinds.update(kinds)
        return redacted

    def feed(self, text: str) -> str:
        self._buffer += text
        limit = len(self._buffer) - self._hold
        if limit <= 0:
            return ""
        cut = -1
        for i in range(limit, 0, -1):
            if self._buffer[i].isspace():
                before = self._buffer[i - 1] if i > 0 else " "
                after = self._buffer[i + 1] if i + 1 < len(self._buffer) else " "
                if not (before.isdigit() and after.isdigit()):
                    cut = i
                    break
        if cut == -1:
            if len(self._buffer) < 4 * self._hold:
                return ""
            cut = limit
        out, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return self._emit(out)

    def flush(self) -> str:
        out, self._buffer = self._buffer, ""
        return self._emit(out) if out else ""


def pii_kinds_in(texts: Sequence[str]) -> set[str]:
    kinds: set[str] = set()
    for t in texts:
        kinds.update(redact_pii(t)[1])
    return kinds
