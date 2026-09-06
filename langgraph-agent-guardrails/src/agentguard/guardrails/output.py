"""Output rails: PII leakage, numeric grounding, advice language, length.

*Numeric grounding* is the cheap, deterministic cousin of a faithfulness judge: every number
in the answer must appear in the evidence the agent actually saw (user input and tool
outputs). Numbers the model invents are flagged and counted in the evaluation as
``ungrounded_number_rate``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from agentguard.config import GuardrailPolicy
from agentguard.guardrails import pii
from agentguard.schemas import GuardrailEvent

_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")
_ADVICE_RE = re.compile(
    r"\b(you should|i recommend|we recommend|my advice|best option|you ought to)\b", re.IGNORECASE
)
DISCLAIMER = (
    "\n\n(This is information about bank policy and portfolio data, not personal financial advice.)"
)


def _normalise_number(token: str) -> str:
    cleaned = token.replace(",", "").lstrip("+")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def extract_numbers(text: str) -> set[str]:
    return {_normalise_number(m.group()) for m in _NUMBER_RE.finditer(text)}


def ungrounded_numbers(answer: str, evidence: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    for chunk in evidence:
        seen |= extract_numbers(chunk)
    return sorted(n for n in extract_numbers(answer) if n not in seen)


@dataclass(slots=True)
class OutputVerdict:
    answer: str
    blocked: bool = False
    events: list[GuardrailEvent] = field(default_factory=list)


def check_output(answer: str, *, evidence: Sequence[str], policy: GuardrailPolicy) -> OutputVerdict:
    verdict = OutputVerdict(answer=answer)

    redacted, matches = pii.redact(answer)
    if matches:
        kinds = sorted({m.kind for m in matches})
        if policy.pii_output_action == "block":
            verdict.blocked = True
            verdict.answer = "I can't share that response because it contains personal information."
            verdict.events.append(
                GuardrailEvent(
                    rail="pii", stage="output", action="block", score=1.0, detail=", ".join(kinds)
                )
            )
            return verdict
        verdict.answer = redacted
        verdict.events.append(
            GuardrailEvent(
                rail="pii", stage="output", action="redact", score=1.0, detail=", ".join(kinds)
            )
        )

    if policy.require_grounded_numbers:
        missing = ungrounded_numbers(verdict.answer, evidence)
        if missing:
            verdict.events.append(
                GuardrailEvent(
                    rail="numeric_grounding",
                    stage="output",
                    action="flag",
                    score=min(1.0, len(missing) / 3),
                    detail=", ".join(missing[:8]),
                )
            )

    if policy.advice_disclaimer and _ADVICE_RE.search(verdict.answer):
        verdict.answer = verdict.answer.rstrip() + DISCLAIMER
        verdict.events.append(
            GuardrailEvent(rail="advice_language", stage="output", action="modify", score=0.5)
        )

    if len(verdict.answer) > policy.max_output_chars:
        verdict.answer = verdict.answer[: policy.max_output_chars].rstrip() + " [truncated]"
        verdict.events.append(
            GuardrailEvent(rail="length", stage="output", action="modify", score=0.2)
        )
    return verdict
