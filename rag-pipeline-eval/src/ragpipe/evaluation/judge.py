"""LLM-as-judge plumbing: strict JSON extraction, schema validation, one repair retry, and
an audit trail of every call. A judge that cannot produce valid JSON yields ``None`` — the
metric is then reported as *missing*, never silently as 0 or 1."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ragpipe.llm import LLM

JUDGE_SYSTEM_PROMPT = (
    "You are a meticulous evaluator. Follow the instructions exactly and reply with a single "
    "JSON object and nothing else — no prose, no markdown fences."
)
REPAIR_SUFFIX = (
    "\n\nYour previous reply was not valid JSON matching the required schema. "
    "Reply again with ONLY the JSON object."
)

T = TypeVar("T", bound=BaseModel)


class JudgeParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class JudgeCall:
    metric: str
    attempt: int
    ok: bool
    latency_s: float
    raw: str
    error: str = ""


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        stripped = stripped[first_newline + 1 :] if first_newline != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def extract_json(text: str) -> Any:
    """Return the first JSON object/array embedded in ``text``."""
    body = _strip_fences(text)
    starts = [i for i in (body.find("{"), body.find("[")) if i != -1]
    if not starts:
        msg = "no JSON object found"
        raise JudgeParseError(msg)
    start = min(starts)
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(body[start:])
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON: {exc.msg}"
        raise JudgeParseError(msg) from exc
    return value


class Judge:
    def __init__(
        self,
        llm: LLM,
        *,
        max_retries: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: str = JUDGE_SYSTEM_PROMPT,
    ) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._system_prompt = system_prompt
        self.calls: list[JudgeCall] = []

    @property
    def name(self) -> str:
        return self._llm.name

    @property
    def parse_failures(self) -> int:
        """Number of prompts that never produced a valid reply (after retries)."""
        return sum(1 for c in self.calls if not c.ok and c.attempt == self._max_retries)

    def ask(self, metric: str, prompt: str, schema: type[T]) -> T | None:
        current = prompt
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            response = self._llm.complete(
                current,
                system=self._system_prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            latency = time.perf_counter() - started
            try:
                parsed = schema.model_validate(extract_json(response.text))
            except (JudgeParseError, ValidationError) as exc:
                self.calls.append(
                    JudgeCall(
                        metric=metric,
                        attempt=attempt,
                        ok=False,
                        latency_s=latency,
                        raw=response.text,
                        error=str(exc)[:200],
                    )
                )
                current = prompt + REPAIR_SUFFIX
                continue
            self.calls.append(
                JudgeCall(
                    metric=metric, attempt=attempt, ok=True, latency_s=latency, raw=response.text
                )
            )
            return parsed
        return None
