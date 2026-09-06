from __future__ import annotations

import pytest
from pydantic import BaseModel

from ragpipe.evaluation.judge import REPAIR_SUFFIX, Judge, JudgeParseError, extract_json
from ragpipe.llm import FakeLLM


class Verdict(BaseModel):
    ok: bool
    n: int = 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"ok": true}', {"ok": True}),
        ('```json\n{"ok": true, "n": 2}\n```', {"ok": True, "n": 2}),
        ('Sure! Here is the JSON: {"ok": false} and some trailing prose', {"ok": False}),
        ("[1, 2, 3]", [1, 2, 3]),
        ('```\n{"a": {"b": [1]}}```', {"a": {"b": [1]}}),
    ],
)
def test_extract_json_variants(text: str, expected: object) -> None:
    assert extract_json(text) == expected


@pytest.mark.parametrize("text", ["no json here", "{not: valid}", ""])
def test_extract_json_failures(text: str) -> None:
    with pytest.raises(JudgeParseError):
        extract_json(text)


def test_judge_parses_first_attempt_and_logs_call() -> None:
    judge = Judge(FakeLLM(default='{"ok": true, "n": 3}'))
    out = judge.ask("m", "prompt", Verdict)
    assert out == Verdict(ok=True, n=3)
    assert len(judge.calls) == 1
    assert judge.calls[0].ok and judge.calls[0].attempt == 0
    assert judge.parse_failures == 0
    assert judge.name == "fake"


def test_judge_retries_with_repair_suffix_then_succeeds() -> None:
    llm = FakeLLM(rules=[(REPAIR_SUFFIX.strip()[:20], '{"ok": true}')], default="garbage")
    judge = Judge(llm, max_retries=1)
    out = judge.ask("m", "prompt", Verdict)
    assert out == Verdict(ok=True)
    assert [c.ok for c in judge.calls] == [False, True]
    assert llm.calls[1].prompt == "prompt" + REPAIR_SUFFIX
    assert judge.parse_failures == 0


def test_judge_returns_none_after_exhausting_retries_and_counts_failure() -> None:
    judge = Judge(FakeLLM(default='{"ok": "not-a-bool"}'), max_retries=2)
    assert judge.ask("m", "p", Verdict) is None
    assert len(judge.calls) == 3
    assert judge.parse_failures == 1
    assert "validation error" in judge.calls[0].error.lower() or judge.calls[0].error
