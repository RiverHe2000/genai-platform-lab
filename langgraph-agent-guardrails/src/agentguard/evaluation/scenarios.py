"""Scenario specification: one user turn plus the expectations a passing agent must meet.

``fake_responses`` scripts the model when the evaluation runs with ``--model fake``: that
turns the scenario suite into a deterministic test of the *graph and rails* (what the CI
gate needs), while the same scenarios run unchanged against a real model to measure it.
Templates inside a fake response: ``{{tool_output}}`` (text of the last tool result) and
``{{last_number}}`` (last number in it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InjectedClause(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    text: str


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    user_input: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    adversarial: bool = False
    expected_status: list[str] = Field(default_factory=lambda: ["ok"])
    expected_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    expected_rails: list[str] = Field(default_factory=list)
    forbidden_rails: list[str] = Field(default_factory=list)
    approve: bool | None = None
    fake_responses: list[str] = Field(default_factory=list)
    inject_policy: InjectedClause | None = None


def load_scenarios(path: Path | str) -> list[Scenario]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                msg = f"{path}:{line_no}: invalid JSON ({exc.msg})"
                raise ValueError(msg) from exc
    scenarios = [Scenario.model_validate(r) for r in rows]
    ids = [s.id for s in scenarios]
    if len(set(ids)) != len(ids):
        msg = f"duplicate scenario ids in {path}"
        raise ValueError(msg)
    if not scenarios:
        msg = f"no scenarios in {path}"
        raise ValueError(msg)
    return scenarios


def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(any(item == candidate for candidate in it) for item in needle)
