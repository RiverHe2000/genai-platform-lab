from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

log = logging.getLogger(__name__)

Risk = Literal["low", "high"]


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    output: str
    data: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class ToolContext:
    thread_id: str
    step: int
    approved_by: str | None = None


Handler = Callable[[BaseModel, ToolContext], ToolResult]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler
    risk: Risk = "low"

    def schema(self) -> dict[str, Any]:
        raw = self.args_model.model_json_schema()
        return {
            "type": "object",
            "properties": {
                k: {kk: vv for kk, vv in v.items() if kk in ("type", "description", "enum")}
                for k, v in raw.get("properties", {}).items()
            },
            "required": raw.get("required", []),
        }


class ToolValidationError(ValueError):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            msg = f"tool {spec.name!r} already registered"
            raise ValueError(msg)
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def describe(self) -> str:
        """Tool catalogue for the system prompt."""
        lines = []
        for spec in self._tools.values():
            flag = " (requires human approval)" if spec.risk == "high" else ""
            lines.append(
                f"- {spec.name}{flag}: {spec.description}\n  args schema: "
                f"{json.dumps(spec.schema(), separators=(',', ':'))}"
            )
        return "\n".join(lines)

    def validate(self, name: str, args: dict[str, Any]) -> tuple[ToolSpec, BaseModel]:
        spec = self._tools.get(name)
        if spec is None:
            msg = f"unknown tool {name!r}; available: {', '.join(self._tools)}"
            raise ToolValidationError(msg)
        try:
            return spec, spec.args_model.model_validate(args)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(x) for x in first["loc"]) or "args"
            msg = f"invalid arguments for {name}: {loc}: {first['msg']}"
            raise ToolValidationError(msg) from exc

    def run(self, spec: ToolSpec, args: BaseModel, ctx: ToolContext) -> tuple[ToolResult, float]:
        started = time.perf_counter()
        try:
            result = spec.handler(args, ctx)
        except Exception as exc:  # a tool bug must not crash the graph
            log.exception("tool %s failed", spec.name)
            result = ToolResult(ok=False, output=f"error: {type(exc).__name__}: {exc}")
        return result, time.perf_counter() - started
