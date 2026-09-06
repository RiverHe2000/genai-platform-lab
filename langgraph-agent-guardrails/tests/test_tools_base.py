from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ConfigDict

from agentguard.tools.base import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ToolValidationError,
)


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    label: str = "default"


def _handler(args: BaseModel, ctx: ToolContext) -> ToolResult:
    return ToolResult(ok=True, output=f"{args} for {ctx.thread_id}")


def _boom(args: BaseModel, ctx: ToolContext) -> ToolResult:
    del args, ctx
    raise RuntimeError("kaboom")


def test_registry_register_describe_validate_run() -> None:
    reg = ToolRegistry()
    spec = ToolSpec(name="double", description="Doubles x.", args_model=Args, handler=_handler)
    reg.register(spec)
    reg.register(
        ToolSpec(name="boom", description="Fails.", args_model=Args, handler=_boom, risk="high")
    )
    with pytest.raises(ValueError, match="already registered"):
        reg.register(spec)
    assert reg.names == ["double", "boom"]
    text = reg.describe()
    assert "- double: Doubles x." in text and "(requires human approval)" in text
    schema = json.loads(text.split("args schema: ")[1].splitlines()[0])
    assert schema["required"] == ["x"] and schema["properties"]["x"]["type"] == "integer"

    found, args = reg.validate("double", {"x": 2})
    assert found is spec and isinstance(args, Args) and args.label == "default"
    with pytest.raises(ToolValidationError, match="unknown tool"):
        reg.validate("nope", {})
    with pytest.raises(ToolValidationError, match="invalid arguments for double: x"):
        reg.validate("double", {"x": "two"})
    with pytest.raises(ToolValidationError, match="Extra inputs"):
        reg.validate("double", {"x": 1, "y": 2})

    result, latency = reg.run(spec, args, ToolContext("t", 1))
    assert result.ok and "for t" in result.output and latency >= 0
    failed, _ = reg.run(reg.get("boom") or spec, args, ToolContext("t", 1))
    assert not failed.ok and "RuntimeError: kaboom" in failed.output
    assert reg.get("missing") is None
