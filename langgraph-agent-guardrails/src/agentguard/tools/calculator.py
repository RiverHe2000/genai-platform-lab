"""A calculator that evaluates arithmetic by walking the AST with an allow-list.

``eval`` on model-generated text is remote code execution; this evaluator accepts only
numeric literals, the arithmetic operators, a handful of maths functions, and rejects
attribute access, subscripts, names, lambdas, comprehensions, huge exponents and deep
nesting. The tests throw the classic payloads at it.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentguard.tools.base import ToolContext, ToolResult, ToolSpec

MAX_EXPRESSION_LENGTH = 200
MAX_DEPTH = 20
MAX_EXPONENT = 512
MAX_ABS = 1e300

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}
_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}


class CalculatorError(ValueError):
    pass


def _check(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = "non-numeric result"
        raise CalculatorError(msg)
    if not math.isfinite(value) or abs(value) > MAX_ABS:
        msg = "result is not a finite number"
        raise CalculatorError(msg)
    return value


def _eval(node: ast.AST, depth: int) -> float:
    if depth > MAX_DEPTH:
        msg = "expression nested too deeply"
        raise CalculatorError(msg)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            msg = f"unsupported literal {node.value!r}"
            raise CalculatorError(msg)
        return _check(node.value)
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        msg = f"unknown name {node.id!r}"
        raise CalculatorError(msg)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _check(_UNARY[type(node.op)](_eval(node.operand, depth + 1)))
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, depth + 1)
        right = _eval(node.right, depth + 1)
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                msg = "exponent too large"
                raise CalculatorError(msg)
            try:
                return _check(float(left) ** float(right))
            except (OverflowError, ZeroDivisionError, ValueError) as exc:
                raise CalculatorError(str(exc)) from exc
        fn = _BINARY.get(type(node.op))
        if fn is None:
            msg = f"unsupported operator {type(node.op).__name__}"
            raise CalculatorError(msg)
        try:
            return _check(fn(left, right))
        except ZeroDivisionError as exc:
            msg = "division by zero"
            raise CalculatorError(msg) from exc
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
            msg = "only the documented functions may be called"
            raise CalculatorError(msg)
        args = [_eval(a, depth + 1) for a in node.args]
        try:
            return _check(_FUNCTIONS[node.func.id](*args))
        except (ValueError, TypeError, OverflowError) as exc:
            raise CalculatorError(str(exc)) from exc
    msg = f"unsupported syntax: {type(node).__name__}"
    raise CalculatorError(msg)


def evaluate(expression: str) -> float:
    if len(expression) > MAX_EXPRESSION_LENGTH:
        msg = "expression too long"
        raise CalculatorError(msg)
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError) as exc:
        msg = f"cannot parse expression: {exc.msg if isinstance(exc, SyntaxError) else exc}"
        raise CalculatorError(msg) from exc
    return _eval(tree.body, 0)


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        description="Arithmetic expression, e.g. '(2500 + 750) * 1.05'. Functions: "
        "abs, round, min, max, sqrt, log, log10, exp."
    )


def _handle(args: BaseModel, ctx: ToolContext) -> ToolResult:
    del ctx
    expression = str(getattr(args, "expression", ""))
    try:
        value = evaluate(expression)
    except CalculatorError as exc:
        return ToolResult(ok=False, output=f"calculator error: {exc}")
    rendered: Any = (
        int(value) if float(value).is_integer() and abs(value) < 1e15 else round(value, 6)
    )
    return ToolResult(ok=True, output=f"{expression.strip()} = {rendered}", data={"value": value})


CALCULATOR = ToolSpec(
    name="calculate",
    description="Evaluate an arithmetic expression exactly. Use it for every number you derive.",
    args_model=CalculatorArgs,
    handler=_handle,
    risk="low",
)
