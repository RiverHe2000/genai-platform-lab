from __future__ import annotations

import math

import pytest

from agentguard.tools.base import ToolContext
from agentguard.tools.calculator import CALCULATOR, CalculatorArgs, CalculatorError, evaluate

CTX = ToolContext(thread_id="t", step=1)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(1250000 * 0.8) / 12", 83333.333333),
        ("2 ** 10", 1024),
        ("-5 % 3", 1),
        ("7 // 2", 3),
        ("sqrt(16) + abs(-2)", 6),
        ("round(2.567, 2)", 2.57),
        ("max(1, 5, 3) - min(4, 2)", 3),
        ("log(e) + pi - pi", 1),
        ("log10(1000)", 3),
        ("+3", 3),
    ],
)
def test_evaluates_arithmetic(expr: str, expected: float) -> None:
    assert evaluate(expr) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "payload",
    [
        "__import__('os').system('dir')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('secrets.txt').read()",
        "x = 1",
        "lambda: 1",
        "[1, 2, 3]",
        "'a' * 1000",
        "1 if True else 2",
        "abs.__doc__",
        "sqrt(x=4)",
        "exec('1')",
        "1 < 2",
        "1 and 2",
        "pow(2, 3)",
        "unknown_name + 1",
        "True + 1",
    ],
)
def test_rejects_non_arithmetic(payload: str) -> None:
    with pytest.raises(CalculatorError):
        evaluate(payload)


def test_rejects_resource_exhaustion() -> None:
    with pytest.raises(CalculatorError, match="exponent"):
        evaluate("9 ** 99999")
    with pytest.raises(CalculatorError):
        evaluate("-" * 25 + "1")  # 25 nested unary minuses
    with pytest.raises(CalculatorError, match="too long"):
        evaluate("1+" * 150 + "1")
    with pytest.raises(CalculatorError):
        evaluate("10 ** 400 * 10 ** 400")


def test_division_by_zero_and_domain_errors() -> None:
    with pytest.raises(CalculatorError, match="division by zero"):
        evaluate("1 / 0")
    with pytest.raises(CalculatorError):
        evaluate("sqrt(-1)")
    with pytest.raises(CalculatorError):
        evaluate("1 +")


def test_tool_spec_handles_errors_and_formats_results() -> None:
    ok = CALCULATOR.handler(CalculatorArgs(expression="1250000 * 0.8"), CTX)
    assert ok.ok and ok.output == "1250000 * 0.8 = 1000000"
    assert ok.data["value"] == 1000000.0
    frac = CALCULATOR.handler(CalculatorArgs(expression="10 / 3"), CTX)
    assert frac.output == "10 / 3 = 3.333333"
    bad = CALCULATOR.handler(CalculatorArgs(expression="__import__('os')"), CTX)
    assert not bad.ok and bad.output.startswith("calculator error")
    assert CALCULATOR.risk == "low"
    assert math.isfinite(evaluate("exp(1)"))
