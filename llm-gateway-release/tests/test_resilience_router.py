from __future__ import annotations

import asyncio

import pytest

from llmgate.config import RoutingSettings
from llmgate.resilience import Bulkhead, BulkheadFullError, CircuitBreaker, with_retries
from llmgate.router import Router, UnknownModelError, stable_bucket


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def state(b: CircuitBreaker) -> str:
    return b.state


def test_breaker_state_machine() -> None:
    clock = Clock()
    b = CircuitBreaker("x", failure_threshold=3, recovery_s=10, clock=clock)
    assert state(b) == "closed" and b.allow()
    b.record_failure()
    b.record_failure()
    assert state(b) == "closed" and b.failures == 2
    b.record_success()
    assert b.failures == 0
    for _ in range(3):
        b.record_failure()
    assert state(b) == "open" and not b.allow()
    clock.t += 9
    assert state(b) == "open"
    clock.t += 1.5
    assert state(b) == "half_open"
    assert b.allow() and not b.allow()  # a single probe
    b.record_failure()
    assert state(b) == "open" and not b.allow()
    clock.t += 11
    assert b.allow()
    b.record_success()
    assert state(b) == "closed" and b.allow() and b.failures == 0


async def test_with_retries_only_on_retriable() -> None:
    calls = {"n": 0}
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("retry me")
        return "ok"

    assert (
        await with_retries(
            flaky,
            retries=3,
            backoff_s=0.1,
            retriable=lambda e: isinstance(e, ValueError),
            sleep=sleep,
        )
        == "ok"
    )
    assert calls["n"] == 3 and sleeps == pytest.approx([0.1, 0.2])

    async def fatal() -> str:
        raise KeyError("no")

    with pytest.raises(KeyError):
        await with_retries(
            fatal,
            retries=3,
            backoff_s=0.1,
            retriable=lambda e: isinstance(e, ValueError),
            sleep=sleep,
        )

    async def always() -> str:
        raise ValueError("always")

    with pytest.raises(ValueError):
        await with_retries(always, retries=2, backoff_s=0.0, retriable=lambda _e: True, sleep=sleep)


async def test_bulkhead_limits_and_times_out() -> None:
    bh = Bulkhead(1, queue_timeout_s=0.05)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold() -> None:
        async with bh.slot():
            entered.set()
            await release.wait()

    task = asyncio.create_task(hold())
    await entered.wait()
    assert bh.in_flight == 1 and bh.limit == 1
    with pytest.raises(BulkheadFullError):
        async with bh.slot():
            pass
    release.set()
    await task
    assert bh.in_flight == 0
    async with bh.slot():
        assert bh.in_flight == 1


def test_router_strategies() -> None:
    names = ["p", "f", "c", "s"]
    r = Router(RoutingSettings(name="gw", primary="p", fallbacks=["f"]), names)
    assert r.resolve("p").candidates == ["p"] and r.resolve("p").reason == "direct"
    d = r.resolve("gw")
    assert (
        d.candidates == ["p", "f"] and d.reason == "primary" and not d.canary and d.shadow is None
    )
    assert r.served_models == ["gw", "p", "f", "c", "s"] and r.virtual_model == "gw"
    with pytest.raises(UnknownModelError):
        r.resolve("nope")

    canary = Router(
        RoutingSettings(
            name="gw",
            strategy="canary",
            primary="p",
            fallbacks=["f"],
            canary="c",
            canary_percent=50,
        ),
        names,
    )
    decisions = {
        key: canary.resolve("gw", routing_key=key) for key in [f"user{i}" for i in range(200)]
    }
    canary_share = sum(d.canary for d in decisions.values()) / len(decisions)
    assert 0.35 < canary_share < 0.65
    for key, d in decisions.items():
        assert d == canary.resolve("gw", routing_key=key)  # stable per key
        assert d.candidates == (["c", "p", "f"] if d.canary else ["p", "f"])
        assert 0 <= d.extra["bucket"] < 100 and d.extra["bucket"] == stable_bucket(key)
    assert canary.resolve("gw").canary is False  # no routing key -> never the canary

    shadow = Router(RoutingSettings(name="gw", strategy="shadow", primary="p", shadow="s"), names)
    d = shadow.resolve("gw")
    assert d.candidates == ["p"] and d.shadow == "s"
    assert shadow.resolve("s").shadow is None
