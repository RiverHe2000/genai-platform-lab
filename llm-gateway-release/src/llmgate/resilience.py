"""Retries with exponential backoff, a circuit breaker and a bulkhead. All clocks and sleeps
are injectable so the state machines are tested deterministically."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal

BreakerState = Literal["closed", "open", "half_open"]


class CircuitOpenError(Exception):
    pass


class BulkheadFullError(Exception):
    pass


class CircuitBreaker:
    """Closed → (failures ≥ threshold) → open → (recovery elapsed) → half-open → one probe:
    success closes, failure re-opens."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._recovery_s = recovery_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._recovery_s:
            return "half_open"
        return "open"

    @property
    def failures(self) -> int:
        return self._failures

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        self._probe_in_flight = False
        if self._opened_at is not None or self._failures >= self._threshold:
            self._opened_at = self._clock()


class Bulkhead:
    """Bounded concurrency per backend; waiting longer than ``queue_timeout_s`` is a fast
    503 rather than an unbounded queue."""

    def __init__(self, limit: int, *, queue_timeout_s: float = 10.0) -> None:
        self._sem = asyncio.Semaphore(limit)
        self._limit = limit
        self._timeout = queue_timeout_s
        self.in_flight = 0

    @property
    def limit(self) -> int:
        return self._limit

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self._timeout)
        except TimeoutError as exc:
            msg = f"bulkhead full ({self._limit} in flight)"
            raise BulkheadFullError(msg) from exc
        self.in_flight += 1
        try:
            yield
        finally:
            self.in_flight -= 1
            self._sem.release()


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int,
    backoff_s: float,
    retriable: Callable[[BaseException], bool],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call ``fn`` up to ``retries + 1`` times, sleeping ``backoff·2^attempt`` between
    attempts, but only when the failure is retriable."""
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            if attempt >= retries or not retriable(exc):
                raise
            await sleep(backoff_s * (2**attempt))
            attempt += 1
