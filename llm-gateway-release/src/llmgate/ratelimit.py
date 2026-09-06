"""Per-principal token-bucket rate limiting and daily token budgets (injectable clock)."""

from __future__ import annotations

import time
from collections.abc import Callable

from llmgate.config import RateLimitSettings


class RateLimitedError(Exception):
    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"rate limited; retry after {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s


class BudgetExceededError(Exception):
    pass


class TokenBucket:
    def __init__(
        self, rate_per_s: float, burst: int, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._rate = rate_per_s
        self._burst = float(burst)
        self._clock = clock
        self._tokens = float(burst)
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + (now - self._last) * self._rate)
        self._last = now

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens

    def try_acquire(self, n: int = 1) -> bool:
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True
        return False

    def retry_after(self, n: int = 1) -> float:
        self._refill()
        deficit = n - self._tokens
        return max(0.0, deficit / self._rate) if deficit > 0 else 0.0


class RateLimiter:
    def __init__(
        self, settings: RateLimitSettings, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._s = settings
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}

    def check(self, principal: str) -> None:
        bucket = self._buckets.get(principal)
        if bucket is None:
            bucket = TokenBucket(
                self._s.requests_per_minute / 60.0, self._s.burst, clock=self._clock
            )
            self._buckets[principal] = bucket
        if not bucket.try_acquire():
            raise RateLimitedError(bucket.retry_after())


class TokenBudget:
    """Daily token budget per principal (UTC day boundary from the injected clock, in
    seconds since epoch)."""

    def __init__(self, daily_limit: int, *, clock: Callable[[], float] = time.time) -> None:
        self._limit = daily_limit
        self._clock = clock
        self._used: dict[tuple[str, int], int] = {}

    def _day(self) -> int:
        return int(self._clock() // 86_400)

    def used(self, principal: str) -> int:
        return self._used.get((principal, self._day()), 0)

    def check(self, principal: str) -> None:
        if self._limit and self.used(principal) >= self._limit:
            msg = f"daily token budget of {self._limit} exhausted for {principal}"
            raise BudgetExceededError(msg)

    def charge(self, principal: str, tokens: int) -> int:
        key = (principal, self._day())
        self._used[key] = self._used.get(key, 0) + max(0, tokens)
        return self._used[key]
