"""Model → ordered list of backends to try.

* A request for a concrete backend name goes straight there (no fallback: the caller asked
  for *that* deployment, which is what an evaluation run needs).
* A request for the virtual model (``routing.name``) applies the strategy:
  ``primary_fallback``, ``canary`` (a stable share of routing keys goes to the canary first,
  so a user always lands on the same variant) or ``shadow`` (primary answers; the shadow
  backend receives a copy for offline comparison and never affects the response).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field

from llmgate.config import RoutingSettings


class UnknownModelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RouteDecision:
    candidates: list[str]
    reason: str
    canary: bool = False
    shadow: str | None = None
    extra: dict[str, float] = field(default_factory=dict)


def stable_bucket(key: str) -> float:
    """Deterministic value in [0, 100) for a routing key."""
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") % 10_000 / 100.0


class Router:
    def __init__(self, settings: RoutingSettings, backend_names: Iterable[str]) -> None:
        self._settings = settings
        self._names = list(backend_names)

    @property
    def virtual_model(self) -> str:
        return self._settings.name

    @property
    def served_models(self) -> list[str]:
        return [self._settings.name, *self._names]

    def resolve(self, model: str, *, routing_key: str | None = None) -> RouteDecision:
        s = self._settings
        if model in self._names:
            return RouteDecision(candidates=[model], reason="direct")
        if model != s.name:
            msg = f"unknown model {model!r}; served: {self.served_models}"
            raise UnknownModelError(msg)
        chain = [s.primary, *s.fallbacks]
        if s.strategy == "canary" and s.canary is not None:
            bucket = stable_bucket(routing_key or "")
            if routing_key is not None and bucket < s.canary_percent:
                return RouteDecision(
                    candidates=[s.canary, *chain],
                    reason="canary",
                    canary=True,
                    extra={"bucket": bucket},
                )
            return RouteDecision(candidates=chain, reason="primary", extra={"bucket": bucket})
        if s.strategy == "shadow":
            return RouteDecision(candidates=chain, reason="primary", shadow=s.shadow)
        return RouteDecision(candidates=chain, reason="primary")
