"""The gateway core: route → guard → (breaker, bulkhead, retries) → backend → guard, with
fallback across candidates, shadow traffic and metrics. Transport-agnostic; ``api.py`` puts
HTTP in front of it."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from llmgate.backends.base import Backend, BackendError
from llmgate.config import GatewaySettings
from llmgate.guardrails import (
    GuardEvent,
    GuardrailBlockedError,
    RequestGuard,
    ResponseGuard,
    StreamRedactor,
)
from llmgate.observability import (
    BREAKER,
    FALLBACKS,
    GUARDRAIL_EVENTS,
    LATENCY,
    REQUESTS,
    SHADOW,
    TOKENS,
    TTFT,
    log_event,
)
from llmgate.protocol import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
)
from llmgate.resilience import Bulkhead, BulkheadFullError, CircuitBreaker, with_retries
from llmgate.router import RouteDecision, Router

log = logging.getLogger("llmgate")

JSON_REPAIR_PROMPT = (
    "Your previous reply was not valid JSON for the requested schema ({error}). "
    "Reply again with ONLY the JSON object."
)


class AllBackendsFailedError(Exception):
    def __init__(self, message: str, *, status: int = 503) -> None:
        super().__init__(message)
        self.status = status


@dataclass(slots=True)
class ChatOutcome:
    response: ChatCompletionResponse
    backend: str
    decision: RouteDecision
    events: list[GuardEvent] = field(default_factory=list)
    attempts: int = 1
    latency_s: float = 0.0


class Gateway:
    def __init__(
        self,
        settings: GatewaySettings,
        backends: Mapping[str, Backend],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._backends = dict(backends)
        self._router = Router(settings.routing, self._backends)
        r = settings.resilience
        self._breakers = {
            name: CircuitBreaker(
                name,
                failure_threshold=r.breaker_failure_threshold,
                recovery_s=r.breaker_recovery_s,
                clock=clock,
            )
            for name in self._backends
        }
        self._bulkheads = {
            b.name: Bulkhead(b.max_concurrency, queue_timeout_s=b.queue_timeout_s)
            for b in settings.backends
        }
        self._request_guard = RequestGuard(settings.guardrails)
        self._response_guard = ResponseGuard(settings.guardrails)
        self._sleep = sleep
        self._clock = clock
        self.shadow_log: deque[dict[str, Any]] = deque(maxlen=200)
        self._shadow_tasks: set[asyncio.Task[None]] = set()

    # ----- introspection ----------------------------------------------------------------

    @property
    def router(self) -> Router:
        return self._router

    @property
    def backends(self) -> Mapping[str, Backend]:
        return self._backends

    def breaker(self, name: str) -> CircuitBreaker:
        return self._breakers[name]

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "backend": name,
                "model": backend.model,
                "breaker": self._breakers[name].state,
                "failures": self._breakers[name].failures,
                "in_flight": self._bulkheads[name].in_flight,
                "limit": self._bulkheads[name].limit,
            }
            for name, backend in self._backends.items()
        ]

    async def ready(self) -> bool:
        primary = self._settings.routing.primary
        if self._breakers[primary].state == "open":
            return False
        return await self._backends[primary].health()

    async def close(self) -> None:
        for task in list(self._shadow_tasks):
            task.cancel()
        for backend in self._backends.values():
            await backend.close()

    # ----- helpers --------------------------------------------------------------------------

    @staticmethod
    def _record_events(events: list[GuardEvent]) -> None:
        for e in events:
            GUARDRAIL_EVENTS.labels(rail=e.rail, action=e.action).inc()

    def _guard_request(
        self, request: ChatCompletionRequest
    ) -> tuple[ChatCompletionRequest, list[GuardEvent]]:
        try:
            guarded, events = self._request_guard.check(request)
        except GuardrailBlockedError as exc:
            GUARDRAIL_EVENTS.labels(rail=exc.rail, action="block").inc()
            raise
        self._record_events(events)
        return guarded, events

    async def _call(
        self, name: str, request: ChatCompletionRequest, request_id: str
    ) -> ChatCompletionResponse:
        backend = self._backends[name]
        r = self._settings.resilience

        async def once() -> ChatCompletionResponse:
            async with self._bulkheads[name].slot():
                return await backend.chat(request, request_id=request_id)

        return await with_retries(
            once,
            retries=r.max_retries,
            backoff_s=r.backoff_s,
            retriable=lambda exc: isinstance(exc, BackendError) and exc.retriable,
            sleep=self._sleep,
        )

    def _mark(self, name: str, ok: bool) -> None:
        breaker = self._breakers[name]
        if ok:
            breaker.record_success()
        else:
            breaker.record_failure()
        BREAKER.labels(backend=name).set(1 if breaker.state == "open" else 0)

    # ----- non-streaming --------------------------------------------------------------------

    async def chat(
        self, request: ChatCompletionRequest, *, request_id: str, routing_key: str | None = None
    ) -> ChatOutcome:
        started = self._clock()
        decision = self._router.resolve(request.model, routing_key=routing_key)
        guarded, events = self._guard_request(request)
        last_error: Exception | None = None
        attempts = 0
        previous: str | None = None
        for name in decision.candidates:
            if not self._breakers[name].allow():
                log_event(log, "breaker_open", backend=name)
                continue
            if previous is not None:
                FALLBACKS.labels(from_backend=previous, to_backend=name).inc()
            attempts += 1
            try:
                response = await self._call(name, guarded, request_id)
            except (BackendError, BulkheadFullError) as exc:
                self._mark(name, ok=False)
                REQUESTS.labels(route=decision.reason, backend=name, status="error").inc()
                log_event(log, "backend_failed", backend=name, error=str(exc)[:200])
                last_error = exc
                previous = name
                continue
            self._mark(name, ok=True)
            response, out_events = await self._guard_response(name, guarded, response, request_id)
            events.extend(out_events)
            latency = self._clock() - started
            REQUESTS.labels(route=decision.reason, backend=name, status="ok").inc()
            LATENCY.labels(backend=name).observe(latency)
            TOKENS.labels(backend=name, kind="prompt").inc(response.usage.prompt_tokens)
            TOKENS.labels(backend=name, kind="completion").inc(response.usage.completion_tokens)
            if decision.shadow and decision.shadow != name:
                self._spawn_shadow(decision.shadow, guarded, request_id, response)
            return ChatOutcome(
                response=response,
                backend=name,
                decision=decision,
                events=events,
                attempts=attempts,
                latency_s=latency,
            )
        msg = f"all candidate backends failed for model {request.model!r}: {last_error}"
        raise AllBackendsFailedError(msg)

    async def _guard_response(
        self,
        name: str,
        request: ChatCompletionRequest,
        response: ChatCompletionResponse,
        request_id: str,
    ) -> tuple[ChatCompletionResponse, list[GuardEvent]]:
        verdict = self._response_guard.check(response.text, request)
        events = list(verdict.events)
        repairs = 0
        while verdict.json_invalid and repairs < self._settings.guardrails.json_repair_retries:
            repairs += 1
            repair_request = request.model_copy(
                update={
                    "messages": [
                        *request.messages,
                        ChatMessage(role="assistant", content=response.text),
                        ChatMessage(
                            role="user", content=JSON_REPAIR_PROMPT.format(error=verdict.json_error)
                        ),
                    ]
                }
            )
            try:
                response = await self._call(name, repair_request, request_id)
            except (BackendError, BulkheadFullError):
                break
            verdict = self._response_guard.check(response.text, request)
            events.append(GuardEvent("json_schema", "modify", f"repair attempt {repairs}"))
            events.extend(verdict.events)
        self._record_events(events)
        if verdict.blocked:
            text = "The response was withheld because it contained personal information."
        else:
            text = verdict.text
        response = response.model_copy(
            update={
                "choices": [
                    ChatChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=text),
                        finish_reason=response.choices[0].finish_reason
                        if response.choices
                        else "stop",
                    )
                ]
            }
        )
        return response, events

    def _spawn_shadow(
        self,
        shadow: str,
        request: ChatCompletionRequest,
        request_id: str,
        primary: ChatCompletionResponse,
    ) -> None:
        async def run() -> None:
            started = self._clock()
            try:
                shadow_response = await self._backends[shadow].chat(
                    request, request_id=f"{request_id}-shadow"
                )
            except (BackendError, BulkheadFullError) as exc:
                SHADOW.labels(backend=shadow, outcome="error").inc()
                self.shadow_log.append(
                    {"request_id": request_id, "shadow": shadow, "error": str(exc)[:200]}
                )
                return
            SHADOW.labels(backend=shadow, outcome="ok").inc()
            self.shadow_log.append(
                {
                    "request_id": request_id,
                    "shadow": shadow,
                    "latency_s": self._clock() - started,
                    "primary_text": primary.text[:200],
                    "shadow_text": shadow_response.text[:200],
                    "same": primary.text.strip() == shadow_response.text.strip(),
                }
            )

        task = asyncio.create_task(run())
        self._shadow_tasks.add(task)
        task.add_done_callback(self._shadow_tasks.discard)

    async def drain_shadow(self) -> None:
        if self._shadow_tasks:
            await asyncio.gather(*list(self._shadow_tasks), return_exceptions=True)

    # ----- streaming ------------------------------------------------------------------------

    async def stream(
        self, request: ChatCompletionRequest, *, request_id: str, routing_key: str | None = None
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Streams from the first candidate whose breaker allows it and whose first chunk
        arrives; failures *before* the first chunk fall through to the next candidate,
        failures mid-stream terminate the stream with an error chunk (a client cannot be
        given a second beginning)."""
        started = self._clock()
        decision = self._router.resolve(request.model, routing_key=routing_key)
        guarded, events = self._guard_request(request)
        del events
        redact = (
            self._settings.guardrails.redact_output_pii
            or self._settings.guardrails.block_output_pii
        )
        last_error: Exception | None = None
        for name in decision.candidates:
            if not self._breakers[name].allow():
                continue
            backend = self._backends[name]
            redactor = StreamRedactor()
            first = True
            completion_tokens = 0
            try:
                async with self._bulkheads[name].slot():
                    async for chunk in backend.stream(guarded, request_id=request_id):
                        if first:
                            first = False
                            self._mark(name, ok=True)
                            TTFT.labels(backend=name).observe(self._clock() - started)
                        content = chunk.content
                        finish = chunk.choices[0].finish_reason if chunk.choices else None
                        if content:
                            completion_tokens += 1
                        if redact:
                            emitted = redactor.feed(content) if content else ""
                            if finish is not None:
                                emitted += redactor.flush()
                            if content and not emitted and finish is None and chunk.usage is None:
                                continue  # held back until a safe boundary
                            chunk = _with_content(chunk, emitted or None)
                        yield chunk
            except (BackendError, BulkheadFullError) as exc:
                if first:
                    self._mark(name, ok=False)
                    REQUESTS.labels(route=decision.reason, backend=name, status="error").inc()
                    last_error = exc
                    continue
                REQUESTS.labels(route=decision.reason, backend=name, status="stream_error").inc()
                tail = redactor.flush() if redact else ""
                if tail:
                    yield ChatCompletionChunk(
                        model=backend.model,
                        choices=[ChunkChoice(delta=ChunkDelta(content=tail))],
                    )
                yield ChatCompletionChunk(
                    model=backend.model,
                    choices=[ChunkChoice(delta=ChunkDelta(content=""), finish_reason="error")],
                )
                return
            if redact and redactor.kinds:
                GUARDRAIL_EVENTS.labels(rail="pii", action="redact").inc()
            REQUESTS.labels(route=decision.reason, backend=name, status="ok").inc()
            LATENCY.labels(backend=name).observe(self._clock() - started)
            TOKENS.labels(backend=name, kind="completion").inc(completion_tokens)
            return
        msg = f"all candidate backends failed for model {request.model!r}: {last_error}"
        raise AllBackendsFailedError(msg)


def _with_content(chunk: ChatCompletionChunk, content: str | None) -> ChatCompletionChunk:
    choice = chunk.choices[0] if chunk.choices else ChunkChoice()
    return chunk.model_copy(
        update={
            "choices": [
                ChunkChoice(
                    index=choice.index,
                    delta=ChunkDelta(role=choice.delta.role, content=content),
                    finish_reason=choice.finish_reason,
                )
            ]
        }
    )
