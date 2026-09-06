"""HTTP surface: OpenAI-compatible endpoints, health/readiness, metrics, admin, and the
error envelope every OpenAI SDK understands."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from llmgate import __version__
from llmgate.auth import Authenticator, AuthError, Principal
from llmgate.config import GatewaySettings
from llmgate.gateway import AllBackendsFailedError, Gateway
from llmgate.guardrails import GuardrailBlockedError
from llmgate.observability import REJECTED, log_event, render_metrics, request_id_var
from llmgate.protocol import (
    SSE_DONE,
    ChatCompletionRequest,
    CompletionRequest,
    CompletionResponse,
    ErrorBody,
    ErrorResponse,
    ModelList,
    ModelObject,
    sse_encode,
)
from llmgate.ratelimit import BudgetExceededError, RateLimitedError, RateLimiter, TokenBudget
from llmgate.resilience import BulkheadFullError
from llmgate.router import UnknownModelError

log = logging.getLogger("llmgate.api")


def _error(
    status: int,
    message: str,
    *,
    type_: str,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(message=message, type=type_, code=code))
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


def create_app(
    gateway: Gateway,
    settings: GatewaySettings,
    *,
    authenticator: Authenticator | None = None,
    limiter: RateLimiter | None = None,
    budget: TokenBudget | None = None,
) -> FastAPI:
    app = FastAPI(title="llmgate", version=__version__)
    auth = authenticator or Authenticator(settings.auth)
    limiter = limiter or RateLimiter(settings.ratelimit)
    budget = budget or TokenBudget(settings.ratelimit.daily_token_budget)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        log_event(
            log,
            "http",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    # ----- error envelope -----------------------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _validation(_r: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        REJECTED.labels(reason="validation").inc()
        return _error(
            400, f"{loc}: {first.get('msg', 'invalid request')}", type_="invalid_request_error"
        )

    @app.exception_handler(GuardrailBlockedError)
    async def _blocked(_r: Request, exc: GuardrailBlockedError) -> JSONResponse:
        REJECTED.labels(reason=f"guardrail:{exc.rail}").inc()
        return _error(400, str(exc), type_="invalid_request_error", code=f"guardrail_{exc.rail}")

    @app.exception_handler(UnknownModelError)
    async def _unknown(_r: Request, exc: UnknownModelError) -> JSONResponse:
        REJECTED.labels(reason="unknown_model").inc()
        return _error(404, str(exc), type_="invalid_request_error", code="model_not_found")

    @app.exception_handler(AuthError)
    async def _auth(_r: Request, exc: AuthError) -> JSONResponse:
        REJECTED.labels(reason="auth").inc()
        return _error(401, str(exc), type_="authentication_error", code="invalid_api_key")

    @app.exception_handler(RateLimitedError)
    async def _limited(_r: Request, exc: RateLimitedError) -> JSONResponse:
        REJECTED.labels(reason="rate_limit").inc()
        return _error(
            429,
            str(exc),
            type_="rate_limit_error",
            code="rate_limit_exceeded",
            headers={"Retry-After": str(max(1, int(exc.retry_after_s + 0.999)))},
        )

    @app.exception_handler(BudgetExceededError)
    async def _budget(_r: Request, exc: BudgetExceededError) -> JSONResponse:
        REJECTED.labels(reason="budget").inc()
        return _error(429, str(exc), type_="rate_limit_error", code="insufficient_quota")

    @app.exception_handler(AllBackendsFailedError)
    async def _failed(_r: Request, exc: AllBackendsFailedError) -> JSONResponse:
        return _error(exc.status, str(exc), type_="server_error", code="backends_unavailable")

    @app.exception_handler(BulkheadFullError)
    async def _bulkhead(_r: Request, exc: BulkheadFullError) -> JSONResponse:
        return _error(503, str(exc), type_="server_error", code="overloaded")

    # ----- helpers ------------------------------------------------------------------------

    def principal_of(request: Request) -> Principal:
        principal = auth.authenticate(request.headers.get("authorization"))
        limiter.check(principal.name)
        budget.check(principal.name)
        return principal

    async def sse(
        request: ChatCompletionRequest, request_id: str, principal: Principal
    ) -> AsyncIterator[bytes]:
        async for chunk in gateway.stream(
            request, request_id=request_id, routing_key=request.user or principal.name
        ):
            yield sse_encode(chunk)
        yield SSE_DONE

    async def handle_chat(body: ChatCompletionRequest, request: Request) -> Response:
        principal = principal_of(request)
        request_id = request_id_var.get()
        routing_key = body.user or principal.name
        if body.stream:
            return StreamingResponse(
                sse(body, request_id, principal),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        outcome = await gateway.chat(body, request_id=request_id, routing_key=routing_key)
        budget.charge(principal.name, outcome.response.usage.total_tokens)
        headers = {
            "X-Backend": outcome.backend,
            "X-Route": outcome.decision.reason,
            "X-Attempts": str(outcome.attempts),
            "X-Guardrails": ";".join(f"{e.rail}:{e.action}" for e in outcome.events) or "none",
        }
        return JSONResponse(content=outcome.response.model_dump(), headers=headers)

    # ----- routes -------------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
        return await handle_chat(body, request)

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request) -> Response:
        if body.stream:
            return _error(
                400,
                "streaming is only supported on /v1/chat/completions",
                type_="invalid_request_error",
            )
        principal = principal_of(request)
        outcome = await gateway.chat(
            body.to_chat(), request_id=request_id_var.get(), routing_key=body.user or principal.name
        )
        budget.charge(principal.name, outcome.response.usage.total_tokens)
        return JSONResponse(
            content=CompletionResponse.from_chat(outcome.response).model_dump(),
            headers={"X-Backend": outcome.backend},
        )

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, Any]:
        principal_of(request)
        return ModelList(
            data=[ModelObject(id=m) for m in gateway.router.served_models]
        ).model_dump()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    async def ready() -> Response:
        ok = await gateway.ready()
        return JSONResponse(
            status_code=200 if ok else 503, content={"ready": ok, "backends": gateway.status()}
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/admin/backends")
    async def admin_backends(request: Request) -> dict[str, Any]:
        principal_of(request)
        return {"backends": gateway.status(), "shadow_samples": list(gateway.shadow_log)[-20:]}

    return app
