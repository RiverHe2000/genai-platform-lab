"""HTTP surface: chat, approve, inspect. Validation at the boundary (Pydantic), stable
error codes, correlation id on every response."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from agentguard import __version__
from agentguard.agent import Agent


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = Field(default=None, max_length=64)


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=64)
    approved: bool
    approver: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=500)


def create_app(agent: Agent) -> FastAPI:
    app = FastAPI(title="agentguard", version=__version__)

    @app.middleware("http")
    async def request_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__, "model": agent._deps.model.name}

    @app.post("/chat")
    def chat(body: ChatRequest) -> dict[str, Any]:
        thread_id = body.thread_id or uuid.uuid4().hex[:12]
        return agent.run(thread_id, body.message).to_dict()

    @app.post("/approve")
    def approve(body: ApproveRequest) -> dict[str, Any]:
        try:
            result = agent.resume(
                body.thread_id, approved=body.approved, approver=body.approver, note=body.note
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return result.to_dict()

    @app.get("/threads/{thread_id}")
    def thread(thread_id: str) -> dict[str, Any]:
        state = agent.state(thread_id)
        if not state.get("messages"):
            raise HTTPException(status_code=404, detail="unknown thread")
        return {
            "thread_id": thread_id,
            "waiting": state.get("waiting", False),
            "status": state.get("status"),
            "turn": state.get("turn"),
            "messages": state.get("messages", []),
        }

    @app.get("/audit/{thread_id}")
    def audit(thread_id: str) -> dict[str, Any]:
        entries = agent.audit.entries(thread_id)
        if not entries:
            raise HTTPException(status_code=404, detail="no audit entries")
        return {"thread_id": thread_id, "entries": entries}

    return app
