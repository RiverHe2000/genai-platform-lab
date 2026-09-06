from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from llmgate.protocol import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse


class BackendError(Exception):
    """A backend failed. ``retriable`` drives retries, fallbacks and the circuit breaker."""

    def __init__(self, message: str, *, status: int = 502, retriable: bool = True) -> None:
        super().__init__(message)
        self.status = status
        self.retriable = retriable


class Backend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def chat(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> ChatCompletionResponse: ...

    def stream(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def health(self) -> bool: ...

    async def close(self) -> None: ...


def count_tokens_approx(text: str) -> int:
    """Whitespace token count — used only where the backend reports no usage."""
    return len(text.split())
