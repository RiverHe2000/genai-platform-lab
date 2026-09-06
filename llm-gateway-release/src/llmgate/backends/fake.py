"""Deterministic backend for tests, CI and load-testing the gateway's own overhead.

Answers come from a scripted list (cycled), a keyword table, or an echo of the last user
message; latency and periodic failures are configurable so retries, fallbacks and the
circuit breaker can be exercised without a real model.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence

from llmgate.backends.base import BackendError, count_tokens_approx
from llmgate.protocol import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
    Usage,
    new_id,
)

Sleep = Callable[[float], Awaitable[None]]


class FakeBackend:
    def __init__(
        self,
        name: str,
        *,
        model: str = "fake-model",
        responses: Sequence[str] = (),
        keyword_answers: Mapping[str, str] | None = None,
        latency_ms: float = 0.0,
        fail_every: int = 0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._name = name
        self._model = model
        self._responses = list(responses)
        self._keyword_answers = dict(keyword_answers or {})
        self._latency_ms = latency_ms
        self._fail_every = fail_every
        self._sleep = sleep
        self.calls = 0
        self.requests: list[ChatCompletionRequest] = []
        self.healthy = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def _answer(self, request: ChatCompletionRequest) -> str:
        user = request.last_user_content()
        if self._responses:
            return self._responses[(self.calls - 1) % len(self._responses)]
        lowered = user.lower()
        for key, answer in self._keyword_answers.items():
            if key.lower() in lowered:
                return answer
        return f"echo: {user}"

    def _tick(self, request: ChatCompletionRequest) -> str:
        self.calls += 1
        self.requests.append(request)
        if self._fail_every and self.calls % self._fail_every == 0:
            msg = f"{self._name}: simulated failure on call {self.calls}"
            raise BackendError(msg, status=503, retriable=True)
        return self._answer(request)

    def _usage(self, request: ChatCompletionRequest, text: str) -> Usage:
        prompt = sum(count_tokens_approx(m.content) for m in request.messages)
        completion = count_tokens_approx(text)
        return Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )

    async def chat(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> ChatCompletionResponse:
        del request_id
        text = self._tick(request)
        if self._latency_ms:
            await self._sleep(self._latency_ms / 1000.0)
        if request.max_tokens is not None:
            words = text.split()
            if len(words) > request.max_tokens:
                text = " ".join(words[: request.max_tokens])
        return ChatCompletionResponse(
            model=self._model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
            usage=self._usage(request, text),
        )

    async def stream(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        del request_id
        text = self._tick(request)
        words = text.split(" ")
        chunk_id = new_id("chatcmpl")
        per_word = (self._latency_ms / 1000.0) / max(len(words), 1) if self._latency_ms else 0.0
        yield ChatCompletionChunk(
            id=chunk_id,
            model=self._model,
            choices=[ChunkChoice(delta=ChunkDelta(role="assistant"))],
        )
        for i, word in enumerate(words):
            if per_word:
                await self._sleep(per_word)
            piece = word if i == 0 else " " + word
            yield ChatCompletionChunk(
                id=chunk_id,
                model=self._model,
                choices=[ChunkChoice(delta=ChunkDelta(content=piece))],
            )
        yield ChatCompletionChunk(
            id=chunk_id,
            model=self._model,
            choices=[ChunkChoice(delta=ChunkDelta(), finish_reason="stop")],
            usage=self._usage(request, text),
        )

    async def health(self) -> bool:
        return self.healthy

    async def close(self) -> None:
        return None
