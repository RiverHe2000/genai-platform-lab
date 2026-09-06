"""Any server that speaks the OpenAI chat API: OpenAI itself, vLLM, TGI, Ollama, LiteLLM …

Error mapping (drives retries / fallback / breaker):

* 408, 409, 425, 429, 5xx and transport errors → ``retriable=True``
* other 4xx (bad request, auth, context length) → ``retriable=False`` — retrying would fail
  again and would waste the fallback's capacity too.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from llmgate.backends.base import BackendError
from llmgate.protocol import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    parse_sse_line,
)

RETRIABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class OpenAICompatBackend:
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get(api_key_env) or "not-needed"
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._extra_body = dict(extra_body or {})

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self, request_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "X-Request-ID": request_id}

    def payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model or request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.stop is not None:
            body["stop"] = request.stop
        if request.seed is not None:
            body["seed"] = request.seed
        if request.user is not None:
            body["user"] = request.user
        if request.response_format is not None:
            body["response_format"] = request.response_format.model_dump(
                by_alias=True, exclude_none=True
            )
        if stream:
            body["stream_options"] = {"include_usage": True}
        body.update(self._extra_body)
        return body

    @staticmethod
    def _raise_for_status(resp: httpx.Response, name: str) -> None:
        if resp.status_code == 200:
            return
        text = resp.text[:300]
        retriable = resp.status_code in RETRIABLE_STATUS
        msg = f"{name}: HTTP {resp.status_code}: {text}"
        raise BackendError(msg, status=resp.status_code, retriable=retriable)

    async def chat(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> ChatCompletionResponse:
        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=self.payload(request, stream=False),
                headers=self._headers(request_id),
            )
        except httpx.TransportError as exc:
            msg = f"{self._name}: transport error: {exc}"
            raise BackendError(msg, status=503, retriable=True) from exc
        self._raise_for_status(resp, self._name)
        try:
            parsed = ChatCompletionResponse.model_validate(resp.json())
        except Exception as exc:  # malformed JSON or schema
            msg = f"{self._name}: malformed response: {exc}"
            raise BackendError(msg, status=502, retriable=False) from exc
        if not parsed.choices:
            msg = f"{self._name}: response had no choices"
            raise BackendError(msg, status=502, retriable=True)
        return parsed

    async def stream(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=self.payload(request, stream=True),
                headers=self._headers(request_id),
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    self._raise_for_status(resp, self._name)
                async for line in resp.aiter_lines():
                    chunk = parse_sse_line(line)
                    if chunk is None:
                        continue
                    if chunk == "done":
                        return
                    yield chunk
        except httpx.TransportError as exc:
            msg = f"{self._name}: transport error during stream: {exc}"
            raise BackendError(msg, status=503, retriable=True) from exc

    async def health(self) -> bool:
        try:
            resp = await self._client.get(
                f"{self._base_url}/models", headers=self._headers("health")
            )
        except httpx.TransportError:
            return False
        return resp.status_code == 200

    async def close(self) -> None:
        await self._client.aclose()
