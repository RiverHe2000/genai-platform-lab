from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM

from llmgate.backends.base import BackendError
from llmgate.backends.fake import FakeBackend
from llmgate.backends.hf_local import HFLocalBackend
from llmgate.backends.openai_compat import OpenAICompatBackend
from llmgate.backends.vllm import RECOMMENDED_FLAGS, VLLMBackend
from llmgate.protocol import ChatCompletionRequest, ChatMessage, ResponseFormat
from tests.conftest import no_sleep


def req(text: str = "hello there", **kw: Any) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content=text)], **kw)


# ----- fake ---------------------------------------------------------------------------------


async def test_fake_backend_echo_script_keyword_and_failures() -> None:
    echo = FakeBackend("f", sleep=no_sleep)
    r = await echo.chat(req("ping"), request_id="1")
    assert r.text == "echo: ping" and r.usage.total_tokens == 3 and r.model == "fake-model"
    scripted = FakeBackend("f", responses=["a", "b"], sleep=no_sleep)
    assert [(await scripted.chat(req(), request_id="1")).text for _ in range(3)] == ["a", "b", "a"]
    kw = FakeBackend("f", keyword_answers={"lvr": "loan-to-value"}, sleep=no_sleep, latency_ms=5)
    assert (await kw.chat(req("What is LVR?"), request_id="1")).text == "loan-to-value"
    capped = (
        await echo.chat(
            req("one two three four"),
            request_id="1",
        )
        if False
        else await echo.chat(req("one two three four", max_tokens=2), request_id="1")
    )
    assert capped.text == "echo: one"
    flaky = FakeBackend("f", fail_every=2, sleep=no_sleep)
    await flaky.chat(req(), request_id="1")
    with pytest.raises(BackendError) as exc:
        await flaky.chat(req(), request_id="2")
    assert exc.value.retriable and exc.value.status == 503
    assert await flaky.health() and flaky.calls == 2
    await flaky.close()


async def test_fake_backend_stream_reassembles_text() -> None:
    backend = FakeBackend("f", responses=["alpha beta gamma"], sleep=no_sleep, latency_ms=3)
    chunks = [c async for c in backend.stream(req(), request_id="1")]
    assert chunks[0].choices[0].delta.role == "assistant"
    assert "".join(c.content for c in chunks) == "alpha beta gamma"
    assert chunks[-1].choices[0].finish_reason == "stop" and chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 3


# ----- OpenAI-compatible --------------------------------------------------------------------


def _client(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")


async def test_openai_backend_payload_and_parse() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "served",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    backend = OpenAICompatBackend(
        "o",
        "http://test/v1/",
        "served-model",
        api_key="k",
        client=_client(handler),
        extra_body={"top_k": 5},
    )
    request = req(
        "hi",
        max_tokens=9,
        stop=["\n"],
        seed=1,
        user="u",
        response_format=ResponseFormat(type="json_object"),
    )
    resp = await backend.chat(request, request_id="rid")
    assert resp.text == "ok" and resp.usage.total_tokens == 4 and resp.model == "served"
    body = json.loads(seen[0].content)
    assert body["model"] == "served-model" and body["max_tokens"] == 9 and body["stop"] == ["\n"]
    assert (
        body["seed"] == 1 and body["user"] == "u" and body["top_k"] == 5 and body["stream"] is False
    )
    assert body["response_format"] == {"type": "json_object"}
    assert (
        seen[0].headers["authorization"] == "Bearer k" and seen[0].headers["x-request-id"] == "rid"
    )
    assert seen[0].url.path == "/v1/chat/completions"
    assert (
        backend.name == "o"
        and backend.model == "served-model"
        and backend.base_url == "http://test/v1"
    )
    assert await backend.health() is True


@pytest.mark.parametrize(
    ("status", "retriable"), [(429, True), (503, True), (400, False), (401, False)]
)
async def test_openai_backend_error_mapping(status: int, retriable: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="nope")

    backend = OpenAICompatBackend("o", "http://test/v1", "m", client=_client(handler))
    with pytest.raises(BackendError) as exc:
        await backend.chat(req(), request_id="1")
    assert exc.value.retriable is retriable and exc.value.status == status
    assert await backend.health() is False


async def test_openai_backend_transport_malformed_and_empty() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(BackendError, match="transport error"):
        await OpenAICompatBackend("o", "http://test/v1", "m", client=_client(down)).chat(
            req(), request_id="1"
        )

    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(BackendError, match="malformed") as exc:
        await OpenAICompatBackend("o", "http://test/v1", "m", client=_client(malformed)).chat(
            req(), request_id="1"
        )
    assert exc.value.retriable is False

    def empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "m", "choices": []})

    with pytest.raises(BackendError, match="no choices"):
        await OpenAICompatBackend("o", "http://test/v1", "m", client=_client(empty)).chat(
            req(), request_id="1"
        )

    async def stream_down(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("gone")

    backend = OpenAICompatBackend("o", "http://test/v1", "m", client=_client(stream_down))
    with pytest.raises(BackendError, match="during stream"):
        async for _ in backend.stream(req(), request_id="1"):
            pass
    await backend.close()


async def test_openai_backend_streaming_parses_sse() -> None:
    lines = [
        'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        "",
        ": keepalive",
        'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}',
        'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}',
        "data: [DONE]",
        'data: {"should":"not be read"}',
    ]
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, content="\n".join(lines).encode(), headers={"content-type": "text/event-stream"}
        )

    backend = OpenAICompatBackend("o", "http://test/v1", "m", client=_client(handler))
    chunks = [c async for c in backend.stream(req(), request_id="1")]
    assert "".join(c.content for c in chunks) == "Hello"
    assert chunks[-1].usage is not None and chunks[-1].usage.total_tokens == 3
    assert seen[0]["stream"] is True and seen[0]["stream_options"] == {"include_usage": True}

    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(BackendError) as exc:
        async for _ in OpenAICompatBackend("o", "http://test/v1", "m", client=_client(bad)).stream(
            req(), request_id="1"
        ):
            pass
    assert exc.value.status == 500


# ----- vLLM ---------------------------------------------------------------------------------


async def test_vllm_backend_guided_json_and_health() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, text="")
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"role": "assistant", "content": '{"a": 1}'}}],
            },
        )

    backend = VLLMBackend("v", "http://test/v1", "m", client=_client(handler))
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    fmt = ResponseFormat.model_validate(
        {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}
    )
    body = backend.payload(req(response_format=fmt), stream=False)
    assert body["guided_json"] == schema and "response_format" not in body
    body2 = backend.payload(req(response_format=ResponseFormat(type="json_object")), stream=False)
    assert body2["response_format"] == {"type": "json_object"}
    assert "guided_json" not in backend.payload(req(), stream=False)
    assert (await backend.chat(req(response_format=fmt), request_id="1")).text == '{"a": 1}'
    assert await backend.health() is True
    assert calls[-1].url.path == "/health"
    assert "--enable-prefix-caching" in RECOMMENDED_FLAGS

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert await VLLMBackend("v", "http://test/v1", "m", client=_client(down)).health() is False


# ----- local HF -----------------------------------------------------------------------------


async def test_hf_backend_chat_and_stream(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    backend = HFLocalBackend("hf", "tiny", device="cpu", model=tiny_model, tokenizer=tiny_tokenizer)
    a = await backend.chat(
        req("the loan value ratio", max_tokens=6, temperature=0.0), request_id="1"
    )
    b = await backend.chat(
        req("the loan value ratio", max_tokens=6, temperature=0.0), request_id="2"
    )
    assert a.text == b.text and a.usage.prompt_tokens == 8 and 1 <= a.usage.completion_tokens <= 6
    assert a.choices[0].finish_reason in ("stop", "length")
    sampled = await backend.chat(req("bank", max_tokens=4, temperature=0.9, seed=3), request_id="3")
    assert sampled.usage.completion_tokens >= 1
    chunks = [c async for c in backend.stream(req("the bank", max_tokens=5), request_id="4")]
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason == "stop" and chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == sum(1 for c in chunks if c.content)
    assert await backend.health() and backend.model == "tiny" and backend.name == "hf"
    await backend.close()


async def test_hf_backend_generation_error_is_backend_error(
    tiny_tokenizer: PreTrainedTokenizerFast,
) -> None:
    class Broken:
        def to(self, _device: str) -> Broken:
            return self

        def eval(self) -> None:
            return None

        def generate(self, **_kwargs: Any) -> Any:
            raise RuntimeError("cuda meltdown")

    backend = HFLocalBackend("hf", "tiny", device="cpu", model=Broken(), tokenizer=tiny_tokenizer)
    with pytest.raises(BackendError, match="cuda meltdown") as exc:
        await backend.chat(req("x"), request_id="1")
    assert exc.value.retriable is False
    with pytest.raises(BackendError, match="cuda meltdown"):
        async for _ in backend.stream(req("x"), request_id="1"):
            pass
