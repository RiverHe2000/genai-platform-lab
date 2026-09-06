from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM

from agentguard.llm import (
    ChatMessage,
    FakeChatModel,
    HFChatModel,
    ModelError,
    OpenAICompatibleChatModel,
    to_provider_messages,
)


def test_fake_model_queue_rules_default_and_calls() -> None:
    model = FakeChatModel(
        responses=["first"], rules=[(r"lvr", "rule-hit"), (r".*", lambda ms: f"n={len(ms)}")]
    )
    msgs = [ChatMessage("system", "s"), ChatMessage("user", "what is the LVR?")]
    assert model.chat(msgs).text == "first"
    assert model.chat(msgs).text == "rule-hit"
    assert model.chat([ChatMessage("user", "other")]).text == "n=1"
    assert FakeChatModel().chat([ChatMessage("user", "x")]).text.startswith('{"type": "final"')
    assert len(model.calls) == 3 and model.name == "fake"


def test_provider_messages_map_tool_role_to_user() -> None:
    out = to_provider_messages([ChatMessage("tool", "data"), ChatMessage("assistant", "a")])
    assert out == [{"role": "user", "content": "data"}, {"role": "assistant", "content": "a"}]


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_openai_chat_model_request_and_retries() -> None:
    seen: list[dict[str, Any]] = []
    attempts = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={
                "model": "m2",
                "choices": [{"message": {"content": '{"type":"final","answer":"ok"}'}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            },
        )

    model = OpenAICompatibleChatModel(
        "http://test/v1",
        "m",
        api_key="k",
        client=_client(handler),
        sleep=sleeps.append,
        backoff_s=0.1,
    )
    resp = model.chat(
        [ChatMessage("system", "s"), ChatMessage("tool", "t")], max_tokens=9, temperature=0.1
    )
    assert resp.text == '{"type":"final","answer":"ok"}' and resp.model == "m2"
    assert (resp.prompt_tokens, resp.completion_tokens) == (5, 7)
    assert seen[0]["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "t"},
    ]
    assert seen[0]["max_tokens"] == 9 and sleeps == [0.1]
    assert model.name == "openai[m]"


def test_openai_chat_model_errors() -> None:
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="nope")

    with pytest.raises(ModelError) as exc:
        OpenAICompatibleChatModel("http://test/v1", "m", client=_client(bad)).chat(
            [ChatMessage("user", "x")]
        )
    assert exc.value.status == 401

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(ModelError, match="giving up"):
        OpenAICompatibleChatModel(
            "http://test/v1", "m", client=_client(down), max_retries=1, sleep=lambda _s: None
        ).chat([ChatMessage("user", "x")])

    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    m = OpenAICompatibleChatModel("http://test/v1", "m", client=_client(malformed))
    with pytest.raises(ModelError, match="malformed"):
        m.chat([ChatMessage("user", "x")])
    m.close()


def test_hf_chat_model_with_and_without_template(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    model = HFChatModel("tiny", device="cpu", model=tiny_model, tokenizer=tiny_tokenizer, seed=1)
    msgs = [
        ChatMessage("system", "policy"),
        ChatMessage("user", "loan ?"),
        ChatMessage("tool", "1 2 3"),
    ]
    a = model.chat(msgs, max_tokens=5)
    b = model.chat(msgs, max_tokens=5)
    assert a.text == b.text and a.prompt_tokens is not None and a.prompt_tokens > 0
    sampled = model.chat(msgs, max_tokens=5, temperature=0.8)
    assert sampled.completion_tokens is not None
    tiny_tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} . {% endfor %}assistant :"
    )
    try:
        templated = model.chat(msgs, max_tokens=3)
        assert templated.prompt_tokens == 17
    finally:
        tiny_tokenizer.chat_template = None
    assert model.name == "hf[tiny]"
