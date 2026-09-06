from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM

from ragpipe.llm import FakeLLM, HFLocalLLM, LLMError, OpenAICompatibleLLM

# ----- fake ---------------------------------------------------------------------------------


def test_fake_llm_rules_default_and_call_log() -> None:
    llm = FakeLLM(
        rules=[(r"LVR", "80% [1]"), (r"VaR", lambda p: f"echo:{len(p)}")], default="I don't know."
    )
    assert llm.complete("what is the LVR?", system="sys").text == "80% [1]"
    assert llm.complete("VaR?").text == "echo:4"
    assert llm.complete("unrelated").text == "I don't know."
    assert [c.prompt for c in llm.calls] == ["what is the LVR?", "VaR?", "unrelated"]
    assert llm.calls[0].system == "sys"
    assert llm.name == "fake"
    r = llm.complete("VaR?")
    assert r.prompt_tokens == 1
    assert r.completion_tokens == 1


# ----- OpenAI-compatible --------------------------------------------------------------------


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def test_openai_chat_request_shape_and_parse() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "model": "served-model",
                "choices": [{"message": {"role": "assistant", "content": "  answer [1]  "}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    llm = OpenAICompatibleLLM("http://test/v1/", "m", api_key="k", client=_client(handler))
    resp = llm.complete("Q?", system="S", max_tokens=17, temperature=0.2)
    assert resp.text == "  answer [1]  "
    assert resp.model == "served-model"
    assert (resp.prompt_tokens, resp.completion_tokens) == (12, 3)
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert body == {
        "model": "m",
        "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q?"}],
        "max_tokens": 17,
        "temperature": 0.2,
    }
    assert seen[0].headers["authorization"] == "Bearer k"
    assert llm.name == "openai[m]"


def test_openai_completions_style_concatenates_system() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"text": "T"}]})

    llm = OpenAICompatibleLLM(
        "http://test/v1", "m", api_style="completions", client=_client(handler)
    )
    assert llm.complete("Q", system="S").text == "T"
    assert seen[0]["prompt"] == "S\n\nQ"
    assert "messages" not in seen[0]


def test_openai_retries_retriable_then_succeeds_with_backoff() -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = OpenAICompatibleLLM(
        "http://test/v1",
        "m",
        max_retries=3,
        backoff_s=0.1,
        client=_client(handler),
        sleep=sleeps.append,
    )
    assert llm.complete("q").text == "ok"
    assert attempts["n"] == 3
    assert sleeps == pytest.approx([0.1, 0.2])


def test_openai_does_not_retry_client_errors() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(400, text="bad request")

    llm = OpenAICompatibleLLM(
        "http://test/v1", "m", max_retries=3, client=_client(handler), sleep=lambda _s: None
    )
    with pytest.raises(LLMError) as exc:
        llm.complete("q")
    assert exc.value.status == 400
    assert attempts["n"] == 1


def test_openai_gives_up_after_transport_errors_and_flags_malformed() -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    llm = OpenAICompatibleLLM(
        "http://test/v1", "m", max_retries=2, client=_client(failing), sleep=lambda _s: None
    )
    with pytest.raises(LLMError, match="giving up after 3 attempts"):
        llm.complete("q")

    def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    llm2 = OpenAICompatibleLLM("http://test/v1", "m", client=_client(malformed))
    with pytest.raises(LLMError, match="malformed"):
        llm2.complete("q")
    llm2.close()


# ----- local HF -----------------------------------------------------------------------------


def test_hf_local_llm_is_greedy_deterministic_and_counts_tokens(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    llm = HFLocalLLM("tiny", device="cpu", model=tiny_model, tokenizer=tiny_tokenizer)
    a = llm.complete("the bank risk limit", max_tokens=6)
    b = llm.complete("the bank risk limit", max_tokens=6)
    assert a.text == b.text
    assert a.prompt_tokens == 4
    assert 1 <= (a.completion_tokens or 0) <= 6
    assert a.model == "tiny"
    assert llm.name == "hf[tiny]"
    assert a.latency_s >= 0.0


def test_hf_local_llm_uses_chat_template_when_present(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    tiny_tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} . {% endfor %}assistant :"
    )
    try:
        llm = HFLocalLLM("tiny", device="cpu", model=tiny_model, tokenizer=tiny_tokenizer)
        r = llm.complete("question", system="system", max_tokens=3)
        # "system : system . user : question . assistant :" -> 10 whitespace/punct tokens
        assert r.prompt_tokens == 10
        sampled = llm.complete("question", max_tokens=3, temperature=0.9)
        assert sampled.completion_tokens is not None
    finally:
        tiny_tokenizer.chat_template = None


def test_hf_local_llm_sampling_with_seed_is_reproducible(
    tiny_model: Qwen2ForCausalLM, tiny_tokenizer: PreTrainedTokenizerFast
) -> None:
    llm = HFLocalLLM("tiny", device="cpu", model=tiny_model, tokenizer=tiny_tokenizer, seed=123)
    a = llm.complete("capital ratio", max_tokens=8, temperature=1.0)
    b = llm.complete("capital ratio", max_tokens=8, temperature=1.0)
    assert a.text == b.text
