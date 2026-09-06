from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmgate.config import GatewaySettings, ProcessSettings
from llmgate.protocol import (
    SSE_DONE,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkChoice,
    ChunkDelta,
    CompletionRequest,
    CompletionResponse,
    ResponseFormat,
    parse_sse_line,
    sse_encode,
)
from tests.conftest import ROOT, settings_dict


def test_chat_request_validation() -> None:
    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi", "name": "x"}],
            "logit_bias": {},
        }
    )
    assert req.last_user_content() == "hi" and req.prompt_chars() == 2
    with pytest.raises(ValidationError, match="n=1"):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], n=2)
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[])
    with pytest.raises(ValidationError):
        ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], temperature=3)


def test_response_format_schema_dict() -> None:
    fmt = ResponseFormat.model_validate(
        {"type": "json_schema", "json_schema": {"name": "n", "schema": {"type": "object"}}}
    )
    assert fmt.schema_dict == {"type": "object"}
    assert ResponseFormat(type="json_object").schema_dict is None
    assert ResponseFormat().type == "text"


def test_sse_roundtrip_and_parse_edge_cases() -> None:
    chunk = ChatCompletionChunk(model="m", choices=[ChunkChoice(delta=ChunkDelta(content="hi"))])
    encoded = sse_encode(chunk)
    assert encoded.startswith(b"data: ") and encoded.endswith(b"\n\n")
    parsed = parse_sse_line(encoded.decode().strip())
    assert isinstance(parsed, ChatCompletionChunk) and parsed.content == "hi"
    assert parse_sse_line("data: [DONE]") == "done"
    assert parse_sse_line("") is None and parse_sse_line(": keepalive") is None
    assert parse_sse_line("event: ping") is None and parse_sse_line("data: {not json") is None
    assert SSE_DONE == b"data: [DONE]\n\n"
    assert ChatCompletionChunk(model="m").content == ""


def test_completion_request_maps_to_chat_and_back() -> None:
    req = CompletionRequest(model="m", prompt=["a", "b"], max_tokens=5)
    chat = req.to_chat()
    assert chat.messages[0].content == "a\nb" and chat.max_tokens == 5
    resp = ChatCompletionResponse.model_validate(
        {"model": "m", "choices": [{"message": {"role": "assistant", "content": "out"}}]}
    )
    legacy = CompletionResponse.from_chat(resp)
    assert legacy.choices[0].text == "out" and legacy.object == "text_completion"
    assert resp.text == "out"
    assert ChatCompletionResponse(model="m", choices=[]).text == ""


def test_settings_load_and_validation(tmp_path: Path) -> None:
    fake = GatewaySettings.load(ROOT / "deploy" / "gateway.fake.yaml")
    assert fake.routing.primary == "fake-flaky" and "fake-baseline" in fake.backend_names
    local = GatewaySettings.load(ROOT / "deploy" / "gateway.local.yaml")
    assert local.routing.strategy == "canary"
    vllm = GatewaySettings.load(ROOT / "deploy" / "gateway.vllm.yaml")
    assert vllm.routing.shadow == "vllm-candidate" and vllm.auth.required
    compose = GatewaySettings.load(ROOT / "deploy" / "gateway.compose.yaml")
    assert compose.backends[0].kind == "vllm"

    js = tmp_path / "g.json"
    js.write_text(json.dumps(settings_dict()), encoding="utf-8")
    assert GatewaySettings.load(js).routing.name == "gw"

    with pytest.raises(ValidationError, match="unknown backends"):
        GatewaySettings.model_validate(settings_dict(routing={"name": "gw", "primary": "ghost"}))
    with pytest.raises(ValidationError, match="unique"):
        GatewaySettings.model_validate(
            settings_dict(
                backends=[{"name": "a", "kind": "fake"}, {"name": "a", "kind": "fake"}],
                routing={"name": "gw", "primary": "a"},
            )
        )
    with pytest.raises(ValidationError, match="differ"):
        GatewaySettings.model_validate(
            settings_dict(routing={"name": "primary", "primary": "primary"})
        )
    with pytest.raises(ValidationError, match="canary"):
        GatewaySettings.model_validate(
            settings_dict(routing={"name": "gw", "strategy": "canary", "primary": "primary"})
        )
    with pytest.raises(ValidationError, match="shadow"):
        GatewaySettings.model_validate(
            settings_dict(routing={"name": "gw", "strategy": "shadow", "primary": "primary"})
        )


def test_process_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMGATE_PORT", "9999")
    assert ProcessSettings().port == 9999
