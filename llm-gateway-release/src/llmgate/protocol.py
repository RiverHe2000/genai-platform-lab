"""The OpenAI wire contract the gateway speaks on both sides: it *serves* these shapes to
clients and *consumes* them from vLLM / OpenAI-compatible backends.

Requests use ``extra="ignore"`` on purpose: real SDKs send fields we do not implement
(``logit_bias``, ``tools`` …) and rejecting them would break every client; unsupported
values we cannot honour (``n > 1``) are rejected explicitly instead.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Role
    content: str = ""


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "response"
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    strict: bool | None = None


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchemaSpec | None = None

    @property
    def schema_dict(self) -> dict[str, Any] | None:
        if self.type == "json_schema" and self.json_schema is not None:
            return self.json_schema.schema_
        return None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    n: int = Field(1, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None
    response_format: ResponseFormat | None = None

    @field_validator("n")
    @classmethod
    def _only_one_choice(cls, value: int) -> int:
        if value != 1:
            msg = "only n=1 is supported"
            raise ValueError(msg)
        return value

    def last_user_content(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return ""

    def prompt_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = "stop"


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: new_id("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage = Field(default_factory=Usage)

    @property
    def text(self) -> str:
        return self.choices[0].message.content if self.choices else ""


class ChunkDelta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = 0
    delta: ChunkDelta = Field(default_factory=ChunkDelta)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: new_id("chatcmpl"))
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChunkChoice] = Field(default_factory=list)
    usage: Usage | None = None

    @property
    def content(self) -> str:
        return self.choices[0].delta.content or "" if self.choices else ""


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    prompt: str | list[str]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(1.0, ge=0.0, le=2.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    stream: bool = False
    stop: str | list[str] | None = None
    user: str | None = None

    def to_chat(self) -> ChatCompletionRequest:
        prompt = self.prompt if isinstance(self.prompt, str) else "\n".join(self.prompt)
        return ChatCompletionRequest(
            model=self.model,
            messages=[ChatMessage(role="user", content=prompt)],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=self.stream,
            stop=self.stop,
            user=self.user,
        )


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = "stop"


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Field(default_factory=Usage)

    @classmethod
    def from_chat(cls, chat: ChatCompletionResponse) -> CompletionResponse:
        return cls(
            model=chat.model,
            choices=[
                CompletionChoice(
                    index=c.index, text=c.message.content, finish_reason=c.finish_reason
                )
                for c in chat.choices
            ],
            usage=chat.usage,
        )


class ErrorBody(BaseModel):
    message: str
    type: str
    code: str | None = None
    param: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "llmgate"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


# ----- server-sent events -------------------------------------------------------------------

SSE_DONE = b"data: [DONE]\n\n"


def sse_encode(chunk: ChatCompletionChunk) -> bytes:
    return b"data: " + chunk.model_dump_json(exclude_none=True).encode() + b"\n\n"


def parse_sse_line(line: str) -> ChatCompletionChunk | Literal["done"] | None:
    """One line of an SSE stream → chunk, ``"done"`` for the terminator, ``None`` to skip."""
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    payload = stripped[5:].strip()
    if payload == "[DONE]":
        return "done"
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return ChatCompletionChunk.model_validate(data)
