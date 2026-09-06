"""An in-process Hugging Face causal LM behind the same async protocol.

Generation runs in a worker thread under a lock (one model, one GPU), so the event loop
keeps serving health checks and other backends. Streaming uses ``TextIteratorStreamer`` so
time-to-first-token is real, not simulated. Used for the reported experiments (Qwen2.5
0.5B vs 1.5B on one RTX 4070) and, with a randomly initialised tiny model, in the tests.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

from llmgate.backends.base import BackendError
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


class HFLocalBackend:
    def __init__(
        self,
        name: str,
        model_name: str,
        *,
        device: str = "auto",
        model: Any | None = None,
        tokenizer: Any | None = None,
        max_new_tokens_default: int = 256,
    ) -> None:
        self._name = name
        self._model_name = model_name
        self._device_spec = device
        self._model: Any = model
        self._tokenizer: Any = tokenizer
        self._default_max = max_new_tokens_default
        self._lock = threading.Lock()
        self._loaded = model is not None and tokenizer is not None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model_name

    def _load(self) -> tuple[Any, Any, str]:
        import torch

        if self._device is None:
            device = self._device_spec
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if not self._loaded:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                self._model = AutoModelForCausalLM.from_pretrained(self._model_name, dtype=dtype)
                self._loaded = True
            self._model.to(device)
            self._model.eval()
            self._device = device
        return self._model, self._tokenizer, self._device

    def _encode(
        self, request: ChatCompletionRequest, tokenizer: Any, device: str
    ) -> dict[str, Any]:
        messages = [
            {"role": "user" if m.role == "tool" else m.role, "content": m.content}
            for m in request.messages
        ]
        if getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            flat = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\n\nassistant:"
            encoded = tokenizer(flat, return_tensors="pt")
        return {k: v.to(device) for k, v in encoded.items()}

    def _gen_kwargs(self, request: ChatCompletionRequest, tokenizer: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_tokens or self._default_max,
            "do_sample": request.temperature > 0.0,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        }
        if request.temperature > 0.0:
            kwargs["temperature"] = request.temperature
            kwargs["top_p"] = request.top_p
        return kwargs

    def _generate_sync(self, request: ChatCompletionRequest) -> tuple[str, int, int, str]:
        import torch

        model, tokenizer, device = self._load()
        with self._lock:
            inputs = self._encode(request, tokenizer, device)
            kwargs = self._gen_kwargs(request, tokenizer)
            if request.seed is not None:
                torch.manual_seed(request.seed)
            with torch.no_grad():
                out = model.generate(**inputs, **kwargs)
        n_prompt = int(inputs["input_ids"].shape[1])
        new_tokens = out[0, n_prompt:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finish = "length" if int(new_tokens.shape[0]) >= int(kwargs["max_new_tokens"]) else "stop"
        return text, n_prompt, int(new_tokens.shape[0]), finish

    async def chat(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> ChatCompletionResponse:
        del request_id
        try:
            text, n_prompt, n_new, finish = await asyncio.to_thread(self._generate_sync, request)
        except Exception as exc:
            msg = f"{self._name}: generation failed: {type(exc).__name__}: {exc}"
            raise BackendError(msg, status=500, retriable=False) from exc
        return ChatCompletionResponse(
            model=self._model_name,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=text), finish_reason=finish
                )
            ],
            usage=Usage(
                prompt_tokens=n_prompt, completion_tokens=n_new, total_tokens=n_prompt + n_new
            ),
        )

    async def stream(
        self, request: ChatCompletionRequest, *, request_id: str
    ) -> AsyncIterator[ChatCompletionChunk]:
        del request_id
        from transformers import TextIteratorStreamer

        model, tokenizer, device = await asyncio.to_thread(self._load)
        inputs = self._encode(request, tokenizer, device)
        kwargs = self._gen_kwargs(request, tokenizer)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        errors: list[BaseException] = []

        def run() -> None:
            import torch

            with self._lock:
                try:
                    with torch.no_grad():
                        model.generate(**inputs, streamer=streamer, **kwargs)
                except BaseException as exc:  # surfaced to the consumer below
                    errors.append(exc)
                    streamer.end()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        chunk_id = new_id("chatcmpl")
        sentinel = object()
        yield ChatCompletionChunk(
            id=chunk_id,
            model=self._model_name,
            choices=[ChunkChoice(delta=ChunkDelta(role="assistant"))],
        )
        n_new = 0
        while True:
            piece = await asyncio.to_thread(next, streamer, sentinel)
            if piece is sentinel:
                break
            text = str(piece)
            if not text:
                continue
            n_new += 1
            yield ChatCompletionChunk(
                id=chunk_id,
                model=self._model_name,
                choices=[ChunkChoice(delta=ChunkDelta(content=text))],
            )
        thread.join()
        if errors:
            msg = f"{self._name}: generation failed: {errors[0]}"
            raise BackendError(msg, status=500, retriable=False)
        n_prompt = int(inputs["input_ids"].shape[1])
        yield ChatCompletionChunk(
            id=chunk_id,
            model=self._model_name,
            choices=[ChunkChoice(delta=ChunkDelta(), finish_reason="stop")],
            usage=Usage(
                prompt_tokens=n_prompt, completion_tokens=n_new, total_tokens=n_prompt + n_new
            ),
        )

    async def health(self) -> bool:
        return True

    async def close(self) -> None:
        return None
