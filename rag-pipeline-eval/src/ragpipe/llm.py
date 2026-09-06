"""Language-model backends behind one protocol.

* ``FakeLLM`` — regex-scripted responses; every unit test and the CI smoke run use it.
* ``OpenAICompatibleLLM`` — ``/v1/chat/completions`` or ``/v1/completions`` over HTTP, i.e.
  vLLM, OpenAI, Ollama or the companion ``llmserve`` project; retries with exponential backoff
  on 429/5xx/transport errors only.
* ``HFLocalLLM`` — a Hugging Face causal LM in-process (the RTX 4070 runs Qwen2.5-1.5B).
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

RETRIABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0


class LLM(Protocol):
    @property
    def name(self) -> str: ...

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


# ----- fake ---------------------------------------------------------------------------------

Responder = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class FakeCall:
    prompt: str
    system: str | None


@dataclass
class FakeLLM:
    """First rule whose regex matches ``prompt`` (or the system prompt) wins."""

    rules: Sequence[tuple[str, str | Responder]] = ()
    default: str | Responder = "I don't know."
    name_: str = "fake"
    calls: list[FakeCall] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.name_

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,  # noqa: ARG002 - protocol signature
        temperature: float = 0.0,  # noqa: ARG002
    ) -> LLMResponse:
        self.calls.append(FakeCall(prompt=prompt, system=system))
        haystack = f"{system or ''}\n{prompt}"
        response: str | Responder = self.default
        for pattern, candidate in self.rules:
            if re.search(pattern, haystack, flags=re.DOTALL):
                response = candidate
                break
        text = response(prompt) if callable(response) else response
        return LLMResponse(
            text=text,
            model=self.name_,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            latency_s=0.0,
        )


# ----- OpenAI-compatible HTTP ---------------------------------------------------------------


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        api_style: Literal["chat", "completions"] = "chat",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_s: float = 0.5,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key or os.environ.get(api_key_env) or "not-needed"
        self._api_style = api_style
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def name(self) -> str:
        return f"openai[{self._model}]"

    def _body(
        self, prompt: str, system: str | None, max_tokens: int, temperature: float
    ) -> tuple[str, dict[str, Any]]:
        if self._api_style == "chat":
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return "/chat/completions", {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        return "/completions", {
            "model": self._model,
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    @staticmethod
    def _parse(payload: dict[str, Any], model: str, latency: float) -> LLMResponse:
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] if "message" in choice else choice["text"]
        except (KeyError, IndexError, TypeError) as exc:
            msg = f"malformed response: {payload!r}"
            raise LLMError(msg) from exc
        usage = payload.get("usage") or {}
        return LLMResponse(
            text=str(text or ""),
            model=str(payload.get("model", model)),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=latency,
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        path, body = self._body(prompt, system, max_tokens, temperature)
        headers = {"Authorization": f"Bearer {self._api_key}"}
        last_error: str = ""
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                resp = self._client.post(self._base_url + path, json=body, headers=headers)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
            else:
                if resp.status_code == 200:
                    return self._parse(resp.json(), self._model, time.perf_counter() - started)
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in RETRIABLE_STATUS:
                    raise LLMError(last_error, status=resp.status_code)
            if attempt < self._max_retries:
                self._sleep(self._backoff_s * (2**attempt))
        msg = f"giving up after {self._max_retries + 1} attempts: {last_error}"
        raise LLMError(msg)

    def close(self) -> None:
        self._client.close()


# ----- local Hugging Face -------------------------------------------------------------------


class HFLocalLLM:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        model: Any | None = None,
        tokenizer: Any | None = None,
        seed: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._device_spec = device
        self._model: Any = model
        self._tokenizer: Any = tokenizer
        self._seed = seed

    @property
    def name(self) -> str:
        return f"hf[{self._model_name}]"

    def _resolve_device(self) -> str:
        import torch

        if self._device_spec != "auto":
            return self._device_spec
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> tuple[Any, Any, str]:
        import torch

        device = self._resolve_device()
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name, dtype=dtype)
        self._model.to(device)
        self._model.eval()
        return self._model, self._tokenizer, device

    def _encode(self, tokenizer: Any, prompt: str, system: str | None, device: str) -> Any:
        if getattr(tokenizer, "chat_template", None):
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            full = f"{system}\n\n{prompt}" if system else prompt
            encoded = tokenizer(full, return_tensors="pt")
        return {k: v.to(device) for k, v in encoded.items()}

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        import torch

        model, tokenizer, device = self._load()
        inputs = self._encode(tokenizer, prompt, system, device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0.0,
            "pad_token_id": tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["temperature"] = temperature
            if self._seed is not None:
                torch.manual_seed(self._seed)
        started = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        latency = time.perf_counter() - started
        n_prompt = int(inputs["input_ids"].shape[1])
        new_tokens = out[0, n_prompt:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return LLMResponse(
            text=text,
            model=self._model_name,
            prompt_tokens=n_prompt,
            completion_tokens=int(new_tokens.shape[0]),
            latency_s=latency,
        )
