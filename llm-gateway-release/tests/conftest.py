"""Offline fixtures: fake backends, gateway settings, an in-process ASGI client and a
randomly initialised tiny Qwen2 model. Nothing downloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from fastapi import FastAPI
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from llmgate.api import create_app
from llmgate.backends.base import Backend
from llmgate.backends.fake import FakeBackend
from llmgate.config import GatewaySettings
from llmgate.gateway import Gateway

ROOT = Path(__file__).resolve().parents[1]


async def no_sleep(_seconds: float) -> None:
    return None


def settings_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "backends": [
            {"name": "primary", "kind": "fake", "model": "primary-model"},
            {"name": "backup", "kind": "fake", "model": "backup-model"},
            {"name": "canary", "kind": "fake", "model": "canary-model"},
        ],
        "routing": {
            "name": "gw",
            "strategy": "primary_fallback",
            "primary": "primary",
            "fallbacks": ["backup"],
        },
        "resilience": {
            "max_retries": 1,
            "backoff_s": 0.0,
            "breaker_failure_threshold": 3,
            "breaker_recovery_s": 10,
        },
        "ratelimit": {"requests_per_minute": 6000, "burst": 1000},
    }
    base.update(overrides)
    return base


@pytest.fixture
def settings() -> GatewaySettings:
    return GatewaySettings.model_validate(settings_dict())


def make_backends(**kwargs: Any) -> dict[str, Backend]:
    return {
        "primary": FakeBackend(
            "primary", model="primary-model", sleep=no_sleep, **kwargs.get("primary", {})
        ),
        "backup": FakeBackend(
            "backup", model="backup-model", sleep=no_sleep, **kwargs.get("backup", {})
        ),
        "canary": FakeBackend(
            "canary", model="canary-model", sleep=no_sleep, **kwargs.get("canary", {})
        ),
    }


GatewayFactory = Callable[..., Gateway]


@pytest.fixture
def make_gateway() -> GatewayFactory:
    def factory(
        overrides: Mapping[str, Any] | None = None,
        *,
        backends: dict[str, Backend] | None = None,
        **backend_kwargs: Any,
    ) -> Gateway:
        cfg = GatewaySettings.model_validate(settings_dict(**(overrides or {})))
        return Gateway(cfg, backends or make_backends(**backend_kwargs), sleep=no_sleep)

    return factory


@pytest.fixture
def app(make_gateway: GatewayFactory) -> FastAPI:
    gateway = make_gateway()
    return create_app(gateway, gateway._settings)


@pytest.fixture
def asgi_client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


# ----- tiny HF model ------------------------------------------------------------------------

WORDS = [
    "the",
    "a",
    "of",
    "to",
    "in",
    "and",
    "is",
    "loan",
    "value",
    "ratio",
    "bank",
    ":",
    ".",
    "?",
    "1",
    "2",
    "3",
]


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
    for i, w in enumerate(WORDS, start=3):
        vocab[w] = i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer: PreTrainedTokenizerFast) -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tiny_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=True,
        eos_token_id=tiny_tokenizer.eos_token_id,
        pad_token_id=tiny_tokenizer.pad_token_id,
    )
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model
