"""Offline fixtures: in-memory loan book, tool registry, settings, scripted models and a
randomly initialised tiny Qwen2 model. Nothing downloads."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from agentguard.agent import Agent, build_agent, build_registry
from agentguard.config import Settings
from agentguard.llm import ChatModel, FakeChatModel
from agentguard.tools.base import ToolRegistry
from agentguard.tools.loanbook import LoanBook
from agentguard.tools.policy_search import PolicySearch


def tool(name: str, **args: object) -> str:
    import json

    return json.dumps({"type": "tool", "tool": name, "args": args})


def final(answer: str) -> str:
    import json

    return json.dumps({"type": "final", "answer": answer})


@pytest.fixture
def settings() -> Settings:
    return Settings(loanbook_size=60)


@pytest.fixture
def book() -> LoanBook:
    return LoanBook(":memory:", seed=7, n_loans=60)


@pytest.fixture
def registry(book: LoanBook) -> ToolRegistry:
    return build_registry(book, PolicySearch())


AgentFactory = Callable[..., Agent]


@pytest.fixture
def make_agent(settings: Settings) -> AgentFactory:
    def factory(
        responses: Sequence[str] = (), *, model: ChatModel | None = None, **overrides: object
    ) -> Agent:
        cfg = settings.model_copy(update=dict(overrides)) if overrides else settings
        return build_agent(cfg, model=model or FakeChatModel(responses=list(responses)))

    return factory


# ----- tiny HF model ------------------------------------------------------------------------

WORDS = [
    "the",
    "a",
    "of",
    "to",
    "in",
    "and",
    "is",
    "type",
    "final",
    "answer",
    "tool",
    "loan",
    "policy",
    "system",
    "user",
    "assistant",
    ":",
    ".",
    ",",
    "?",
    "{",
    "}",
    '"',
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
