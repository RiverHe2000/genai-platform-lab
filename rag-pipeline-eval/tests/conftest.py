"""Offline fixtures: a four-document toy corpus, the hashing embedder, an in-memory index
and a randomly initialised tiny Qwen2 model with an in-memory tokenizer. Nothing downloads."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from ragpipe.chunking import RecursiveChunker
from ragpipe.documents import Document
from ragpipe.embeddings import HashEmbedder
from ragpipe.llm import FakeLLM
from ragpipe.pipeline import IndexBundle, build_index

DOCS = [
    Document(
        doc_id="mortgages",
        title="Residential Mortgages",
        text=(
            "# Residential Mortgages\n\n"
            "The maximum loan-to-value ratio without lenders mortgage insurance is 80%. "
            "With LMI the maximum LVR is 95% for owner-occupiers.\n\n"
            "Serviceability is assessed at the contract rate plus a buffer of 3 percentage "
            "points or the floor rate of 5.5%, whichever is higher."
        ),
    ),
    Document(
        doc_id="var",
        title="Market Risk VaR",
        text=(
            "# Market Risk VaR\n\n"
            "The trading book value-at-risk limit is AUD 5 million at the 99% level. "
            "The foreign exchange desk limit is AUD 2 million.\n\n"
            "Ten or more backtesting exceptions in 250 days put the model in the red zone."
        ),
    ),
    Document(
        doc_id="liquidity",
        title="Liquidity",
        text=(
            "# Liquidity\n\n"
            "The liquidity coverage ratio internal trigger is 110% and the management target "
            "is 115%. The regulatory minimum LCR is 100%.\n\n"
            "At least 60% of high-quality liquid assets must be Level 1 assets."
        ),
    ),
    Document(
        doc_id="privacy",
        title="Privacy",
        text=(
            "# Privacy\n\n"
            "Tax file numbers are stored only in encrypted fields and never in free-text "
            "fields, emails or application logs.\n\n"
            "A suspected eligible data breach must be assessed within 30 days."
        ),
    ),
]


@pytest.fixture
def docs() -> list[Document]:
    return list(DOCS)


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dim=256)


@pytest.fixture
def bundle(docs: list[Document], embedder: HashEmbedder) -> IndexBundle:
    return build_index(docs, RecursiveChunker(max_words=40), embedder)


@pytest.fixture
def corpus_dir(tmp_path: Path, docs: list[Document]) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    for d in docs:
        (root / f"{d.doc_id}.md").write_text(d.text, encoding="utf-8")
    return root


@pytest.fixture
def abstaining_llm() -> FakeLLM:
    return FakeLLM()


# ----- tiny HF model ------------------------------------------------------------------------

WORDS = [
    "the",
    "a",
    "of",
    "to",
    "in",
    "and",
    "is",
    "limit",
    "ratio",
    "maximum",
    "minimum",
    "percent",
    "million",
    "risk",
    "capital",
    "liquidity",
    "mortgage",
    "answer",
    "question",
    "context",
    "passage",
    "policy",
    "bank",
    "i",
    "don't",
    "know",
    "user",
    "system",
    "assistant",
    ":",
    ".",
    ",",
    "?",
    "[",
    "]",
    "1",
    "2",
    "3",
    "80",
    "95",
    "110",
    "115",
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
