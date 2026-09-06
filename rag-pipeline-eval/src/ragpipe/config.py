"""Twelve-factor configuration: every knob is a ``RAGPIPE_*`` environment variable.

Nested groups use a double underscore, e.g. ``RAGPIPE_GENERATOR__KIND=hf`` or
``RAGPIPE_JUDGE__BASE_URL=http://vllm:8000/v1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMKind = Literal["fake", "openai", "hf"]
EmbedderKind = Literal["hash", "sentence-transformers"]
RetrieverKind = Literal["bm25", "dense", "hybrid"]
FusionKind = Literal["rrf", "convex"]
ChunkerKind = Literal["fixed", "recursive"]


class LLMSettings(BaseModel):
    """One language-model endpoint (used twice: generator and judge)."""

    model_config = ConfigDict(extra="forbid")

    kind: LLMKind = "fake"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # OpenAI-compatible backends: vLLM, llmserve, OpenAI, Ollama, ...
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    api_style: Literal["chat", "completions"] = "chat"
    timeout_s: float = Field(60.0, gt=0)
    max_retries: int = Field(3, ge=0)
    # Local Hugging Face backend
    device: str = "auto"
    # Decoding
    max_tokens: int = Field(256, ge=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGPIPE_", env_nested_delimiter="__", extra="ignore"
    )

    # --- chunking -------------------------------------------------------------------------
    chunker: ChunkerKind = "recursive"
    chunk_words: int = Field(180, ge=8)
    chunk_overlap_words: int = Field(40, ge=0)

    # --- lexical ---------------------------------------------------------------------------
    bm25_k1: float = Field(1.5, gt=0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)

    # --- dense -----------------------------------------------------------------------------
    embedder: EmbedderKind = "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    hash_dim: int = Field(1024, ge=64)
    embedding_cache: Path | None = None

    # --- retrieval -------------------------------------------------------------------------
    retriever: RetrieverKind = "hybrid"
    top_k: int = Field(5, ge=1)
    candidate_k: int = Field(20, ge=1)
    fusion: FusionKind = "rrf"
    rrf_k: int = Field(60, ge=1)
    dense_weight: float = Field(0.5, ge=0.0, le=1.0)
    rerank: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- models ----------------------------------------------------------------------------
    generator: LLMSettings = Field(default_factory=LLMSettings)
    judge: LLMSettings = Field(default_factory=LLMSettings)

    # --- misc ------------------------------------------------------------------------------
    seed: int = 0
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        if self.chunk_overlap_words >= self.chunk_words:
            msg = "chunk_overlap_words must be smaller than chunk_words"
            raise ValueError(msg)
        if self.candidate_k < self.top_k:
            msg = "candidate_k must be >= top_k"
            raise ValueError(msg)
        return self
