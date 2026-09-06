"""Configuration: every knob is an ``AGENTGUARD_*`` environment variable; nested groups use
``__`` (e.g. ``AGENTGUARD_MODEL__KIND=openai``, ``AGENTGUARD_MODEL__BASE_URL=http://vllm:8000/v1``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelKind = Literal["fake", "openai", "hf"]


class ModelSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ModelKind = "fake"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = Field(60.0, gt=0)
    max_retries: int = Field(3, ge=0)
    device: str = "auto"
    max_tokens: int = Field(400, ge=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class GuardrailPolicy(BaseModel):
    """Which rails are on and what they do when they fire."""

    model_config = ConfigDict(extra="forbid")

    pii_input_action: Literal["redact", "block"] = "redact"
    pii_output_action: Literal["redact", "block"] = "redact"
    injection_threshold: float = Field(0.5, ge=0.0, le=1.0)
    block_out_of_scope: bool = True
    block_restricted: bool = True
    require_grounded_numbers: bool = True
    advice_disclaimer: bool = True
    max_input_chars: int = Field(4000, ge=1)
    max_output_chars: int = Field(4000, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTGUARD_", env_nested_delimiter="__", extra="ignore"
    )

    model: ModelSettings = Field(default_factory=ModelSettings)
    guardrails: GuardrailPolicy = Field(default_factory=GuardrailPolicy)

    max_steps: int = Field(8, ge=1)
    max_parse_retries: int = Field(2, ge=0)
    require_approval_for_high_risk: bool = True

    # Durable state. ``None`` keeps everything in memory (tests, CI).
    checkpoint_path: Path | None = None
    loanbook_path: Path | None = None
    audit_path: Path | None = None
    loanbook_seed: int = 7
    loanbook_size: int = Field(200, ge=1)
    sql_max_rows: int = Field(50, ge=1)

    seed: int = 0
    log_level: str = "INFO"
