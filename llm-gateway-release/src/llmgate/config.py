"""Gateway configuration: a YAML/JSON file for the topology (backends, routing, policies)
plus ``LLMGATE_*`` environment variables for process-level settings. Secrets (API keys)
never live in the file — backends read them from the environment variable they name, and
client keys are stored as SHA-256 hashes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BackendKind = Literal["fake", "openai", "vllm", "hf"]
Strategy = Literal["primary_fallback", "canary", "shadow"]


class BackendSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    kind: BackendKind
    model: str = ""
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = Field(60.0, gt=0)
    max_concurrency: int = Field(8, ge=1)
    queue_timeout_s: float = Field(10.0, gt=0)
    device: str = "auto"
    # fake-backend knobs (ignored by other kinds)
    fake_responses: list[str] = Field(default_factory=list)
    fake_latency_ms: float = Field(0.0, ge=0)
    fake_fail_every: int = Field(0, ge=0)


class RoutingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "gateway-default"
    strategy: Strategy = "primary_fallback"
    primary: str
    fallbacks: list[str] = Field(default_factory=list)
    canary: str | None = None
    canary_percent: float = Field(0.0, ge=0.0, le=100.0)
    shadow: str | None = None


class ResilienceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(2, ge=0)
    backoff_s: float = Field(0.2, ge=0)
    breaker_failure_threshold: int = Field(5, ge=1)
    breaker_recovery_s: float = Field(30.0, gt=0)


class RateLimitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requests_per_minute: float = Field(600.0, gt=0)
    burst: int = Field(60, ge=1)
    daily_token_budget: int = Field(0, ge=0)  # 0 = unlimited


class GuardrailSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    redact_input_pii: bool = True
    redact_output_pii: bool = True
    block_output_pii: bool = False
    injection_threshold: float = Field(0.7, ge=0.0, le=1.0)
    max_prompt_chars: int = Field(32_000, ge=1)
    max_tokens_cap: int = Field(2048, ge=1)
    blocked_terms: list[str] = Field(default_factory=list)
    enforce_json_schema: bool = True
    json_repair_retries: int = Field(1, ge=0)


class AuthSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = False
    # sha256(api key) -> principal name
    api_keys: dict[str, str] = Field(default_factory=dict)


class GatewaySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backends: list[BackendSettings] = Field(min_length=1)
    routing: RoutingSettings
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)

    @property
    def backend_names(self) -> list[str]:
        return [b.name for b in self.backends]

    @model_validator(mode="after")
    def _check_references(self) -> GatewaySettings:
        names = self.backend_names
        if len(set(names)) != len(names):
            msg = "backend names must be unique"
            raise ValueError(msg)
        routing = self.routing
        refs = [routing.primary, *routing.fallbacks]
        if routing.canary:
            refs.append(routing.canary)
        if routing.shadow:
            refs.append(routing.shadow)
        unknown = [r for r in refs if r not in names]
        if unknown:
            msg = f"routing references unknown backends: {unknown}"
            raise ValueError(msg)
        if routing.name in names:
            msg = "routing.name must differ from every backend name"
            raise ValueError(msg)
        if routing.strategy == "canary" and (routing.canary is None or routing.canary_percent <= 0):
            msg = "canary strategy needs a canary backend and canary_percent > 0"
            raise ValueError(msg)
        if routing.strategy == "shadow" and routing.shadow is None:
            msg = "shadow strategy needs a shadow backend"
            raise ValueError(msg)
        return self

    @classmethod
    def load(cls, path: Path | str) -> GatewaySettings:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)
        return cls.model_validate(data)


class ProcessSettings(BaseSettings):
    """Process-level knobs (``LLMGATE_HOST`` etc.)."""

    model_config = SettingsConfigDict(env_prefix="LLMGATE_", extra="ignore")

    config_path: Path = Path("deploy/gateway.fake.yaml")
    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"
