from __future__ import annotations

import pytest

from ragpipe.config import Settings


def test_defaults_are_offline_friendly() -> None:
    s = Settings()
    assert s.embedder == "hash"
    assert s.generator.kind == "fake"
    assert s.retriever == "hybrid"


def test_env_overrides_including_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGPIPE_TOP_K", "7")
    monkeypatch.setenv("RAGPIPE_GENERATOR__KIND", "openai")
    monkeypatch.setenv("RAGPIPE_GENERATOR__BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("RAGPIPE_JUDGE__MAX_TOKENS", "999")
    s = Settings()
    assert s.top_k == 7
    assert s.generator.kind == "openai"
    assert s.generator.base_url == "http://vllm:8000/v1"
    assert s.judge.max_tokens == 999


def test_validation_rules() -> None:
    with pytest.raises(ValueError, match="overlap"):
        Settings(chunk_words=50, chunk_overlap_words=50)
    with pytest.raises(ValueError, match="candidate_k"):
        Settings(top_k=10, candidate_k=5)
    with pytest.raises(ValueError):
        Settings(generator={"kind": "nope"})
