from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mcpeval.bench.attention import (
    ATTENTION_NAME,
    efficient_attention,
    register_attention,
    repeated_kv,
)


@pytest.mark.parametrize("mode", ["prefill", "decode", "mask"])
def test_repeated_kv_preserves_grouped_attention_mathematics(mode: str) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    q_len = 1 if mode == "decode" else 7
    q = torch.randn(1, 4, q_len, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 7, 8, dtype=torch.float64)
    v = torch.randn_like(k)
    mask = None
    if mode == "mask":
        mask = torch.ones(q_len, 7, dtype=torch.bool).tril()
        mask[:, 2] = False
    arguments: dict[str, Any] = {"is_causal": mode == "prefill", "attn_mask": mask}
    expanded_k, expanded_v = repeated_kv(q, k, v)
    original = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, enable_gqa=True, **arguments
    )
    expanded = torch.nn.functional.scaled_dot_product_attention(
        q, expanded_k, expanded_v, **arguments
    )
    torch.testing.assert_close(expanded, original, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="requires CUDA"):
        efficient_attention(SimpleNamespace(is_causal=True), q, k, v, None)
    with pytest.raises(ValueError, match="divisible"):
        repeated_kv(q[:, :3], k, v)


def test_registration_keeps_default_hf_attention_and_mask_unchanged() -> None:
    pytest.importorskip("torch")
    modeling = pytest.importorskip("transformers.modeling_utils")
    masking = pytest.importorskip("transformers.masking_utils")
    original = modeling.ALL_ATTENTION_FUNCTIONS["sdpa"]
    original_mask = masking.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    assert register_attention() == ATTENTION_NAME
    assert modeling.ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    assert masking.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"] is original_mask
    assert masking.ALL_MASK_ATTENTION_FUNCTIONS[ATTENTION_NAME] is original_mask
    assert modeling.ALL_ATTENTION_FUNCTIONS[ATTENTION_NAME] is efficient_attention
