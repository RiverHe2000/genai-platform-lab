"""Opt-in HF attention registration for CUDA builds without Flash GQA.

Expand K/V heads explicitly, then require PyTorch's efficient SDPA kernel. The
original HF SDPA mask/causal handling is retained. This avoids the native-GQA math
fallback observed on Windows; failure to use the efficient kernel is an error.
No installed library function or existing attention registration is replaced.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

ATTENTION_NAME = "mcpeval_sdpa_repeat_kv_efficient"


def repeated_kv(query: Any, key: Any, value: Any) -> tuple[Any, Any]:
    """Group expansion is algebraically the same head assignment as native GQA."""
    if key.shape[1] != value.shape[1] or query.shape[1] % key.shape[1]:
        raise ValueError("query heads must be divisible by matching key/value heads")
    groups = query.shape[1] // key.shape[1]
    return key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)


def efficient_attention(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Use the existing HF SDPA semantics with GQA expanded and no math fallback."""
    attention = importlib.import_module("torch.nn.attention")
    sdpa = importlib.import_module("transformers.integrations.sdpa_attention")
    if query.device.type != "cuda":
        raise ValueError("the frozen efficient attention protocol requires CUDA")
    key, value = repeated_kv(query, key, value)
    # HF SDPA only reads these two module properties. Setting the already-expanded
    # group count to one prevents it from asking PyTorch for native GQA again.
    proxy = SimpleNamespace(is_causal=getattr(module, "is_causal", True), num_key_value_groups=1)
    with attention.sdpa_kernel(attention.SDPBackend.EFFICIENT_ATTENTION):
        output, weights = sdpa.sdpa_attention_forward(
            proxy,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **kwargs,
        )
    if weights is not None:
        raise RuntimeError("unexpected attention weights from HF SDPA")
    return output, None


def register_attention() -> str:
    """Register one new implementation and its standard SDPA mask factory."""
    modeling = importlib.import_module("transformers.modeling_utils")
    masking = importlib.import_module("transformers.masking_utils")
    modeling.AttentionInterface.register(ATTENTION_NAME, efficient_attention)
    masking.AttentionMaskInterface.register(
        ATTENTION_NAME, masking.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    )
    return ATTENTION_NAME
