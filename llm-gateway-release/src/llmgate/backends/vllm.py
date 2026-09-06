"""vLLM's OpenAI-compatible server (``vllm serve <model> --port 8000``).

Beyond the generic OpenAI adapter this one:

* maps ``response_format: json_schema`` to vLLM's ``guided_json`` field (grammar-constrained
  decoding, so structured output is *guaranteed* valid JSON rather than repaired after the
  fact) and ``json_object`` to ``guided_decoding_backend`` defaults;
* probes vLLM's dedicated ``/health`` endpoint (cheaper than ``/v1/models``);
* documents the server flags that matter in production (see ``RECOMMENDED_FLAGS``).

The adapter is exercised in tests against a recorded fake of vLLM's responses; the compose
file in ``deploy/`` runs the real thing (Linux + NVIDIA container runtime).
"""

from __future__ import annotations

from typing import Any

import httpx

from llmgate.backends.openai_compat import OpenAICompatBackend
from llmgate.protocol import ChatCompletionRequest

RECOMMENDED_FLAGS: dict[str, str] = {
    "--max-model-len": "cap context to what the product needs; KV-cache memory scales with it",
    "--gpu-memory-utilization": "0.85-0.90 on a dedicated GPU; leave headroom for CUDA graphs",
    "--enable-prefix-caching": "shared system prompts are computed once",
    "--max-num-seqs": "upper bound on concurrent sequences (continuous batching)",
    "--dtype": "bfloat16 on Ampere+; fp16 otherwise",
    "--quantization": "awq/gptq/fp8 to fit larger models; validate quality with `llmgate promote`",
    "--guided-decoding-backend": "xgrammar (default) or outlines for JSON-schema output",
    "--served-model-name": "stable public name so clients do not depend on the checkpoint path",
    "--api-key": "shared secret; the gateway holds it in an environment variable",
}


class VLLMBackend(OpenAICompatBackend):
    def payload(self, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
        body = super().payload(request, stream=stream)
        fmt = request.response_format
        if fmt is not None and fmt.type == "json_schema" and fmt.schema_dict is not None:
            body["guided_json"] = fmt.schema_dict
            body.pop("response_format", None)
        elif fmt is not None and fmt.type == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    async def health(self) -> bool:
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        try:
            resp = await self._client.get(f"{root}/health")
        except httpx.TransportError:
            return False
        return resp.status_code == 200
