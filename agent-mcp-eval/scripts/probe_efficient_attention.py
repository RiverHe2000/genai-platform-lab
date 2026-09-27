"""Unscored CUDA compatibility probe and synthetic long-prompt model smoke."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mcpeval.bench.attention import efficient_attention, register_attention


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("use a fresh output file")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    torch.manual_seed(0)
    record: dict[str, Any] = {
        "purpose": "unscored synthetic tensor and long-prompt resource feasibility check",
        "torch": torch.__version__,
        "flash_compiled": torch.backends.cuda.is_flash_attention_available(),
        "modes": {},
        "status": "running",
    }
    for mode in ("prefill", "decode", "mask"):
        q = torch.randn(
            1, 32, 1 if mode == "decode" else 128, 128, device="cuda", dtype=torch.bfloat16
        )
        k = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        mask = None
        if mode == "mask":
            mask = torch.ones(128, 128, device="cuda", dtype=torch.bool).tril()
            mask[:, 2] = False
        with torch.inference_mode():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                original = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, enable_gqa=True, attn_mask=mask, is_causal=mode == "prefill"
                )
                torch.cuda.synchronize()
            default_ops = [e.key for e in profile.key_averages() if "attention" in e.key]
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                efficient, _ = efficient_attention(SimpleNamespace(is_causal=True), q, k, v, mask)
                torch.cuda.synchronize()
            ops = [e.key for e in profile.key_averages() if "attention" in e.key]
        expected = original.transpose(1, 2)
        close = torch.allclose(efficient, expected, atol=0.02, rtol=0.02)
        record["modes"][mode] = {
            "original_ops": default_ops,
            "efficient_ops": ops,
            "max_absolute_diff": float((efficient - expected).abs().max()),
            "allclose_atol_0.02_rtol_0.02": close,
        }
        if not close or "aten::_scaled_dot_product_efficient_attention" not in ops:
            raise RuntimeError(f"{mode} did not pass numerical/kernel verification")
        del q, k, v, original, efficient, expected, mask
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    name = register_attention()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False
    )
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            str(args.model_dir),
            dtype=torch.bfloat16,
            attn_implementation=name,
            local_files_only=True,
            trust_remote_code=False,
        )
        .to("cuda")
        .eval()
    )
    torch.cuda.synchronize()
    record["load_seconds"] = time.perf_counter() - began
    # No benchmark question, gold, task ID, or model quality score appears here.
    messages = [
        {"role": "system", "content": "Return a brief JSON status object."},
        {
            "role": "user",
            "content": ("Synthetic record: owner A, amount 10, state ready. " * 360)
            + "Reply with status ready.",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    began = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    record.update(
        generation_seconds=time.perf_counter() - began,
        prompt_tokens=inputs["input_ids"].shape[1],
        completion_tokens=output.shape[1] - inputs["input_ids"].shape[1],
        reply=tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        attention_implementation=name,
        dtype=str(model.dtype),
        status="passed",
    )
    # Sample while model weights and the generated output are still resident.
    counters = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory "
            f"| Where-Object {{ $_.Name -like 'pid_{os.getpid()}_*' }} "
            "| Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted "
            "| ConvertTo-Json -Compress",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    record["windows_gpu_process_memory"] = json.loads(counters.stdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
