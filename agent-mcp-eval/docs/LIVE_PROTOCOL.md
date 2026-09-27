# Frozen local model study v1

This is a **new exploratory model comparison**, separate from the historical 72-task
Qwen2.5-1.5B runs and the scripted CI regression fixtures. The model performs real
inference, while every tool operates on the generated synthetic world through the
in-process MCP server. No real customer records, brokerage, email, filesystem tools,
or external APIs are exposed to the agent. This is a tool sandbox, not an operating
system/network isolation boundary.

## Choices fixed before the new outcomes

| Choice | Fixed value |
|---|---|
| Baseline | Qwen/Qwen2.5-1.5B-Instruct, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Candidate | Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Precision and attention | bfloat16, SDPA; no quantization or CPU offload |
| Architectures | single, supervisor; same existing prompts and tool policy |
| World | seed 7 |
| Selection | 18 tasks, eight families; explicit IDs in both protocol JSON files |
| Turns / output ceiling | 20 model calls per task, 512 new tokens per call |
| Decoding | greedy, temperature 0; no response cache |
| Repeats / order | single, supervisor, supervisor, single; fresh world/log per arm |
| Concurrency | one task at a time |
| Approval | deny every requested write; grade whether approval was sought |
| Resume | prohibited; used output directories fail |
| Comparison | within-task paired bootstrap; average the two repeats first |

Selection uses task-builder order: first two tasks per family, except the first
reconciliation break and first clean account, the first note and first order action,
and the first four injection tasks (the two planted attacks plus two clean controls).
It is chosen for structural coverage, not new model outcomes. These tasks and their
historical outcomes were already visible; this is **not an unseen holdout**. Do not
retune prompts, drop failures, change budgets, or choose the better repeat after running.
An amended experiment needs a new protocol and separately reported outcomes.

There are 72 task attempts per model and 144 total. The ceilings are 1,440 backend
requests and 737,280 generated tokens per model; actual use should be much lower and
is recorded. Equal turn/token ceilings are not equal wall-clock or memory budgets.
The 4B candidate is not assumed to be empirically better.

## Run from a clean checkout

Install the HF optional extra in the intended environment. The runner itself never
downloads weights. Cache the exact revisions beforehand, or provide an existing HF
`local_dir` whose `.cache/huggingface/download/*.metadata` records that revision.
The local cache resolver rejects missing model/tokenizer files and missing shards.

```bash
pip install -e ".[hf,dev]"

# These inspect the protocol and local files, without importing torch or loading weights.
python -m mcpeval.bench.live --protocol protocols/local-qwen15-v1.json --preflight
python -m mcpeval.bench.live --protocol protocols/local-qwen4b-v1.json --preflight \
  --model-dir /path/to/Qwen3-4B-Instruct-2507

# Run sequentially so the models do not compete for GPU memory.
python -m mcpeval.bench.live --protocol protocols/local-qwen15-v1.json \
  --out docs/experiments/local-live-v1/qwen15
python -m mcpeval.bench.live --protocol protocols/local-qwen4b-v1.json \
  --model-dir /path/to/Qwen3-4B-Instruct-2507 \
  --out docs/experiments/local-live-v1/qwen4b

# No model inference in verification or report comparison.
python -m mcpeval.bench.live --verify --out docs/experiments/local-live-v1/qwen15
python -m mcpeval.bench.live --verify --out docs/experiments/local-live-v1/qwen4b
python -m mcpeval.bench.live_compare docs/experiments/local-live-v1/qwen15 \
  docs/experiments/local-live-v1/qwen4b --out docs/experiments/local-live-v1/comparison
```

The default device is CUDA and an unavailable GPU fails explicitly. A CPU run must
be requested explicitly and cannot be combined with a CUDA run by the comparator.
Out-of-memory/backend exceptions are recorded as failed attempts, never silently
replaced by a smaller model, quantization, shorter context, or cached responses.
If an arm contains a backend exception, the study is marked incomplete and the
cross-model comparison refuses it; raw attempts remain available for diagnosis.

## Evidence and interpretation

Each study writes the protocol, full source-file and model-file SHA-256 inventory,
Git HEAD, Python/package/hardware versions, start/end timestamps, model loading time,
completion status, and every output checksum to `receipt.json`. Each arm keeps the
raw trajectories, per-call request/response hashes, actual tokenizer usage and
wall time, grades, aggregates, manifest and report. Failed backend calls are retained;
token use for an exception that supplies no completion is unknown and not invented.
Loading time is reported separately from model-call time. Local API spend is zero;
these measurements do not estimate energy or amortized hardware cost.

Verification requires the exact source version, every expected task exactly once
and in protocol order, compatible run manifests, fresh (unresumed) runs, consistent
backend/trajectory token totals, and grades recomputed from the raw trajectories.
Python source hashes normalize Windows/Unix newlines to LF. Model and output hashes
use exact bytes; the committed artifact directory sets `-text` in `.gitattributes`
so Git cannot rewrite them between platforms. Checksums are an accident/mismatch
check, **not an externally signed attestation**.
Use the receipt's Git HEAD to recover the source if the grader later changes.

Reports show success by task family, attack and clean-control counts separately,
failure taxonomy, approval failures, forbidden violations, refused calls, executed
writes, request/token/runtime costs, and repeat stability. Zero executed writes can
come from the permission layer refusing a model's attempt; it must not be reported
as proof of perfect model behaviour.

The comparison uses 18 paired task means, not 36 allegedly independent repetitions.
The slice is deliberately small and nonrandom. Intervals describe this diagnostic
slice; two architecture-specific comparisons are exploratory, not a familywise
significance claim. Neither a nonsignificant difference nor a better point estimate
establishes equivalence or general superiority. No deployment promotion gate is
inferred from these results.

The separate 4B loading smoke is excluded from all study denominators and scores.
Its short-prompt success only establishes that the model loads at the stated
precision; longer tool trajectories may still exhaust memory.
