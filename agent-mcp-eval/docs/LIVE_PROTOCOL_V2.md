# Local model comparison, version 2

This study reruns both local models with explicit repeated key/value heads and
forced efficient CUDA SDPA. It changes no model weights, bf16 precision, tasks,
gold, permission policy, decoding, 20-turn limit or 512-output-token ceiling.
Both protocols retain all 18 v1 tasks, eight families, two architectures and two
greedy repeats: 72 attempts per model. The source and protocols are committed
before either model starts. Only the attention execution backend changes.

## Why a new version

The frozen v1 source is commit `29d7ee3c350c47491db83d9f46536af326f993b5`.
Its 1.5B run completed. Its 4B run was stopped for resource feasibility when
Windows reported 13.279 GB of shared GPU memory and system commit approached its
limit. This was **not an observed CUDA OOM**, not a quality-based exclusion, and
not a completed model comparison. Preserve its partial calls and trajectories;
see [the operator stop record](experiments/local-live-v1/FEASIBILITY_STOP.md).
The Windows launcher interrupted before Python could finalize the old receipt,
which still says `running` and is rejected by the verifier. The separate operator
record records that interruption without rewriting the historical receipt.

The installed Windows CUDA build does not contain Flash Attention. Profiling
confirmed native GQA SDPA selected the math kernel. Explicitly expanding K/V
heads allows the efficient kernel. The new registered attention implementation
retains Hugging Face's existing SDPA mask and causal logic and forces EFFICIENT;
an unavailable efficient kernel raises an error rather than falling back.
No installed library function or existing attention registration is replaced.

CPU float64 tests cover causal prefill, one-token decode and explicit masks at
1e-12 tolerance. The unscored GPU probe confirms efficient-kernel execution in all
three modes and bf16 agreement at atol=rtol=0.02 (maximum differences 0.015625,
0.00390625 and 0.015625). This is algebraic equivalence with floating-point
differences, not bitwise identity. Therefore **both models start fresh under v2**;
no v1 score is pooled with v2.

The independent synthetic 5,066-token prompt smoke loaded the actual 4B bf16
model and generated seven tokens in 2.011 seconds; peak allocation was 9.262 GB,
reservation 9.609 GB, and resident-process shared GPU memory was 0.086 GB.
These are resource diagnostics, not task scores or throughput guarantees.
See [the raw probe receipt](experiments/local-live-v2/unscored-resource-probe.json)
and `scripts/probe_efficient_attention.py`. Formal runs retain separate actual
call/token/time measurements and peak memory.

## Reproduce and verify

Use the model revisions and dependency versions in the protocol/receipts. Install
the project with `pip install -e ".[hf,dev]"`. The measured stack was Python 3.12,
PyTorch 2.11.0+cu128, Transformers 5.16.1 and an RTX 4070 12 GB on Windows.
The runner requires CUDA and local pinned model files; it downloads nothing and
uses no paid model API. Its in-process MCP tools contain synthetic data and do
not connect to a real customer or external operational system.

```powershell
python -m mcpeval.bench.live_v2 --protocol protocols/local-qwen15-v2.json --out runs/live-v2/qwen15
python -m mcpeval.bench.live_v2 --protocol protocols/local-qwen4b-v2.json --model-dir D:\models\Qwen3-4B-Instruct-2507 --out runs/live-v2/qwen4b
python -m mcpeval.bench.live_v2 --verify --out runs/live-v2/qwen15
python -m mcpeval.bench.live_v2 --verify --out runs/live-v2/qwen4b
python -m mcpeval.bench.live_v2 --compare runs/live-v2/qwen15 runs/live-v2/qwen4b --out runs/live-v2/comparison
```

Run directories must be fresh. Backend errors make the study incomplete; retain
all attempts and do not rerun chosen tasks or tune budgets. Verification requires
the recorded source version, complete output inventory, file hashes, protocol
identity and regraded trajectories. Source hashes normalize LF/CRLF; committed
raw outputs use `.gitattributes -text` to preserve bytes on another clone. The
receipt is reproducibility evidence, not an externally signed attestation.

## Interpretation boundaries

The 18 task units share synthetic world data and templates. They are not 18
statistically independent real-world users. Average repeats within each task
before resampling; correlated families can make task-bootstrap intervals
understate wider uncertainty. These historically visible tasks are a fixed
diagnostic slice, not an unseen production holdout. Both architecture-specific
comparisons remain descriptive; there is no promotion or general superiority
claim. The legacy per-model summary key `independent_tasks` denotes only task
resampling units; the v2 comparison names them `resampling_task_units`.

Qwen2.5-1.5B and Qwen3-4B differ in generation and training as well as size, so
differences cannot be attributed to parameter count alone. Equal output-token
and turn ceilings do not imply equal total compute. Actual calls, repeated input
tokens, output tokens and latency are reported. Zero API fees do not mean zero
hardware ownership or electricity cost. Denying an unsafe tool is not task
success: approval failures, forbidden violations and executed writes remain
separate. Original scripted regression fixtures and historical 72-task model
artifacts remain unchanged.
