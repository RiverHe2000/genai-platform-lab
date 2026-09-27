# Same-budget local model comparison

Baseline: `Qwen/Qwen2.5-1.5B-Instruct`. Candidate: `Qwen/Qwen3-4B-Instruct-2507`.

Both models use the same tasks, tool permission policy, greedy decoding, 20 model turns per task, and 512 new tokens per turn. Each architecture runs twice. This controls token/turn ceilings, not equal wall-clock or memory consumption.

| Architecture | Baseline success | Candidate success | Difference | Paired 95% CI |
|---|---:|---:|---:|---:|
| single | 0.000 | 0.667 | +0.667 | [+0.444, +0.889] |
| supervisor | 0.111 | 0.722 | +0.611 | [+0.389, +0.833] |

**Exploratory only:** 18 fixed diagnostic tasks; two repeats are averaged per task before bootstrap. Two architecture-specific comparisons are descriptive, not a familywise significance claim. These are historically visible synthetic tasks, not a held-out production evaluation. Neither size nor a higher point score proves general model superiority. See each model's report for permission failures, attacks vs clean controls, repeat stability, backend errors and actual runtime/token cost.

## Version 2 interpretation

The 18 tasks are resampling units, not statistically independent users or real-world
cases. They share a synthetic world and template families; task bootstrap intervals
can understate wider uncertainty. The legacy summary key `independent_tasks` means
only 18 task units here. Repeats are averaged within task before resampling.

Qwen2.5-1.5B and Qwen3-4B differ in generation and training as well as parameter
count. Differences cannot be attributed to parameter count alone. Identical task,
output-token and turn ceilings do not equal total compute: inspect observed calls,
input/output tokens and latency. Zero API fees exclude hardware ownership and
electricity costs. Denied dangerous actions are not task successes; approval
failures, permission violations and executed writes are reported separately.

Both models use explicit repeated K/V heads and forced efficient SDPA, without
math fallback or quantization. Floating-point results are not bitwise identical
to v1. No v1 score is combined with v2; v1's 4B resource-feasibility interruption
was not an observed CUDA OOM or a completed model comparison.
