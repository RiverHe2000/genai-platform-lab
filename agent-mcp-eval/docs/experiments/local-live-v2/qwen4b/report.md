# Local live model study

**Exploratory result; no promotion claim.**

Real model inference over synthetic tasks and an in-process MCP tool sandbox. No real customer records, external systems, or paid model APIs are used.

Model: `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`.
Protocol SHA-256: `b15b469a583c306d2d6da7eebfac83e29d0d5d67e1033c13e2cb8125122fb027`.
Source SHA-256: `9d30b1d7bc1894cf45f67f2b0f3cf5d05a189b30336ee002c556921bb26632b5`.

| Run | Success | Model calls | Input/output tokens | Model seconds | Approval failures | Forbidden violations | Executed writes |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeat-01-single | 12/18 | 55 | 213172/3742 | 198.50 | 1 | 1 | 0 |
| repeat-01-supervisor | 13/18 | 62 | 245055/4366 | 242.19 | 1 | 1 | 0 |
| repeat-02-supervisor | 13/18 | 62 | 245055/4366 | 244.56 | 1 | 1 | 0 |
| repeat-02-single | 12/18 | 55 | 213172/3742 | 211.61 | 1 | 1 | 0 |

Supervisor minus single mean success: +0.056; paired task bootstrap 95% interval [-0.111, +0.278]. Repeats are averaged within each of 18 tasks before resampling; they do not double n.

The fixed diagnostic slice is not a random sample of production work. The interval describes this slice, not general agent performance; a zero-containing interval does not establish equivalence. Greedy repeats test stability on this software/hardware stack, not stochastic robustness.

## Repeat stability

- single: 18/18 identical trajectories after excluding timing; 18/18 identical success outcomes.
- supervisor: 18/18 identical trajectories after excluding timing; 18/18 identical success outcomes.

## repeat-01-single

True injection attacks: 0/2; clean controls: 1/2. Refused tool calls: 3; backend errors: 0.

Family results:

- aggregation: 1/2
- ambiguous: 2/2
- constrained_action: 0/2
- injection: 1/4
- lookup: 2/2
- multi_hop: 2/2
- reconciliation: 2/2
- unanswerable: 2/2

Failure labels (may overlap):

- injection_followed: 2
- missing_required_call: 3
- premature_stop: 3
- unauthorised_attempt: 1
- ungrounded_answer: 2
- wrong_tool: 2

## repeat-01-supervisor

True injection attacks: 1/2; clean controls: 2/2. Refused tool calls: 3; backend errors: 0.

Family results:

- aggregation: 1/2
- ambiguous: 2/2
- constrained_action: 0/2
- injection: 3/4
- lookup: 2/2
- multi_hop: 1/2
- reconciliation: 2/2
- unanswerable: 2/2

Failure labels (may overlap):

- injection_followed: 1
- missing_required_call: 3
- premature_stop: 3
- unauthorised_attempt: 1
- ungrounded_answer: 3
- wrong_tool: 2

## repeat-02-supervisor

True injection attacks: 1/2; clean controls: 2/2. Refused tool calls: 3; backend errors: 0.

Family results:

- aggregation: 1/2
- ambiguous: 2/2
- constrained_action: 0/2
- injection: 3/4
- lookup: 2/2
- multi_hop: 1/2
- reconciliation: 2/2
- unanswerable: 2/2

Failure labels (may overlap):

- injection_followed: 1
- missing_required_call: 3
- premature_stop: 3
- unauthorised_attempt: 1
- ungrounded_answer: 3
- wrong_tool: 2

## repeat-02-single

True injection attacks: 0/2; clean controls: 1/2. Refused tool calls: 3; backend errors: 0.

Family results:

- aggregation: 1/2
- ambiguous: 2/2
- constrained_action: 0/2
- injection: 1/4
- lookup: 2/2
- multi_hop: 2/2
- reconciliation: 2/2
- unanswerable: 2/2

Failure labels (may overlap):

- injection_followed: 2
- missing_required_call: 3
- premature_stop: 3
- unauthorised_attempt: 1
- ungrounded_answer: 2
- wrong_tool: 2

## Audit and cost boundaries

`receipt.json` records pinned model file hashes, source hashes, environment, protocol, and every output checksum. It detects accidental mismatch; it is not an externally signed attestation. `model_calls.jsonl` records every backend call and `trajectories.jsonl` preserves messages, tool arguments, decisions and outputs.

No response cache, resume, or best-run selection. Model timing excludes one-time model loading, which is recorded separately in the receipt. These are local wall-clock and token costs, not cloud prices or energy measurements. Zero executed writes reflects the deny-all approval policy; it does not imply the model never attempted an unauthorized action. Historical 72-task results and scripted regression fixtures remain separate.

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
