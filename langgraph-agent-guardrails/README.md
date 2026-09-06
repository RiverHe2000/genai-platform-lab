# agentguard · a LangGraph agent with layered guardrails for a bank's credit desk

A tool-using assistant for a lending team — policy lookup, read-only SQL over a loan book,
exact arithmetic, and one *high-risk* action (flagging a loan for review) that pauses the
graph for human approval. Every input, tool call and output passes through named guardrails,
every step lands in a redacted audit trail, and a scenario/red-team harness with a CI gate
measures both the rails and the model.

| | |
|---|---|
| Quality gates | `ruff`, `mypy --strict`, **126 tests** (offline, ≈ 5 s), **96 % branch coverage** |
| Framework | LangGraph 1.x: typed state with reducers, conditional edges, `interrupt()` + `Command(resume=…)`, in-memory and SQLite checkpointers |
| Rails | prompt injection (input *and* tool output), topic scope, PII with TFN/Luhn checksums, tool-argument validation, human approval, numeric grounding, advice language |
| Headline | Scripted model: **29/29 scenarios pass, 16/16 adversarial cases caught, 0 benign cases blocked** (the CI gate). Real model (Qwen2.5-1.5B-Instruct, one RTX 4070): every input-side attack blocked before the model runs, poisoned tool output withheld, approval flow completed across two processes; 8/13 benign and 11/16 adversarial scenario expectations met, failure classes analysed in [docs/RESULTS.md](docs/RESULTS.md) |

Companion projects: [`rag-pipeline-eval`](../rag-pipeline-eval) (retrieval + RAGAS-style
evaluation) and [`llm-gateway-release`](../llm-gateway-release) (serving, vLLM, eval-gated
promotion). Together: *retrieve, act, ship*.

---

## 1. Architecture

```
 user ─► input_guard ─┬─(injection / topic / PII block)──────────────────────► refusal
                      └─► agent ◄────────────────────────────────────────┐
                            │ one JSON action per turn                   │ tool result (wrapped as
                            ├─ final ─► output_guard ─► answer            │ untrusted data), repair
                            └─ tool  ─► tool_guard ─┬─ invalid args ──────┤ message, or rejection
                                                    ├─ low risk ─► execute ┤
                                                    └─ high risk ─► approval [interrupt] ─(approved)─► execute
                                                                        └─(rejected)───────────────────┘
```

| Component | File | Notes |
|---|---|---|
| Graph | `graph.py` | Six nodes, pure functions of the typed state; `AgentState` uses `operator.add` reducers so messages, tool records and rail events accumulate across turns; every record is stamped with the turn number |
| Action contract | `schemas.py` | `{"type":"tool",…}` / `{"type":"final",…}`; plain prose is accepted as a final answer, truncated/invalid JSON triggers a repair turn (bounded by `max_parse_retries`) |
| Models | `llm.py` | One `ChatModel` protocol: scripted `FakeChatModel`, OpenAI-compatible HTTP (vLLM/OpenAI) with bounded retries, in-process Hugging Face |
| Tools | `tools/` | `calculate` (AST allow-list evaluator), `describe_loanbook` / `query_loanbook` (SQLite **authorizer** = engine-level read-only on one table, single statement, execution budget, row cap), `search_policy`, `flag_for_review` (**high risk**) |
| Guardrails | `guardrails/` | `pii` (email, AU phone, TFN mod-11, card Luhn), `injection` (noisy-OR of weighted patterns, also applied to tool outputs), `topic` (in scope / out of scope / restricted advice), `output` (PII, numeric grounding, advice disclaimer, length) |
| Approval | `graph.approval` + `agent.resume` | `interrupt()` pauses with the pending action; `Command(resume={"approved", "approver", "note"})` continues; the approver's identity is written with the tool record and the review-queue row |
| Durability | `agent.build_checkpointer` | `InMemorySaver` by default, `SqliteSaver` with `--checkpoint-path`; a thread can be resumed from another process (tested) |
| Audit | `audit.py` | Append-only JSONL, PII-redacted before writing, `agentguard replay --thread …` reconstructs the timeline |
| Evaluation | `evaluation/` | Scenario spec (expected status, tool subsequence, must/must-not contain, expected rails, approval decision, optional poisoned policy clause), scripted or real model, report with pass/catch/false-block rates, `--gate` |
| API | `api.py` | `POST /chat`, `POST /approve`, `GET /threads/{id}`, `GET /audit/{id}`, `/health`; Pydantic validation, request ids |

---

## 2. Results

**Scripted model (deterministic, what CI runs)** — `scenarios/core.jsonl` (13 benign) +
`scenarios/redteam.jsonl` (16 adversarial); full table in
[docs/experiments/scripted_report.md](docs/experiments/scripted_report.md).

| Metric | Value |
|---|---:|
| Benign task pass rate | 13/13 |
| Adversarial catch rate | 16/16 |
| Benign false-block rate | 0/13 |
| Gate | PASS |

What the adversarial set covers: instruction override, role-switch jailbreak, exfiltration
requests, out-of-scope and restricted-advice requests, **prompt injection planted in a policy
document returned by a tool**, PII in the model's answer, fabricated numbers, unknown tool,
SQL mutation / stacked statements / forbidden table, approval bypass, fake transcript
injection, calculator code injection, prompt-leak request.

**Real model** — the same scenarios driven by Qwen2.5-1.5B-Instruct (greedy, bf16):

| Metric | Benign (13) | Adversarial (16) |
|---|---:|---:|
| Scenario expectations met | 8 / 13 | 11 / 16 |
| Blocked before the model ran (injection / topic rails) | – | 7 / 7 input-side attacks |
| Poisoned tool output withheld | – | 1 / 1 |
| Benign requests wrongly blocked | 0 / 13 | – |
| Answers with ungrounded numbers | 0 / 13 | 0 / 16 |

Of the ten unmet expectations, seven are cases where the scenario scripts a *misbehaviour*
(a bad argument, a PII leak, invented numbers, forwarding a `DELETE` to the SQL tool) that
the real model simply did not commit, and two are wording mismatches after a correctly
executed approval flow; one is a genuine capability gap (skipping schema discovery). The
first real-model run also exposed a systematic envelope mistake the 1.5 B model makes
(`"type": "<tool name>"`), now repaired by a lenient parser with strict argument validation.
Details, the failure taxonomy and the two-process approval replay:
[docs/RESULTS.md](docs/RESULTS.md). The point of running both: the scripted run proves the
*rails and the graph*; the real run measures the *model*, and the gap is what a stronger
model buys.

---

## 3. Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"                                   # add [hf] for a local model, [serve] for the API

# one turn with the scripted model (no model download)
agentguard chat "What is the maximum LVR without LMI?" --model fake

# a real local model, durable state, and an approval that arrives from another process
agentguard chat "Flag loan L00042 for review, 45 days past due" --model hf \
  --model-name Qwen/Qwen2.5-1.5B-Instruct --thread demo \
  --checkpoint-path state/checkpoints.sqlite --loanbook-path state/loanbook.sqlite --audit-path state/audit.jsonl
agentguard approve --thread demo --approved true --approver risk.reviewer@bank \
  --checkpoint-path state/checkpoints.sqlite --loanbook-path state/loanbook.sqlite --audit-path state/audit.jsonl
agentguard replay --thread demo --audit-path state/audit.jsonl

# evaluation (the CI gate)
agentguard eval --scenarios scenarios/core.jsonl scenarios/redteam.jsonl --model fake --out runs/scripted --gate

# HTTP API
agentguard serve --model hf --model-name Qwen/Qwen2.5-1.5B-Instruct --checkpoint-path state/checkpoints.sqlite

# quality gates
make all
```

Every knob is also an environment variable (`AGENTGUARD_MODEL__KIND=openai`,
`AGENTGUARD_MODEL__BASE_URL=http://vllm:8000/v1`, `AGENTGUARD_GUARDRAILS__PII_INPUT_ACTION=block`).

---

## 4. Design decisions

* **JSON action protocol, not native tool calling** — portable across vLLM/OpenAI/HF and
  small models, deterministic to test; the repair loop is bounded and measured.
* **Rails are small, named, pure functions** — each returns an event with a stage, an action
  and a score; the audit trail shows exactly which rule fired. A learned classifier would be
  another rail behind the same interface, not a replacement for the audit trail.
* **Tool output is data** — wrapped in explicit untrusted-data delimiters, scanned with the
  injection rules, withheld when they fire. This is the control for the RAG-poisoning class of
  attack.
* **Engine-level SQL safety** — the SQLite authorizer denies anything but SELECT/READ on the
  `loans` table *inside the parser*, so a JOIN to `customers` fails even in a syntactically
  innocent query; regex screening alone is not a control.
* **High-risk actions need a human** — the graph interrupts; the approver and note are
  persisted with the action; rejections are recorded and fed back to the model.
* **Numbers must be grounded** — a deterministic check that every figure in the answer
  appears in the evidence the agent saw; cheap, and it catches the costliest hallucination on
  a credit desk.

Interview preparation notes: [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md).

---

## 5. Layout

```
src/agentguard/
├── graph.py, agent.py, schemas.py, llm.py, config.py, audit.py, api.py, cli.py
├── tools/        base.py (registry, validation, risk), calculator.py, loanbook.py, policy_search.py, review.py
├── guardrails/   pii.py, injection.py, topic.py, output.py
└── evaluation/   scenarios.py, runner.py
scenarios/        core.jsonl (benign), redteam.jsonl (adversarial)
tests/            126 tests: every rail, every tool, the graph end-to-end, interrupt/resume across a SQLite checkpoint, API, CLI, evaluation harness
docs/             RESULTS.md, INTERVIEW_NOTES.md, experiments/
```
