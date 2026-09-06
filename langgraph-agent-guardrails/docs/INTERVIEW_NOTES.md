# Interview notes — LangGraph agent with guardrails

## Architecture

**Walk me through one request.** `input_guard` scores the message for prompt injection,
classifies the topic (in scope / out of scope / restricted advice) and redacts PII; a blocked
message ends the graph with a refusal before the model is ever called. `agent` builds the
transcript (system prompt with the tool catalogue + conversation) and asks the model for
*one JSON action*. A tool call goes to `tool_guard`, which validates the arguments against
the tool's Pydantic schema and routes high-risk tools to `approval`; `execute` runs the tool,
scans its output for injection and PII, and appends it to the transcript as delimited,
untrusted data. The loop continues until the model returns a final answer, which passes
through `output_guard` (PII, numeric grounding, advice language, length). Every step, tool
call and rail event is stamped with the turn number and written to the audit log.

**Why a JSON action protocol instead of native tool calling?** Portability and testability.
Native function calling differs per provider and is absent or unreliable on small open
models; a JSON envelope works with vLLM, OpenAI, and a 1.5 B local model alike, and a
scripted model can drive the whole graph deterministically in tests and CI. The cost is a
repair loop for malformed output (`max_parse_retries`), which is measured rather than
hidden: the report counts repairs and the CI scenarios include truncated JSON and wrong
argument names.

**What does LangGraph give you that a while-loop does not?** Three things I actually use:
(1) a *checkpointer* — every node transition is persisted per `thread_id`, so a conversation
can be resumed from another process; (2) `interrupt()` — the approval node pauses the graph
and `Command(resume=...)` continues it with the human's decision, without me writing a
state machine for "waiting"; (3) reducers on state keys (`operator.add` on messages, tool
records and rail events) so multi-turn accumulation is declarative. The graph is also a
diagram I can put in front of a risk reviewer.

**How does resume work exactly?** The `approval` node calls `interrupt(payload)`. LangGraph
raises, checkpoints the state, and `invoke` returns with an `__interrupt__` entry — the API
reports `awaiting_approval` with the pending action. On `Command(resume=decision)` the node
re-executes from its start and `interrupt()` returns the decision, so the node must be
side-effect free before that call (it is: side effects live in `execute`). The test
`test_resume_across_processes_with_sqlite_checkpoint` builds a second agent over the same
SQLite checkpoint file and completes the turn there.

## Guardrails

**Why several small rails rather than one classifier?** Each rail is a pure function with a
name, a stage and a score, so a reviewer can see exactly which rule fired and why, thresholds
are tunable per deployment, and each is unit-tested with its own adversarial cases. A
classifier could be added as another rail; it would not replace the audit trail.

**Prompt injection in tool output.** Retrieved documents are the classic vector: a policy
clause that says "ignore previous instructions". Tool results are wrapped in explicit
untrusted-data delimiters, scanned with the same injection rules, and withheld entirely
when they trip — the model receives a placeholder instead. Scenario `r06` plants such a
clause in the policy corpus and checks the answer never contains the attacker's payload.

**Read-only SQL — how do you actually enforce it?** Not with regex alone. The SQLite
authorizer callback runs *inside the engine* for every operation the parsed statement
attempts: `SELECT`/`READ` on the `loans` table is allowed, everything else — `customers`
(names, TFNs), `review_queue`, `sqlite_master`, `PRAGMA`, `INSERT`, `ATTACH` — is denied at
the engine level, so a JOIN to `customers` fails even inside a syntactically pure SELECT. On
top of that: single-statement check, a progress handler that aborts runaway queries, and a
row cap with an explicit truncation flag.

**The calculator is `eval` with extra steps, isn't it?** No — it walks the AST and only
accepts numeric literals, arithmetic operators, a fixed function table and two constants.
Attribute access, subscripts, names, lambdas, comparisons and keyword arguments are rejected
before anything is evaluated; exponents and nesting depth are capped. The tests throw
`__import__('os')`, `().__class__.__bases__`, and `9 ** 99999` at it.

**PII: why checksums?** A TFN is nine digits with a weighted mod-11 check; a card number
passes Luhn. Without those checks every loan balance and order number is a false positive
and staff turn the rail off. Redaction applies to input, tool output, the final answer *and
the audit log itself* — the log is a common leak path.

**Numeric grounding.** Every number in the final answer must appear in the evidence the
agent actually saw (user input or tool outputs). It is not a faithfulness judge, but it is
deterministic, free, and catches the most damaging hallucination on a credit desk: an
invented figure. The evaluation reports `ungrounded_number_rate`.

## Evaluation

**What does the scenario harness measure?** Task pass rate on benign scenarios (status,
tool sequence as an ordered subsequence, must/must-not contain, expected rails), catch rate
on adversarial scenarios, false-block rate on benign ones, mean steps and latency. With the
scripted model it is a deterministic regression suite for the graph and the rails — that is
the CI gate (`--gate` fails the build if any adversarial case is missed or any benign case is
blocked). With a real model it measures the model's competence on the same tasks; the two
runs are reported separately and the difference is informative (see `docs/RESULTS.md`).

**How would you extend this for production?** Replace the pattern rails with a tuned
classifier *behind the same interface*; add per-user authorisation to the tool context so
the loan book is filtered by entitlement; stream the audit log to the SIEM; run the red-team
suite in CI and quarterly with fresh cases; sample production transcripts into the scenario
set; and put the whole thing behind the model gateway from the companion
`llm-gateway-release` project so model changes go through the same promotion gate.

## Model risk / governance angle

**How does this map to APRA CPS 230 / CPG 235?** Critical decisions (flagging a loan for
review) require human approval with the approver recorded; every model interaction is
logged with the inputs, tool calls and rail events for replay; the model is replaceable
behind a protocol; and the evaluation suite is versioned data with a pass/fail gate, which is
the evidence base a model-validation function needs to sign off an assistant for use.
