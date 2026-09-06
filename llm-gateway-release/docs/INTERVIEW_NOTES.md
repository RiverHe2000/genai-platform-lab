# Interview notes — LLM gateway and eval-gated releases

## Why a gateway at all

**What problem does it solve?** Every team calling a model directly means every team
re-implementing retries, timeouts, key handling, PII controls and logging — differently.
A gateway gives one OpenAI-compatible endpoint with those concerns in one audited place,
and one place to swap or A/B a model without touching clients. It is also where a bank's
model-risk controls (which model served which request, with which guardrails) get their
evidence.

**Why OpenAI-compatible rather than your own API?** Every SDK, framework and evaluation
tool already speaks it; vLLM, TGI and Ollama serve it; so the gateway is a drop-in
`base_url` change for clients and a drop-in consumer for backends. Requests use
`extra="ignore"` so real SDKs (which send fields I do not implement) keep working, and the
one thing I cannot honour (`n > 1`) is rejected explicitly.

## vLLM

**What does vLLM do that a plain Hugging Face server does not?** Continuous (iteration-level)
batching — new sequences join at every decode step — and PagedAttention, which allocates the
KV cache in blocks so memory is not reserved for the maximum context up front. Together they
give an order of magnitude more throughput at the same latency. My companion `llmserve`
project implements the simpler *dynamic* batching so I understand exactly what the gap is.

**Which flags matter in production?** `--max-model-len` (KV-cache memory scales with it),
`--gpu-memory-utilization`, `--enable-prefix-caching` (shared system prompts are computed
once), `--max-num-seqs`, `--dtype`/`--quantization`, `--served-model-name` (stable public
name), and `--api-key`. They are documented in `backends/vllm.py` and set in the compose file
and the Kubernetes manifest.

**Structured output.** For `response_format: json_schema` the vLLM adapter sends
`guided_json`, so the grammar-constrained decoder *guarantees* schema-valid JSON. For
backends without guided decoding the gateway validates the JSON against the schema and does
one repair round-trip; both paths surface as `json_schema` guardrail events in the headers
and metrics.

**Why is vLLM not running in the reported results?** vLLM is Linux/CUDA-only and this
machine is Windows without WSL or Docker. The adapter is tested against a recorded fake of
vLLM's wire behaviour (payload shape, `/health`, guided decoding fields, SSE streaming), and
the real numbers were produced with the in-process Hugging Face backend on the same GPU. The
compose file and manifests are how it runs on a Linux host.

## Resilience

**Retries, fallback, circuit breaker — how do they interact?** Retries handle transient
failures of one backend (429/5xx/transport errors only — a 400 is never retried). Fallback
moves to the next candidate when retries are exhausted. The breaker stops sending traffic to
a backend that keeps failing (closed → open after N failures → half-open after the recovery
window → one probe). The bulkhead caps in-flight requests per backend so one slow model
cannot exhaust the event loop, and returns a fast 503 instead of queueing forever. All four
are injectable-clock state machines with their own tests.

**What about streaming?** A failure *before the first token* falls through to the next
candidate transparently. A failure *mid-stream* cannot be retried — the client has already
received part of an answer — so the gateway flushes what it holds and ends the stream with
`finish_reason: "error"`. That asymmetry is deliberate and tested.

**Canary and shadow.** Canary hashes a routing key (the `user` field or the API-key
principal) into a stable bucket, so a given user always sees the same variant — required for
a fair comparison and for debugging. Shadow sends a copy to the candidate and records both
answers for offline comparison without affecting the response; the gateway never awaits it.

## Guardrails

**Where do they run and what do they cost?** Request rails (prompt size, blocked terms,
injection heuristics, PII redaction of user turns) run before routing; response rails (PII,
JSON schema) after. All are regex/checksum based — microseconds — because a gateway is on
the latency path of every call. Heavier classifiers belong in an async shadow path or in the
application (see the `langgraph-agent-guardrails` project).

**PII in a token stream.** Tokens split a TFN or an e-mail across chunks, so a per-chunk
regex misses them. The stream redactor holds back a tail window and only emits up to a
whitespace boundary that does not split a digit group; the property test feeds text with PII
in random chunk sizes and asserts the output equals the fully-redacted text every time.

## Release process

**What is "eval-gated promotion"?** A candidate deployment is evaluated on the same cases
as the incumbent, through the same gateway path, and promoted only if it clears an absolute
quality floor, is non-inferior to the baseline (paired bootstrap on per-case differences,
plus an exact McNemar test on the win/loss pattern), and meets the p95-latency and error-rate
SLOs. The CLI's exit code *is* the decision, so it drops into a pipeline. The report is
structured like a model-validation memo: scope, data, results, statistical tests, policy
checks, limitations, decision.

**Why paired statistics?** With 33 cases, a 0.52 vs 0.45 mean score is well inside noise.
Pairing on the same cases removes the between-case variance; McNemar looks only at the
discordant pairs. With small suites the honest verdict is often "non-inferior" or
"inconclusive", and the policy makes that explicit with a margin rather than pretending a
point estimate is a fact.

**How would you scale the evaluation?** More cases per tag (the by-tag table shows where a
model regresses), LLM-judged rubrics for open-ended tasks with a judge whose agreement with
humans has been measured (see `rag-pipeline-eval`), production traffic replayed through the
shadow path, and periodic re-evaluation because the incumbent's data drifts too.

## Observability and operations

Prometheus metrics by route/backend/status, latency and time-to-first-token histograms,
token counters, breaker state, guardrail events, rejections by reason; JSON logs with the
request id on every line; `/ready` reflects the primary's breaker and health so Kubernetes
stops routing to a pod whose model is down; the image is non-root with a health check; the
HPA scales the gateway on CPU while vLLM scales separately on GPU.
