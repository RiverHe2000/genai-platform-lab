# llmgate · an OpenAI-compatible LLM gateway and an eval-gated release workflow

One endpoint in front of any number of model deployments — vLLM, OpenAI-compatible servers,
in-process Hugging Face models, deterministic fakes — with the production concerns a bank's
platform team actually owns: routing (primary/fallback, stable canary, shadow), resilience
(retries, circuit breaker, bulkhead), guardrails on the request *and* the token stream, API
keys, rate limits and budgets, Prometheus metrics, and a promotion workflow that evaluates a
candidate against the incumbent with paired statistics before it is allowed to serve.

| | |
|---|---|
| Quality gates | `ruff`, `mypy --strict`, **50 tests** (async, offline, ≈ 3 s), **96 % branch coverage** |
| Protocol | `/v1/chat/completions` (incl. SSE streaming), `/v1/completions`, `/v1/models`, OpenAI error envelope; `/health`, `/ready`, `/metrics`, `/admin/backends` |
| Backends | `VLLMBackend` (guided-decoding JSON, `/health`), `OpenAICompatBackend`, `HFLocalBackend` (real streaming via `TextIteratorStreamer`), `FakeBackend` |
| Headline | Gateway overhead ≈ **1 ms p50 / 1 ms p95 at 1 100 req/s** (fake backend, in-process); baseline-vs-candidate promotion of Qwen2.5-0.5B → 1.5B with paired bootstrap + McNemar: see [docs/RESULTS.md](docs/RESULTS.md) |

Companion projects: [`rag-pipeline-eval`](../rag-pipeline-eval) (retrieval + RAGAS-style
evaluation) and [`langgraph-agent-guardrails`](../langgraph-agent-guardrails) (LangGraph
agent with rails). Together: *retrieve, act, ship*.

---

## 1. Architecture

```
 client (any OpenAI SDK)
   │  Bearer key · rate limit · budget · request id
   ▼
 api.py ──► gateway.py ──► router.py ─► [candidates: primary, fallbacks | canary-first | + shadow]
              │  request rails: size, blocked terms, injection, PII redaction
              ▼  per backend: breaker.allow() → bulkhead slot → retries (429/5xx/transport only)
        backends/ (vllm · openai_compat · hf_local · fake)
              │  response rails: PII redact/block, JSON-schema validate + one repair round-trip
              ▼  streaming: StreamRedactor holds a tail window so PII split across chunks never leaks
        metrics (requests, latency, TTFT, tokens, breaker, rails, fallbacks, shadow) + JSON logs
```

| Concern | Where | Behaviour |
|---|---|---|
| Routing | `router.py` | direct backend name → that backend only; virtual model → strategy. Canary buckets a routing key (`user` or API-key principal) with SHA-256 so a user always lands on the same variant; shadow copies the request to the candidate and records both answers without touching the response |
| Resilience | `resilience.py`, `gateway.py` | retries with exponential backoff **only** for retriable failures; circuit breaker closed → open → half-open (one probe); bulkhead per backend with a queue timeout → fast 503; fallback across candidates; mid-stream failure ends the stream with `finish_reason: error` instead of a second beginning |
| Guardrails | `guardrails.py` | request: prompt-size cap, `max_tokens` cap, blocked terms, injection heuristics, PII redaction of user turns (TFN mod-11, Luhn, e-mail, AU phone); response: PII redact/block, JSON-schema enforcement (`jsonschema`) with a repair round-trip; streaming redactor with a property test |
| vLLM | `backends/vllm.py`, `deploy/` | `response_format: json_schema` → `guided_json`; `/health` probe; recommended server flags; compose file and Kubernetes manifests (GPU reservation, readiness on `/health`, HF cache PVC) |
| Auth / limits | `auth.py`, `ratelimit.py` | API keys stored as SHA-256 hashes; per-principal token bucket; daily token budget; OpenAI error codes (`invalid_api_key`, `rate_limit_exceeded`, `insufficient_quota`) with `Retry-After` |
| Observability | `observability.py` | Prometheus counters/histograms by route, backend and status; TTFT; breaker gauge; rail events; rejections by reason; JSON logs carrying the request id |
| Load test | `loadtest.py` | closed-loop generator: p50/p95/p99, throughput, tokens/s, TTFT for streaming, status histogram |
| Evaluation | `evaluation/` | JSONL suites (`exact`, `contains`, `regex`, `numeric`, `json_schema`, `refusal`), per-tag scores, paired bootstrap + exact McNemar comparison, promotion policy → `PROMOTE`/`HOLD` with a model-validation-style report; the CLI exit code is the decision |

---

## 2. Results

**Gateway overhead** — fake backends (zero model time), in-process ASGI, 300 requests per
row ([docs/experiments/loadtest_fake.md](docs/experiments/loadtest_fake.md)):

| Concurrency | p50 | p95 | p99 | req/s |
|---:|---:|---:|---:|---:|
| 1 | 1 ms | 1 ms | 1 ms | 1 112 |
| 8 | 5 ms | 8 ms | 18 ms | 1 428 |
| 32 | 18 ms | 40 ms | 40 ms | 1 541 |
| 8, streaming | 7 ms (TTFT 7 ms) | 10 ms | 12 ms | 1 041 |

So the full path — auth, rate limit, request rails, routing, breaker, bulkhead, response
rails, metrics — costs about a millisecond per request; the model dominates everything else.

**Release workflow on real models** — Qwen2.5-0.5B-Instruct (incumbent) vs
Qwen2.5-1.5B-Instruct (candidate) served through the gateway on one RTX 4070, evaluated on
`evalsets/finance_qa.jsonl` (33 cases: acronyms, facts, arithmetic, instruction following,
structured JSON, refusals), promotion policy in `deploy/promotion_policy.yaml`. The
per-case scores, the paired comparison, the SLO checks and the decision are in
[docs/RESULTS.md](docs/RESULTS.md) together with the canary split and the real-model load
test.

---

## 3. Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev,serve]"                             # add [hf] for in-process models

llmgate check-config --config deploy/gateway.fake.yaml
llmgate serve --config deploy/gateway.fake.yaml --port 8080

# any OpenAI SDK works: base_url=http://127.0.0.1:8080/v1
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "gateway-default", "messages": [{"role": "user", "content": "hello"}]}' -D -

# load test (a running gateway, or in-process from a config), streaming or not
llmgate loadtest --base-url http://127.0.0.1:8080/v1 --model gateway-default --concurrency 1 8 32 --requests 300 --stream

# evaluate two deployments and decide (exit 0 = PROMOTE, 1 = HOLD)
llmgate eval --base-url http://127.0.0.1:8080/v1 --model qwen05 --suite evalsets/finance_qa.jsonl --out runs/base
llmgate eval --base-url http://127.0.0.1:8080/v1 --model qwen15 --suite evalsets/finance_qa.jsonl --out runs/cand
llmgate promote --candidate runs/cand/report.json --baseline runs/base/report.json --policy deploy/promotion_policy.yaml --out runs/promotion

# vLLM on a Linux GPU host
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct docker compose -f deploy/docker-compose.yml up
kubectl apply -f deploy/k8s/

make all      # ruff + mypy --strict + pytest
```

CI runs the tests, then the whole release path with fake backends (`eval` ×2 → `promote`),
then builds the Docker image.

---

## 4. Design decisions

* **The wire contract is OpenAI's**, on both sides. Unknown request fields are ignored (real
  SDKs send them); unsupported values (`n > 1`) are rejected explicitly with the standard
  error envelope.
* **Retry only what can succeed.** 429/5xx/transport errors retry with backoff; a 400 never
  does. Fallback is a routing decision, the breaker is a health decision, the bulkhead is a
  capacity decision — three separate, separately tested state machines.
* **Streaming is first-class**, including its failure semantics and its PII problem: a tail
  window is held back and emitted only at boundaries that do not split digit groups.
* **vLLM where it belongs.** Guided decoding makes structured output a guarantee instead of a
  repair; prefix caching, `max-model-len` and `served-model-name` are set in the deployment
  files, not left to defaults. The adapter is contract-tested; the real server runs on Linux
  (this machine is Windows without WSL/Docker, so the reported model numbers use the
  in-process backend on the same GPU).
* **Promotion is a statistical decision with an exit code**: absolute floor, non-inferiority
  margin on a paired bootstrap, McNemar on discordant pairs, p95 latency and error-rate SLOs,
  minimum sample size — and a report a model-risk reviewer can read.

Interview preparation notes: [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md).

---

## 5. Layout

```
src/llmgate/
├── api.py, gateway.py, router.py, resilience.py, guardrails.py, auth.py, ratelimit.py
├── observability.py, protocol.py, config.py, loadtest.py, cli.py
├── backends/     base.py, fake.py, openai_compat.py, vllm.py, hf_local.py
└── evaluation/   suite.py (cases + scorers), runner.py, compare.py, promote.py
deploy/           gateway.{fake,local,vllm,compose}.yaml, promotion_policy{,.ci}.yaml, Dockerfile, docker-compose.yml, k8s/
evalsets/         finance_qa.jsonl
tests/            50 tests: protocol, config, every backend (incl. a tiny random HF model), breaker/retry/bulkhead, router, rails, auth/limits, gateway end-to-end (fallback, breaker, JSON repair, canary, shadow, streaming), HTTP API, load test, evaluation + promotion, CLI
docs/             RESULTS.md, INTERVIEW_NOTES.md, experiments/
```
