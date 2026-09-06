# genai-platform-lab

[![rag-pipeline-eval](https://github.com/ChuanHe-PhD/genai-platform-lab/actions/workflows/rag-pipeline-eval-ci.yml/badge.svg)](https://github.com/ChuanHe-PhD/genai-platform-lab/actions/workflows/rag-pipeline-eval-ci.yml)
[![langgraph-agent-guardrails](https://github.com/ChuanHe-PhD/genai-platform-lab/actions/workflows/langgraph-agent-guardrails-ci.yml/badge.svg)](https://github.com/ChuanHe-PhD/genai-platform-lab/actions/workflows/langgraph-agent-guardrails-ci.yml)
[![llm-gateway-release](https://github.com/ChuanHe-PhD/genai-platform-lab/actions/workflows/llm-gateway-release-ci.yml/badge.svg)](https://github.com/ChuanHe-PhD/genai-platform-lab/actions/workflows/llm-gateway-release-ci.yml)

Three projects covering what an enterprise GenAI team actually has to build around a model —
**retrieve, act, ship**: grounded retrieval with an evaluation harness that can gate a
release, a tool-using agent whose every input, tool call and output passes named guardrails
with a human in the loop for high-risk actions, and an OpenAI-compatible gateway that routes,
protects and promotes model deployments on evidence.

| # | Project | What it demonstrates | Headline result |
|---|---|---|---|
| 01 | [rag-pipeline-eval](rag-pipeline-eval/) — `ragpipe` | Offset-preserving chunking, BM25 from the formula, dense + reciprocal rank fusion, cross-encoder reranking, the four RAGAS metrics implemented from their definitions, bootstrap CIs, regression gates, cross-check against the official `ragas` | Hybrid retrieval lifts **hit_rate@1 0.852 → 0.926**; + reranker → **1.000** on every retrieval metric. End-to-end with a local judge: faithfulness 0.885 [0.82, 0.94]; **143 tests, 96 % coverage** |
| 02 | [langgraph-agent-guardrails](langgraph-agent-guardrails/) — `agentguard` | LangGraph state machine with typed state, `interrupt()` human approval, SQLite checkpoints, read-only SQL enforced by the SQLite authorizer, AST-allow-list calculator, injection/PII/topic/grounding rails, red-team harness with a CI gate | Scripted model: **29/29 scenarios, 16/16 adversarial cases caught, 0 benign blocked**. Real model: every input-side attack blocked before the model ran, poisoned tool output withheld; **126 tests, 96 % coverage** |
| 03 | [llm-gateway-release](llm-gateway-release/) — `llmgate` | OpenAI-compatible gateway: vLLM/OpenAI/local-HF/fake backends, retries + circuit breaker + bulkhead, primary/canary/shadow routing, streaming PII redaction, JSON-schema repair, Prometheus, load testing, paired-statistics promotion decision | Gateway overhead ≈ **1 ms p50 at 1 100 req/s**; promotion gate with paired bootstrap + exact McNemar and an exit code CI can act on; **50 tests, 96 % coverage** |

Companion repositories: [`llm-engineering-lab`](https://github.com/ChuanHe-PhD/llm-engineering-lab)
(Transformer internals, LoRA, an inference server) and [`mlops-lab`](https://github.com/ChuanHe-PhD/mlops-lab)
(MLflow lifecycle, SageMaker deployment, drift monitoring).

---

## The through-line

The three answer the three questions an enterprise asks about a GenAI feature, in order:

1. **Is the answer grounded, and can you prove it?** (`ragpipe`) — retrieval metrics against
   a gold set, RAGAS metrics with an LLM judge whose failures are reported as *missing*
   rather than absorbed into an average, bootstrap intervals, and gates that can be applied
   to the conservative bound of the interval.
2. **What happens when the model can act?** (`agentguard`) — tool arguments validated before
   execution, tool output treated as untrusted data, high-risk actions paused for a named
   approver, everything written to a PII-redacted audit trail that replays.
3. **How does it reach production and change safely?** (`llmgate`) — one endpoint, one place
   for keys/limits/rails, and a model change that only ships if it is non-inferior to the
   incumbent on the same evaluation cases.

The corpus is a fictional Australian bank ("Meridian Bank") with policy documents in the
APRA / IFRS 9 / CPS 230 idiom, so the examples are the ones a bank interview will ask about.

---

## Engineering standard (identical across the three)

| Gate | Tooling |
|---|---|
| Lint + format | `ruff` with a broad rule set |
| Types | `mypy --strict` on `src/` **and** `tests/` |
| Tests | `pytest` with branch-coverage gates ≥ 90 %; **319 tests total**, all offline on CPU in seconds — scripted models and hashing embedders stand in for real ones |
| Determinism | every judge/model interaction is scriptable, so the CI gates measure the *pipeline*, not a model's mood |
| CI | one path-filtered workflow per project on Python 3.12/3.13, `HF_HUB_OFFLINE=1`; each runs that project's own release gate (retrieval gate, red-team gate, promotion dry run) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
make install
make all                                  # ruff + mypy + pytest for all three
make PROJECT=rag-pipeline-eval test
```

---

## Layout

```
genai-platform-lab/
├── rag-pipeline-eval/           ragpipe: chunking, BM25, dense, fusion, rerank, RAGAS metrics, gates
├── langgraph-agent-guardrails/  agentguard: LangGraph graph, tools, rails, audit, red-team harness
├── llm-gateway-release/         llmgate: protocol, backends, routing, resilience, rails, promotion
├── .github/workflows/           one path-filtered CI workflow per project
└── Makefile                     install / lint / type / test / all
```
