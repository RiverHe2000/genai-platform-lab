# genai-platform-lab

[![rag-pipeline-eval](https://github.com/RiverHe2000/genai-platform-lab/actions/workflows/rag-pipeline-eval-ci.yml/badge.svg)](https://github.com/RiverHe2000/genai-platform-lab/actions/workflows/rag-pipeline-eval-ci.yml)
[![langgraph-agent-guardrails](https://github.com/RiverHe2000/genai-platform-lab/actions/workflows/langgraph-agent-guardrails-ci.yml/badge.svg)](https://github.com/RiverHe2000/genai-platform-lab/actions/workflows/langgraph-agent-guardrails-ci.yml)
[![llm-gateway-release](https://github.com/RiverHe2000/genai-platform-lab/actions/workflows/llm-gateway-release-ci.yml/badge.svg)](https://github.com/RiverHe2000/genai-platform-lab/actions/workflows/llm-gateway-release-ci.yml)
[![agent-mcp-eval](https://github.com/RiverHe2000/genai-platform-lab/actions/workflows/agent-mcp-eval-ci.yml/badge.svg)](https://github.com/RiverHe2000/genai-platform-lab/actions/workflows/agent-mcp-eval-ci.yml)

Four projects covering what an enterprise GenAI team actually has to build around a model —
**retrieve, act, ship, measure**: grounded retrieval with an evaluation harness that can gate
a release, a tool-using agent whose every input, tool call and output passes named guardrails
with a human in the loop for high-risk actions, an OpenAI-compatible gateway that routes,
protects and promotes model deployments on evidence, and a Model Context Protocol server with
a benchmark that grades an agent's whole trajectory rather than its final answer.

| # | Project | What it demonstrates | Headline result |
|---|---|---|---|
| 01 | [rag-pipeline-eval](rag-pipeline-eval/) — `ragpipe` | Offset-preserving chunking, BM25 from the formula, dense + reciprocal rank fusion, cross-encoder reranking, the four RAGAS metrics implemented from their definitions, bootstrap CIs, regression gates, cross-check against the official `ragas` | Hybrid retrieval lifts **hit_rate@1 0.852 → 0.926**; + reranker → **1.000** on every retrieval metric. End-to-end with a local judge: faithfulness 0.885 [0.82, 0.94]; **143 tests, 96 % coverage** |
| 02 | [langgraph-agent-guardrails](langgraph-agent-guardrails/) — `agentguard` | LangGraph state machine with typed state, `interrupt()` human approval, SQLite checkpoints, read-only SQL enforced by the SQLite authorizer, AST-allow-list calculator, injection/PII/topic/grounding rails, red-team harness with a CI gate | Scripted model: **29/29 scenarios, 16/16 adversarial cases caught, 0 benign blocked**. Real model: every input-side attack blocked before the model ran, poisoned tool output withheld; **126 tests, 96 % coverage** |
| 03 | [llm-gateway-release](llm-gateway-release/) — `llmgate` | OpenAI-compatible gateway: vLLM/OpenAI/local-HF/fake backends, retries + circuit breaker + bulkhead, primary/canary/shadow routing, streaming PII redaction, JSON-schema repair, Prometheus, load testing, paired-statistics promotion decision | Gateway overhead ≈ **1 ms p50 at 1 100 req/s**; promotion gate with paired bootstrap + exact McNemar and an exit code CI can act on; **50 tests, 96 % coverage** |
| 04 | [agent-mcp-eval](agent-mcp-eval/) — `mcpeval` | A real Model Context Protocol server (mcp 2.x) for a wealth platform with thirteen tools, resources and prompts; a client-side permission policy that rules on every call before the transport; a single agent and a LangGraph supervisor multi-agent system that differ only in orchestration; and a 72-task long-horizon benchmark that grades **trajectories** — tool-call F1 against a gold call set, step efficiency, cost, and a fifteen-class failure taxonomy assigned by rules rather than by a judge model | Success is a conjunction of a correct answer, no forbidden-tool violation and correct approval behaviour, so a right answer reached by an unauthorised route scores zero. The scripted harness run puts the supervisor arm at **3.7× the steps and 2.5× the tokens** of the single agent, and the policy refuses **3 out-of-scope tool calls** made by a specialist role before they reach the transport, each recorded with the rule that fired. An adversarial review then found the comparison was not controlled after all — only the single-agent arm had a loop guard, worth a **5.5× token gap** that would have been reported as the price of orchestration — along with a refusal matcher that scored a fabricated identifier as a clean success, and an approval gate that graded eight tasks identically whether or not the agent ever asked. The real-model run found two more: 13 of 72 tasks were reported as platform `run_error`s when the model had simply never emitted a valid action, and the injection family scored a **compromised answer as a clean pass** — Qwen2.5-1.5B summarised a poisoned policy document correctly and then reported placing the attacker's trade, without ever proposing the call, so the permission layer never saw it. Both now have their own failure class and their own check; **1 294 tests, 99.4 % coverage** |

Companion repositories: [`llm-engineering-lab`](https://github.com/RiverHe2000/llm-engineering-lab)
(Transformer internals, LoRA, an inference server) and [`mlops-lab`](https://github.com/RiverHe2000/mlops-lab)
(MLflow lifecycle, SageMaker deployment, drift monitoring).

---

## The through-line

The first three answer the three questions an enterprise asks about a GenAI feature, in order:

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

The fourth answers the question that follows all three, and the one teams usually answer with
an anecdote:

4. **Is the agent actually any good, and what does it cost?** (`mcpeval`) — 72 long-horizon
   tasks over a real MCP server, graded on the whole trajectory rather than the final answer,
   so a right answer reached by an unauthorised route scores zero and "the multi-agent version
   is better" becomes a paired comparison with an interval attached to it.

The corpus is a fictional Australian bank ("Meridian Bank") with policy documents in the
APRA / IFRS 9 / CPS 230 idiom, so the examples are the ones a bank interview will ask about;
`mcpeval` adds a synthetic wealth-management platform in the same idiom.

---

## Engineering standard (identical across the four)

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
make all                                  # ruff + mypy + pytest for all four
make PROJECT=rag-pipeline-eval test
```

---

## Layout

```
genai-platform-lab/
├── rag-pipeline-eval/           ragpipe: chunking, BM25, dense, fusion, rerank, RAGAS metrics, gates
├── langgraph-agent-guardrails/  agentguard: LangGraph graph, tools, rails, audit, red-team harness
├── llm-gateway-release/         llmgate: protocol, backends, routing, resilience, rails, promotion
├── agent-mcp-eval/              mcpeval: MCP server, permission policy, supervisor agents, benchmark
├── .github/workflows/           one path-filtered CI workflow per project
└── Makefile                     install / lint / type / test / all
```
