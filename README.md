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
| 01 | [rag-pipeline-eval](rag-pipeline-eval/) | Grounded retrieval, evaluation and release checks | Hit rate@1 **0.852 → 0.926 → 1.000** on 54 answerable questions / 35 chunks; answer-quality gate **HOLD**, official RAGAS cross-check unresolved. [Evidence](rag-pipeline-eval/docs/RESULTS.md) |
| 02 | [langgraph-agent-guardrails](langgraph-agent-guardrails/) | Tool permissions, human approval and redacted replay | Scripted harness **29/29**; Qwen 1.5B meets **8/13 benign, 11/16 adversarial** scenario expectations. [What these scores mean](langgraph-agent-guardrails/docs/RESULTS.md) |
| 03 | [llm-gateway-release](llm-gateway-release/) | Routing, resilience, streaming controls and release decisions | About **1 ms** overhead with an in-process fake backend; real 1.5B candidate quality improves but promotion is **HOLD** on latency. [Evidence](llm-gateway-release/docs/RESULTS.md) |
| 04 | [agent-mcp-eval](agent-mcp-eval/) | MCP permissions and trajectory-based comparison of single vs supervisor agents | Latest 18-task real-model slice: Qwen 1.5B **0/18 vs 2/18**, Qwen3-4B **12/18 vs 13/18**, repeated twice. Architecture advantage remains inconclusive; security and strict-matcher failures remain. [Results, cost and limits](agent-mcp-eval/docs/experiments/local-live-v2/INTERPRETATION.md) |

Companion repositories: [`llm-engineering-lab`](https://github.com/RiverHe2000/llm-engineering-lab)
(Transformer internals, LoRA, an inference server) and [`mlops-lab`](https://github.com/RiverHe2000/mlops-lab)
(MLflow lifecycle, SageMaker deployment, drift monitoring).

---

## Start here

For agent engineering, start with `agent-mcp-eval`: inspect a failed trajectory, run the
scripted demo, then read the comparison limits. For retrieval roles, use the lightweight
RAG quickstart and the paired retrieval/answer-quality report. These are measured laboratory
examples on fictional bank and wealth-platform data; scripted checks prove harness behaviour.
The reports separate historical runs from the latest controlled local-model comparison;
all are laboratory evidence rather than production deployments.

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
| Tests | `pytest` with per-project branch-coverage gates; offline CPU suites using scripted models and hashing embedders. The linked CI runs provide current counts |
| Determinism | every judge/model interaction is scriptable, so the CI gates measure the *pipeline*, not a model's mood |
| CI | one path-filtered workflow per project on Python 3.12/3.13, `HF_HUB_OFFLINE=1`; retrieval/red-team/release checks plus the MCP per-task harness regression gate; architecture comparison HOLD is a valid research result |

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
