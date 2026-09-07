# ragpipe · hybrid-retrieval RAG with a RAGAS-style evaluation harness and release gates

A retrieval-augmented question-answering pipeline over a bank policy corpus, built the way
an enterprise team has to build one: structure-aware chunking with traceable offsets, BM25 +
dense embeddings fused by reciprocal rank fusion, a cross-encoder reranker, grounded
generation with citations and abstention — and an evaluation harness that implements the
four RAGAS metrics from their definitions, reports bootstrap confidence intervals, compares
two runs with paired statistics, enforces thresholds as a CI gate, and cross-checks itself
against the official `ragas` package with the same judge.

| | |
|---|---|
| Quality gates | `ruff`, `mypy --strict`, **143 tests** (offline, CPU, ≈ 6 s), **96 % branch coverage** |
| Corpus / eval set | 20 policy documents (credit, IFRS 9, capital, liquidity, CPS 230, AML, privacy, GenAI governance …) → 35 chunks; 60 questions: 42 single-hop, 5 multi-hop, 12 paraphrased, 6 unanswerable |
| Headline | Hybrid (bge-small + BM25, RRF) lifts **hit_rate@1 from 0.852 (BM25) to 0.926**; adding the cross-encoder reranker reaches **1.000 on every retrieval metric** at 189 ms/query. End-to-end RAGAS metrics with a local judge, the hybrid-vs-BM25 paired comparison, the production-gate decision and the status of the official-`ragas` cross-check: [docs/RESULTS.md](docs/RESULTS.md) |

Companion projects: [`langgraph-agent-guardrails`](../langgraph-agent-guardrails) (LangGraph
agent with rails) and [`llm-gateway-release`](../llm-gateway-release) (serving, vLLM,
eval-gated promotion). Together: *retrieve, act, ship*.

---

## 1. Architecture

```
 corpus/*.md ─► RecursiveChunker (paragraph → sentence → word, offsets kept) ─► chunks.jsonl
                     ├─► BM25Index (inverted index, Okapi formula)            ─► bm25.json
                     └─► Embedder (bge-small | hashing) ─► SqliteVectorStore  ─► vectors.sqlite
                                                                                manifest.json (embedder, corpus fingerprint)
 question ─► HybridRetriever (RRF over BM25 + dense candidates) ─► CrossEncoderReranker ─► top-k
          ─► grounded prompt with numbered passages ─► LLM (fake | OpenAI-compatible/vLLM | local HF)
          ─► answer + resolved [n] citations + abstention flag + timings
 evalset ─► runner: retrieval metrics · lexical proxies · RAGAS metrics via Judge ─► report.json/.md
          ─► gates (thresholds, optionally on the CI bound) · compare (paired bootstrap) · ragas-crosscheck
```

| Layer | Files | What is worth knowing |
|---|---|---|
| Chunking | `textproc.py`, `chunking.py` | Fixed-window baseline and a recursive chunker; every chunk is `doc.text[start:end]`; property tests: ordering, no overlap, full coverage, size bound |
| Lexical | `bm25.py` | Okapi BM25 with inverted index, tested against hand-computed scores, JSON-persisted |
| Dense | `embeddings.py`, `vectorstore.py` | `HashEmbedder` (offline, deterministic) and `SentenceTransformerEmbedder` (bge query instruction); SQLite cache; NumPy and SQLite stores property-tested for identical results |
| Fusion / rerank | `retrieval.py` | RRF and convex fusion (both tested on the formula), cross-encoder reranking over the top-20; every hit carries its per-source scores |
| Generation | `llm.py`, `prompts.py`, `pipeline.py` | One `LLM` protocol (fake / HTTP with bounded retries / HF); citations resolved to chunk ids; abstention detection; index manifest guards against mismatched embedders |
| Evaluation | `evaluation/` | `retrieval_metrics` (hit rate, recall, precision, MRR, MAP, nDCG), `ragas_metrics` (faithfulness, answer relevancy, context precision, context recall + lexical F1/EM/coverage), `judge` (strict JSON, schema validation, one repair, audit trail), `runner` (bootstrap CIs, per-tag slices), `gates`, `retrieval_bench`, `ragas_adapter` |

---

## 2. Results

### Retrieval (deterministic; `ragpipe retrieval-bench`, k = 5, 54 questions with gold documents, 95 % bootstrap CIs)

| Retriever | hit_rate@1 | hit_rate@5 | MRR | nDCG@5 | ms / query |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.852 [0.76, 0.94] | 1.000 | 0.911 [0.85, 0.97] | 0.934 | 0.1 |
| dense, hashing embedder (offline) | 0.759 [0.65, 0.87] | 0.889 | 0.809 | 0.827 | 0.1 |
| dense, bge-small-en-v1.5 | 0.889 [0.81, 0.96] | 0.981 | 0.932 | 0.945 | 8.7 |
| hybrid RRF (BM25 + bge) | **0.926** [0.85, 0.98] | 1.000 | 0.963 [0.93, 0.99] | 0.973 | 7.5 |
| hybrid convex (BM25 + bge) | 0.926 | 1.000 | 0.960 | 0.970 | 7.8 |
| hybrid RRF + cross-encoder rerank | **1.000** | 1.000 | **1.000** | **1.000** | 189.3 |

Full tables: [docs/experiments/retrieval_bench_bge.md](docs/experiments/retrieval_bench_bge.md),
[retrieval_bench_hash.md](docs/experiments/retrieval_bench_hash.md). The `paraphrase` slice
(questions rephrased to avoid the document's vocabulary) is where BM25 loses and the dense
retriever wins; fusion keeps the best of both, and the reranker fixes the remaining
rank-1 misses at ~25× the latency — the classic retrieve-cheap-then-rerank trade.

### End-to-end (generation + RAGAS metrics)

Generator and judge: Qwen2.5-1.5B-Instruct on one RTX 4070; retriever: hybrid RRF + rerank;
baseline for the paired comparison: BM25 only. Faithfulness, answer relevancy, context
precision/recall with confidence intervals, abstention behaviour on the six unanswerable
questions, the `ragpipe compare` verdicts, the production gate outcome, and the status of the
cross-check against the official `ragas` implementation are in [docs/RESULTS.md](docs/RESULTS.md).

---

## 3. Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -e ".[dev]"                                   # add [dense] for bge/reranker, [hf] for a local model, [ragas] for the cross-check

ragpipe ingest --corpus corpus --index index/bge --embedder sentence-transformers
ragpipe query --index index/bge --embedder sentence-transformers --rerank --show-contexts \
  --generator hf --generator-model Qwen/Qwen2.5-1.5B-Instruct \
  "How quickly must APRA be told about a disruption to a critical operation?"

# evaluation → runs/hybrid/report.{json,md}
ragpipe eval --index index/bge --evalset evalsets/policy_qa.jsonl --out runs/hybrid \
  --embedder sentence-transformers --retriever hybrid --rerank \
  --generator hf --generator-model Qwen/Qwen2.5-1.5B-Instruct --judge same

ragpipe gate --report runs/hybrid/report.json --gates gates/production.yaml      # exit 1 on failure
ragpipe compare --candidate runs/hybrid/report.json --baseline runs/bm25/report.json
ragpipe ragas-crosscheck --report runs/hybrid/report.json --embedder sentence-transformers --judge hf --judge-model Qwen/Qwen2.5-1.5B-Instruct
ragpipe retrieval-bench --corpus corpus --evalset evalsets/policy_qa.jsonl --embedder sentence-transformers --rerank

make all           # ruff + mypy --strict + pytest
make results       # regenerates docs/experiments (scripts/run_experiments.sh)
```

Any OpenAI-compatible server works as generator or judge:
`--generator openai --generator-base-url http://vllm:8000/v1 --generator-model <served name>`
(or `RAGPIPE_GENERATOR__KIND=openai …` as environment variables). CI runs the tests and then
a deterministic gate (hashing embedder + scripted LLM) on every push.

---

## 4. Design decisions

* **Offline by default.** The hashing embedder and the scripted LLM make every test and the
  CI gate run without a download; real models are extras behind the same protocols.
* **Traceability.** Chunks keep character offsets and content-addressed ids; the index
  manifest records the embedder and a corpus fingerprint; every judge call is logged.
* **Metrics you can explain.** Each RAGAS metric is ~30 lines implementing the paper's
  definition; a judge that cannot produce valid JSON yields a *missing* value, never a 0.
  Prompt-engineering lessons from running a small judge are recorded in the module docstring
  and checked by tests (id-aligned verdicts, placeholder filtering, corroborated
  non-committal flags).
* **Statistics before thresholds.** Bootstrap intervals on every metric; paired bootstrap
  with an explicit non-inferiority margin for run-vs-run comparisons; gates can be applied to
  the conservative bound of the interval.
* **Cross-checked, not trusted.** The optional adapter runs the official `ragas` metrics with
  the same local judge and reports Pearson/Spearman/mean absolute difference per metric.

Interview preparation notes: [docs/INTERVIEW_NOTES.md](docs/INTERVIEW_NOTES.md).

---

## 5. Layout

```
src/ragpipe/
├── documents.py, textproc.py, chunking.py, bm25.py, embeddings.py, vectorstore.py, retrieval.py
├── llm.py, prompts.py, pipeline.py, factory.py, config.py, cli.py, logging_utils.py
└── evaluation/   dataset.py, retrieval_metrics.py, judge.py, ragas_metrics.py, runner.py, gates.py,
                  retrieval_bench.py, ragas_adapter.py
corpus/           20 policy documents (fictional Meridian Bank, Australian regulatory context)
evalsets/         policy_qa.jsonl (60 questions, gold documents, tags)
gates/            retrieval_smoke.yaml (CI), production.yaml (release)
tests/            143 tests
docs/             RESULTS.md, INTERVIEW_NOTES.md, experiments/
```
