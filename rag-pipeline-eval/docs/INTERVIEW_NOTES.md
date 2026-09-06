# Interview notes — RAG pipeline and evaluation

## Retrieval

**Why hybrid retrieval, and why RRF rather than a weighted score sum?** BM25 wins on exact
identifiers and numbers ("CPS 230", "10.75%", "DSCR"), dense embeddings win on paraphrase
("money trouble" → "financial hardship"). The evaluation set has a `paraphrase` slice
precisely to show that gap (see `docs/RESULTS.md`). Reciprocal rank fusion only needs the
*ranks* from each retriever, so it is immune to the fact that BM25 scores are unbounded and
cosine scores live in [-1, 1]; a convex combination needs min–max normalisation per query
and is sensitive to outliers. Both are implemented and benchmarked; RRF is the default.

**What does the cross-encoder add?** Bi-encoders embed query and passage independently, so
they cannot model interactions between the two. A cross-encoder reads the pair jointly and
is far more precise, but it costs one forward pass per candidate — hence "retrieve 20
cheaply, rerank to 5". In the benchmark the reranker mostly improves `hit_rate@1` and MRR,
i.e. *which passage is first*, which is what the generator sees first.

**Why chunk by structure instead of a fixed window?** A fixed window with overlap is the
baseline (implemented, tested). The recursive chunker splits paragraph → sentence → word and
packs greedily, so policy clauses stay intact. That matters twice: a half clause retrieves
worse, and a faithfulness judge cannot mark a statement "supported" by a truncated
sentence. Chunks keep character offsets into the source document so every citation can be
traced to the exact span — an audit requirement in a bank.

**Why not a vector database?** At thousands of chunks exact cosine over a matrix is
faster than any ANN index's overhead, deterministic and trivially verifiable. The
`VectorStore` protocol is the seam: NumPy and SQLite backends are property-tested for
identical results, and a FAISS/pgvector/Qdrant adapter would implement the same three
methods. I would switch when the corpus outgrows RAM or needs multi-tenant filtering, not
before.

**How is the index kept consistent with the corpus?** The manifest stores the embedder
name, the corpus fingerprint (SHA-256 over document ids and texts) and the chunker config.
Loading with a different embedder raises `IndexMismatchError`; chunk ids are content
addressed, so an unchanged document produces the same ids after a rebuild and evaluation
sets that reference chunks stay valid.

## Generation

**How do you stop the model making things up?** Three layers: the prompt restricts the
model to numbered passages and asks for citations; the pipeline resolves `[n]` citations to
chunk ids and records them; the evaluation measures faithfulness (statements supported by
the retrieved context) and the abstention behaviour on unanswerable questions. Six
questions in the set have no answer in the corpus; `correct_abstention` and
`false_abstention` are reported separately, because a model can look faithful simply by
refusing everything.

**Why is the LLM behind a protocol?** The same pipeline runs against a scripted `FakeLLM`
in tests and CI, an OpenAI-compatible HTTP endpoint (vLLM, the companion `llmserve`
project, OpenAI) or an in-process Hugging Face model. Retries only fire on 429/5xx and
transport errors with exponential backoff; a 400 is never retried because it will fail
again.

## Evaluation

**Explain the four RAGAS metrics and what each catches.**
*Faithfulness*: extract atomic statements from the answer, ask the judge which are
supported by the context — catches hallucination. *Answer relevancy*: generate questions
the answer would answer, embed them, compare with the real question — catches evasive or
off-target answers (non-committal answers score 0). *Context precision*: for each retrieved
passage, is it useful for the reference answer? Weighted by rank so junk at rank 1 hurts
more — catches noisy retrieval. *Context recall*: which sentences of the reference answer
are attributable to the context — catches missing retrieval. The first two need no
reference answer; the last two do.

**Why implement them yourself when `ragas` exists?** Three reasons. (1) The definitions are
short and I wanted to be able to explain every number in a validation meeting. (2) The
official package pins a large LangChain/OpenAI dependency tree; at the time of writing
`ragas 0.4.3` fails to import against the current `langchain-community` (a removed Vertex AI
shim), which is exactly the kind of fragility I do not want in a release gate. (3) The
adapter runs the official implementation with the *same* local judge and reports the
agreement (Pearson, Spearman, mean absolute difference), so any disagreement is about the
metric definition, not the model.

**How do you handle a judge that returns garbage?** Strict JSON extraction, Pydantic
validation of the schema, one repair retry with an explicit instruction, then `None`. A
missing value is reported as missing (`n_missing` in every summary) — never coerced to 0 or
1, which would silently bias the mean. `judge_parse_failures` is in every report so a judge
model that cannot follow the format is visible.

**Why bootstrap confidence intervals and paired comparison?** With 60 questions a mean
faithfulness of 0.87 vs 0.84 is noise. The report carries percentile-bootstrap intervals
for every metric, and `ragpipe compare` resamples the *per-question differences* between
two runs on the same questions — a paired design is much more powerful than comparing two
independent intervals. The verdict vocabulary (`better`, `non-inferior`, `worse`,
`inconclusive`) with an explicit non-inferiority margin is the language a model-risk
function uses for a model change.

**What is a gate and where would it run?** `gates/*.yaml` declares thresholds; `ragpipe
gate` exits non-zero if any fails. The CI workflow runs a deterministic gate (hashing
embedder + FakeLLM, no downloads) on every push so a chunking or fusion regression fails the
build; the production gate (`use_ci: true`, i.e. the *lower* bound of the interval must
clear the threshold) runs against the real model before a release. The thresholds mirror
the GenAI governance standard in the corpus: faithfulness ≥ 0.85, relevancy ≥ 0.70.

**Judge quality.** The reported numbers use Qwen2.5-1.5B-Instruct as generator *and* judge
because that is what fits alongside everything else on one 12 GB GPU without network
access. A 1.5 B judge is noisier than a frontier model; the harness is judge-agnostic (swap
`--judge openai --judge-model ...`), and I would use a stronger judge than the generator in
production and calibrate it against a human-labelled slice first.

## Model risk / governance angle

**How would you validate a RAG assistant under CPG 235 / SR 11-7?** Treat retrieval and
generation as two components with their own tests: retrieval metrics against a gold set
(this repo), grounded-generation metrics with a judge whose agreement with humans has been
measured, abstention behaviour on unanswerable questions, a red-team set, monitoring of the
same metrics on production traffic samples, and re-evaluation on every change to model,
corpus or prompt. Everything here — versioned corpus, evaluation set, gates, reports with
intervals — exists so that the validation report can point at reproducible artefacts.
