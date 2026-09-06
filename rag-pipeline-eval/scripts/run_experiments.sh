#!/usr/bin/env bash
# Regenerates every number quoted in docs/RESULTS.md.
#
# Stages 1–2 need only the [dense] extra (bge-small + MiniLM cross-encoder, ~220 MB, CPU is
# fine). Stages 3–6 run a local Hugging Face model as generator *and* judge and want a GPU
# (Qwen2.5-1.5B-Instruct takes ~3.5 GB in bf16). Override MODEL / K / PY as needed:
#   MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
K=${K:-5}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
OUT=docs/experiments
EVALSET=evalsets/policy_qa.jsonl
mkdir -p "$OUT" runs

echo "== 1. retrieval bench: hashing embedder (no downloads)"
$PY -m ragpipe --plain-logs retrieval-bench --corpus corpus --evalset "$EVALSET" \
  --embedder hash --top-k "$K" --out "$OUT/retrieval_bench_hash.md"

echo "== 2. retrieval bench: bge-small + cross-encoder rerank"
$PY -m ragpipe --plain-logs retrieval-bench --corpus corpus --evalset "$EVALSET" \
  --embedder sentence-transformers --rerank --top-k "$K" --out "$OUT/retrieval_bench_bge.md"

echo "== 3. ingest with bge-small"
$PY -m ragpipe --plain-logs ingest --corpus corpus --index index/bge --embedder sentence-transformers

COMMON=(--index index/bge --evalset "$EVALSET" --embedder sentence-transformers --top-k "$K"
        --generator hf --generator-model "$MODEL" --generator-max-tokens 200
        --judge same --judge-max-tokens 400)

echo "== 4. end-to-end: hybrid (RRF) + rerank, $MODEL as generator and judge"
$PY -m ragpipe eval "${COMMON[@]}" --retriever hybrid --rerank --out runs/hybrid_rerank \
  2> runs/hybrid_rerank.log

echo "== 5. end-to-end: BM25 only (baseline for the paired comparison)"
$PY -m ragpipe eval "${COMMON[@]}" --retriever bm25 --out runs/bm25_only \
  2> runs/bm25_only.log

echo "== 6. gates and paired comparison"
$PY -m ragpipe gate --report runs/hybrid_rerank/report.json --gates gates/production.yaml \
  | tee "$OUT/gate_hybrid_rerank.md" || true
$PY -m ragpipe compare --candidate runs/hybrid_rerank/report.json --baseline runs/bm25_only/report.json \
  --metrics faithfulness,answer_relevancy,context_precision,context_recall,retrieval/mrr,retrieval/recall@$K,lexical/reference_coverage,false_abstention,correct_abstention \
  | tee "$OUT/compare_hybrid_vs_bm25.md" || true

echo "== 7. official ragas cross-check (optional: needs the [ragas] extra)"
$PY -m ragpipe ragas-crosscheck --report runs/hybrid_rerank/report.json \
  --embedder sentence-transformers --judge hf --judge-model "$MODEL" --judge-max-tokens 400 \
  --out "$OUT/ragas_agreement.md" 2> runs/ragas_crosscheck.log \
  || echo "ragas cross-check failed — see runs/ragas_crosscheck.log"

cp runs/hybrid_rerank/report.md "$OUT/report_hybrid_rerank.md"
cp runs/bm25_only/report.md "$OUT/report_bm25_only.md"
echo "done — artefacts in $OUT"
