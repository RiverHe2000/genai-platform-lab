#!/usr/bin/env bash
# Regenerates docs/experiments/*: gateway-overhead load tests (fake backends, CPU), then the
# release workflow against two real local models served through the gateway (needs the [hf]
# extra and a GPU): evaluate baseline and candidate, decide promotion, load-test the candidate.
#   PORT=8091 bash scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
PORT=${PORT:-8091}
OUT=docs/experiments
SUITE=evalsets/finance_qa.jsonl
mkdir -p "$OUT" runs

echo "== 1. gateway overhead: fake backends, in-process (no model, no network)"
$PY -m llmgate --plain-logs loadtest --config deploy/gateway.fake.yaml --model fake-baseline \
  --concurrency 1 8 32 --requests 300 --out "$OUT/loadtest_fake.md" 2> runs/loadtest_fake.log
$PY -m llmgate --plain-logs loadtest --config deploy/gateway.fake.yaml --model fake-baseline \
  --concurrency 8 --requests 200 --stream --out "$OUT/loadtest_fake_stream.md" 2>> runs/loadtest_fake.log

echo "== 2. serve two local HF models (deploy/gateway.local.yaml) on port $PORT"
$PY -m llmgate serve --config deploy/gateway.local.yaml --port "$PORT" > runs/serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  sleep 1
done
BASE="http://127.0.0.1:$PORT/v1"

echo "== 3. warm-up (loads each model once so the first request does not skew latency)"
for m in qwen05 qwen15; do
  curl -fsS -X POST "$BASE/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$m\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello.\"}], \"max_tokens\": 4}" > /dev/null
done

echo "== 4. evaluate baseline (qwen05) and candidate (qwen15) through the gateway"
$PY -m llmgate --plain-logs eval --base-url "$BASE" --model qwen05 --suite "$SUITE" \
  --out runs/eval_qwen05 --concurrency 2 2> runs/eval_qwen05.log
$PY -m llmgate --plain-logs eval --base-url "$BASE" --model qwen15 --suite "$SUITE" \
  --out runs/eval_qwen15 --concurrency 2 2> runs/eval_qwen15.log
cp runs/eval_qwen05/report.md "$OUT/eval_qwen05.md"
cp runs/eval_qwen15/report.md "$OUT/eval_qwen15.md"

echo "== 5. promotion decision"
$PY -m llmgate --plain-logs promote --candidate runs/eval_qwen15/report.json \
  --baseline runs/eval_qwen05/report.json --policy deploy/promotion_policy.yaml \
  --out "$OUT/promotion" > "$OUT/promotion_stdout.txt" 2> runs/promote.log || echo "decision: HOLD (exit 1)"

echo "== 6. load test the candidate through the gateway (real model, streaming and not)"
$PY -m llmgate --plain-logs loadtest --base-url "$BASE" --model qwen15 --concurrency 1 4 \
  --requests 24 --max-tokens 48 --out "$OUT/loadtest_qwen15.md" 2> runs/loadtest_qwen15.log
$PY -m llmgate --plain-logs loadtest --base-url "$BASE" --model qwen15 --concurrency 4 \
  --requests 16 --max-tokens 48 --stream --out "$OUT/loadtest_qwen15_stream.md" 2>> runs/loadtest_qwen15.log

echo "== 7. canary routing sample + metrics snapshot"
for i in $(seq 1 20); do
  curl -sS -X POST "$BASE/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"gateway-default\", \"user\": \"user-$i\", \"messages\": [{\"role\": \"user\", \"content\": \"Reply with one word: ok\"}], \"max_tokens\": 3}" \
    -D - -o /dev/null | grep -i '^X-Backend' || true
done | sort | uniq -c > "$OUT/canary_split.txt"
curl -sS "http://127.0.0.1:$PORT/metrics" | grep -E '^llmgate_(requests_total|fallbacks_total|breaker_open)' > "$OUT/metrics_snapshot.txt" || true
curl -sS "http://127.0.0.1:$PORT/admin/backends" > "$OUT/admin_backends.json" || true
echo "done — artefacts in $OUT"
