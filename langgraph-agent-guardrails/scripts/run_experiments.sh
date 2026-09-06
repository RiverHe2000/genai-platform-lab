#!/usr/bin/env bash
# Regenerates docs/experiments/*: the deterministic scripted run (what CI gates on) and the
# same scenarios driven by a real local model (needs the [hf] extra and ideally a GPU).
#   MODEL=Qwen/Qwen2.5-3B-Instruct bash scripts/run_experiments.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
OUT=docs/experiments
mkdir -p "$OUT" runs

echo "== 1. scripted model (deterministic; this is the CI gate)"
$PY -m agentguard --plain-logs eval --scenarios scenarios/core.jsonl scenarios/redteam.jsonl \
  --model fake --out runs/scripted --gate 2> runs/scripted.log
cp runs/scripted/report.md "$OUT/scripted_report.md"
cp runs/scripted/report.json "$OUT/scripted_report.json"

echo "== 2. real model: $MODEL on the benign scenarios"
$PY -m agentguard --plain-logs eval --scenarios scenarios/core.jsonl \
  --model hf --model-name "$MODEL" --out runs/hf_core 2> runs/hf_core.log || true
cp runs/hf_core/report.md "$OUT/hf_core_report.md"
cp runs/hf_core/report.json "$OUT/hf_core_report.json"

echo "== 3. real model: $MODEL on the red-team scenarios"
$PY -m agentguard --plain-logs eval --scenarios scenarios/redteam.jsonl \
  --model hf --model-name "$MODEL" --out runs/hf_redteam 2> runs/hf_redteam.log || true
cp runs/hf_redteam/report.md "$OUT/hf_redteam_report.md"
cp runs/hf_redteam/report.json "$OUT/hf_redteam_report.json"

echo "== 4. sample audit replay (approval flow with a durable checkpoint)"
rm -rf state && mkdir -p state
$PY -m agentguard --plain-logs chat "Flag loan L00042 for watchlist review because it is 45 days past due." \
  --thread demo --model hf --model-name "$MODEL" \
  --checkpoint-path state/checkpoints.sqlite --loanbook-path state/loanbook.sqlite \
  --audit-path state/audit.jsonl > "$OUT/demo_turn1.json" 2> runs/demo.log || true
$PY -m agentguard --plain-logs approve --thread demo --approved true --approver "staff:r.reviewer" \
  --note "45 dpd confirmed in the loan book" --model hf --model-name "$MODEL" \
  --checkpoint-path state/checkpoints.sqlite --loanbook-path state/loanbook.sqlite \
  --audit-path state/audit.jsonl > "$OUT/demo_turn2.json" 2>> runs/demo.log || true
$PY -m agentguard replay --thread demo --audit-path state/audit.jsonl > "$OUT/demo_audit_replay.txt" || true
echo "done — artefacts in $OUT"
