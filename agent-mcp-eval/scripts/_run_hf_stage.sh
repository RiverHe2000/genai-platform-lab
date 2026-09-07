#!/usr/bin/env bash
# Stage 3 of run_experiments.sh on its own: the 1.5B real-model comparison.
# Split out only so it can be launched while the 4B checkpoint is still downloading.
#
# No response cache here, deliberately. A cached replay reproduces every trajectory exactly
# --- greedy decoding, same batch composition --- but its `wall_ms` measures SQLite lookups,
# and a committed aggregate whose timing column means something other than what it says is
# worse than one with no timing column at all.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=docs/experiments
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
MODEL=Qwen/Qwen2.5-1.5B-Instruct
PY=${PYTHON:-python}

"$PY" -m mcpeval bench run --arch single --model hf --model-name "$MODEL" \
  --concurrency 1 --out "$OUT/qwen15-single"
"$PY" -m mcpeval bench run --arch supervisor --model hf --model-name "$MODEL" \
  --concurrency 1 --out "$OUT/qwen15-supervisor"
"$PY" -m mcpeval bench compare "$OUT/qwen15-single" "$OUT/qwen15-supervisor" \
  --margin 0.05 | tee "$OUT/qwen15_compare.md"
for d in "$OUT"/qwen15-*/; do
  "$PY" -m mcpeval bench report "$d" > "$d/report.md"
done
echo "HF stage done"
