#!/usr/bin/env bash
# Regenerates everything docs/RESULTS.md cites.
#
# Two stages. The scripted stage is deterministic, runs on CPU in seconds and is the same
# thing CI gates on. The Hugging Face stage needs a CUDA device and takes an hour or two; it
# is what turns "the harness works" into "here is what a real model actually does".
#
# Usage:  bash scripts/run_experiments.sh [--scripted-only] [--limit N]
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=docs/experiments
mkdir -p "$OUT"

SCRIPTED_ONLY=0
LIMIT_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --scripted-only) SCRIPTED_ONLY=1 ;;
    --limit) LIMIT_ARG="--limit $2"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

echo "== 1. World summary and task set"
mcpeval world summary | tee "$OUT/world_summary.txt"
mcpeval tasks list | tee "$OUT/task_list.txt"

echo
echo "== 2. Scripted model: the deterministic harness check (this is the CI gate)"
mcpeval bench run --arch single     --model scripted --out "$OUT/scripted-single"
mcpeval bench run --arch supervisor --model scripted --out "$OUT/scripted-supervisor"
mcpeval bench compare "$OUT/scripted-single" "$OUT/scripted-supervisor" \
  --margin 0.05 | tee "$OUT/scripted_compare.md"

if [ "$SCRIPTED_ONLY" = "1" ]; then
  echo
  echo "Scripted stage complete. Skipping the Hugging Face stage as requested."
  exit 0
fi

echo
echo "== 3. Real model: Qwen2.5-1.5B-Instruct, both architectures"
# 1.5B rather than 0.5B because 0.5B does not reliably emit the action protocol at all, and
# rather than 4B because 4B runs at roughly a third of the speed for 60 long-horizon tasks.
# The 4B slice below measures what that extra capacity buys.
#
# No HF_HUB_OFFLINE and no response cache, on purpose. Forcing offline mode makes the first
# run on a fresh machine fail before it starts; and a cached replay reproduces every
# trajectory exactly -- greedy decoding, same batch composition -- but its `wall_ms` column
# then measures SQLite lookups, and a committed aggregate whose timing column means something
# other than what it says is worse than one with no timing column at all. `--resume` is
# kept: it skips tasks whose trajectory is already on disk, which costs nothing in fidelity.
export TOKENIZERS_PARALLELISM=false
MODEL=Qwen/Qwen2.5-1.5B-Instruct

# One attempt at a time. The Hugging Face backend holds a single model in this process and
# `complete` is a blocking call, so concurrency buys nothing and only interleaves the
# trajectories.
# shellcheck disable=SC2086
mcpeval bench run --arch single --model hf --model-name "$MODEL" \
  --concurrency 1 --resume $LIMIT_ARG --out "$OUT/qwen15-single"
# shellcheck disable=SC2086
mcpeval bench run --arch supervisor --model hf --model-name "$MODEL" \
  --concurrency 1 --resume $LIMIT_ARG --out "$OUT/qwen15-supervisor"
mcpeval bench compare "$OUT/qwen15-single" "$OUT/qwen15-supervisor" \
  --margin 0.05 | tee "$OUT/qwen15_compare.md"

echo
echo "== 4. Model-size ablation: does a larger model close the gap the architecture opens?"
# A hub id, not a local path: this script has to run on a machine that is not mine.
# Point $MCPEVAL_BIG_MODEL at a local directory to use one that is already downloaded.
# Not run for the committed results -- see section 4 of docs/RESULTS.md for why.
BIG=${MCPEVAL_BIG_MODEL:-Qwen/Qwen3-4B-Instruct-2507}
mcpeval bench run --arch single --model hf --model-name "$BIG" \
  --concurrency 1 --resume --limit 24 --out "$OUT/qwen4b-single"
mcpeval bench run --arch supervisor --model hf --model-name "$BIG" \
  --concurrency 1 --resume --limit 24 --out "$OUT/qwen4b-supervisor"
mcpeval bench compare "$OUT/qwen4b-single" "$OUT/qwen4b-supervisor" \
  --margin 0.05 | tee "$OUT/qwen4b_compare.md"

echo
echo "== 5. Reports and the rendered sample trajectories"
for d in "$OUT"/*/; do
  [ -f "$d/aggregate.json" ] || continue
  mcpeval bench report "$d" > "$d/report.md"
done
python scripts/render_trajectories.py --out "$OUT/sample-trajectories.md"
