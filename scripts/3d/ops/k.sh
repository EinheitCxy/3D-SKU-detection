#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/Output"
export CUDA_VISIBLE_DEVICES="${GPU:-1}"

for spec in 2:4 3:4 4:10 5:7 6:11 7:14 8:7 9:7 10:6 11:12 12:16; do
  IFS=: read -r floor max_idx <<< "$spec"
  bash "$SCRIPT_DIR/batch.sh" "floor_display$floor" "$max_idx"
done

bash "$PROJECT_ROOT/scripts/3d/evaluation/batch_accuracy_evaluation.sh" \
  --backend da3 --start 2 --end 12 --save-root "$OUTPUT_ROOT"
