#!/usr/bin/env bash
set -euo pipefail

[ "$#" -eq 1 ] || { echo "usage: $0 OUTPUT_DIR" >&2; exit 2; }

OUTPUT_DIR="$1"
[ "${OUTPUT_DIR#/}" = "$OUTPUT_DIR" ] && OUTPUT_DIR="$(pwd)/$OUTPUT_DIR"
[ ! -e "$OUTPUT_DIR" ] || {
  echo "candidate environment already exists: $OUTPUT_DIR" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHONPATH="$PROJECT_ROOT/Depth-Anything-3/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
uv venv "$OUTPUT_DIR" --python 3.11
VIRTUAL_ENV="$OUTPUT_DIR" uv sync --active --frozen --extra dev
uv pip check --python "$OUTPUT_DIR/bin/python"
PYTHONPATH="$PYTHONPATH" "$OUTPUT_DIR/bin/python" -c \
  'import torch, xformers, depth_anything_3, omegaconf, e3nn, evo, sam3'
