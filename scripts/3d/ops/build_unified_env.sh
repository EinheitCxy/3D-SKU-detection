#!/usr/bin/env bash
set -euo pipefail

[ "$#" -eq 1 ] || { echo "usage: $0 OUTPUT_DIR" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPENCV_WHEEL_DIR="$PROJECT_ROOT/docker/wheels"
OPENCV_HEADLESS_WHEEL="opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
OUTPUT_DIR="$(realpath -m "$1")"
ROOT_VENV="$(realpath -m "$PROJECT_ROOT/.venv")"
DA3_VENV="$(realpath -m "$PROJECT_ROOT/Depth-Anything-3/.venv")"
case "$OUTPUT_DIR" in
  "$ROOT_VENV" | "$ROOT_VENV"/* | "$DA3_VENV" | "$DA3_VENV"/*)
    echo "candidate environment is inside protected environment: $OUTPUT_DIR" >&2
    exit 1
    ;;
esac
[ ! -e "$OUTPUT_DIR" ] || {
  echo "candidate environment already exists: $OUTPUT_DIR" >&2
  exit 1
}
test -f "$OPENCV_WHEEL_DIR/$OPENCV_HEADLESS_WHEEL"

PYTHONPATH="$PROJECT_ROOT/Depth-Anything-3/src:$PROJECT_ROOT/sam3:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"
uv venv "$OUTPUT_DIR" --python 3.11 --relocatable
VIRTUAL_ENV="$OUTPUT_DIR" uv sync --active --frozen --offline --extra dev
uv pip check --python "$OUTPUT_DIR/bin/python"
PYTHONPATH="$PYTHONPATH" "$OUTPUT_DIR/bin/python" -c \
  'from sam3.model_builder import build_sam3_image_model; import torch, xformers, depth_anything_3, omegaconf, e3nn, evo'
