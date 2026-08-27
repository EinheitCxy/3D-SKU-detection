#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_IMAGE="harbor-cn.lingmouai.com/alg/sku-classifier-base:0.0.4"
HOST_UV="${HOST_UV:-/home/xingyu/.local/bin/uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/xingyu/.cache/uv}"
DA3_MODEL_CACHE="${DA3_MODEL_CACHE:-/home/xingyu/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1}"
DA3_SNAPSHOT="b2359bdf726fb44ef62acca04d629dcf158053e7"
SAM3_CHECKPOINT="$REPO_ROOT/sam3/checkpoints/sam3.pt"
IMAGE_TAG="${IMAGE_TAG:-global-id-mapping:da3-self-contained}"
BUILD_WORK_ROOT="${BUILD_WORK_ROOT:-/data/www/comfyui/3d-recognition-build}"
OPENCV_WHEEL_DIR="${OPENCV_WHEEL_DIR:-$BUILD_WORK_ROOT/runtime/wheels}"
OPENCV_HEADLESS_WHEEL="opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

[ -d "$OPENCV_WHEEL_DIR" ] || {
  echo "OPENCV_WHEEL_DIR must be an existing directory: $OPENCV_WHEEL_DIR" >&2
  exit 1
}
OPENCV_WHEEL_DIR="$(realpath -e "$OPENCV_WHEEL_DIR")"

docker image inspect "$BASE_IMAGE" >/dev/null
test "$(docker image inspect --format '{{.Architecture}}' "$BASE_IMAGE")" = "amd64"
test -x "$HOST_UV"
test -d "$UV_CACHE_DIR"
test -w "$UV_CACHE_DIR"
test -d "$DA3_MODEL_CACHE/blobs"
test -d "$DA3_MODEL_CACHE/refs"
test -d "$DA3_MODEL_CACHE/snapshots/$DA3_SNAPSHOT"
test -f "$DA3_MODEL_CACHE/snapshots/$DA3_SNAPSHOT/config.json"
test -f "$DA3_MODEL_CACHE/snapshots/$DA3_SNAPSHOT/model.safetensors"
test -f "$DA3_MODEL_CACHE/refs/main"
grep -Fx "$DA3_SNAPSHOT" "$DA3_MODEL_CACHE/refs/main" >/dev/null
test -f "$SAM3_CHECKPOINT"
test -f "$REPO_ROOT/pyproject.toml"
test -f "$REPO_ROOT/uv.lock"
test -d "$BUILD_WORK_ROOT"
test -w "$BUILD_WORK_ROOT"
test -f "$OPENCV_WHEEL_DIR/$OPENCV_HEADLESS_WHEEL"

BUILD_ROOT="$(mktemp -d "$BUILD_WORK_ROOT/global-id-mapping.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT
APP_CONTEXT="$BUILD_ROOT/app"
VENV_CONTEXT="$BUILD_ROOT/venv"
mkdir -p "$APP_CONTEXT/Depth-Anything-3" "$APP_CONTEXT/sam3" "$APP_CONTEXT/docker" "$VENV_CONTEXT"
cp -a "$REPO_ROOT/main.py" "$REPO_ROOT/config.yaml" "$APP_CONTEXT/"
cp -a "$REPO_ROOT/src" "$REPO_ROOT/utils" "$APP_CONTEXT/"
cp -a "$REPO_ROOT/Depth-Anything-3/src" "$APP_CONTEXT/Depth-Anything-3/"
cp -a "$REPO_ROOT/sam3/sam3" "$APP_CONTEXT/sam3/"
cp -a "$REPO_ROOT/docker/__init__.py" "$REPO_ROOT/docker/api.py" \
  "$REPO_ROOT/docker/processor.py" "$REPO_ROOT/docker/request_runner.py" "$APP_CONTEXT/docker/"
find "$APP_CONTEXT" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$APP_CONTEXT" -type f -name '*.py[co]' -delete
rm -rf "$APP_CONTEXT/Depth-Anything-3/src/depth_anything_3/app"
rm -rf "$APP_CONTEXT/Depth-Anything-3/src/depth_anything_3/bench"
rm -rf "$APP_CONTEXT/Depth-Anything-3/src/depth_anything_3/services"
rm -f "$APP_CONTEXT/Depth-Anything-3/src/depth_anything_3/cli.py"
rm -rf "$APP_CONTEXT/sam3/sam3/perflib/tests"

docker run --rm --pull=never --network none --user "$(id -u):$(id -g)" --entrypoint /bin/bash \
  -v "$REPO_ROOT:/workspace:ro" \
  -v "$HOST_UV:/usr/local/bin/uv:ro" \
  -v "$UV_CACHE_DIR:/uv-cache" \
  -v "$OPENCV_WHEEL_DIR:/workspace/docker/wheels:ro" \
  -v "$VENV_CONTEXT:/output" \
  "$BASE_IMAGE" -c '
    set -euo pipefail
    export UV_CACHE_DIR=/uv-cache
    export UV_LINK_MODE=copy
    export VIRTUAL_ENV=/output/.venv
    uv venv --relocatable --python /opt/conda/bin/python "$VIRTUAL_ENV"
    uv sync --active --project /workspace --frozen --offline --no-dev --no-install-project --no-editable
    uv pip check --python "$VIRTUAL_ENV/bin/python"
    "$VIRTUAL_ENV/bin/python" -c "import bson, cv2, fastapi, torch, torchvision, xformers"
  '

test -x "$VENV_CONTEXT/.venv/bin/python"

DOCKER_BUILDKIT=1 docker build --network=none --pull=false \
  -f "$SCRIPT_DIR/Dockerfile" \
  --build-context app="$APP_CONTEXT" \
  --build-context venv="$VENV_CONTEXT/.venv" \
  --build-context da3_model="$DA3_MODEL_CACHE" \
  --build-context sam3_checkpoint="$REPO_ROOT/sam3/checkpoints" \
  -t "$IMAGE_TAG" \
  "$SCRIPT_DIR"
