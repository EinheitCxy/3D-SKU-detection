#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_REPO_ROOT="${CORE_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CORE_REPO_ROOT="$(realpath -e "$CORE_REPO_ROOT")"
BASE_IMAGE="harbor-cn.lingmouai.com/alg/sku-classifier-base:0.0.4"
HOST_UV="${HOST_UV:-/home/xingyu/.local/bin/uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/home/xingyu/.cache/uv}"
DA3_MODEL_CACHE="${DA3_MODEL_CACHE:-/home/xingyu/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1}"
DA3_SNAPSHOT="b2359bdf726fb44ef62acca04d629dcf158053e7"
SAM3_CHECKPOINT="$CORE_REPO_ROOT/sam3/checkpoints/sam3.pt"
IMAGE_TAG="${IMAGE_TAG:-global-id-mapping:da3-self-contained}"
BUILD_WORK_ROOT="${BUILD_WORK_ROOT:-/data/www/comfyui/3d-recognition-build}"
OPENCV_WHEEL_DIR="${OPENCV_WHEEL_DIR:-$BUILD_WORK_ROOT/runtime/wheels}"
OPENCV_HEADLESS_WHEEL="opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
SYSTEM_DEB_DIR="${SYSTEM_DEB_DIR:-$BUILD_WORK_ROOT/runtime/system-debs/ubuntu-22.04-amd64}"
PREPARE_SYSTEM_DEPS_ONLY="${PREPARE_SYSTEM_DEPS_ONLY:-0}"

# 可选的一次在线准备：下载 Ubuntu 运行时 .deb，供后续离线镜像组装。
if [ "$PREPARE_SYSTEM_DEPS_ONLY" = "1" ]; then
  docker image inspect "$BASE_IMAGE" >/dev/null
  test "$(docker image inspect --format '{{.Architecture}}' "$BASE_IMAGE")" = "amd64"
  docker run --rm --pull=never --network=none --entrypoint /bin/bash \
    "$BASE_IMAGE" -ceu '
      source /etc/os-release
      test "$ID" = "ubuntu"
      test "$VERSION_ID" = "22.04"
    '
  mkdir -p "$SYSTEM_DEB_DIR"
  SYSTEM_DEB_DIR="$(realpath -e "$SYSTEM_DEB_DIR")"
  docker run --rm --pull=never --entrypoint /bin/bash \
    -e OUTPUT_UID="$(id -u)" \
    -e OUTPUT_GID="$(id -g)" \
    -v "$SYSTEM_DEB_DIR:/output" \
    "$BASE_IMAGE" -ceu '
      rm -f /output/*.deb /output/lock
      rm -rf /output/partial
      mkdir -p /output/partial
      apt-get update
      apt-get install --download-only --yes --no-install-recommends \
        -o Dir::Cache::archives=/output libx11-6 libgl1
      rm -f /output/lock
      rm -rf /output/partial
      chown "$OUTPUT_UID:$OUTPUT_GID" /output /output/*.deb
    '
  exit 0
fi

[ -d "$OPENCV_WHEEL_DIR" ] || {
  echo "OPENCV_WHEEL_DIR must be an existing directory: $OPENCV_WHEEL_DIR" >&2
  exit 1
}
OPENCV_WHEEL_DIR="$(realpath -e "$OPENCV_WHEEL_DIR")"
[ -d "$SYSTEM_DEB_DIR" ] || {
  echo "SYSTEM_DEB_DIR must be an existing directory: $SYSTEM_DEB_DIR" >&2
  exit 1
}
SYSTEM_DEB_DIR="$(realpath -e "$SYSTEM_DEB_DIR")"

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
test -f "$CORE_REPO_ROOT/pyproject.toml"
test -f "$CORE_REPO_ROOT/uv.lock"
test -d "$BUILD_WORK_ROOT"
test -w "$BUILD_WORK_ROOT"
test -f "$OPENCV_WHEEL_DIR/$OPENCV_HEADLESS_WHEEL"
compgen -G "$SYSTEM_DEB_DIR/libx11-6_*.deb" >/dev/null
compgen -G "$SYSTEM_DEB_DIR/libgl1_*.deb" >/dev/null

BUILD_ROOT="$(mktemp -d "$BUILD_WORK_ROOT/global-id-mapping.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT
APP_CONTEXT="$BUILD_ROOT/app"
VENV_CONTEXT="$BUILD_ROOT/venv"
# 最小应用 context：只保留 Mapping API 所需源码。
mkdir -p "$APP_CONTEXT/Depth-Anything-3" "$APP_CONTEXT/sam3" "$APP_CONTEXT/docker" "$VENV_CONTEXT"
cp -a "$CORE_REPO_ROOT/main.py" "$CORE_REPO_ROOT/config.yaml" "$APP_CONTEXT/"
cp -a "$CORE_REPO_ROOT/src" "$CORE_REPO_ROOT/utils" "$APP_CONTEXT/"
cp -a "$CORE_REPO_ROOT/Depth-Anything-3/src" "$APP_CONTEXT/Depth-Anything-3/"
cp -a "$CORE_REPO_ROOT/sam3/sam3" "$APP_CONTEXT/sam3/"
cp -a "$SCRIPT_DIR/__init__.py" "$SCRIPT_DIR/api.py" \
  "$SCRIPT_DIR/processor.py" "$SCRIPT_DIR/request_runner.py" "$APP_CONTEXT/docker/"
find "$APP_CONTEXT" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$APP_CONTEXT" -type f -name '*.py[co]' -delete
rm -rf "$APP_CONTEXT/Depth-Anything-3/src/depth_anything_3/app"
rm -rf "$APP_CONTEXT/Depth-Anything-3/src/depth_anything_3/bench"
rm -rf "$APP_CONTEXT/Depth-Anything-3/src/depth_anything_3/services"
rm -f "$APP_CONTEXT/Depth-Anything-3/src/depth_anything_3/cli.py"
rm -rf "$APP_CONTEXT/sam3/sam3/perflib/tests"

# 离线 Python 环境：从冻结 lock 和本地 uv cache 创建唯一 venv。
docker run --rm --pull=never --network none --user "$(id -u):$(id -g)" --entrypoint /bin/bash \
  -v "$CORE_REPO_ROOT:/workspace:ro" \
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

test -f "$VENV_CONTEXT/.venv/pyvenv.cfg"
test -L "$VENV_CONTEXT/.venv/bin/python"

# 最终离线镜像组装：所有运行时输入均通过 named context 提供。
DOCKER_BUILDKIT=1 docker build --network=none --pull=false \
  -f "$SCRIPT_DIR/Dockerfile" \
  --build-context app="$APP_CONTEXT" \
  --build-context venv="$VENV_CONTEXT/.venv" \
  --build-context da3_model="$DA3_MODEL_CACHE" \
  --build-context sam3_checkpoint="$CORE_REPO_ROOT/sam3/checkpoints" \
  --build-context system_debs="$SYSTEM_DEB_DIR" \
  -t "$IMAGE_TAG" \
  "$SCRIPT_DIR"
