#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_IMAGE="${BASE_IMAGE:-harbor-cn.lingmouai.com/asu/global-id-mapping:4.0}"
IMAGE_TAG="${IMAGE_TAG:-global-id-mapping:4.0-traceback}"

docker image inspect "$BASE_IMAGE" >/dev/null

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/global-id-mapping-code-update.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT
APP_CONTEXT="$BUILD_ROOT/app"
mkdir -p "$APP_CONTEXT"
cp "$SCRIPT_DIR/processor.py" "$APP_CONTEXT/processor.py"
cp "$SCRIPT_DIR/cos_upload.py" "$APP_CONTEXT/cos_upload.py"

DOCKER_BUILDKIT=1 docker build --network=none --pull=false \
  -f "$SCRIPT_DIR/Dockerfile.code-update" \
  --build-context app="$APP_CONTEXT" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$IMAGE_TAG" \
  "$SCRIPT_DIR"
