#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_REPO_ROOT="${CORE_REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CORE_REPO_ROOT="$(realpath -e "$CORE_REPO_ROOT")"
COS_SITE_PACKAGES="${COS_SITE_PACKAGES:-$CORE_REPO_ROOT/.venv/lib/python3.11/site-packages}"
COS_SITE_PACKAGES="$(realpath -e "$COS_SITE_PACKAGES")"
BASE_IMAGE="${BASE_IMAGE:-global-id-mapping:4.0-traceback}"
IMAGE_TAG="${IMAGE_TAG:-global-id-mapping:4.0-traceback}"

docker image inspect "$BASE_IMAGE" >/dev/null
test -f "$COS_SITE_PACKAGES/qcloud_cos/__init__.py"
test -d "$COS_SITE_PACKAGES/crcmod"
test -d "$COS_SITE_PACKAGES/Crypto"
test -f "$COS_SITE_PACKAGES/xmltodict.py"

BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/global-id-mapping-code-update.XXXXXX")"
trap 'rm -rf "$BUILD_ROOT"' EXIT
APP_CONTEXT="$BUILD_ROOT/app"
mkdir -p "$APP_CONTEXT/cos-sdk"
cp "$SCRIPT_DIR/processor.py" "$APP_CONTEXT/processor.py"
cp "$SCRIPT_DIR/cos_upload.py" "$APP_CONTEXT/cos_upload.py"
cp -a "$COS_SITE_PACKAGES/qcloud_cos" "$COS_SITE_PACKAGES/crcmod" \
  "$COS_SITE_PACKAGES/Crypto" "$COS_SITE_PACKAGES/xmltodict.py" "$APP_CONTEXT/cos-sdk/"

DOCKER_BUILDKIT=1 docker build --network=none --pull=false \
  -f "$SCRIPT_DIR/Dockerfile.code-update" \
  --build-context app="$APP_CONTEXT" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$IMAGE_TAG" \
  "$SCRIPT_DIR"
