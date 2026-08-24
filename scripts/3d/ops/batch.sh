#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
FLOOR_DISPLAY="${1:-floor_display2}"
MAX_IDX="${2:-4}"
ALGORITHM="${3:-3d}"
BACKEND="${4:-da3}"
IMAGE_FOLDER="$PROJECT_ROOT/imdata/$FLOOR_DISPLAY/images"
DETECTION_DIR="$PROJECT_ROOT/imdata/$FLOOR_DISPLAY/detections_results"
OUTPUT_DIR="$PROJECT_ROOT/Output/$FLOOR_DISPLAY"

[ -d "$IMAGE_FOLDER" ] || { echo "[ERROR] 图片目录不存在: $IMAGE_FOLDER" >&2; exit 1; }
[ -d "$DETECTION_DIR" ] || { echo "[ERROR] 检测目录不存在: $DETECTION_DIR" >&2; exit 1; }

cd "$PROJECT_ROOT"
if [ "$FLOOR_DISPLAY" = "floor_display3" ]; then
  uv run python utils/process_image_orientation.py --input_dir "$IMAGE_FOLDER"
fi

processed=0
for reference_idx in $(seq 0 "$MAX_IDX"); do
  [ -f "$DETECTION_DIR/$reference_idx.json" ] || continue
  uv run python src/inference.py \
    --algorithm "$ALGORITHM" \
    --backend "$BACKEND" \
    --reference_idx "$reference_idx" \
    --image_folder "$IMAGE_FOLDER" \
    --detection_dir "$DETECTION_DIR" \
    --output_dir "$OUTPUT_DIR"
  processed=$((processed + 1))
done

[ "$processed" -gt 0 ] || { echo "[ERROR] 未找到 0..$MAX_IDX 的检测 JSON" >&2; exit 1; }
echo "完成: $OUTPUT_DIR"
