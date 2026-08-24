#!/bin/bash
# video -> 去重后的 SKU 数目 + 去重后带检测框的图片
#
# 链路:
#   [1] 抽帧   cv2 按 fps 采样 -> images/0.JPG,1.JPG,... (直接 0-based, 对齐 core 输入约定)
#   [2] SKU检测 -> detections_results/0.json,1.json,...（modules/sku_detector）
#   [3] pipeline main.py --mode pipeline (da3 + 3d) -> 匹配+去重+global_id
#   [4] 提取   去重SKU数目 = global_mapping.json 的 key 数; 带框图 = dedup_imgs_w_bboxes/
#
# 用法:
#   bash video_to_dedup.sh <video> <fps> <gpu> [detections_dir]
set -euo pipefail

# ===== 参数 =====
FPS="${2:-2.0}"
GPU="${3:-0}"
DETECTIONS_SRC="${4:-}"   # 已有 detections 目录(每帧 <i>.json); 空=报错

# ===== 路径 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CORE_DIR="$REPO_DIR"
VIDEO_ARG="${1:-$REPO_DIR/small_fd_video/video-test/6-1.mp4}"
VIDEO="$(realpath "$VIDEO_ARG" 2>/dev/null || echo "$VIDEO_ARG")"
DATASET_NAME="$(basename "$VIDEO")"; DATASET_NAME="${DATASET_NAME%.*}"
WORK_ROOT="$REPO_DIR/runtime/video_to_dedup"
DATASET_DIR="$WORK_ROOT/$DATASET_NAME"
SAVE_ROOT="$REPO_DIR/Output"
DETECTOR_ROOT="${DETECTOR_ROOT:-$REPO_DIR/modules/sku_detector}"
DETECTOR_ENV="${DETECTOR_ENV:-$REPO_DIR/runtime/sku_detector/.venv}"
CORE_ENV="${CORE_ENV:-$REPO_DIR/.venv}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-cpu}"
export CUDA_VISIBLE_DEVICES="$GPU"

[ -f "$VIDEO" ] || { echo "[ERROR] 视频不存在: $VIDEO" >&2; exit 1; }
[ -x "$DETECTOR_ENV/bin/python" ] || { echo "[ERROR] 检测环境不存在: $DETECTOR_ENV" >&2; exit 1; }
[ -x "$CORE_ENV/bin/python" ] || { echo "[ERROR] 核心环境不存在: $CORE_ENV" >&2; exit 1; }
echo "=== video -> 去重结果 ==="
echo "  video      : $VIDEO"
echo "  dataset    : $DATASET_DIR"
echo "  save_root  : $SAVE_ROOT"
echo "  fps/gpu    : $FPS / $GPU"
echo

run_python() {
  local env_dir="$1"; shift
  VIRTUAL_ENV="$env_dir" uv run --active --no-project python "$@"
}

# ---------- [1] 抽帧 (cv2, 直接输出 0-based) ----------
run_frame_sample() {
  echo "[1/4] 抽帧 -> $DATASET_DIR/images/"
  mkdir -p "$DATASET_DIR/images"
  cd "$CORE_DIR"
  run_python "$CORE_ENV" - "$VIDEO" "$DATASET_DIR/images" "$FPS" <<'PYEOF'
import sys, cv2, os
from pathlib import Path
video, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
output_dir = Path(out)
output_dir.mkdir(parents=True, exist_ok=True)
for path in output_dir.iterdir():
    if path.stem.isdigit() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        path.unlink()
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"[ERROR] 无法打开视频: {video}")
vfps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(round(vfps / fps)))
idx = fi = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if fi % step == 0:
        cv2.imwrite(str(output_dir / f"{idx}.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1
    fi += 1
cap.release()
print(f"  抽帧完成: {idx} 张 -> {out} (视频fps={vfps:.2f}, 采样间隔={step}帧)")
PYEOF
}

run_detection() {
  echo "[2/4] SKU 检测 -> $DATASET_DIR/detections_results/"
  mkdir -p "$DATASET_DIR/detections_results"
  find "$DATASET_DIR/detections_results" -maxdepth 1 -type f -name '*.json' -delete
  if [ -n "$DETECTIONS_SRC" ]; then
    DET_ABS="$(realpath "$DETECTIONS_SRC" 2>/dev/null || echo "$DETECTIONS_SRC")"
    cp "$DET_ABS"/*.json "$DATASET_DIR/detections_results/" 2>/dev/null || true
    echo "  使用已有 detections: $DET_ABS"
  else
    run_python "$DETECTOR_ENV" "$DETECTOR_ROOT/bbox_gen.py" "$DATASET_DIR/images" \
      -o "$DATASET_DIR" --device "$DETECTOR_DEVICE"
  fi
  local n expected
  n=$(find "$DATASET_DIR/detections_results" -maxdepth 1 -name '*.json' | wc -l)
  expected=$(find "$DATASET_DIR/images" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
  [ "$n" -eq "$expected" ] || { echo "[ERROR] 检测 JSON 数量 $n 与帧数 $expected 不一致" >&2; exit 1; }
}

# ---------- [3] 运行 pipeline (main.py --mode pipeline, pi3/3d) ----------
run_pipeline() {
  echo "[3/4] 运行 pipeline: main.py --mode pipeline (da3 + 3d)"
  cd "$CORE_DIR"
  nvidia-smi -L >/dev/null 2>&1 || { echo "[ERROR] DA3 匹配需要可用 NVIDIA GPU；检测结果已保留，未生成 global_mapping.json。" >&2; exit 2; }
  run_python "$CORE_ENV" main.py --mode pipeline --dataset "$DATASET_DIR" \
    --algorithm 3d --match_backend da3 --recon_backend da3 \
    --save_root "$SAVE_ROOT"
}

# ---------- [4] 提取结果 ----------
extract_results() {
  echo "[4/4] 提取结果"
  local OUT_DIR="$SAVE_ROOT/$DATASET_NAME"
  local DEDUP_DIR="$OUT_DIR/dedup_detections"
  local VIZ_DIR="$OUT_DIR/dedup_imgs_w_bboxes"
  local GM="$DEDUP_DIR/global_mapping.json"
  [ -f "$GM" ] || { echo "[ERROR] 未找到 $GM, pipeline 去重未成功" >&2; exit 1; }
  cd "$CORE_DIR"
  local n; n=$(run_python "$CORE_ENV" -c "import json; print(len(json.load(open('$GM'))))")
  echo "  ============================================"
  echo "  去重后 SKU 数目 : $n   (global_id 个数)"
  echo "  去重后带框图片  : $VIZ_DIR"
  echo "  去重结果 JSON    : $DEDUP_DIR/{global_mapping.json, global_skus.json, *.json}"
  echo "  ============================================"
}

run_frame_sample
run_detection
run_pipeline
extract_results
