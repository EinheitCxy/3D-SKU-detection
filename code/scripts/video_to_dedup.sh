#!/bin/bash
# video -> 去重后的 SKU 数目 + 去重后带检测框的图片
#
# 链路:
#   [1] 抽帧   cv2 按 fps 采样 -> images/0.JPG,1.JPG,... (直接 0-based, 对齐 code 输入约定)
#   [2] SKU检测 -> detections_results/0.json,1.json,...   ⚠️ 缺口, 见下
#   [3] pipeline main.py --mode pipeline (da3 + 3d) -> 匹配+去重+global_id
#   [4] 提取   去重SKU数目 = global_mapping.json 的 key 数; 带框图 = dedup_imgs_w_bboxes/
#
# ⚠️ 关键缺口 - SKU 检测:
#   code/main.py 是核心编排服务, 但【不做 SKU 图像检测】, 它要求输入已有 detections_results/.
#   仓库内无检测器(frame_sampler 只抽帧; processor 注释"接收上游 images+skus").
#   因此必须提供 detections 来源之一:
#     (a) 已有 detections 目录(每帧一个 <i>.json):
#         bash video_to_dedup.sh <video> <fps> <gpu> <detections_dir>
#     (b) 接入外部检测服务: 编辑下方 run_detection() 函数
#
# 用法:
#   bash video_to_dedup.sh                              # 默认 video-test/6-1.mp4, fps=2.0, gpu=0 (会因缺检测报错)
#   bash video_to_dedup.sh <video> <fps> <gpu> <detections_dir>
#   示例: bash video_to_dedup.sh ../../small_fd_video/video-test/6-1.mp4 2.0 0 /path/to/detections_results
set -euo pipefail

# ===== 参数 =====
VIDEO_ARG="${1:-../../small_fd_video/video-test/6-1.mp4}"
FPS="${2:-2.0}"
GPU="${3:-0}"
DETECTIONS_SRC="${4:-}"   # 已有 detections 目录(每帧 <i>.json); 空=报错

# ===== 路径 =====
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"          # code/
REPO_DIR="$(cd "$CODE_DIR/.." && pwd)"            # 仓库根
VIDEO="$(realpath "$VIDEO_ARG" 2>/dev/null || echo "$VIDEO_ARG")"
DATASET_NAME="$(basename "$VIDEO")"; DATASET_NAME="${DATASET_NAME%.*}"
WORK_ROOT="$CODE_DIR/video_dedup_runs"
DATASET_DIR="$WORK_ROOT/$DATASET_NAME"
SAVE_ROOT="$CODE_DIR/Output"
export CUDA_VISIBLE_DEVICES="$GPU"

[ -f "$VIDEO" ] || { echo "[ERROR] 视频不存在: $VIDEO" >&2; exit 1; }
echo "=== video -> 去重结果 ==="
echo "  video      : $VIDEO"
echo "  dataset    : $DATASET_DIR"
echo "  save_root  : $SAVE_ROOT"
echo "  fps/gpu    : $FPS / $GPU"
echo

# ---------- [1] 抽帧 (cv2, 直接输出 0-based) ----------
run_frame_sample() {
  echo "[1/4] 抽帧 -> $DATASET_DIR/images/"
  mkdir -p "$DATASET_DIR/images"
  cd "$CODE_DIR"
  uv run python - "$VIDEO" "$DATASET_DIR/images" "$FPS" <<'PYEOF'
import sys, cv2, os
video, out, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
os.makedirs(out, exist_ok=True)
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
        cv2.imwrite(os.path.join(out, f"{idx}.JPG"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        idx += 1
    fi += 1
cap.release()
print(f"  抽帧完成: {idx} 张 -> {out} (视频fps={vfps:.2f}, 采样间隔={step}帧)")
PYEOF
}

# ---------- [2] SKU 检测 (缺口: 需外部提供) ----------
run_detection() {
  echo "[2/4] SKU 检测 -> $DATASET_DIR/detections_results/"
  mkdir -p "$DATASET_DIR/detections_results"
  if [ -n "$DETECTIONS_SRC" ]; then
    DET_ABS="$(realpath "$DETECTIONS_SRC" 2>/dev/null || echo "$DETECTIONS_SRC")"
    cp "$DET_ABS"/*.json "$DATASET_DIR/detections_results/" 2>/dev/null || true
    local n; n=$(ls "$DATASET_DIR/detections_results"/*.json 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "[ERROR] detections_dir 内无 json: $DET_ABS" >&2; exit 1; }
    echo "  使用已有 detections: $DET_ABS ($n 个 json)"
  else
    cat >&2 <<EOF
[ERROR] 缺少 SKU 检测结果.
  code/main.py 不做检测(只匹配/去重), 仓库内无检测器.
  请任选其一:
    (a) 已有 detections 目录(每帧 <i>.json, 格式 {skus:[{classes:{det:[...]}, objects:[{position:[x1,y1,x2,y2],confidences:...}]}]}):
        bash "$0" "$VIDEO_ARG" "$FPS" "$GPU" <detections_dir>
    (b) 编辑本脚本 run_detection() 接入外部检测服务.
EOF
    exit 1
  fi
}

# ---------- [3] 运行 pipeline (main.py --mode pipeline, da3/3d) ----------
run_pipeline() {
  echo "[3/4] 运行 pipeline: main.py --mode pipeline (da3 + 3d)"
  cd "$CODE_DIR"
  # pipeline 末尾的 accuracy_evaluation 对无 benchmark 的 video 数据可能失败, 但去重结果在此之前已生成;
  # 故允许非0退出, 由 extract_results 检查 global_mapping.json 是否存在来判定.
  uv run python main.py --mode pipeline --dataset "$DATASET_DIR" \
    --algorithm 3d --save_root "$SAVE_ROOT" || echo "  [warn] pipeline 退出码非0, 继续检查结果..."
}

# ---------- [4] 提取结果 ----------
extract_results() {
  echo "[4/4] 提取结果"
  local OUT_DIR="$SAVE_ROOT/$DATASET_NAME"
  local DEDUP_DIR="$OUT_DIR/dedup_detections"
  local VIZ_DIR="$OUT_DIR/dedup_imgs_w_bboxes"
  local GM="$DEDUP_DIR/global_mapping.json"
  [ -f "$GM" ] || { echo "[ERROR] 未找到 $GM, pipeline 去重未成功" >&2; exit 1; }
  cd "$CODE_DIR"
  local n; n=$(uv run python -c "import json; print(len(json.load(open('$GM'))))" 2>/dev/null)
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
