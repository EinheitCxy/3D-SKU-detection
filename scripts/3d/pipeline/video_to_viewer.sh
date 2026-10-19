#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VIDEO_TO_DEDUP="$PROJECT_ROOT/modules/video_to_dedup/run.sh"
VIEWER_ROOT="$PROJECT_ROOT/modules/viewer_web"
DEFAULT_VIEWER_OUTPUT="$VIEWER_ROOT/public/data"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  bash scripts/3d/pipeline/video_to_viewer.sh --video PATH [options]" \
        "" \
        "Pipeline:" \
        "  video -> frames -> detection -> DA3 pipeline -> dedup/global ID" \
        "        + early geometry/texture preparation -> complete Surfel bundle -> optional Vite server" \
        "" \
        "Required:" \
        "  --video PATH                  输入视频；相对路径按仓库根解析" \
        "" \
        "Sampling and inference:" \
        "  --fps FPS                     抽帧频率，默认 2.0" \
        "  --gpu INDEX                   物理 GPU 编号，默认 0" \
        "  --detections-dir DIR          复用已有逐帧 JSON；省略时运行 SKU detector" \
        "  --detector-device DEVICE      detector 设备，默认 cpu" \
        "  --classifier-device DEVICE    GPU mask 后的 classifier 设备，默认 cuda:0" \
        "" \
        "Outputs:" \
        "  --save-root DIR               DA3/matching/dedup 输出根，默认 Output" \
        "  --viewer-output DIR           bundle 目录，默认 modules/viewer_web/public/data" \
        "  --glb-output PATH             可选 compact GLB；默认不生成" \
        "  --glb-bake-device DEVICE      cpu（默认）或隔离后的 cuda:0" \
        "  --glb-encoder-workers N       1/2/4，默认 2" \
        "  --glb-fixed-shape             显式启用实验 fixed-shape CUDA sampler" \
        "" \
        "Viewer server:" \
        "  --serve                       导出后以前台 Vite 服务运行" \
        "  --host HOST                   Vite 监听地址，默认 127.0.0.1" \
        "  --port PORT                   Vite 监听端口，默认 5173" \
        "" \
        "Notes:" \
        "  CUDA_VISIBLE_DEVICES=INDEX 后，classifier 的 cuda:0 指向该物理 GPU。" \
        "  --serve 仅支持默认 viewer-output；自定义 bundle 需自行挂载到 /data/."
}

require_value() {
    [ "$#" -ge 2 ] || { printf '参数 %s 缺少值\n' "$1" >&2; exit 2; }
}

resolve_from_root() {
    case "$1" in
        /*) realpath -m "$1" ;;
        *) realpath -m "$PROJECT_ROOT/$1" ;;
    esac
}

VIDEO=""
FPS="2.0"
GPU="0"
DETECTIONS_DIR=""
DETECTOR_DEVICE="cpu"
CLASSIFIER_DEVICE="cuda:0"
SAVE_ROOT="$PROJECT_ROOT/Output"
VIEWER_OUTPUT="$DEFAULT_VIEWER_OUTPUT"
GLB_OUTPUT=""
GLB_BAKE_DEVICE="cpu"
GLB_ENCODER_WORKERS="2"
GLB_FIXED_SHAPE="false"
SERVE="false"
HOST="127.0.0.1"
PORT="5173"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --video) require_value "$@"; VIDEO="$2"; shift 2 ;;
        --fps) require_value "$@"; FPS="$2"; shift 2 ;;
        --gpu) require_value "$@"; GPU="$2"; shift 2 ;;
        --detections-dir) require_value "$@"; DETECTIONS_DIR="$2"; shift 2 ;;
        --detector-device) require_value "$@"; DETECTOR_DEVICE="$2"; shift 2 ;;
        --classifier-device) require_value "$@"; CLASSIFIER_DEVICE="$2"; shift 2 ;;
        --save-root) require_value "$@"; SAVE_ROOT="$2"; shift 2 ;;
        --viewer-output) require_value "$@"; VIEWER_OUTPUT="$2"; shift 2 ;;
        --glb-output) require_value "$@"; GLB_OUTPUT="$2"; shift 2 ;;
        --glb-bake-device) require_value "$@"; GLB_BAKE_DEVICE="$2"; shift 2 ;;
        --glb-encoder-workers) require_value "$@"; GLB_ENCODER_WORKERS="$2"; shift 2 ;;
        --glb-fixed-shape) GLB_FIXED_SHAPE="true"; shift ;;
        --serve) SERVE="true"; shift ;;
        --host) require_value "$@"; HOST="$2"; shift 2 ;;
        --port) require_value "$@"; PORT="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf '未知参数: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$VIDEO" ] || { printf '必须指定 --video PATH\n' >&2; exit 2; }
[[ "$FPS" =~ ^[0-9]+([.][0-9]+)?$ ]] && ! [[ "$FPS" =~ ^0+([.]0+)?$ ]] || { printf '无效 --fps: %s\n' "$FPS" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { printf '无效 --gpu: %s\n' "$GPU" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] && [ "$PORT" -ge 1 ] && [ "$PORT" -le 65535 ] || { printf '无效 --port: %s\n' "$PORT" >&2; exit 2; }
[[ "$GLB_BAKE_DEVICE" = "cpu" || "$GLB_BAKE_DEVICE" = "cuda:0" ]] || { printf '无效 --glb-bake-device: %s\n' "$GLB_BAKE_DEVICE" >&2; exit 2; }
[[ "$GLB_ENCODER_WORKERS" =~ ^(1|2|4)$ ]] || { printf '无效 --glb-encoder-workers: %s\n' "$GLB_ENCODER_WORKERS" >&2; exit 2; }
[ "$GLB_FIXED_SHAPE" != "true" ] || [ "$GLB_BAKE_DEVICE" = "cuda:0" ] || { printf '%s\n' '--glb-fixed-shape 要求 cuda:0' >&2; exit 2; }

VIDEO="$(resolve_from_root "$VIDEO")"
SAVE_ROOT="$(resolve_from_root "$SAVE_ROOT")"
VIEWER_OUTPUT="$(resolve_from_root "$VIEWER_OUTPUT")"
[ -z "$GLB_OUTPUT" ] || GLB_OUTPUT="$(resolve_from_root "$GLB_OUTPUT")"
[ -z "$GLB_OUTPUT" ] || [ ! -e "$GLB_OUTPUT" ] || { printf 'GLB输出已存在: %s\n' "$GLB_OUTPUT" >&2; exit 2; }
[ -z "$DETECTIONS_DIR" ] || DETECTIONS_DIR="$(resolve_from_root "$DETECTIONS_DIR")"
[ -f "$VIDEO" ] || { printf '视频不存在: %s\n' "$VIDEO" >&2; exit 1; }
[ -z "$DETECTIONS_DIR" ] || [ -d "$DETECTIONS_DIR" ] || { printf 'detections 目录不存在: %s\n' "$DETECTIONS_DIR" >&2; exit 1; }
[ "$SERVE" != "true" ] || [ "$VIEWER_OUTPUT" = "$DEFAULT_VIEWER_OUTPUT" ] || { printf '%s\n' '--serve 只能使用默认 viewer-output；自定义目录必须挂载到 /data/' >&2; exit 2; }

VIDEO_NAME="$(basename "$VIDEO")"
DATASET_NAME="${VIDEO_NAME%.*}"
DATASET_DIR="$PROJECT_ROOT/runtime/video_to_dedup/$DATASET_NAME"
CORE_ENV="${CORE_ENV:-$PROJECT_ROOT/.venv}"
[ -x "$CORE_ENV/bin/python" ] || { printf '核心环境不存在: %s\n' "$CORE_ENV" >&2; exit 1; }

printf 'Video pipeline: %s -> %s\n' "$VIDEO" "$DATASET_DIR"
SAVE_ROOT="$SAVE_ROOT" \
PIPELINE_SURFEL_OUTPUT="$VIEWER_OUTPUT" \
DETECTOR_DEVICE="$DETECTOR_DEVICE" \
CLASSIFIER_DEVICE="$CLASSIFIER_DEVICE" \
CUDA_VISIBLE_DEVICES="$GPU" \
bash "$VIDEO_TO_DEDUP" "$VIDEO" "$FPS" "$GPU" "$DETECTIONS_DIR"

[ -f "$VIEWER_OUTPUT/CURRENT" ] || { printf 'Viewer bundle 未完整发布: %s/CURRENT\n' "$VIEWER_OUTPUT" >&2; exit 1; }
printf 'Viewer bundle: %s\n' "$VIEWER_OUTPUT"

if [ -n "$GLB_OUTPUT" ]; then
    GLB_ARGS=(--viewer-root "$VIEWER_OUTPUT" --output "$GLB_OUTPUT" --bake-device "$GLB_BAKE_DEVICE" --encoder-workers "$GLB_ENCODER_WORKERS")
    [ "$GLB_FIXED_SHAPE" != "true" ] || GLB_ARGS+=(--bake-fixed-shape)
    # The core command has returned; explicitly apply its GPU isolation again.
    (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES="$GPU" uv run --no-project --python "$CORE_ENV/bin/python" python -m src.surfel_glb_delivery "${GLB_ARGS[@]}")
fi

if [ "$SERVE" = "true" ]; then
    printf 'Viewer URL: http://%s:%s/\n' "$HOST" "$PORT"
    exec npm --prefix "$VIEWER_ROOT" run dev -- --host "$HOST" --port "$PORT" --strictPort
fi

printf '%s\n' "启动命令: npm --prefix $VIEWER_ROOT run dev -- --host $HOST --port $PORT --strictPort"
