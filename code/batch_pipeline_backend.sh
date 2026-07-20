#!/bin/bash
# 批量跑 3D pipeline (da3: 重建+matching+dedup, batch_all_refs)，输出隔离到 --save-root。
# 用法: bash batch_pipeline_backend.sh --start 2 --end 4 --gpu 1 --save-root ../Output/da3
set -euo pipefail

START=2; END=12; GPU=""; SAVE_ROOT=""; MODEL_PATH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --start)           START="$2"; shift 2 ;;
        --end)             END="$2"; shift 2 ;;
        --gpu)             GPU="$2"; shift 2 ;;
        --save-root)       SAVE_ROOT="${2%/}"; shift 2 ;;
        --recon-model-path) MODEL_PATH="$2"; shift 2 ;;
        *)                 echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

[ -z "$SAVE_ROOT" ] && { echo "错误: 必须指定 --save-root 以隔离输出"; exit 1; }
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")"

BACKEND="da3"
echo "=== 批量 pipeline (da3) floor_display${START}..${END} | GPU=${GPU:-默认} | save=${SAVE_ROOT} ==="
for i in $(seq "$START" "$END"); do
    DS="../imdata/floor_display${i}"
    [ -d "$DS" ] || { echo "跳过(无目录): floor_display${i}"; continue; }
    echo "--- [da3] floor_display${i} ---"
    ARGS=(--mode pipeline --dataset "$DS" \
        --algorithm 3d --save_root "$SAVE_ROOT")
    [ -n "$MODEL_PATH" ] && ARGS+=(--recon_model_path "$MODEL_PATH")
    uv run python main.py "${ARGS[@]}"
done
echo "完成: da3 floor_display${START}..${END} -> ${SAVE_ROOT}"
