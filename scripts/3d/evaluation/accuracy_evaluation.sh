#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BENCHMARK_CSV="$PROJECT_ROOT/imdata/picture_mapping_benchmark.csv"
ACCURACY_SCRIPT="$PROJECT_ROOT/accuracy_annotation.py"
FD=""
BACKEND="pt"
SAVE_ROOT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        --save-root) SAVE_ROOT="${2%/}"; shift 2 ;;
        -*) echo "未知选项: $1" >&2; exit 1 ;;
        *) FD="${FD:-$1}"; shift ;;
    esac
done
FD="${FD:-floor_display3}"
case "$BACKEND" in
    pt|point_tracking) OUT_SUB="output_pt"; BACKEND="pt" ;;
    pi3|pi3x|da3|mapanything) OUT_SUB="output_3dmapping_${BACKEND}" ;;
    *) echo "错误: 未知 backend: $BACKEND (pt|pi3|pi3x|da3|mapanything)" >&2; exit 1 ;;
esac
DATA_ROOT="${SAVE_ROOT:-$PROJECT_ROOT/Output}"
OUTPUT_BASE_DIR="$DATA_ROOT/$FD/$OUT_SUB"
RESULT_BASE_DIR="$DATA_ROOT/$FD/accuracy_evaluation_${BACKEND}"
[ -f "$BENCHMARK_CSV" ] || { echo "人工标注文件不存在: $BENCHMARK_CSV" >&2; exit 1; }
[ -f "$ACCURACY_SCRIPT" ] || { echo "准确性标注脚本不存在: $ACCURACY_SCRIPT" >&2; exit 1; }
[ -d "$OUTPUT_BASE_DIR" ] || { echo "匹配输出目录不存在: $OUTPUT_BASE_DIR" >&2; exit 1; }
mkdir -p "$RESULT_BASE_DIR"
SUMMARY_REPORT="$RESULT_BASE_DIR/summary.txt"
printf 'SKU匹配准确性批量评估汇总报告\n生成时间: %s\n\n' "$(date)" > "$SUMMARY_REPORT"
cd "$PROJECT_ROOT"
total_pairs=0
for ref_dir in "$OUTPUT_BASE_DIR"/*/; do
    [ -d "$ref_dir" ] || continue
    ref_num=$(basename "$ref_dir")
    matching_summary="$ref_dir/matching_summary.txt"
    [ -f "$matching_summary" ] || { echo "匹配结果不存在: $matching_summary" >&2; exit 1; }
    temp_report="$RESULT_BASE_DIR/temp_${ref_num}.txt"
    uv run python "$ACCURACY_SCRIPT" \
        --benchmark-csv "$BENCHMARK_CSV" \
        --vggt-result "$matching_summary" \
        --dataset-filter "$FD" \
        --output "$temp_report"
    image_pair=$(awk '/图片对.*详细分析/ { print $2; exit }' "$temp_report")
    output_report="$RESULT_BASE_DIR/${image_pair:-ref_${ref_num}}.txt"
    mv "$temp_report" "$output_report"
    printf '参考图片 %s 评估结果:\n' "$ref_num" >> "$SUMMARY_REPORT"
    awk '/总体召回率|VGGT有效率|模型有效率|Reference ID映射准确率/ { print "  " $0 }' "$output_report" >> "$SUMMARY_REPORT"
    printf '\n' >> "$SUMMARY_REPORT"
    total_pairs=$((total_pairs + 1))
done
[ "$total_pairs" -gt 0 ] || { echo "没有可评估的参考帧: $OUTPUT_BASE_DIR" >&2; exit 1; }
printf '整体统计:\n  处理的图片对总数: %s\n  成功评估数量: %s\n  失败评估数量: 0\n  成功率: 100.00%%\n' "$total_pairs" "$total_pairs" >> "$SUMMARY_REPORT"
printf '评估完成: %s 个参考帧；汇总报告: %s\n' "$total_pairs" "$SUMMARY_REPORT"
