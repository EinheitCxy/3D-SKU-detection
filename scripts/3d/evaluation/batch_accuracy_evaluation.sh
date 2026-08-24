#!/bin/bash
# 批量评估 floor_display<start..end> 的 SKU 匹配准确率，按后端隔离。
# 用法: bash batch_accuracy_evaluation.sh --backend <pt|pi3|da3> --start 2 --end 12 --save-root Output
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; B='\033[1m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

BACKEND="pt"; START=2; END=12; SAVE_ROOT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --backend)    BACKEND="$2"; shift 2 ;;
        --start)      START="$2"; shift 2 ;;
        --end)        END="$2"; shift 2 ;;
        --save-root)  SAVE_ROOT="${2%/}"; shift 2 ;;
        *)            echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

case "$BACKEND" in
    pt|point_tracking) OUT_SUB="output_pt"; BACKEND="pt" ;;
    pi3)               OUT_SUB="output_3dmapping_pi3" ;;
    da3)               OUT_SUB="output_3dmapping_da3" ;;
    *)                 echo -e "${RED}未知 backend: $BACKEND (pt|pi3|da3)${NC}"; exit 1 ;;
esac

DATA_ROOT="${SAVE_ROOT:-$PROJECT_ROOT/Output}"
BENCHMARK_CSV="$PROJECT_ROOT/imdata/picture_mapping_benchmark.csv"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY_DIR="$DATA_ROOT/batch_accuracy_results_${BACKEND}"
GLOBAL_REPORT="$SUMMARY_DIR/global_summary_${BACKEND}_${TS}.txt"
mkdir -p "$SUMMARY_DIR"

[ -f "$BENCHMARK_CSV" ] || { echo -e "${RED}缺 benchmark CSV: $BENCHMARK_CSV${NC}"; exit 1; }
[ -f "$SCRIPT_DIR/accuracy_evaluation.sh" ] || { echo -e "${RED}缺 accuracy_evaluation.sh${NC}"; exit 1; }

echo -e "${B}${CYAN}=== 批量准确率评估 [${BACKEND}] floor_display${START}..${END} ===${NC}"
echo -e "${BLUE}数据根: $DATA_ROOT | 匹配目录: $OUT_SUB${NC}"
echo -e "${BLUE}开始: $(date)${NC}"

{
echo "================================================================================"
echo "SKU匹配准确率批量评估 [${BACKEND}]"
echo "================================================================================"
echo "生成时间: $(date)"
echo "后端: $BACKEND | 数据根: $DATA_ROOT | 范围: floor_display${START}..${END}"
echo ""
} > "$GLOBAL_REPORT"

total=0; ok=0; fail=0; skip=0
declare -a results=()

for fd in $(seq "$START" "$END"); do
    FD="floor_display${fd}"
    FD_DIR="$DATA_ROOT/$FD"
    MATCH_DIR="$FD_DIR/$OUT_SUB"
    total=$((total+1))
    echo -e "${YELLOW}[$fd/$END] $FD${NC}"

    [ -d "$FD_DIR" ] || { echo -e "  ${YELLOW}跳过: 无目录${NC}"; skip=$((skip+1)); echo "[$FD] 跳过-无目录" >> "$GLOBAL_REPORT"; continue; }
    [ -d "$MATCH_DIR" ] || { echo -e "  ${YELLOW}跳过: 无 $OUT_SUB${NC}"; skip=$((skip+1)); echo "[$FD] 跳过-无匹配输出目录" >> "$GLOBAL_REPORT"; continue; }
    mc=$(find "$MATCH_DIR" -name "matching_summary.txt" 2>/dev/null | wc -l)
    [ "$mc" -eq 0 ] && { echo -e "  ${YELLOW}跳过: 无 matching_summary.txt${NC}"; skip=$((skip+1)); echo "[$FD] 跳过-无matching_summary" >> "$GLOBAL_REPORT"; continue; }

    echo -e "  ${BLUE}发现 $mc 个 matching_summary，评估中...${NC}"
    set +e
    ACC_ARGS=("$FD" "--backend" "$BACKEND")
    [ -n "$SAVE_ROOT" ] && ACC_ARGS+=("--save-root" "$SAVE_ROOT")
    bash "$SCRIPT_DIR/accuracy_evaluation.sh" "${ACC_ARGS[@]}" 2>&1 | tee "$SUMMARY_DIR/${FD}_log.txt"
    rc=${PIPESTATUS[0]}
    set -e

    if [ "$rc" -eq 0 ]; then
        ok=$((ok+1)); echo -e "  ${GREEN}✓ $FD 完成${NC}"
        S="$FD_DIR/accuracy_evaluation_${BACKEND}/summary.txt"
        if [ -f "$S" ]; then
            {
            echo "--------------------------------------------------------------------------------"
            echo "[$FD] 评估成功"
            grep -E "总体召回率|有效率|映射准确率" "$S" 2>/dev/null || true
            echo ""
            } >> "$GLOBAL_REPORT"
            r=$(grep "总体召回率" "$S" | grep -oE '[0-9.]+%' | head -1); r="${r:-0%}"
            results+=("$FD:$r")
        fi
    else
        fail=$((fail+1)); echo -e "  ${RED}✗ $FD 失败($rc)${NC}"
        echo "[$FD] 失败($rc)" >> "$GLOBAL_REPORT"
    fi
done

{
echo ""
echo "================================================================================"
echo "全局统计 [${BACKEND}]"
echo "================================================================================"
echo "总: $total | 成功: $ok | 失败: $fail | 跳过: $skip"
echo ""
echo "各数据集召回率:"
if [ ${#results[@]} -gt 0 ]; then
    for r in "${results[@]}"; do echo "  $r"; done
else
    echo "  (无成功数据集)"
fi
echo ""
echo "报告时间: $(date)"
} >> "$GLOBAL_REPORT"

echo ""
echo -e "${B}${CYAN}=== 完成 [${BACKEND}] ===${NC}"
echo "总:$total 成功:$ok 失败:$fail 跳过:$skip"
echo -e "${BLUE}报告: $GLOBAL_REPORT${NC}"
echo -e "${BLUE}结束: $(date)${NC}"
