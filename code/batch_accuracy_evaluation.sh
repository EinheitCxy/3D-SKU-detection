#!/bin/bash
# 该脚本用于批量处理 floor_display2 到 floor_display12 的所有数据集，
# 依次调用 accuracy_evaluation.sh 进行准确性评估，并生成跨数据集汇总报告。

# 设置脚本参数
set -e  # 遇到错误时退出
set -u  # 使用未定义变量时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 配置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 数据集范围配置
START_FD="${1:-2}"   # 起始数据集编号，默认2
END_FD="${2:-12}"    # 结束数据集编号，默认12

# 跨数据集汇总报告
GLOBAL_SUMMARY_DIR="$PROJECT_ROOT/imdata/batch_accuracy_results"
GLOBAL_SUMMARY_REPORT="$GLOBAL_SUMMARY_DIR/global_summary_${TIMESTAMP}.txt"

# 创建全局结果目录
mkdir -p "$GLOBAL_SUMMARY_DIR"

# 初始化
echo ""
echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║          SKU匹配准确性 - 多数据集批量评估                      ║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}开始时间: $(date)${NC}"
echo -e "${BLUE}数据集范围: floor_display${START_FD} ~ floor_display${END_FD}${NC}"
echo ""

# 检查单数据集评估脚本是否存在
if [ ! -f "$SCRIPT_DIR/accuracy_evaluation.sh" ]; then
    echo -e "${RED}错误: 单数据集评估脚本不存在: $SCRIPT_DIR/accuracy_evaluation.sh${NC}"
    exit 1
fi

# 统计变量
total_datasets=0
successful_datasets=0
failed_datasets=0
skipped_datasets=0

# 存储各数据集结果
declare -a dataset_results=()

# 初始化全局汇总报告
cat > "$GLOBAL_SUMMARY_REPORT" << EOF
================================================================================
SKU匹配准确性 - 多数据集批量评估汇总报告
================================================================================
生成时间: $(date)
数据集范围: floor_display${START_FD} ~ floor_display${END_FD}

================================================================================
各数据集评估结果
================================================================================

EOF

# 遍历所有数据集
for fd_num in $(seq $START_FD $END_FD); do
    FD="floor_display${fd_num}"
    FD_DIR="$PROJECT_ROOT/imdata/$FD"
    OUTPUT_PT_DIR="$FD_DIR/output_pt"

    total_datasets=$((total_datasets + 1))

    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}[$fd_num/$END_FD] 处理数据集: $FD${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # 检查数据集目录是否存在
    if [ ! -d "$FD_DIR" ]; then
        echo -e "${YELLOW}  ⚠ 跳过: 数据集目录不存在 ($FD_DIR)${NC}"
        skipped_datasets=$((skipped_datasets + 1))
        echo "[$FD] 跳过 - 数据集目录不存在" >> "$GLOBAL_SUMMARY_REPORT"
        echo "" >> "$GLOBAL_SUMMARY_REPORT"
        continue
    fi

    # 检查output_pt目录是否存在
    if [ ! -d "$OUTPUT_PT_DIR" ]; then
        echo -e "${YELLOW}  ⚠ 跳过: output_pt目录不存在 ($OUTPUT_PT_DIR)${NC}"
        skipped_datasets=$((skipped_datasets + 1))
        echo "[$FD] 跳过 - output_pt目录不存在" >> "$GLOBAL_SUMMARY_REPORT"
        echo "" >> "$GLOBAL_SUMMARY_REPORT"
        continue
    fi

    # 检查是否有matching_summary.txt文件
    matching_count=$(find "$OUTPUT_PT_DIR" -name "matching_summary.txt" 2>/dev/null | wc -l)
    if [ "$matching_count" -eq 0 ]; then
        echo -e "${YELLOW}  ⚠ 跳过: 无匹配结果文件${NC}"
        skipped_datasets=$((skipped_datasets + 1))
        echo "[$FD] 跳过 - 无matching_summary.txt文件" >> "$GLOBAL_SUMMARY_REPORT"
        echo "" >> "$GLOBAL_SUMMARY_REPORT"
        continue
    fi

    echo -e "${BLUE}  发现 $matching_count 个匹配结果文件${NC}"

    # 调用单数据集评估脚本
    echo -e "${BLUE}  执行评估...${NC}"

    # 使用set +e临时允许错误，以便捕获失败
    set +e
    bash "$SCRIPT_DIR/accuracy_evaluation.sh" "$FD" 2>&1 | tee "$GLOBAL_SUMMARY_DIR/${FD}_log.txt"
    eval_result=$?
    set -e

    if [ $eval_result -eq 0 ]; then
        successful_datasets=$((successful_datasets + 1))
        echo -e "${GREEN}  ✓ $FD 评估完成${NC}"

        # 提取该数据集的汇总信息
        DATASET_SUMMARY="$FD_DIR/accuracy_evaluation/summary.txt"
        if [ -f "$DATASET_SUMMARY" ]; then
            echo "--------------------------------------------------------------------------------" >> "$GLOBAL_SUMMARY_REPORT"
            echo "[$FD] 评估成功" >> "$GLOBAL_SUMMARY_REPORT"
            echo "--------------------------------------------------------------------------------" >> "$GLOBAL_SUMMARY_REPORT"

            # 提取关键统计信息
            grep -E "处理的图片对总数|成功评估数量|失败评估数量|成功率" "$DATASET_SUMMARY" >> "$GLOBAL_SUMMARY_REPORT" 2>/dev/null || true
            echo "" >> "$GLOBAL_SUMMARY_REPORT"

            # 提取各图片对的指标
            grep -A3 "参考图片.*评估结果" "$DATASET_SUMMARY" >> "$GLOBAL_SUMMARY_REPORT" 2>/dev/null || true
            echo "" >> "$GLOBAL_SUMMARY_REPORT"

            # 存储结果供后续统计
            success_count=$(grep "成功评估数量" "$DATASET_SUMMARY" | grep -oE '[0-9]+' | head -1 || echo "0")
            dataset_results+=("$FD:$success_count")
        fi
    else
        failed_datasets=$((failed_datasets + 1))
        echo -e "${RED}  ✗ $FD 评估失败${NC}"
        echo "[$FD] 评估失败 (退出码: $eval_result)" >> "$GLOBAL_SUMMARY_REPORT"
        echo "" >> "$GLOBAL_SUMMARY_REPORT"
    fi

    echo ""
done

# 生成全局统计
cat >> "$GLOBAL_SUMMARY_REPORT" << EOF

================================================================================
全局统计汇总
================================================================================
数据集范围: floor_display${START_FD} ~ floor_display${END_FD}
总数据集数量: $total_datasets
成功评估数量: $successful_datasets
失败评估数量: $failed_datasets
跳过数量: $skipped_datasets

EOF

# 计算成功率
if [ $total_datasets -gt 0 ]; then
    success_rate=$(echo "scale=2; $successful_datasets * 100 / $total_datasets" | bc -l)
    echo "数据集评估成功率: ${success_rate}%" >> "$GLOBAL_SUMMARY_REPORT"
fi

# 各数据集评估图片对数量统计
if [ ${#dataset_results[@]} -gt 0 ]; then
    echo "" >> "$GLOBAL_SUMMARY_REPORT"
    echo "各数据集评估图片对数量:" >> "$GLOBAL_SUMMARY_REPORT"
    total_pairs=0
    for result in "${dataset_results[@]}"; do
        fd_name=$(echo "$result" | cut -d: -f1)
        pair_count=$(echo "$result" | cut -d: -f2)
        echo "  $fd_name: $pair_count 对" >> "$GLOBAL_SUMMARY_REPORT"
        total_pairs=$((total_pairs + pair_count))
    done
    echo "  总计: $total_pairs 对" >> "$GLOBAL_SUMMARY_REPORT"
fi

echo "" >> "$GLOBAL_SUMMARY_REPORT"
echo "报告生成时间: $(date)" >> "$GLOBAL_SUMMARY_REPORT"

# 输出最终统计
echo ""
echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║                      批量评估完成                              ║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}统计信息:${NC}"
echo "  数据集范围: floor_display${START_FD} ~ floor_display${END_FD}"
echo "  总数据集数量: $total_datasets"
echo -e "  成功: ${GREEN}$successful_datasets${NC}"
echo -e "  失败: ${RED}$failed_datasets${NC}"
echo -e "  跳过: ${YELLOW}$skipped_datasets${NC}"
echo ""
echo -e "${BLUE}输出文件:${NC}"
echo "  全局汇总报告: $GLOBAL_SUMMARY_REPORT"
echo "  各数据集日志: $GLOBAL_SUMMARY_DIR/floor_display*_log.txt"
echo "  各数据集详细报告: \$PROJECT_ROOT/imdata/floor_display*/accuracy_evaluation/"
echo ""
echo -e "${BLUE}结束时间: $(date)${NC}"
echo ""

# 显示汇总报告预览
if [ -f "$GLOBAL_SUMMARY_REPORT" ]; then
    echo -e "${CYAN}全局汇总报告预览:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    head -40 "$GLOBAL_SUMMARY_REPORT"
    echo "..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${CYAN}完整报告请查看: $GLOBAL_SUMMARY_REPORT${NC}"
fi
