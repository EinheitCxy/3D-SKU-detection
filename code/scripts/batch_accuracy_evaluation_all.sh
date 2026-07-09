#!/bin/bash
# 该脚本用于批量评估所有三种算法（pt, vggt, pi3）的准确性

# 设置脚本参数
set -e  # 遇到错误时退出
set -u  # 使用未定义变量时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FD="${1:-floor_display12}"  # 支持命令行参数，默认floor_display3
BENCHMARK_CSV="$PROJECT_ROOT/imdata/picture_mapping_benchmark.csv"

# 定义三种算法类型及其对应的目录
declare -A ALGO_CONFIGS=(
    ["pt"]="output_pt:accuracy_evaluation_pt:Point Tracking"
    ["vggt"]="output_3dmapping_vggt:accuracy_evaluation_vggt:3D Mapping (VGGT)"
    ["pi3"]="output_3dmapping_pi3:accuracy_evaluation_pi3:3D Mapping (Pi3)"
)

# 初始化
echo "============================================"
echo "SKU匹配准确性批量评估 - 全算法模式"
echo "数据集: $FD"
echo "开始时间: $(date)"
echo "============================================"
echo ""

# 检查必要文件
if [ ! -f "$BENCHMARK_CSV" ]; then
    echo -e "${RED}错误: 人工标注文件不存在: $BENCHMARK_CSV${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/accuracy_annotation.py" ]; then
    echo -e "${RED}错误: 准确性标注脚本不存在: $SCRIPT_DIR/accuracy_annotation.py${NC}"
    exit 1
fi

# 全局统计变量
global_total_algos=0
global_successful_algos=0

# 遍历所有算法类型
for algo_type in pt vggt pi3; do
    echo ""
    echo "================================================================================"
    IFS=':' read -r output_subdir result_subdir algo_name <<< "${ALGO_CONFIGS[$algo_type]}"
    echo -e "${BLUE}开始评估算法: $algo_name${NC}"
    echo "================================================================================"

    OUTPUT_BASE_DIR="$PROJECT_ROOT/code/Output/$FD/$output_subdir"
    RESULT_BASE_DIR="$PROJECT_ROOT/code/Output/$FD/$result_subdir"

    # 检查输出目录是否存在
    if [ ! -d "$OUTPUT_BASE_DIR" ]; then
        echo -e "${YELLOW}警告: 输出目录不存在，跳过 $algo_name: $OUTPUT_BASE_DIR${NC}"
        continue
    fi

    global_total_algos=$((global_total_algos + 1))

    # 创建结果目录
    mkdir -p "$RESULT_BASE_DIR"

    # 统计变量
    total_pairs=0
    successful_pairs=0
    failed_pairs=0

    # 创建汇总报告文件
    SUMMARY_REPORT="$RESULT_BASE_DIR/summary.txt"

    echo "================================================================================" > "$SUMMARY_REPORT"
    echo "SKU匹配准确性批量评估汇总报告 - $algo_name" >> "$SUMMARY_REPORT"
    echo "================================================================================" >> "$SUMMARY_REPORT"
    echo "数据集: $FD" >> "$SUMMARY_REPORT"
    echo "生成时间: $(date)" >> "$SUMMARY_REPORT"
    echo "" >> "$SUMMARY_REPORT"

    # 存储所有评估结果
    declare -a all_results=()

    echo -e "${BLUE}扫描输出目录: $OUTPUT_BASE_DIR${NC}"

    # 遍历所有子目录
    for ref_dir in "$OUTPUT_BASE_DIR"/*/; do
        if [ -d "$ref_dir" ]; then
            ref_num=$(basename "$ref_dir")
            matching_summary="$ref_dir/matching_summary.txt"

            if [ -f "$matching_summary" ]; then
                echo -e "${YELLOW}处理参考图片 $ref_num 的匹配结果...${NC}"

                temp_report="$RESULT_BASE_DIR/temp_${ref_num}.txt"
                total_pairs=$((total_pairs + 1))

                # 切换到脚本目录并运行
                cd "$SCRIPT_DIR"

                if uv run python accuracy_annotation.py \
                    --benchmark-csv "$BENCHMARK_CSV" \
                    --vggt-result "$matching_summary" \
                    --dataset-filter "$FD" \
                    --output "$temp_report" 2>&1; then

                    successful_pairs=$((successful_pairs + 1))
                    echo -e "${GREEN}✓ 参考图片 $ref_num 评估完成${NC}"

                    # 从临时报告中提取图片对信息，并创建对应的文件名
                    if [ -f "$temp_report" ]; then
                        image_pair=$(grep "图片对.*详细分析" "$temp_report" | head -1 | sed 's/图片对 \([^ ]*\) 详细分析:.*/\1/')

                        if [ -n "$image_pair" ]; then
                            final_report="$RESULT_BASE_DIR/${image_pair}.txt"
                            mv "$temp_report" "$final_report"
                            output_report="$final_report"
                        else
                            final_report="$RESULT_BASE_DIR/ref_${ref_num}.txt"
                            mv "$temp_report" "$final_report"
                            output_report="$final_report"
                        fi
                    fi

                    # 提取关键指标并添加到汇总报告
                    if [ -f "$output_report" ]; then
                        echo "参考图片 $ref_num 评估结果:" >> "$SUMMARY_REPORT"
                        echo "----------------------------------------" >> "$SUMMARY_REPORT"

                        recall=$(grep "总体召回率" "$output_report" | head -1)
                        effectiveness=$(grep "VGGT有效率" "$output_report" | head -1)
                        precision=$(grep "Reference ID映射准确率" "$output_report" | head -1)

                        if [ -n "$recall" ] && [ -n "$effectiveness" ] && [ -n "$precision" ]; then
                            echo "  $recall" >> "$SUMMARY_REPORT"
                            echo "  $effectiveness" >> "$SUMMARY_REPORT"
                            echo "  $precision" >> "$SUMMARY_REPORT"
                            all_results+=("$ref_num:$recall:$effectiveness:$precision")
                        else
                            echo "  警告: 无法提取评估指标" >> "$SUMMARY_REPORT"
                        fi

                        echo "" >> "$SUMMARY_REPORT"
                    fi

                else
                    failed_pairs=$((failed_pairs + 1))
                    echo -e "${RED}✗ 参考图片 $ref_num 评估失败${NC}"
                    echo "参考图片 $ref_num: 评估失败" >> "$SUMMARY_REPORT"
                    echo "" >> "$SUMMARY_REPORT"
                fi

                echo ""

            else
                echo -e "${YELLOW}跳过 $ref_num: matching_summary.txt 不存在${NC}"
            fi
        fi
    done

    # 计算整体统计
    echo "================================================================================" >> "$SUMMARY_REPORT"
    echo "整体统计:" >> "$SUMMARY_REPORT"
    echo "  处理的图片对总数: $total_pairs" >> "$SUMMARY_REPORT"
    echo "  成功评估数量: $successful_pairs" >> "$SUMMARY_REPORT"
    echo "  失败评估数量: $failed_pairs" >> "$SUMMARY_REPORT"
    if [ $total_pairs -gt 0 ]; then
        echo "  成功率: $(echo "scale=2; $successful_pairs * 100 / $total_pairs" | bc -l)%" >> "$SUMMARY_REPORT"
    fi

    echo "" >> "$SUMMARY_REPORT"
    echo "报告生成时间: $(date)" >> "$SUMMARY_REPORT"

    # 输出算法统计
    echo ""
    echo -e "${GREEN}$algo_name 评估完成！${NC}"
    echo "  总处理数量: $total_pairs"
    echo "  成功数量: $successful_pairs"
    echo "  失败数量: $failed_pairs"
    echo "  汇总报告: $SUMMARY_REPORT"

    if [ $successful_pairs -gt 0 ]; then
        global_successful_algos=$((global_successful_algos + 1))
    fi

    # 清理数组
    unset all_results
done

# 返回原始目录
cd "$SCRIPT_DIR"

# 全局总结
echo ""
echo "================================================================================"
echo -e "${GREEN}所有算法评估完成！${NC}"
echo "================================================================================"
echo "评估的算法总数: $global_total_algos"
echo "成功评估的算法: $global_successful_algos"
echo ""
echo "结果目录: $PROJECT_ROOT/code/Output/$FD/"
echo "  - accuracy_evaluation_pt/       (Point Tracking)"
echo "  - accuracy_evaluation_vggt/     (3D Mapping VGGT)"
echo "  - accuracy_evaluation_pi3/      (3D Mapping Pi3)"
echo "================================================================================"
