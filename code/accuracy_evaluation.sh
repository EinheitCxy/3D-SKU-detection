# 该脚本用于批量处理floor_display数据集中的所有图片对匹配结果，
# 自动运行准确性评估并生成详细的报告。

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
BENCHMARK_CSV="$PROJECT_ROOT/imdata/picture_mapping_benchmark.csv"

# 参数: 位置=数据集名(兼容旧用法), --backend pt|pi3|da3 (默认pt=point_tracking), --save-root <dir>(默认 imdata)
FD=""; BACKEND="pt"; SAVE_ROOT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --backend)    BACKEND="$2"; shift 2 ;;
        --save-root)  SAVE_ROOT="${2%/}"; shift 2 ;;
        -*)           echo "未知选项: $1" >&2; exit 1 ;;
        *)            FD="${FD:-$1}"; shift ;;
    esac
done
FD="${FD:-floor_display3}"

case "$BACKEND" in
    pt|point_tracking) OUT_SUB="output_pt"; BACKEND="pt" ;;
    pi3)               OUT_SUB="output_3dmapping_pi3" ;;
    da3)               OUT_SUB="output_3dmapping_da3" ;;
    *)                 echo "错误: 未知 backend: $BACKEND (pt|pi3|da3)" >&2; exit 1 ;;
esac

DATA_ROOT="${SAVE_ROOT:-$PROJECT_ROOT/imdata}"
OUTPUT_BASE_DIR="$DATA_ROOT/$FD/$OUT_SUB"
RESULT_BASE_DIR="$DATA_ROOT/$FD/accuracy_evaluation_${BACKEND}"
echo "[accuracy_eval] backend=$BACKEND 匹配输出=$OUTPUT_BASE_DIR 结果=$RESULT_BASE_DIR"


# 创建结果目录
mkdir -p "$RESULT_BASE_DIR"

# 初始化
echo "============================================"
echo "SKU匹配准确性批量评估"
echo "开始时间: $(date)"
echo "============================================"

# 检查必要文件
if [ ! -f "$BENCHMARK_CSV" ]; then
    echo -e "${RED}错误: 人工标注文件不存在: $BENCHMARK_CSV${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/accuracy_annotation.py" ]; then
    echo -e "${RED}错误: 准确性标注脚本不存在: $SCRIPT_DIR/accuracy_annotation.py${NC}"
    exit 1
fi

if [ ! -d "$OUTPUT_BASE_DIR" ]; then
    echo -e "${RED}错误: 匹配输出目录不存在: $OUTPUT_BASE_DIR${NC}"
    exit 1
fi

# 统计变量
total_pairs=0
successful_pairs=0
failed_pairs=0

# 创建汇总报告文件
SUMMARY_REPORT="$RESULT_BASE_DIR/summary.txt"

echo "================================================================================" > "$SUMMARY_REPORT"
echo "SKU匹配准确性批量评估汇总报告" >> "$SUMMARY_REPORT"
echo "================================================================================" >> "$SUMMARY_REPORT"
echo "生成时间: $(date)" >> "$SUMMARY_REPORT"
echo "" >> "$SUMMARY_REPORT"

# 存储所有评估结果
declare -a all_results=()

echo -e "${BLUE}开始扫描匹配输出目录...${NC}"

# 遍历所有output_pt子目录
for ref_dir in "$OUTPUT_BASE_DIR"/*/; do
    if [ -d "$ref_dir" ]; then
        ref_num=$(basename "$ref_dir")
        matching_summary="$ref_dir/matching_summary.txt"
        
        if [ -f "$matching_summary" ]; then
            echo -e "${YELLOW}处理参考图片 $ref_num 的匹配结果...${NC}"
            
            # 从匹配结果中解析实际的图片对信息
            # 根据我们的推理逻辑，参考图片索引需要+1
            actual_ref_img=$((ref_num + 1))
            
            # 查找匹配的图片对，生成相应的报告文件
            # 先运行一次获取所有图片对信息
            temp_report="$RESULT_BASE_DIR/temp_${ref_num}.txt"
            
            total_pairs=$((total_pairs + 1))
            
            # 运行准确性评估到临时文件
            echo "  执行命令: uv run python accuracy_annotation.py --benchmark-csv '$BENCHMARK_CSV' --vggt-result '$matching_summary' --dataset-filter "$FD" --output '$temp_report'"
            
            # 切换到脚本目录并运行
            cd "$SCRIPT_DIR"
            
            if uv run python accuracy_annotation.py \
                --benchmark-csv "$BENCHMARK_CSV" \
                --vggt-result "$matching_summary" \
                --dataset-filter "$FD" \
                --output "$temp_report" 2>&1; then
                
                successful_pairs=$((successful_pairs + 1))
                echo -e "${GREEN}参考图片 $ref_num 评估完成${NC}"
                
                # 从临时报告中提取图片对信息，并创建对应的文件名
                if [ -f "$temp_report" ]; then
                    # 查找图片对信息
                    image_pair=$(grep "图片对.*详细分析" "$temp_report" | head -1 | sed 's/图片对 \([^ ]*\) 详细分析:.*/\1/')
                    
                    if [ -n "$image_pair" ]; then
                        # 使用图片对名称作为文件名
                        final_report="$RESULT_BASE_DIR/${image_pair}.txt"
                        mv "$temp_report" "$final_report"
                        output_report="$final_report"
                    else
                        # 如果无法提取图片对信息，使用默认名称
                        final_report="$RESULT_BASE_DIR/ref_${ref_num}.txt"
                        mv "$temp_report" "$final_report"
                        output_report="$final_report"
                    fi
                fi
                
                # 提取关键指标并添加到汇总报告
                if [ -f "$output_report" ]; then
                    echo "参考图片 $ref_num 评估结果:" >> "$SUMMARY_REPORT"
                    echo "----------------------------------------" >> "$SUMMARY_REPORT"
                    
                    # 提取关键指标（适应新的百分比格式）
                    recall=$(grep "总体召回率" "$output_report" | head -1)
                    effectiveness=$(grep -E "VGGT有效率|模型有效率" "$output_report" | head -1)
                    precision=$(grep "Reference ID映射准确率" "$output_report" | head -1)
                    
                    if [ -n "$recall" ] && [ -n "$effectiveness" ] && [ -n "$precision" ]; then
                        echo "  $recall" >> "$SUMMARY_REPORT"
                        echo "  $effectiveness" >> "$SUMMARY_REPORT"
                        echo "  $precision" >> "$SUMMARY_REPORT"
                        
                        # 存储结果供后续统计
                        all_results+=("$ref_num:$recall:$effectiveness:$precision")
                    else
                        echo "  警告: 无法提取评估指标" >> "$SUMMARY_REPORT"
                    fi
                    
                    echo "" >> "$SUMMARY_REPORT"
                fi
                
            else
                failed_pairs=$((failed_pairs + 1))
                echo -e "${RED}参考图片 $ref_num 评估失败${NC}"
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
echo "  成功率: $(echo "scale=2; $successful_pairs * 100 / $total_pairs" | bc -l)%" >> "$SUMMARY_REPORT"

# 如果有成功的评估结果，计算平均指标
if [ $successful_pairs -gt 0 ]; then
    echo "" >> "$SUMMARY_REPORT"
    echo "平均性能指标:" >> "$SUMMARY_REPORT"
    
    total_precision=0
    total_recall=0
    total_f1=0
    count=0
    
    # 检查数组是否有元素，避免unbound variable错误
    if [ ${#all_results[@]} -gt 0 ]; then
        for result in "${all_results[@]}"; do
            if [[ $result == *"总体召回率"* ]]; then
                # 这里可以添加更复杂的指标提取和计算逻辑
                # 目前只是占位符
                count=$((count + 1))
            fi
        done
    fi
    
    if [ $count -gt 0 ]; then
        echo "  注意: 平均指标计算需要进一步实现" >> "$SUMMARY_REPORT"
    fi
fi

echo "" >> "$SUMMARY_REPORT"
echo "报告生成时间: $(date)" >> "$SUMMARY_REPORT"

# 输出最终统计
echo ""
echo "============================================"
echo -e "${GREEN}批量评估完成！${NC}"
echo "总处理数量: $total_pairs"
echo "成功数量: $successful_pairs"
echo "失败数量: $failed_pairs"
echo ""
echo "结果文件:"
echo "  汇总报告: $SUMMARY_REPORT"
echo "  各图片对详细报告: $RESULT_BASE_DIR/*_to_*.txt"
echo "============================================"

# 返回原始目录
cd "$SCRIPT_DIR"

# 显示汇总报告预览
if [ -f "$SUMMARY_REPORT" ]; then
    echo -e "${BLUE}汇总报告预览:${NC}"
    head -20 "$SUMMARY_REPORT"
    echo "..."
    echo -e "${BLUE}完整报告请查看: $SUMMARY_REPORT${NC}"
fi
