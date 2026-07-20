#!/bin/bash

# SKU匹配批量处理脚本
# 用法: ./batch.sh [floor_display] [max_idx]
# 示例: ./batch.sh floor_display2 4

# 参数设置
FLOOR_DISPLAY="${1:-floor_display2}"
MAX_IDX="${2:-4}"

# 路径变量
ALGORITHM="3d"
IMAGE_FOLDER="../imdata/$FLOOR_DISPLAY/images"
DETECTION_DIR="../imdata/$FLOOR_DISPLAY/detections_results"
OUTPUT_DIR="../imdata/$FLOOR_DISPLAY"

echo "开始批量运行SKU匹配算法"
echo "========================================"
echo "数据集: $FLOOR_DISPLAY"
echo "参考图像: 0 到 $MAX_IDX"
echo "算法: $ALGORITHM"
echo "图像文件夹: $IMAGE_FOLDER"
echo "检测结果目录: $DETECTION_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# CUDA内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 动态检测有效的图像-检测对数量
echo "检测有效图像-检测对..."
VALID_PAIRS=0
for img_file in "$IMAGE_FOLDER"/*; do
    if [ -f "$img_file" ]; then
        filename=$(basename "$img_file")
        basename_no_ext="${filename%.*}"
        if [[ "$basename_no_ext" =~ ^[0-9]+$ ]]; then  # 检查是否为纯数字
            detection_file="$DETECTION_DIR/${basename_no_ext}.json"
            if [ -f "$detection_file" ]; then
                # 检查检测文件是否有有效内容
                if uv run python -c "
import json, sys
try:
    with open('$detection_file', 'r') as f:
        data = json.load(f)
    # 检查不同格式
    valid = False
    if isinstance(data, list) and len(data) > 0 and 'objects' in data[0] and data[0]['objects']:
        valid = True
    elif isinstance(data, dict):
        if 'skus' in data and isinstance(data['skus'], list) and len(data['skus']) > 0:
            if 'objects' in data['skus'][0] and data['skus'][0]['objects']:
                valid = True
        elif 'objects' in data and data['objects']:
            valid = True
    sys.exit(0 if valid else 1)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
                    ((VALID_PAIRS++))
                fi
            fi
        fi
    fi
done

echo "找到 $VALID_PAIRS 个有效的图像-检测对"
ACTUAL_MAX_IDX=$((VALID_PAIRS - 1))

# 调整MAX_IDX
if [ $MAX_IDX -gt $ACTUAL_MAX_IDX ]; then
    echo "⚠️  用户指定MAX_IDX($MAX_IDX)超过有效范围，调整为实际最大索引($ACTUAL_MAX_IDX)"
    MAX_IDX=$ACTUAL_MAX_IDX
elif [ $ACTUAL_MAX_IDX -eq -1 ]; then
    echo "❌ 没有找到有效的图像-检测对"
    exit 1
fi
if [ "$FLOOR_DISPLAY" = "floor_display3" ]; then
    echo ""
    echo "检测到floor_display3数据集，先修复图像方向..."
    echo "========================================"
    
    uv run python process_image_orientation.py --input_dir "$IMAGE_FOLDER"
    
    if [ $? -eq 0 ]; then
        echo "✅ 图像方向修复完成"
    else
        echo "❌ 图像方向修复失败"
        exit 1
    fi
    echo "========================================"
fi

# 批量处理
for i in $(seq 0 $MAX_IDX); do
    echo ""
    echo "处理参考图像 $i ..."
    
    uv run inference.py --algorithm "$ALGORITHM" --reference_idx $i --image_folder "$IMAGE_FOLDER" --detection_dir "$DETECTION_DIR" --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✅ 图像 $i 完成"
    else
        echo "❌ 图像 $i 失败"
    fi
done

echo ""
echo "========================================"
echo "批量处理完成!"
echo "结果保存在: $OUTPUT_DIR/output_3dmapping_da3/"
echo "========================================"