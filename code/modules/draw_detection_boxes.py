#!/usr/bin/env python3
"""
检出框可视化脚本（统一使用公共数据与可视化模块）
将 detections_results 中的检测结果绘制到对应的图片上
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging

# 添加父目录到路径以便导入utils模块
sys.path.insert(0, str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 统一复用模块能力
from utils import (
    SKUMatchingConfig,
    load_detections,
    extract_bboxes_from_detections,
)
from utils.visualization import draw_bbox_with_id, generate_colors_for_objects

def main():
    parser = argparse.ArgumentParser(description="绘制检出框到图片上")
    parser.add_argument("--image_dir", type=str, default="../imdata/floor_display3/images", 
                       help="图片目录路径")
    parser.add_argument("--detection_dir", type=str, default="../imdata/floor_display3/detections_results",
                       help="检测结果目录路径")
    parser.add_argument("--output_dir", type=str, default="../imdata/floor_display3/imdata_with_bbox",
                       help="输出目录路径 ")
    parser.add_argument("--confidence_threshold", type=float, default=0.3,
                       help="置信度阈值")
    parser.add_argument("--no_confidence", action="store_true",
                       help="不显示置信度信息")
    parser.add_argument("--no_class", action="store_true", 
                       help="不显示类别信息")
    parser.add_argument("--image_format", type=str, default="jpg",
                       help="图片格式 (default: jpg)")
    
    args = parser.parse_args()
    
    # 路径设置
    image_dir = Path(args.image_dir)
    detection_dir = Path(args.detection_dir)
    output_dir = Path(args.output_dir)
    
    # 检查输入目录
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        return
    
    if not detection_dir.exists():
        logger.error(f"Detection directory not found: {detection_dir}")
        return
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # 查找图片文件（按数字排序）
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*.{ext}"))
        image_files.extend(image_dir.glob(f"*.{ext.upper()}"))
    
    # 仅保留数字文件名，避免无法对齐
    numeric_images = []
    for p in image_files:
        try:
            num = int(p.stem)
            numeric_images.append((num, p))
        except ValueError:
            logger.warning(f"Skipping non-numeric image: {p.name}")
    numeric_images.sort(key=lambda x: x[0])
    logger.info(f"Found {len(numeric_images)} numeric images in {image_dir}")
    
    if not image_files:
        logger.error("No images found!")
        return
    
    # 加载检测结果并建立编号映射
    try:
        detections_indexed = load_detections(str(detection_dir), return_index_map=True)
    except Exception as e:
        logger.error(f"Failed to load detections: {e}")
        return

    det_map = {num: det for num, det in detections_indexed}

    # 构建配置（仅使用检测阈值相关项）
    cfg = SKUMatchingConfig(
        detection_confidence_threshold=float(args.confidence_threshold),
        # 其余使用默认值
    )

    # 处理每张图片
    success_count = 0
    total_count = 0
    
    for num, image_path in numeric_images:
        logger.info(f"\n--- Processing {image_path.name} ---")
        total_count += 1

        if num not in det_map:
            logger.warning(f"Detection file not found for image index: {num}")
            continue

        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            continue

        # 提取边界框（使用统一逻辑）
        try:
            bboxes = extract_bboxes_from_detections([det_map[num]], 0, cfg)
        except Exception as e:
            logger.error(f"Failed to extract bboxes for {image_path.name}: {e}")
            continue

        if not bboxes:
            logger.info(f"No boxes above threshold for {image_path.name}")
            continue

        # 为对象生成稳定颜色
        object_ids = [b['object_id'] for b in bboxes]
        colors = generate_colors_for_objects(object_ids)

        # 绘制
        for bx in bboxes:
            color = colors.get(bx['object_id'], (0, 255, 0))
            draw_bbox_with_id(image, bx['bbox'], bx['object_id'], color, thickness=2, font_scale=0.6)
            if not args.no_confidence:
                conf = bx.get('confidence', 0.0)
                x1, y1, x2, y2 = [int(c) for c in bx['bbox']]
                y_text = max(15, y1 - 18)
                cv2.putText(image, f"{conf:.2f}", (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存结果
        output_path = output_dir / f"{image_path.stem}_with_boxes.{args.image_format}"
        if cv2.imwrite(str(output_path), image):
            success_count += 1
        else:
            logger.error(f"Failed to save image to: {output_path}")
    
    # 统计结果
    logger.info(f"\n=== 处理完成 ===")
    logger.info(f"总共处理: {total_count} 张图片")
    logger.info(f"成功处理: {success_count} 张图片") 
    logger.info(f"失败数量: {total_count - success_count} 张图片")
    logger.info(f"输出目录: {output_dir}")
    
    if success_count > 0:
        logger.info(f"\n可以查看输出目录中的可视化结果。")

if __name__ == "__main__":
    main()
