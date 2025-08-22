#!/usr/bin/env python3
"""
检出框可视化脚本
将 imdata/detections_results 中的检测结果绘制到 imdata/total 中对应的图片上
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_detection_results(json_path: str) -> dict:
    """加载检测结果JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检测结果格式：[{...}]，取第一个元素
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        elif isinstance(data, dict):
            return data
        else:
            logger.warning(f"Unexpected data format in {json_path}")
            return {}
            
    except Exception as e:
        logger.error(f"Failed to load {json_path}: {e}")
        return {}

def draw_detection_boxes(image_path: str, detection_data: dict, 
                        output_path: str, confidence_threshold: float = 0.5,
                        show_confidence: bool = True, show_class: bool = True) -> bool:
    """在图片上绘制检出框"""
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return False
    
    # 获取类别映射
    classes = detection_data.get('classes', {}).get('det', [])
    objects = detection_data.get('objects', [])
    
    if not objects:
        logger.warning(f"No objects found in detection data for {image_path}")
        return False
    
    logger.info(f"Drawing {len(objects)} detection boxes on {os.path.basename(image_path)}")
    
    # 为每个检测框生成颜色
    colors = [
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色  
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 黄色
        (128, 0, 128),  # 紫色
        (255, 165, 0),  # 橙色
        (0, 128, 128),  # 青绿色
        (128, 128, 0),  # 橄榄色
    ]
    
    drawn_count = 0
    
    for idx, obj in enumerate(objects):
        confidence = obj.get('confidences', {}).get('det', 0.0)
        
        # 过滤低置信度的检测框
        if confidence < confidence_threshold:
            continue
        
        position = obj.get('position', [])
        if len(position) != 4:
            logger.warning(f"Invalid position data for object {idx}")
            continue
        
        x1, y1, x2, y2 = map(int, position)
        
        # 确保坐标在图像范围内
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # 选择颜色
        color = colors[idx % len(colors)]
        
        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
        
        # 准备标签文本
        label_parts = []
        
        # 添加类别信息
        if show_class and classes:
            class_idx = obj.get('classes', {}).get('det', 0)
            if 0 <= class_idx < len(classes):
                class_name = classes[class_idx].split('^')[-1] if '^' in classes[class_idx] else classes[class_idx]
                label_parts.append(class_name)
        
        # 添加置信度信息
        if show_confidence:
            label_parts.append(f"{confidence:.2f}")
        
        # 添加对象ID
        label_parts.append(f"ID:{idx}")
        
        label = " | ".join(label_parts)
        
        # 绘制标签背景
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
        
        # 绘制标签文本
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        drawn_count += 1
    
    # 保存结果
    success = cv2.imwrite(output_path, image)
    if success:
        logger.info(f"Saved visualization with {drawn_count} boxes to: {output_path}")
        return True
    else:
        logger.error(f"Failed to save image to: {output_path}")
        return False

def main():
    parser = argparse.ArgumentParser(description="绘制检出框到图片上")
    parser.add_argument("--image_dir", type=str, default="../imdata/total", 
                       help="图片目录路径 (default: imdata/total)")
    parser.add_argument("--detection_dir", type=str, default="../imdata/detections_results",
                       help="检测结果目录路径 (default: imdata/detections_results)")
    parser.add_argument("--output_dir", type=str, default="imdata_with_bbox",
                       help="输出目录路径 (default: output_detection_visualization)")
    parser.add_argument("--confidence_threshold", type=float, default=0.3,
                       help="置信度阈值 (default: 0.5)")
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
    
    # 查找图片文件
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f"*.{ext}"))
        image_files.extend(image_dir.glob(f"*.{ext.upper()}"))
    
    image_files = sorted(image_files)
    logger.info(f"Found {len(image_files)} images in {image_dir}")
    
    if not image_files:
        logger.error("No images found!")
        return
    
    # 处理每张图片
    success_count = 0
    total_count = 0
    
    for image_path in image_files:
        # 构造对应的检测结果文件名
        image_name = image_path.stem  # 不带扩展名的文件名
        detection_file = detection_dir / f"{image_name}.json"
        
        logger.info(f"\n--- Processing {image_name} ---")
        
        if not detection_file.exists():
            logger.warning(f"Detection file not found: {detection_file}")
            total_count += 1
            continue
        
        # 加载检测结果
        detection_data = load_detection_results(str(detection_file))
        if not detection_data:
            logger.warning(f"No valid detection data for {image_name}")
            total_count += 1
            continue
        
        # 输出文件路径
        output_path = output_dir / f"{image_name}_with_boxes.{args.image_format}"
        
        # 绘制检出框
        if draw_detection_boxes(
            str(image_path), 
            detection_data, 
            str(output_path),
            confidence_threshold=args.confidence_threshold,
            show_confidence=not args.no_confidence,
            show_class=not args.no_class
        ):
            success_count += 1
        
        total_count += 1
    
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