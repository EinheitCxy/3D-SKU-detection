"""
SKU匹配系统数据处理模块

包含检测结果加载、边界框提取等数据处理功能
"""

import json
import logging
from typing import Dict, List
from pathlib import Path

from .config import SKUMatchingConfig

# 配置日志
logger = logging.getLogger(__name__)


def load_detections(detection_dir: str) -> List[Dict]:
    """加载检测结果文件
    
    Args:
        detection_dir: 检测结果目录路径，包含按数字命名的JSON文件(1.json, 2.json, ...)
        
    Returns:
        检测结果列表，按文件名数字顺序排列
        
    Raises:
        FileNotFoundError: 目录不存在时抛出
        ValueError: 文件格式不正确时抛出
    """
    detection_path = Path(detection_dir)
    if not detection_path.exists():
        raise FileNotFoundError(f"Detection directory not found: {detection_dir}")
    
    if not detection_path.is_dir():
        raise ValueError(f"Path is not a directory: {detection_dir}")
    
    try:
        # 获取所有JSON文件并按数字顺序排序
        json_files = []
        for file_path in detection_path.glob("*.json"):
            try:
                # 尝试提取文件名中的数字
                file_number = int(file_path.stem)
                json_files.append((file_number, file_path))
            except ValueError:
                logger.warning(f"Skipping non-numeric JSON file: {file_path.name}")
                continue
        
        # 按数字顺序排序
        json_files.sort(key=lambda x: x[0])
        
        if not json_files:
            raise ValueError(f"No valid JSON files found in {detection_dir}")
        
        detections = []
        for file_number, file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_detections = json.load(f)
                
                # 每个文件包含一个列表，取第一个元素
                if isinstance(file_detections, list) and len(file_detections) > 0:
                    detections.append(file_detections[0])
                elif isinstance(file_detections, dict):
                    detections.append(file_detections)
                else:
                    logger.warning(f"Invalid format in file {file_path.name}, skipping")
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to load detection from {file_path.name}: {e}")
                raise
        
        logger.info(f"Loaded {len(detections)} detection files\n")
        return detections
        
    except Exception as e:
        logger.error(f"Failed to load detections from directory {detection_dir}: {e}")
        raise


def extract_bboxes_from_detections(detections: List[Dict], image_idx: int, config: SKUMatchingConfig) -> List[Dict]:
    """从检测结果中提取边界框
    
    Args:
        detections: 检测结果列表
        image_idx: 图像索引
        config: 配置参数
        
    Returns:
        边界框列表
    """
    if image_idx >= len(detections):
        raise ValueError(f"Image index {image_idx} out of range")
    
    detection_data = detections[image_idx]
    if 'objects' not in detection_data:
        raise ValueError(f"No objects found in detection data")
    
    bboxes = []
    for obj_idx, obj in enumerate(detection_data['objects']):
        if 'position' in obj:
            x1, y1, x2, y2 = obj['position']
            confidence = obj.get('confidences', {}).get('det', 0.0)
            if confidence < config.detection_confidence_threshold:
                continue
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if area < config.min_bbox_area:
                continue
            bbox_info = {
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'confidence': confidence,
                'object_id': obj_idx,
                'area': area
            }
            bboxes.append(bbox_info)
    
    # 按面积排序并限制数量
    bboxes.sort(key=lambda x: x['area'], reverse=True)
    if len(bboxes) > config.max_bboxes:
        bboxes = bboxes[:config.max_bboxes]
    
    return bboxes


def save_correspondences_json(
    correspondences: Dict[int, List[Dict]],
    points_per_object: Dict[int, Dict],
    config: SKUMatchingConfig,
    meta: Dict = None,
) -> Path:
    """将匹配结果保存为 JSON 文件
    
    Args:
        correspondences: 匹配结果
        points_per_object: 参考图像对象点信息
        config: 配置
        meta: 可选元数据（如图像路径列表、时间戳等）
    Returns:
        保存文件路径
    """
    try:
        result = {
            "correspondences": correspondences,
            "reference_points": points_per_object if points_per_object is not None else {},
            "meta": meta or {},
        }
        out_path = Path(config.output_dir) / config.json_filename
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved correspondences JSON to '{out_path}'")
        return out_path
    except Exception as e:
        logger.error(f"Failed to save correspondences JSON: {e}")
        raise