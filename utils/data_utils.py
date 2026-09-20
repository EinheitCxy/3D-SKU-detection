"""
SKU匹配系统数据处理模块

包含检测结果加载、边界框提取等数据处理功能
"""

import json
import logging
from typing import Dict, List
from pathlib import Path

from .config import SKUMatchingConfig
from .detection_objects import flatten_detection_objects

# 配置日志
logger = logging.getLogger(__name__)


def load_detections(detection_dir: str, return_index_map: bool = False) -> List[Dict]:
    """按数字文件名加载检测，保留合法空帧并传播坏 JSON 或 schema 错误。"""
    detection_path = Path(detection_dir)
    if not detection_path.is_dir():
        raise FileNotFoundError(f"Detection directory not found: {detection_dir}")
    json_files = sorted(
        (int(path.stem), path)
        for path in detection_path.glob("*.json")
        if path.stem.isdigit()
    )
    if not json_files:
        raise ValueError(f"No valid JSON files found in {detection_dir}")
    indexed_detections = []
    for file_number, file_path in json_files:
        with file_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        objects = flatten_detection_objects(payload)
        indexed_detections.append((file_number, {"objects": objects}))
    logger.info("Loaded %d detection files", len(indexed_detections))
    if return_index_map:
        return indexed_detections
    return [data for _, data in indexed_detections]


def extract_bboxes_from_detections(detections: List[Dict], image_idx: int, config: SKUMatchingConfig) -> List[Dict]:
    """从检测结果中提取边界框
    
    Args:
        detections: 检测结果列表
        image_idx: 图像索引
        config: 配置参数
        
    Returns:
        边界框列表
        
    Raises:
        ValueError: 图像索引超出范围或检测数据无效时抛出
    """
    if not detections:
        raise ValueError("Detections list is empty")
        
    if image_idx >= len(detections):
        raise ValueError(f"Image index {image_idx} out of range (max: {len(detections) - 1})")
    
    detection_data = detections[image_idx]
    if not detection_data:
        raise ValueError(f"Detection data for image {image_idx} is None or empty")
        
    if 'objects' not in detection_data:
        raise ValueError(f"No 'objects' field found in detection data for image {image_idx}")
        
    objects = detection_data['objects']
    if not objects:
        logger.warning(f"No objects found in detection data for image {image_idx}")
        return []
    
    bboxes = []
    total_with_position = 0
    below_conf = 0
    below_area = 0
    for obj_idx, obj in enumerate(objects):
        if 'position' in obj:
            total_with_position += 1
            x1, y1, x2, y2 = obj['position']
            confidence = obj.get('confidences', {}).get('det', 0.0)
            if confidence < config.detection_confidence_threshold:
                below_conf += 1
                continue
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if area < config.min_bbox_area:
                below_area += 1
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
    kept_before_cap = len(bboxes)
    truncated = 0
    if len(bboxes) > config.max_bboxes:
        truncated = len(bboxes) - config.max_bboxes
        bboxes = bboxes[:config.max_bboxes]

    logger.debug(
        "bbox_filter image_idx=%d total=%d with_position=%d below_det_conf=%d below_min_area=%d "
        "kept_before_max=%d truncated_by_max_bboxes=%d kept=%d min_bbox_area=%.1f det_conf_thres=%.2f max_bboxes=%d",
        image_idx,
        len(objects),
        total_with_position,
        below_conf,
        below_area,
        kept_before_cap,
        truncated,
        len(bboxes),
        config.min_bbox_area,
        config.detection_confidence_threshold,
        config.max_bboxes,
    )
    
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
    result = {
        "correspondences": correspondences,
        "reference_points": points_per_object if points_per_object is not None else {},
        "meta": meta or {},
    }
    out_path = Path(config.output_dir) / config.json_filename
    out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved correspondences JSON to '{out_path}'")
    return out_path
