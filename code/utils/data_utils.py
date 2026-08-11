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
    """加载检测结果文件
    
    Args:
        detection_dir: 检测结果目录路径，包含按数字命名的JSON文件(1.json, 2.json, ...)
        return_index_map: 若为True，按 [(file_number, processed_data), ...] 返回，便于对齐图像编号
        
    Returns:
        - 当 return_index_map=False: 检测结果列表，按文件名数字顺序排列
        - 当 return_index_map=True: [(文件编号, 检测结果)] 列表
        
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
        skipped_non_numeric = 0
        for file_path in detection_path.glob("*.json"):
            try:
                # 尝试提取文件名中的数字
                file_number = int(file_path.stem)
                json_files.append((file_number, file_path))
            except ValueError:
                skipped_non_numeric += 1
                logger.debug(f"Skipping non-numeric JSON file: {file_path.name}")
                continue
        
        # 按数字顺序排序
        json_files.sort(key=lambda x: x[0])
        
        if not json_files:
            raise ValueError(f"No valid JSON files found in {detection_dir}")
        
        detections: List[Dict] = []
        detections_with_numbers: List[tuple[int, Dict]] = []
        empty_objects_count = 0
        for file_number, file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_detections = json.load(f)
                
                try:
                    processed_data = {"objects": flatten_detection_objects(file_detections)}
                except ValueError:
                    logger.debug(f"Invalid format in file {file_path.name}, skipping")
                    continue
                
                # 验证处理后的数据是否包含必要字段
                if processed_data and 'objects' in processed_data:
                    # 即使 objects 为空，也保留该检测文件，用于帧对齐
                    detections.append(processed_data)
                    detections_with_numbers.append((file_number, processed_data))
                    if processed_data['objects']:
                        logger.debug(f"Loaded {len(processed_data['objects'])} objects from {file_path.name}")
                    else:
                        logger.debug(f"No objects found in {file_path.name}, keeping empty detection entry")
                        empty_objects_count += 1
                else:
                    logger.debug(f"No 'objects' field found in {file_path.name}, skipping")
                    empty_objects_count += 1
                    continue
                    
            except (FileNotFoundError, json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                logger.error(f"Failed to load detection from {file_path.name}: {e}")
                continue
        
        logger.info(f"Loaded {len(detections)} detection files")
        logger.debug(
            f"load_detections summary: skipped_non_numeric={skipped_non_numeric} empty_objects={empty_objects_count}"
        )
        if return_index_map:
            return detections_with_numbers
        return detections
        
    except (FileNotFoundError, PermissionError) as e:
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
    try:
        result = {
            "correspondences": correspondences,
            "reference_points": points_per_object if points_per_object is not None else {},
            "meta": meta or {},
        }
        out_path = Path(config.output_dir) / config.json_filename
        out_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved correspondences JSON to '{out_path}'")
        return out_path
    except (FileNotFoundError, PermissionError, json.JSONEncodeError, UnicodeEncodeError) as e:
        logger.error(f"Failed to save correspondences JSON: {e}")
        raise
