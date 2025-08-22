"""
SKU匹配系统点处理模块

包含点生成、采样等功能
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional

from .config import SKUMatchingConfig

logger = logging.getLogger(__name__)


def generate_points_from_bboxes(bboxes: List[Dict], image_shape: Tuple[int, int], config: SKUMatchingConfig) -> Tuple[Optional[torch.Tensor], Dict[int, Dict]]:
    """在检测框内随机采样点作为查询点
    
    Args:
        bboxes: 检测框列表
        image_shape: 图像形状 (height, width)
        config: 配置参数
        
    Returns:
        tuple: (查询点张量, 物体点映射字典)
    """
    logger.info(f"Generating query points from {len(bboxes)} bounding boxes...")
    
    all_query_points = []
    points_per_object = {}
    
    height, width = image_shape
    total_points = 0
    
    try:
        for bbox_info in bboxes:
            x1, y1, x2, y2 = bbox_info['bbox']
            object_id = bbox_info['object_id']
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # 在检测框内生成网格点
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # 计算需要的网格密度
            area = bbox_width * bbox_height
            if area == 0:
                logger.warning(f"Invalid bbox area for object {object_id}, skipping")
                continue
            
            # 基于面积计算采样点数，但不超过最大值
            desired_points = min(int(np.sqrt(area) * 2), config.max_points_per_bbox)
            if desired_points < 5:  # 至少5个点
                desired_points = 5
            
            # 计算网格大小
            grid_cols = max(1, int(np.sqrt(desired_points * bbox_width / bbox_height)))
            grid_rows = max(1, int(np.sqrt(desired_points * bbox_height / bbox_width)))
            
            # 生成网格点
            x_points = np.linspace(x1, x2, grid_cols)
            y_points = np.linspace(y1, y2, grid_rows)
            
            # 创建网格
            xx, yy = np.meshgrid(x_points, y_points)
            xx = xx.flatten()
            yy = yy.flatten()
            
            # 随机选择点以达到 desired_points 数量
            if len(xx) > desired_points:
                indices = np.random.choice(len(xx), desired_points, replace=False)
                xx = xx[indices]
                yy = yy[indices]
            
            # 使用numpy操作提高性能
            points = np.stack([xx, yy], axis=-1).astype(np.float32)

            # 全局点数上限控制
            remaining_allowance = config.max_total_points - total_points
            if remaining_allowance <= 0:
                logger.warning("Reached max_total_points limit; stopping further point sampling.")
                break
            if len(points) > remaining_allowance:
                sel_idx = np.random.choice(len(points), remaining_allowance, replace=False)
                points = points[sel_idx]

            all_query_points.append(points)

            num_points = len(points)
            start_idx = total_points
            points_per_object[object_id] = {
                "point_indices": (start_idx, start_idx + num_points),
                "bbox": bbox_info['bbox'],
                "center": bbox_info['center'],
                "confidence": bbox_info['confidence'],
                "area": bbox_info['area'],
                "num_sampled_points": num_points,
                "original_bbox": bbox_info.get('original_bbox', bbox_info['bbox'])
            }
            total_points += num_points
            
        if not all_query_points:
            logger.warning("No query points generated from bounding boxes")
            return None, None
            
        # 一次性转换为torch tensor，减少内存碎片
        all_points_array = np.concatenate(all_query_points, axis=0)
        query_points_tensor = torch.from_numpy(all_points_array).float()
        
        logger.info(f"Generated {total_points} query points from {len(points_per_object)} objects")
        return query_points_tensor, points_per_object
        
    except Exception as e:
        logger.error(f"Failed to generate points from bounding boxes: {e}")
        raise