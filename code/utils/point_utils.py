"""
SKU匹配系统点处理模块

包含点生成、采样等功能
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional

from .config import SKUMatchingConfig
from .bbox_utils import analyze_bbox_overlaps, compute_non_overlap_regions

logger = logging.getLogger(__name__)


def generate_points_from_non_overlap_regions(
    bbox_info: Dict, non_overlap_regions: List[List[float]], 
    image_shape: Tuple[int, int], config: SKUMatchingConfig
) -> np.ndarray:
    """从非重合区域采样点
    
    Args:
        bbox_info: 检出框信息
        non_overlap_regions: 非重合区域列表
        image_shape: 图像形状 (height, width)
        config: 配置参数
        
    Returns:
        采样点数组 (N, 2)
    """
    height, width = image_shape
    object_id = bbox_info['object_id']
    
    if not non_overlap_regions:
        logger.warning(f"对象 {object_id}: 没有非重合区域，跳过采样")
        return np.array([])
    
    all_region_points = []
    total_area = 0
    region_areas = []
    
    # 计算每个区域的面积
    for region in non_overlap_regions:
        x1, y1, x2, y2 = region
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        area = (x2 - x1) * (y2 - y1)
        region_areas.append(area)
        total_area += area
    
    # 只有当总面积小于最小要求面积时才跳过采样
    if total_area < config.min_non_overlap_area:
        logger.warning(f"对象 {object_id}: 总非重合区域面积 ({total_area:.1f}) 小于最小要求 ({config.min_non_overlap_area})，跳过采样")
        return np.array([])
    
    # 根据面积比例分配采样点数
    desired_points = min(int(np.sqrt(total_area) * 2), config.max_points_per_bbox)
    if desired_points < 5:
        desired_points = 5
    
    for i, (region, area) in enumerate(zip(non_overlap_regions, region_areas)):
        if area <= 0:  # 跳过无效区域
            continue
            
        x1, y1, x2, y2 = region
        # 重新确保坐标在图像范围内
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # 按面积比例分配点数
        region_points = max(1, int(desired_points * area / total_area))
        
        # 在区域内采样
        region_width = x2 - x1
        region_height = y2 - y1
        
        # 生成网格点
        grid_cols = max(1, int(np.sqrt(region_points * region_width / region_height)))
        grid_rows = max(1, int(np.sqrt(region_points * region_height / region_width)))
        
        x_points = np.linspace(x1, x2, grid_cols)
        y_points = np.linspace(y1, y2, grid_rows)
        
        xx, yy = np.meshgrid(x_points, y_points)
        xx = xx.flatten()
        yy = yy.flatten()
        
        # 随机选择点以达到 region_points 数量
        if len(xx) > region_points:
            indices = np.random.choice(len(xx), region_points, replace=False)
            xx = xx[indices]
            yy = yy[indices]
        
        region_sample_points = np.stack([xx, yy], axis=-1).astype(np.float32)
        all_region_points.append(region_sample_points)
    
    if not all_region_points:
        return np.array([])
    
    # 合并所有区域的采样点
    all_points = np.concatenate(all_region_points, axis=0)
    
    # 限制最终点数
    if len(all_points) > config.max_points_per_bbox:
        indices = np.random.choice(len(all_points), config.max_points_per_bbox, replace=False)
        all_points = all_points[indices]
    
    logger.debug(f"对象 {object_id}: 从 {len(non_overlap_regions)} 个非重合区域采样了 {len(all_points)} 个点")
    return all_points


def generate_points_from_bboxes(bboxes: List[Dict], image_shape: Tuple[int, int], config: SKUMatchingConfig) -> Tuple[Optional[torch.Tensor], Dict[int, Dict]]:
    """在检测框内采样点作为查询点（支持非重合区域采样）
    
    Args:
        bboxes: 检测框列表
        image_shape: 图像形状 (height, width)
        config: 配置参数
        
    Returns:
        tuple: (查询点张量, 物体点映射字典)
    """
    logger.info(f"Generating query points from {len(bboxes)} bounding boxes...")
    
    # 分析检出框重合情况
    if config.enable_non_overlap_sampling and len(bboxes) > 1:
        overlap_stats = analyze_bbox_overlaps(bboxes, config.overlap_threshold)
        use_non_overlap_sampling = len(overlap_stats['overlapping_boxes']) > 0
        
        if use_non_overlap_sampling:
            logger.info(f"检测到 {len(overlap_stats['overlapping_boxes'])} 个检出框有重合，启用非重合区域采样")
        else:
            logger.info("检出框无重合，使用传统采样方式")
    else:
        use_non_overlap_sampling = False
        logger.info("非重合区域采样未启用，使用传统采样方式")
    
    all_query_points = []
    points_per_object = {}
    height, width = image_shape
    total_points = 0
    
    try:
        for bbox_info in bboxes:
            object_id = bbox_info['object_id']
            
            if use_non_overlap_sampling:
                # 使用非重合区域采样
                other_bboxes = [other['bbox'] for other in bboxes if other['object_id'] != object_id]
                non_overlap_regions = compute_non_overlap_regions(
                    bbox_info['bbox'], other_bboxes, config.overlap_threshold
                )
                
                points = generate_points_from_non_overlap_regions(
                    bbox_info, non_overlap_regions, image_shape, config
                )
                
            else:
                # 传统全框采样
                points = generate_points_from_single_bbox(bbox_info, image_shape, config)
            
            if len(points) == 0:
                logger.warning(f"Invalid bbox or no points generated for object {object_id}, skipping")
                continue
            
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
        
        sampling_method = "非重合区域采样" if use_non_overlap_sampling else "传统采样"
        logger.info(f"Generated {total_points} query points from {len(points_per_object)} objects (使用{sampling_method})")
        return query_points_tensor, points_per_object
        
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Failed to generate points from bounding boxes: {e}")
        raise


def generate_points_from_single_bbox(bbox_info: Dict, image_shape: Tuple[int, int], config: SKUMatchingConfig) -> np.ndarray:
    """从单个检出框采样点（传统方法）
    
    Args:
        bbox_info: 检出框信息
        image_shape: 图像形状 (height, width)
        config: 配置参数
        
    Returns:
        采样点数组 (N, 2)
    """
    x1, y1, x2, y2 = bbox_info['bbox']
    object_id = bbox_info['object_id']
    height, width = image_shape
    
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
        return np.array([])
    
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
    return points