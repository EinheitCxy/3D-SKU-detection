"""  
检出框处理工具模块

包含重合检测、非重合区域计算等功能
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """计算两个检出框的重合比例(IoU)"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算重合区域
    x1_overlap = max(x1_1, x1_2)
    y1_overlap = max(y1_1, y1_2)
    x2_overlap = min(x2_1, x2_2)
    y2_overlap = min(y2_1, y2_2)
    
    # 如果没有重合，返回0
    if x1_overlap >= x2_overlap or y1_overlap >= y2_overlap:
        return 0.0
    
    # 计算重合面积
    overlap_area = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
    
    # 计算两个框的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算IoU
    union_area = area1 + area2 - overlap_area
    if union_area <= 0:
        return 0.0
    
    return overlap_area / union_area


def _get_overlap_region(bbox1: List[float], bbox2: List[float]) -> List[float]:
    """获取两个检出框的重合区域（内部使用）"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算重合区域
    x1_overlap = max(x1_1, x1_2)
    y1_overlap = max(y1_1, y1_2)
    x2_overlap = min(x2_1, x2_2)
    y2_overlap = min(y2_1, y2_2)
    
    return [x1_overlap, y1_overlap, x2_overlap, y2_overlap]


def compute_non_overlap_regions(bbox: List[float], other_bboxes: List[List[float]], 
                               min_overlap_threshold: float = 0.1) -> List[List[float]]:
    """计算一个检出框的非重合区域"""
    x1, y1, x2, y2 = bbox
    
    # 收集所有重合区域
    overlap_regions = []
    for other_bbox in other_bboxes:
        overlap_ratio = calculate_bbox_overlap(bbox, other_bbox)
        if overlap_ratio > min_overlap_threshold:
            overlap_region = _get_overlap_region(bbox, other_bbox)
            if overlap_region[0] < overlap_region[2] and overlap_region[1] < overlap_region[3]:
                overlap_regions.append(overlap_region)
    
    if not overlap_regions:
        # 没有重合，返回原框
        return [bbox]
    
    # 使用简单的区域分割策略
    non_overlap_regions = []
    
    # 策略1：如果重合区域太多或太复杂，采用边缘采样
    if len(overlap_regions) > 3:
        # 只在边缘区域采样，避免复杂的区域分割
        edge_width = min(20, (x2 - x1) * 0.2, (y2 - y1) * 0.2)  # 边缘宽度
        
        # 上边缘
        if y1 + edge_width < y2:
            non_overlap_regions.append([x1, y1, x2, y1 + edge_width])
        
        # 下边缘
        if y2 - edge_width > y1:
            non_overlap_regions.append([x1, y2 - edge_width, x2, y2])
            
        # 左边缘（排除已经包含的角落）
        if x1 + edge_width < x2:
            non_overlap_regions.append([x1, y1 + edge_width, x1 + edge_width, y2 - edge_width])
            
        # 右边缘（排除已经包含的角落）
        if x2 - edge_width > x1:
            non_overlap_regions.append([x2 - edge_width, y1 + edge_width, x2, y2 - edge_width])
    
    else:
        # 策略2：简单区域分割
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 四个象限
        quadrants = [
            [x1, y1, center_x, center_y],  # 左上
            [center_x, y1, x2, center_y],  # 右上
            [x1, center_y, center_x, y2],  # 左下
            [center_x, center_y, x2, y2]   # 右下
        ]
        
        # 检查每个象限是否与重合区域有显著重合
        for quad in quadrants:
            has_significant_overlap = False
            for overlap_region in overlap_regions:
                quad_overlap_ratio = calculate_bbox_overlap(quad, overlap_region)
                if quad_overlap_ratio > 0.5:  # 如果象限与重合区域重合超过50%
                    has_significant_overlap = True
                    break
            
            if not has_significant_overlap:
                # 确保象限有效（面积 > 0）
                if quad[2] > quad[0] and quad[3] > quad[1]:
                    non_overlap_regions.append(quad)
    
    # 过滤掉面积太小的区域
    min_area = 100  # 最小面积阈值
    valid_regions = []
    for region in non_overlap_regions:
        area = (region[2] - region[0]) * (region[3] - region[1])
        if area >= min_area:
            valid_regions.append(region)
    
    # 如果所有区域都被过滤掉了，返回原框的中心区域
    if not valid_regions:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center_size = min((x2 - x1) * 0.3, (y2 - y1) * 0.3)
        center_region = [
            center_x - center_size/2, center_y - center_size/2,
            center_x + center_size/2, center_y + center_size/2
        ]
        valid_regions = [center_region]
        
        logger.debug(f"所有区域被过滤，使用中心区域: {center_region}")
    
    return valid_regions


def analyze_bbox_overlaps(bboxes_info: List[Dict], min_overlap_threshold: float = 0.1) -> Dict:
    """分析检出框重合情况"""
    total_boxes = len(bboxes_info)
    overlap_stats = {
        'total_boxes': total_boxes,
        'overlapping_boxes': set(),
        'overlap_pairs': [],
        'max_overlap': 0.0,
        'avg_overlap': 0.0
    }
    
    if total_boxes < 2:
        return overlap_stats
    
    total_overlap = 0.0
    overlap_count = 0
    
    for i in range(total_boxes):
        for j in range(i + 1, total_boxes):
            bbox1 = bboxes_info[i]['bbox']
            bbox2 = bboxes_info[j]['bbox']
            overlap_ratio = calculate_bbox_overlap(bbox1, bbox2)
            
            if overlap_ratio > min_overlap_threshold:
                overlap_stats['overlap_pairs'].append((i, j, overlap_ratio))
                overlap_stats['overlapping_boxes'].add(i)
                overlap_stats['overlapping_boxes'].add(j)
                overlap_stats['max_overlap'] = max(overlap_stats['max_overlap'], overlap_ratio)
                total_overlap += overlap_ratio
                overlap_count += 1
    
    if overlap_count > 0:
        overlap_stats['avg_overlap'] = total_overlap / overlap_count
    
    overlap_stats['overlapping_boxes'] = list(overlap_stats['overlapping_boxes'])
    
    logger.info(f"检出框重合分析: {len(overlap_stats['overlapping_boxes'])}/{total_boxes} 个框有重合, "
               f"平均重合度: {overlap_stats['avg_overlap']:.3f}, 最大重合度: {overlap_stats['max_overlap']:.3f}")
    
    return overlap_stats