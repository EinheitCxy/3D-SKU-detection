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


def _subtract_bbox_from_bbox(target_bbox: List[float], subtract_bbox: List[float]) -> List[List[float]]:
    """从目标bbox中减去重叠bbox，返回非重叠的矩形区域（Smart矩形分割算法）

    核心思想：将目标bbox分割为最多4个非重叠矩形（上、下、左、右）。
    这个算法比象限分割更精确，能最大化保留非重叠区域。

    Args:
        target_bbox: 目标bbox [x1, y1, x2, y2]
        subtract_bbox: 要减去的bbox [x1, y1, x2, y2]

    Returns:
        非重叠区域列表（0-4个矩形）

    示例：
        target = [0, 0, 100, 100]
        subtract = [50, 50, 150, 150]  # 右下角重叠
        结果 = [
            [0, 0, 100, 50],   # 上方区域
            [0, 50, 50, 100]   # 左侧区域
        ]
    """
    tx1, ty1, tx2, ty2 = target_bbox
    sx1, sy1, sx2, sy2 = subtract_bbox

    # 计算交集区域
    ix1 = max(tx1, sx1)
    iy1 = max(ty1, sy1)
    ix2 = min(tx2, sx2)
    iy2 = min(ty2, sy2)

    # 如果没有交集，返回原始bbox
    if ix1 >= ix2 or iy1 >= iy2:
        return [target_bbox]

    non_overlap_regions = []

    # 上方区域（如果存在）
    if ty1 < iy1:
        non_overlap_regions.append([tx1, ty1, tx2, iy1])

    # 下方区域（如果存在）
    if iy2 < ty2:
        non_overlap_regions.append([tx1, iy2, tx2, ty2])

    # 左侧区域（如果存在，仅覆盖交集的高度范围）
    if tx1 < ix1:
        non_overlap_regions.append([tx1, iy1, ix1, iy2])

    # 右侧区域（如果存在，仅覆盖交集的高度范围）
    if ix2 < tx2:
        non_overlap_regions.append([ix2, iy1, tx2, iy2])

    return non_overlap_regions


def compute_non_overlap_regions(bbox: List[float], other_bboxes: List[List[float]],
                               min_overlap_threshold: float = 0.1) -> List[List[float]]:
    """计算一个检出框的非重合区域（Smart矩形分割算法）

    使用矩形分割算法，逐步减去重叠区域，保留非重叠部分。
    相比旧版本的象限分割，这个算法：
    1. 更精确：完全保留所有非重叠区域，没有浪费
    2. 更通用：可以处理任意复杂的重叠情况
    3. 更高效：时间复杂度O(n)，n为other_bboxes数量

    Args:
        bbox: 目标bbox [x1, y1, x2, y2]
        other_bboxes: 其他bboxes列表
        min_overlap_threshold: 最小重叠阈值（IoU），低于此值的重叠将被忽略

    Returns:
        非重合区域列表（原始坐标系）

    注意：
        - 输入的bbox应该在同一坐标系中（如果需要transform，调用方负责转换）
        - 返回的区域已过滤掉面积过小的部分（<100像素²）
    """
    if not other_bboxes:
        # 没有其他bbox，整个目标bbox就是非重合区域
        return [bbox]

    # 初始化：当前非重叠区域列表
    current_regions = [bbox]

    # 逐个减去重叠的other_bboxes
    for other_bbox in other_bboxes:
        # 检查是否有显著重叠
        overlap_ratio = calculate_bbox_overlap(bbox, other_bbox)
        if overlap_ratio <= min_overlap_threshold:
            continue  # 重叠不显著，跳过

        new_regions = []
        for region in current_regions:
            # 从当前区域减去other_bbox
            subtracted = _subtract_bbox_from_bbox(region, other_bbox)
            new_regions.extend(subtracted)
        current_regions = new_regions

        # 如果所有区域都被减完了，提前退出
        if not current_regions:
            break

    # 过滤掉面积太小的区域
    min_area = 100  # 最小面积阈值（像素²）
    valid_regions = []
    for region in current_regions:
        x1, y1, x2, y2 = region
        area = (x2 - x1) * (y2 - y1)
        if area >= min_area:
            valid_regions.append(region)

    # 如果所有区域都被过滤掉了，返回原框的中心区域作为降级策略
    if not valid_regions:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # 中心区域大小：取bbox较小边的30%，但不超过50像素
        center_size = min((x2 - x1) * 0.3, (y2 - y1) * 0.3, 50)
        center_region = [
            center_x - center_size/2, center_y - center_size/2,
            center_x + center_size/2, center_y + center_size/2
        ]
        valid_regions = [center_region]

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
