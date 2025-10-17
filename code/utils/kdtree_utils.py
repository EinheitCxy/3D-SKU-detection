"""
KDTree工具函数模块 - 统一的KD树构建和最近邻搜索

本模块提供通用的KDTree操作，避免代码重复
"""

import logging
import numpy as np
from scipy.spatial import cKDTree
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def build_kdtree(points: np.ndarray, leafsize: int = 10) -> cKDTree:
    """
    构建KD-Tree用于快速最近邻搜索

    Args:
        points: 点云坐标数组 (N, 3)
        leafsize: 叶节点大小（优化参数）

    Returns:
        构建好的KD-Tree对象
    """
    if points.ndim != 2 or points.shape[1] not in [2, 3]:
        raise ValueError(f"Points must be (N, 2) or (N, 3), got {points.shape}")

    logger.debug(f"Building KD-Tree with {len(points)} points, leafsize={leafsize}")
    tree = cKDTree(points, leafsize=leafsize)
    logger.debug(f"KD-Tree built successfully")
    return tree


def nearest_neighbor_mapping(
    source_points: np.ndarray,
    target_points: np.ndarray,
    k: int = 1,
    distance_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用KDTree进行最近邻映射

    Args:
        source_points: 源点云 (N, D)
        target_points: 目标点云 (M, D)
        k: 最近邻数量
        distance_threshold: 距离阈值，超过此值的匹配将被丢弃

    Returns:
        (distances, indices): 距离和索引数组
    """
    tree = build_kdtree(source_points)
    distances, indices = tree.query(target_points, k=k)

    if distance_threshold is not None:
        valid_mask = distances < distance_threshold
        logger.info(f"Filtered matches: {valid_mask.sum()}/{len(target_points)} within threshold {distance_threshold}")

    return distances, indices


def query_ball_point(
    tree: cKDTree,
    point: np.ndarray,
    radius: float
) -> np.ndarray:
    """
    查询球形区域内的所有点

    Args:
        tree: KD-Tree对象
        point: 查询点坐标
        radius: 查询半径

    Returns:
        索引数组
    """
    indices = tree.query_ball_point(point, r=radius)
    return np.array(indices)
