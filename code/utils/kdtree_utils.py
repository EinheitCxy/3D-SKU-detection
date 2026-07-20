"""
KDTree 工具函数模块（运行期拾取专用）

说明：
- 构建期/批量最近邻映射（KNN）已统一迁移到 utils.nn_search.nn_search，
  该函数内部提供 FAISS-GPU → FAISS-CPU → SciPy cKDTree 的梯度回退。
- 本模块仅保留运行期交互拾取所需的 KDTree 构建（build_kdtree）。
"""

import logging

import numpy as np
from scipy.spatial import cKDTree

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


# 注意：
# - 最近邻映射/搜索请使用 utils.nn_search.nn_search。
# - 这里仅保留 build_kdtree 以支持 viewer 运行期拾取的快速单点查询。
