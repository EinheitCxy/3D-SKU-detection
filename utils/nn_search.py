"""
通用最近邻搜索 nn_search：优先使用 FAISS-GPU，其次 FAISS-CPU，最后回退到 SciPy cKDTree。

统一提供一个入口，避免在各处重复实现 KNN 逻辑。

用法:
    from utils.nn_search import nn_search
    D, I = nn_search(source_points, target_points, k=1)

返回:
    - 当 k == 1: (distances.shape == (M,), indices.shape == (M,))
    - 当 k > 1:  (distances.shape == (M, k), indices.shape == (M, k))
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_shapes(D: np.ndarray, I: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """将 FAISS/KDTree 返回的 D, I 统一到期望形状。

    - k == 1 → 扁平化为 (M,)
    - k > 1  → 确保为 (M, k)
    """
    if k == 1:
        D = D.reshape(-1)
        I = I.reshape(-1)
    else:
        # 保证 2D 形状 (M, k)
        if D.ndim == 1:
            D = D.reshape(-1, 1)
        if I.ndim == 1:
            I = I.reshape(-1, 1)
    return D, I


def nn_search(source_points: np.ndarray, target_points: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """最近邻搜索（FAISS-GPU → FAISS-CPU → KDTree）。

    参数
    -----
    source_points : np.ndarray
        参考点集 (N, D)
    target_points : np.ndarray
        待查询点集 (M, D)
    k : int
        返回最近邻个数，默认 1

    返回
    -----
    distances : np.ndarray
        距离数组，k==1 时形状为 (M,)，否则为 (M, k)
    indices : np.ndarray
        索引数组，k==1 时形状为 (M,)，否则为 (M, k)
    """
    if source_points.size == 0 or target_points.size == 0:
        # 空输入的安全返回
        if k == 1:
            return np.zeros((target_points.shape[0],), dtype=np.float32), np.full((target_points.shape[0],), -1, dtype=np.int64)
        else:
            return (
                np.zeros((target_points.shape[0], k), dtype=np.float32),
                np.full((target_points.shape[0], k), -1, dtype=np.int64),
            )

    # 优先：FAISS（GPU→CPU）
    try:
        import faiss  # type: ignore

        d = int(source_points.shape[1])
        src = source_points.astype(np.float32, copy=False)
        tgt = target_points.astype(np.float32, copy=False)

        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, d)
            logger.info("KNN: Using FAISS-GPU (IndexFlatL2)")
        except (Exception,):  # ImportError/RuntimeError/AttributeError 等统一处理
            index = faiss.IndexFlatL2(d)
            logger.info("KNN: Using FAISS-CPU (IndexFlatL2)")

        index.add(src)
        D, I = index.search(tgt, int(k))
        return _ensure_shapes(D, I, int(k))
    except Exception as e:
        logger.info(f"KNN: FAISS unavailable or failed ({e}); fallback to KDTree")

    # 回退：SciPy KDTree
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(source_points)
        D, I = tree.query(target_points, k=int(k))
        # cKDTree 在 k==1 时返回 1D，在 k>1 时返回 2D；统一一下
        return _ensure_shapes(np.asarray(D), np.asarray(I), int(k))
    except Exception as e:
        # 兜底：返回无效值，但不抛出，避免中断上层流程
        logger.error(f"KNN: KDTree fallback failed: {e}")
        if k == 1:
            return np.zeros((target_points.shape[0],), dtype=np.float32), np.full((target_points.shape[0],), -1, dtype=np.int64)
        else:
            return (
                np.zeros((target_points.shape[0], k), dtype=np.float32),
                np.full((target_points.shape[0], k), -1, dtype=np.int64),
            )

