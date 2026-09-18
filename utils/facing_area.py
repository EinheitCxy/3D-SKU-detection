"""牌面面积（front-facing projection area）计算。

给定 DA3 metric 世界点云 + 每 global_id 标注，计算每个商品正对货架面
的物理投影面积（m²）与牌面占比（Share of Shelf）。

核心思想
--------
front-facing 面积 = 物体 3D 点云在"最佳正面方向"上的正交投影凸包面积。
多视图几何事实：从正面看物体投影面积最大，因此对多个候选法向（各相机
视线方向 + SVD 主平面法向 ±）分别算正交投影凸包面积，取最大值即为牌面
面积。

优点：无需显式提取货架基准平面（脆弱且易错），每个商品独立计算，天然
适配多层架、不同朝向；world_points 已是 DA3 多视图统一的世界坐标系，
跨图 union 同一 global_id 的点云后取凸包即可（凸包只取最外圈，重复点
不影响，对噪声鲁棒）。
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError

logger = logging.getLogger(__name__)


def plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """给定法向 n，构造平面内两个正交单位向量 (u, v)。"""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    n = n / norm
    # 选一个与 n 不平行的参考轴，叉积得平面内第一正交基
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    return u, v


def projected_convex_hull_area(points: np.ndarray, normal: np.ndarray) -> float:
    """把 3D 点正交投影到垂直 normal 的平面，返回 2D 凸包面积（单位 = 点坐标单位的平方）。"""
    if points.shape[0] < 3:
        return 0.0
    u, v = plane_basis(normal)
    rel = points - points.mean(axis=0)
    p2d = np.stack([rel @ u, rel @ v], axis=1)
    try:
        hull = ConvexHull(p2d)
        return float(hull.volume)  # 2D 下 ConvexHull.volume == 面积
    except QhullError:
        return 0.0


def svd_main_normal(points: np.ndarray) -> Optional[np.ndarray]:
    """SVD/协方差拟合点云主平面法向（最小特征值方向）。点太少返回 None。"""
    if points.shape[0] < 3:
        return None
    centered = points - points.mean(axis=0)
    try:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        return eigvecs[:, 0]  # eigh 升序，最小特征值对应平面法向
    except np.linalg.LinAlgError:
        return None


def camera_view_normals(extrinsics: np.ndarray) -> List[np.ndarray]:
    """从 w2c 外参 (N,3,4) 提取每个相机在世界系的光轴方向（视线方向）。

    相机系 +z 朝向场景（depth 为正），世界系视线方向 = R^T @ [0,0,1] = R 的第三列。
    """
    normals: List[np.ndarray] = []
    extr = np.asarray(extrinsics, dtype=np.float64)
    if extr.ndim == 2:
        extr = extr[None, ...]
    for i in range(extr.shape[0]):
        R = extr[i, :3, :3]
        view_dir = R[:, 2]  # 第三列 = 相机系 +z 在世界系的方向
        nrm = np.linalg.norm(view_dir)
        if nrm > 1e-12:
            normals.append(view_dir / nrm)
    return normals


def front_facing_area(
    points: np.ndarray,
    global_candidate_normals: List[np.ndarray],
    min_points: int = 3,
) -> Dict:
    """计算单组点云的 front-facing 投影面积（取候选法向中最大投影面积）。

    Args:
        points: (M,3) metric 世界点云（已过滤无效点）
        global_candidate_normals: 全局候选法向（各相机视线方向等）
        min_points: 少于此数返回 0（无法构成凸包）

    Returns:
        {"area_m2": float, "n_points": int, "used_normal_index": int}
    """
    n = int(points.shape[0])
    if n < min_points:
        return {"area_m2": 0.0, "n_points": n, "used_normal_index": -1}

    candidates: List[np.ndarray] = list(global_candidate_normals)
    svd_n = svd_main_normal(points)
    if svd_n is not None:
        candidates.append(svd_n)
        candidates.append(-svd_n)

    best_area, best_idx = 0.0, -1
    for i, nm in enumerate(candidates):
        area = projected_convex_hull_area(points, nm)
        if area > best_area:
            best_area, best_idx = area, i
    return {"area_m2": best_area, "n_points": n, "used_normal_index": best_idx}


def compute_facing_area_report(
    points_3d: np.ndarray,
    global_ids: np.ndarray,
    gid_meta: Dict[int, Dict],
    candidate_normals: List[np.ndarray],
    min_points: int = 3,
) -> Dict:
    """聚合 per-global_id front-facing 面积 + per-sku_class 牌面占比。

    Args:
        points_3d: (M,3) 所有有效点（metric 世界系，跨图 union）
        global_ids: (M,) 每点所属 global_id
        gid_meta: {gid: {"sku_class": str, "n_instances": int}}
            n_instances = 该 global_id 跨图出现的次数（global_mapping 中 entry 数）
        candidate_normals: 全局候选法向（各相机视线方向）
        min_points: 单商品最少点数

    Returns:
        {
          "total_facing_area_m2": float,
          "per_global_id": {gid_str: {area_m2, share, n_points, sku_class, n_instances}},
          "per_sku_class": {class: {area_m2, share, n_global_ids, n_instances}},
          "n_global_ids": int,
          "n_global_ids_with_area": int,
        }
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    global_ids = np.asarray(global_ids)

    per_gid: Dict[int, Dict] = {}
    for gid in np.unique(global_ids):
        gid = int(gid)
        pts = points_3d[global_ids == gid]
        res = front_facing_area(pts, candidate_normals, min_points=min_points)
        meta = gid_meta.get(gid, {})
        per_gid[gid] = {
            "area_m2": res["area_m2"],
            "n_points": int(res["n_points"]),
            "sku_class": meta.get("sku_class", "unknown"),
            "n_instances": int(meta.get("n_instances", 1)),
        }

    total = sum(g["area_m2"] for g in per_gid.values())

    per_gid_out: Dict[str, Dict] = {}
    for gid, info in per_gid.items():
        per_gid_out[str(gid)] = {
            "area_m2": round(info["area_m2"], 6),
            "share": round(info["area_m2"] / total, 6) if total > 0 else 0.0,
            "n_points": info["n_points"],
            "sku_class": info["sku_class"],
            "n_instances": info["n_instances"],
        }

    # 按 sku_class 聚合（客户关心的"某品类占总牌面多少"）
    class_agg: Dict[str, Dict] = {}
    for info in per_gid.values():
        cls = info["sku_class"]
        a = class_agg.setdefault(cls, {"area_m2": 0.0, "n_global_ids": 0, "n_instances": 0})
        a["area_m2"] += info["area_m2"]
        a["n_global_ids"] += 1
        a["n_instances"] += info["n_instances"]
    per_class_out: Dict[str, Dict] = {}
    for cls, a in class_agg.items():
        per_class_out[cls] = {
            "area_m2": round(a["area_m2"], 6),
            "share": round(a["area_m2"] / total, 6) if total > 0 else 0.0,
            "n_global_ids": a["n_global_ids"],
            "n_instances": a["n_instances"],
        }

    n_with_area = sum(1 for g in per_gid.values() if g["area_m2"] > 0)
    return {
        "total_facing_area_m2": round(total, 6),
        "per_global_id": per_gid_out,
        "per_sku_class": per_class_out,
        "n_global_ids": int(len(per_gid)),
        "n_global_ids_with_area": int(n_with_area),
    }
