"""
简化与裁剪对齐的 2D 边界框 → 3D 点云提取器

本版本重点：
- 直接支持 VGGT 裁剪与批量填充的坐标映射（可选传入 transforms）
- 修正 (image_id, object_id) 反向映射的 off-by-one（image_id 为 1-based）
- 简化实现：默认不再预分配内存，逻辑更清晰
- 可选每 bbox 自适应置信度阈值与边界内缩
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np

try:
    # 仅在调用方需要启用 VGGT 裁剪映射时导入
    from .transforms import VGGTImageTransform
except Exception:  # pragma: no cover - 运行环境可能未用到
    VGGTImageTransform = None  # type: ignore


logger = logging.getLogger(__name__)


def extract_3d_from_bboxes(
    world_points: np.ndarray,      # (S, H, W, 3) or (H, W, 3)
    world_points_conf: np.ndarray, # (S, H, W) or (H, W)
    detections: List[Dict],        # 按图像顺序的检测结果（与 world_points 对齐）
    reverse_mapping: Dict[Tuple[int, int], int],  # (image_id, object_id) -> global_id（image_id为1-based）
    conf_threshold: float = 0.1,
    use_adaptive_threshold: bool = False,
    adaptive_percentile: float = 10.0,
    bbox_margin: int = 0,
    filter_invalid_ids: bool = True,
    return_stats: bool = False,
    vggt_transforms: Optional[List["VGGTImageTransform"]] = None,  # 若提供则进行裁剪/填充对齐
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, Any]]:
    """从2D检测框提取3D点并赋全局ID（兼容VGGT裁剪/填充）

    核心要点：
    - detections 的顺序必须与 world_points 的第0维一致；若传入 vggt_transforms，其长度也需一致。
    - reverse_mapping 使用 (image_id, object_id) 作为键，其中 image_id 按 1 开始计数。

    Args:
        world_points: VGGT 预测的世界坐标点云。
        world_points_conf: 与 world_points 对齐的置信度图。
        detections: 单张图的检测数据，需包含 objects 列表，每个对象有 position[bbox]。
        reverse_mapping: (image_id(1-based), object_id) -> global_id。
        conf_threshold: 置信度绝对阈值。
        use_adaptive_threshold: 使用每个bbox内部的自适应百分位阈值。
        adaptive_percentile: 自适应阈值百分位数。
        bbox_margin: 在 world_points 空间（最终输入空间）内对 bbox 做内缩像素。
        filter_invalid_ids: 过滤未命中映射的 bbox（global_id == -1）。
        return_stats: 返回统计信息。
        vggt_transforms: 若提供，使用其进行原图 → 最终输入空间（含裁剪/填充）的精确映射。

    Returns:
        - 默认返回三元组 (points_3d, global_ids, confidences)。
        - 若 return_stats=True，返回包含上述字段与统计信息的字典。
    """

    # 统一形状为批量维度
    if world_points.ndim == 3:
        world_points = world_points[np.newaxis, ...]
        world_points_conf = world_points_conf[np.newaxis, ...]

    S, H, W, _ = world_points.shape

    if len(detections) < S:
        logger.warning("detections 张数(%d)少于 world_points 帧数(%d)，仅处理前者", len(detections), S)
        S = len(detections)
    elif len(detections) > S:
        logger.warning("detections 张数(%d)多于 world_points 帧数(%d)，将忽略多余部分", len(detections), S)

    if vggt_transforms is not None and len(vggt_transforms) != S:
        logger.warning(
            "vggt_transforms 数量(%d)与处理帧数(%d)不一致，将按最小长度对齐",
            len(vggt_transforms), S,
        )
        S = min(S, len(vggt_transforms))

    # 统计
    stats: Dict[str, Any] = {
        "total_bboxes": 0,
        "valid_bboxes": 0,
        "skipped_outofbound": 0,
        "skipped_lowconf": 0,
        "skipped_invalid_id": 0,
        "per_bbox": [],
        "per_image": [],
    }

    all_points_list: List[np.ndarray] = []
    all_gids_list: List[np.ndarray] = []
    all_confs_list: List[np.ndarray] = []

    for img_idx in range(S):
        det = detections[img_idx]
        img_stats = {"image_idx": img_idx, "bboxes_count": 0, "valid_bboxes": 0, "points_extracted": 0}

        # 将不同格式统一为“扁平 objects 列表”
        objects = _flatten_objects(det)
        img_stats["bboxes_count"] = len(objects)

        # 对齐 global_mapping 的 object_id：按本图内对象出现顺序从 0 递增
        next_object_id = 0

        for obj in objects:
            stats["total_bboxes"] += 1
            bbox_orig = obj.get("position")  # 原图坐标 [x1, y1, x2, y2]
            if bbox_orig is None or len(bbox_orig) != 4:
                stats["skipped_outofbound"] += 1
                next_object_id += 1
                continue

            # 将 bbox 原图坐标映射到 VGGT 最终输入坐标（包含裁剪与批量填充）
            if vggt_transforms is not None:
                t = vggt_transforms[img_idx]
                x1f, y1f, x2f, y2f = _map_bbox_to_final_with_transform(t, bbox_orig)
            else:
                # 无变换信息时，假定检测已与 world_points 空间对齐
                x1f, y1f, x2f, y2f = bbox_orig

            # 应用边界内缩与裁剪到 [0,W)×[0,H)
            x1i, y1i, x2i, y2i = _shrink_and_clip_bbox(x1f, y1f, x2f, y2f, W, H, bbox_margin)
            if x2i <= x1i or y2i <= y1i:
                stats["skipped_outofbound"] += 1
                next_object_id += 1
                continue

            # 取出该区域的 3D 点与置信度
            pts_3d_patch = world_points[img_idx, y1i:y2i, x1i:x2i]  # (h, w, 3)
            conf_patch = world_points_conf[img_idx, y1i:y2i, x1i:x2i]  # (h, w)
            flat_pts = pts_3d_patch.reshape(-1, 3)
            flat_conf = conf_patch.reshape(-1)

            # 按阈值过滤
            if use_adaptive_threshold and flat_conf.size > 0:
                thr = float(np.percentile(flat_conf, adaptive_percentile))
            else:
                thr = float(conf_threshold)

            valid_mask = (flat_conf > thr) & np.isfinite(flat_pts).all(axis=1)
            if not np.any(valid_mask):
                stats["skipped_lowconf"] += 1
                next_object_id += 1
                continue

            # 反向映射获得 global_id（image_id 为 1-based）
            key = (img_idx + 1, next_object_id)
            gid = reverse_mapping.get(key, -1)
            if filter_invalid_ids and gid == -1:
                stats["skipped_invalid_id"] += 1
                next_object_id += 1
                continue

            valid_points = flat_pts[valid_mask]
            valid_confs = flat_conf[valid_mask]

            all_points_list.append(valid_points)
            all_gids_list.append(np.full(len(valid_points), gid, dtype=np.int32))
            all_confs_list.append(valid_confs.astype(np.float32))

            stats["valid_bboxes"] += 1
            img_stats["valid_bboxes"] += 1
            img_stats["points_extracted"] += int(len(valid_points))

            if return_stats:
                stats["per_bbox"].append({
                    "image_idx": img_idx,
                    "object_id": next_object_id,
                    "global_id": gid,
                    "bbox_final": [int(x1i), int(y1i), int(x2i), int(y2i)],
                    "points_count": int(len(valid_points)),
                    "conf_mean": float(np.mean(valid_confs)),
                    "conf_std": float(np.std(valid_confs)),
                })

            next_object_id += 1

        if return_stats:
            stats["per_image"].append(img_stats)

    if not all_points_list:
        logger.warning("未提取到任何有效3D点")
        return _return_empty(return_stats)

    points_3d = np.vstack(all_points_list)
    global_ids = np.concatenate(all_gids_list)
    confidences = np.concatenate(all_confs_list)

    stats["total_points"] = int(len(points_3d))
    stats["unique_global_ids"] = int(len(np.unique(global_ids)))

    logger.info(
        "提取完成: %d 点, %d 唯一ID, 有效bbox %d/%d",
        stats["total_points"], stats["unique_global_ids"], stats["valid_bboxes"], stats["total_bboxes"],
    )

    if return_stats:
        return {"points": points_3d, "global_ids": global_ids, "confidences": confidences, "stats": stats}
    return points_3d, global_ids, confidences


def _flatten_objects(det_data: Dict) -> List[Dict]:
    """将不同格式的检测结果统一为扁平 objects 列表。"""
    # floor_display2: {"skus": [{"objects": [...]}, ...]}
    if isinstance(det_data, dict) and "skus" in det_data:
        skus = det_data.get("skus", []) or []
        objs: List[Dict] = []
        for sku in skus:
            objs.extend(sku.get("objects", []) or [])
        return objs
    # floor_display1: 顶层即列表，或直接包含 objects 的字典
    if isinstance(det_data, list):
        if len(det_data) == 0:
            return []
        # 约定取第一个元素
        first = det_data[0]
        return first.get("objects", []) if isinstance(first, dict) else []
    if isinstance(det_data, dict):
        return det_data.get("objects", []) or []
    return []


def _map_bbox_to_final_with_transform(t: "VGGTImageTransform", bbox: List[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    xf1, yf1 = t.map_xy_to_final(x1, y1)
    xf2, yf2 = t.map_xy_to_final(x2, y2)
    # 确保 x1<=x2, y1<=y2
    x1f, x2f = (xf1, xf2) if xf1 <= xf2 else (xf2, xf1)
    y1f, y2f = (yf1, yf2) if yf1 <= yf2 else (yf2, yf1)
    return x1f, y1f, x2f, y2f


def _shrink_and_clip_bbox(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int, margin: int
) -> Tuple[int, int, int, int]:
    """对最终输入空间中的 bbox 做内缩并裁剪，再转为整数像素索引。"""
    # 内缩
    x1m = x1 + margin
    y1m = y1 + margin
    x2m = x2 - margin
    y2m = y2 - margin

    # 裁剪
    x1c = max(0.0, min(x1m, W - 1))
    y1c = max(0.0, min(y1m, H - 1))
    x2c = max(0.0, min(x2m, W - 1))
    y2c = max(0.0, min(y2m, H - 1))

    # 采样区域：左上取 floor，右下取 ceil（开区间）
    xi1 = int(np.floor(x1c))
    yi1 = int(np.floor(y1c))
    xi2 = int(np.ceil(x2c))
    yi2 = int(np.ceil(y2c))

    # 保证至少 1 像素宽/高
    if xi2 <= xi1:
        xi2 = min(W, xi1 + 1)
    if yi2 <= yi1:
        yi2 = min(H, yi1 + 1)

    return xi1, yi1, xi2, yi2


def _return_empty(return_stats: bool):
    empty_points = np.zeros((0, 3), dtype=np.float32)
    empty_gids = np.zeros(0, dtype=np.int32)
    empty_confs = np.zeros(0, dtype=np.float32)
    if return_stats:
        return {
            "points": empty_points,
            "global_ids": empty_gids,
            "confidences": empty_confs,
            "stats": {"total_bboxes": 0, "valid_bboxes": 0, "total_points": 0},
        }
    return empty_points, empty_gids, empty_confs
