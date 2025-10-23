from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.global_id_mapper import GlobalIDMapper
from utils.kdtree_utils import nearest_neighbor_mapping

logger = logging.getLogger(__name__)


def _require_path(p: Optional[Path], desc: str) -> Path:
    if p is None:
        raise FileNotFoundError(f"缺少必需路径: {desc}")
    if not p.exists():
        raise FileNotFoundError(f"{desc} 不存在: {p}")
    logger.info(f"使用 {desc}: {p}")
    return p


def assign_global_ids_to_points(
    points: np.ndarray,
    reconstruction_path: Path,
    global_mapping_path: Path,
    image_dir: Optional[Path],
    detection_dir: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按策略为点云分配 (gid, conf, frame_idx)。

    - 快速路径：<dataset>/vggt_cache/points_with_gid.npz
    - 回退路径：<dataset>/vggt_cache/predictions.npz + 检测 + 变换
    """
    dataset_root = reconstruction_path.parent
    vggt_cache_dir = dataset_root / "vggt_cache"

    # 1) 优先：预计算的 points_with_gid.npz
    pvg_path = vggt_cache_dir / "points_with_gid.npz"
    if pvg_path.exists():
        logger.info(f"使用预计算VGGT点云: {pvg_path}")
        data = np.load(pvg_path)
        pre_points = data["points"]  # (M,3)
        pre_gids = data["global_ids"]  # (M,)
        pre_confs = data["confidences"] if "confidences" in data else np.ones(len(pre_points), dtype=np.float32)
        pre_frame_indices = data["frame_indices"] if "frame_indices" in data else None

        distances, indices = nearest_neighbor_mapping(pre_points, points, k=1)
        final_gids = pre_gids[indices].astype(np.int32)
        final_confs = pre_confs[indices].astype(np.float32)
        if pre_frame_indices is not None:
            final_frame_indices = pre_frame_indices[indices].astype(np.int32)
        else:
            final_frame_indices = _generate_frame_indices_from_predictions(points, reconstruction_path)
        logger.info(
            f"完成快速映射: {len(np.unique(final_gids))} 个唯一ID，平均距离={float(np.mean(distances)):.4f}"
        )
        return final_gids, final_confs, final_frame_indices

    # 2) 回退：在线计算（需要 predictions.npz）
    predictions_cache = vggt_cache_dir / "predictions.npz"
    if not predictions_cache.exists():
        raise FileNotFoundError(f"缺少 {predictions_cache}，无法分配全局ID")

    gids, confs, frames = _compute_gids_from_predictions(
        predictions_cache, dataset_root, points, global_mapping_path, image_dir, detection_dir
    )
    return gids, confs, frames


def _generate_frame_indices_from_predictions(target_points: np.ndarray, reconstruction_path: Path) -> np.ndarray:
    vggt_cache_dir = reconstruction_path.parent / "vggt_cache"
    predictions_cache = vggt_cache_dir / "predictions.npz"
    if not predictions_cache.exists():
        return np.zeros(len(target_points), dtype=np.int32)
    data = np.load(predictions_cache)
    world_points = data["world_points"]  # (S,H,W,3) or (H,W,3)
    if world_points.ndim == 4:
        S, H, W, _ = world_points.shape
    elif world_points.ndim == 3:
        S, H, W = 1, *world_points.shape[:2]
        world_points = world_points[np.newaxis, ...]
    else:
        return np.zeros(len(target_points), dtype=np.int32)

    flat_points = world_points.reshape(-1, 3)
    source_frame_ids = np.repeat(np.arange(S), H * W)
    from scipy.spatial import cKDTree

    tree = cKDTree(flat_points)
    _, indices = tree.query(target_points, k=1)
    return source_frame_ids[indices].astype(np.int32)


def _compute_gids_from_predictions(
    predictions_path: Path,
    dataset_root: Path,
    target_points: np.ndarray,
    global_mapping_path: Path,
    image_dir: Optional[Path],
    detection_dir: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("加载VGGT predictions...")
    data = np.load(predictions_path, allow_pickle=True)
    world_points = data["world_points"]  # (B,H,W,3) or (H,W,3)
    conf = data["conf"]

    # 展平 world_points 统计
    if world_points.ndim == 4:
        B, H, W, _ = world_points.shape
        world_points_flat = world_points.reshape(-1, 3)
    else:
        H, W, _ = world_points.shape
        B = 1
        world_points_flat = world_points.reshape(-1, 3)

    logger.info(f"   世界坐标点云: {world_points_flat.shape}")
    logger.info(f"   目标点云: {target_points.shape}")

    # 加载 global_mapping
    _require_path(global_mapping_path, "global_mapping.json")
    with global_mapping_path.open("r") as f:
        global_mapping_data: Dict[str, Any] = json.load(f)
    logger.info(f"加载 global_mapping: {len(global_mapping_data)} 个全局ID")

    # 加载检测
    det_dir = _require_path(detection_dir, "检测结果目录")
    from utils.data_utils import load_detections

    detections_with_indices = load_detections(str(det_dir), return_index_map=True)
    detection_indices = [item[0] for item in detections_with_indices]
    detections = [item[1] for item in detections_with_indices]
    logger.info(f"加载检测结果: {len(detections)} 张图片")

    # 帧对齐
    logger.info("验证VGGT-Detection帧对齐...")
    from utils.frame_alignment import VGGTDetectionAligner

    vggt_image_ids = None
    if "image_ids" in data:
        vggt_image_ids = data["image_ids"].tolist()
    else:
        vggt_image_ids = list(range(world_points.shape[0] if world_points.ndim == 4 else 1))

    aligned_vggt_data, aligned_detections, alignment_report = VGGTDetectionAligner.validate_and_align(
        vggt_data={"world_points": world_points, "conf": conf},
        detections=detections,
        detection_indices=detection_indices,
        vggt_image_ids=vggt_image_ids,
        strict_mode=False,  # 启用自动修复，过滤不匹配的帧
    )

    # 反向索引 (image_id, object_id) -> gid
    reverse_mapping: Dict[Tuple[int, int], int] = {}
    for gid_str, instances in global_mapping_data.items():
        for inst in instances:
            reverse_mapping[(inst["image_id"], inst["object_id"])] = int(gid_str)

    aligned_image_ids = alignment_report.get("repaired_image_ids") or alignment_report.get("common_ids")
    if aligned_image_ids is None:
        aligned_image_ids = detection_indices[: len(aligned_detections)]

    # 裁剪对齐变换
    vggt_transforms = None
    try:
        from utils.transforms import build_transforms_from_json

        transforms_path = dataset_root / "vggt_cache" / "transforms.json"
        vggt_transforms = build_transforms_from_json(str(transforms_path), aligned_image_ids)
    except Exception:
        vggt_transforms = None

    if vggt_transforms is None:
        try:
            from utils.transforms import build_vggt_transforms

            search_dir = image_dir if (image_dir is not None and image_dir.exists()) else (dataset_root / "images")
            if search_dir.exists():
                image_paths, found_all = _find_image_paths(aligned_image_ids, search_dir)
                if found_all and image_paths:
                    vggt_transforms = build_vggt_transforms(image_paths, target_size=518)
        except Exception:
            vggt_transforms = None

    # 从检测框提取 3D 点
    from utils.bbox_3d_extractor import extract_3d_from_bboxes

    extracted_points, extracted_gids, extracted_confs = extract_3d_from_bboxes(
        world_points=world_points,
        world_points_conf=conf,
        detections=aligned_detections,
        reverse_mapping=reverse_mapping,
        image_ids=aligned_image_ids,
        conf_threshold=0.1,
        vggt_transforms=vggt_transforms,
    )

    logger.info(
        f"高效提取完成: {len(extracted_points)} 个3D点, {len(np.unique(extracted_gids))} 个唯一ID"
    )

    if len(extracted_points) == 0:
        return (
            np.zeros(len(target_points), dtype=np.int32),
            np.zeros(len(target_points), dtype=np.float32),
            np.zeros(len(target_points), dtype=np.int32),
        )

    # 将提取点映射到目标点云
    distances, indices = nearest_neighbor_mapping(extracted_points, target_points, k=1)
    final_gids = np.full(len(target_points), -1, dtype=np.int32)
    final_confs = np.zeros(len(target_points), dtype=np.float32)

    try:
        med = float(np.median(distances)) if len(distances) > 0 else 0.0
        p90 = float(np.percentile(distances, 90)) if len(distances) > 0 else 0.0
        # 使用更严格的阈值，减少bbox外的点被包含
        thr = max(med * 1.5, p90 * 1.0, 0.005)  # 从3.0→1.5, 从1.5→1.0, 从0.01→0.005
        valid = distances <= thr
        final_gids[valid] = extracted_gids[indices[valid]]
        final_confs[valid] = extracted_confs[indices[valid]]
    except Exception:
        final_gids = extracted_gids[indices]
        final_confs = extracted_confs[indices]

    # 帧索引：用 VGGT 平面点构建 KDTree，查询 target_points 对应帧标签
    world_points_flat = world_points.reshape(-1, 3)
    source_frame_ids = np.repeat(np.arange(world_points.shape[0] if world_points.ndim == 4 else 1), H * W)
    _, glb_to_vggt_indices = nearest_neighbor_mapping(world_points_flat, target_points, k=1)
    final_frame_indices = source_frame_ids[glb_to_vggt_indices].astype(np.int32)
    return final_gids, final_confs, final_frame_indices


def _find_image_paths(detection_indices: List[int], search_dir: Path) -> Tuple[List[str], bool]:
    exts = [".JPG", ".JPEG", ".PNG", ".jpg", ".jpeg", ".png"]
    image_paths: List[str] = []
    found_all = True
    for num in detection_indices:
        found = False
        for ext in exts:
            p = search_dir / f"{num}{ext}"
            if p.exists():
                image_paths.append(str(p))
                found = True
                break
        if not found:
            found_all = False
            break
    return image_paths, found_all

