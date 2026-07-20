"""
Global ID assignment for reconstructed point clouds (da3 backend).

This module maps every point in a reconstruction to a global object ID (gid)
using detection results and a global mapping file. It reads the da3 prediction
cache (`da3_cache/predictions.npz`: depth/world_points/extrinsic/intrinsic/
image_ids/source_model; no transforms.json, no GLB):

- Fast path: use precomputed points_with_gid.npz when available.
- Fallback path: use da3_cache/predictions.npz + detections + transforms
  (rebuilt from images via build_transforms(model_type="da3")).

Frame alignment is performed via utils.frame_alignment.ReconstructionDetectionAligner
(works generically for our prediction structure).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.nn_search import nn_search

from .paths import _resolve_da3_cache_dir, resolve_predictions_npz

logger = logging.getLogger(__name__)


def _require_path(p: Optional[Path], desc: str) -> Path:
    if p is None:
        raise FileNotFoundError(f"缺少必需路径: {desc}")
    if not p.exists():
        raise FileNotFoundError(f"{desc} 不存在: {p}")
    logger.info(f"使用 {desc}: {p}")
    return p


def _infer_dataset_root(
    reconstruction_path: Optional[Path],
    global_mapping_path: Optional[Path],
    detection_dir: Optional[Path],
    image_dir: Optional[Path],
) -> Path:
    """推断 da3_cache 所在的 output_dir 基目录（da3 无 GLB，reconstruction 可能为 None）。

    优先级：reconstruction_path.parent > global_mapping.parent.parent >
    detection_dir.parent > image_dir.parent。
    """
    if reconstruction_path is not None:
        return reconstruction_path.parent
    if global_mapping_path is not None and global_mapping_path.exists():
        return global_mapping_path.parent.parent
    if detection_dir is not None and detection_dir.exists():
        return detection_dir.parent
    if image_dir is not None and image_dir.exists():
        return image_dir.parent
    return Path(".")


def assign_global_ids_to_points(
    points: np.ndarray,
    reconstruction_path: Optional[Path],
    global_mapping_path: Path,
    image_dir: Optional[Path],
    detection_dir: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按策略为点云分配 (gid, conf, frame_idx)。

    - 快速路径：<dataset>/da3_cache/points_with_gid.npz
    - 回退路径：<dataset>/da3_cache/predictions.npz + 检测 + 变换
    """
    dataset_root = _infer_dataset_root(
        reconstruction_path, global_mapping_path, detection_dir, image_dir
    )
    da3_cache_dir = _resolve_da3_cache_dir(dataset_root)

    # 1) 优先：预计算的 points_with_gid.npz
    pvg_path = da3_cache_dir / "points_with_gid.npz"
    if pvg_path.exists():
        logger.info(f"使用预计算点云: {pvg_path}")
        data = np.load(pvg_path)
        pre_points = data["points"]  # (M,3)
        pre_gids = data["global_ids"]  # (M,)
        pre_confs = (
            data["confidences"]
            if "confidences" in data
            else np.ones(len(pre_points), dtype=np.float32)
        )
        pre_frame_indices = data["frame_indices"] if "frame_indices" in data else None

        # 最近邻映射（统一使用 utils.nn_search）
        distances, indices = nn_search(pre_points, points, k=1)
        final_gids = pre_gids[indices].astype(np.int32)
        final_confs = pre_confs[indices].astype(np.float32)
        if pre_frame_indices is not None:
            final_frame_indices = pre_frame_indices[indices].astype(np.int32)
        else:
            final_frame_indices = _generate_frame_indices_from_predictions(
                points, dataset_root
            )
        logger.info(
            f"完成快速映射: {len(np.unique(final_gids))} 个唯一ID，平均距离={float(np.mean(distances)):.4f}"
        )
        return final_gids, final_confs, final_frame_indices

    # 2) 回退：在线计算（需要 predictions.npz）
    predictions_cache = resolve_predictions_npz(dataset_root)
    if not predictions_cache.exists():
        raise FileNotFoundError(f"缺少 {predictions_cache}，无法分配全局ID")

    gids, confs, frames = _compute_gids_from_predictions(
        predictions_cache,
        dataset_root,
        points,
        global_mapping_path,
        image_dir,
        detection_dir,
    )
    return gids, confs, frames


def _generate_frame_indices_from_predictions(
    target_points: np.ndarray, dataset_root: Path
) -> np.ndarray:
    """
    为 target_points 生成帧索引。
    """
    predictions_cache = resolve_predictions_npz(dataset_root)
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

    # 统一的最近邻搜索（FAISS→KDTree）
    _, indices = nn_search(flat_points, target_points, k=1)
    return source_frame_ids[indices].astype(np.int32)


def _compute_gids_from_predictions(
    predictions_path: Path,
    dataset_root: Path,
    target_points: np.ndarray,
    global_mapping_path: Path,
    image_dir: Optional[Path],
    detection_dir: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("加载 predictions 缓存...")
    data = np.load(predictions_path, allow_pickle=True)
    world_points = data["world_points"]  # (B,H,W,3) or (H,W,3)
    # da3 缓存用 world_points_conf/depth_conf；Pi3 用 conf。统一兼容。
    if "conf" in data:
        conf = data["conf"]
    elif "world_points_conf" in data:
        conf = data["world_points_conf"]
    elif "depth_conf" in data:
        conf = data["depth_conf"]
    else:
        conf_shape = (
            world_points.shape[:3] if world_points.ndim == 4 else world_points.shape[:2]
        )
        conf = np.ones(conf_shape, dtype=np.float32)

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
    logger.info("验证预测与检测的帧对齐...")
    from utils.frame_alignment import ReconstructionDetectionAligner

    if "image_ids" in data:
        recon_image_ids = data["image_ids"].tolist()
    else:
        recon_image_ids = list(
            range(world_points.shape[0] if world_points.ndim == 4 else 1)
        )

    aligned_pred_data, aligned_detections, alignment_report = (
        ReconstructionDetectionAligner.validate_and_align(
            reconstruction_data={"world_points": world_points, "conf": conf},
            detections=detections,
            detection_indices=detection_indices,
            reconstruction_image_ids=recon_image_ids,
            strict_mode=False,  # 启用自动修复，过滤不匹配的帧
        )
    )

    # 使用对齐后的 3D 预测与检测结果
    world_points = aligned_pred_data.get("world_points", world_points)
    conf = aligned_pred_data.get("conf", conf)
    detections = aligned_detections

    # 反向索引 (image_id, object_id) -> gid
    reverse_mapping: Dict[Tuple[int, int], int] = {}
    for gid_str, instances in global_mapping_data.items():
        for inst in instances:
            reverse_mapping[(inst["image_id"], inst["object_id"])] = int(gid_str)

    aligned_image_ids = alignment_report.get(
        "repaired_image_ids"
    ) or alignment_report.get("common_ids")
    if aligned_image_ids is None:
        aligned_image_ids = detection_indices[: len(aligned_detections)]

    # 模型类型：da3 唯一后端（默认）；若 npz 含 source_model 且含 depth-anything 则确认。
    model_type = "da3"
    try:
        if "source_model" in data:
            sm = str(data["source_model"]).lower()
            if "depth-anything" in sm:
                model_type = "da3"
    except Exception:
        pass

    # 构建 da3 变换（da3_cache 无 transforms.json，从图像重建）
    transforms_info = None
    try:
        from utils.transforms import build_transforms

        search_dir = (
            image_dir
            if (image_dir is not None and image_dir.exists())
            else (dataset_root / "images")
        )
        if search_dir.exists():
            image_paths, found_all = _find_image_paths(aligned_image_ids, search_dir)
            if found_all and image_paths:
                transforms_info = build_transforms(
                    image_paths, model_type=model_type, process_res=504
                )
    except (FileNotFoundError, ImportError) as e:
        logger.debug(f"Failed to build transforms from images: {e}")
        transforms_info = None

    # 从检测框提取 3D 点
    from utils.bbox_3d_extractor import extract_3d_from_bboxes

    extracted_points, extracted_gids, extracted_confs = extract_3d_from_bboxes(
        world_points=world_points,
        world_points_conf=conf,
        detections=aligned_detections,
        reverse_mapping=reverse_mapping,
        image_ids=aligned_image_ids,
        conf_threshold=0.05,
        transforms=transforms_info,
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

    # 将提取点映射到目标点云（优先使用 FAISS-GPU，加速构建期 KNN 映射）
    distances, indices = nn_search(extracted_points, target_points, k=1)
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
    except (ValueError, IndexError) as e:
        # 如果距离计算失败，使用所有点（无过滤）
        logger.warning(f"Distance threshold calculation failed: {e}, using all points")
        final_gids = extracted_gids[indices]
        final_confs = extracted_confs[indices]

    # 帧索引：用平面点构建 KDTree，查询 target_points 对应帧标签
    if world_points.ndim == 4:
        S, H, W, _ = world_points.shape
        world_points_flat = world_points.reshape(-1, 3)
    else:
        H, W, _ = world_points.shape
        S = 1
        world_points_flat = world_points.reshape(-1, 3)

    logger.info(f"   世界坐标点云: {world_points_flat.shape}")

    source_frame_ids = np.repeat(np.arange(S), H * W)
    # 帧索引：用统一 KNN 将 target_points 映射到平面点以获得帧标签
    _, glb_to_pred_indices = nn_search(world_points_flat, target_points, k=1)
    final_frame_indices = source_frame_ids[glb_to_pred_indices].astype(np.int32)
    return final_gids, final_confs, final_frame_indices


def _find_image_paths(
    detection_indices: List[int], search_dir: Path
) -> Tuple[List[str], bool]:
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
