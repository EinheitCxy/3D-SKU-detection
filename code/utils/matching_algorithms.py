"""
SKU匹配系统核心算法模块

3D-2D投影匹配算法
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import SKUMatchingConfig
from .data_utils import extract_bboxes_from_detections
from .geometry_3d import (
    apply_uniqueness_constraint,
    find_best_matching_bbox_with_3d_validation,
    project_3d_to_2d,
    sample_3d_points_from_non_overlap_regions,
    transform_world_to_camera,
)
from .profiling import StageTimer
from .sam3_utils import (
    maybe_run_sam3_for_reference,
    sample_3d_points_from_mask,
)
from .transforms import ImageTransformBase

logger = logging.getLogger(__name__)

# 场景缓存：避免重复从磁盘加载并拷贝到 device
# key 形如 "<npz_path>::<device>"，value 为包含 depth/world_points 等张量的字典
SCENE_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def find_object_correspondences(
    detections: List[Dict],
    images: torch.Tensor,
    config: SKUMatchingConfig,
    reference_image_idx: int = 0,
    transforms_info: Optional[List[ImageTransformBase]] = None,
    image_paths: Optional[List[str]] = None,
    target_indices: Optional[List[int]] = None,
) -> Tuple[Dict[int, List[Dict]], Optional[Dict[int, Dict]]]:
    """查找物体对应关系的主函数

    Args:
        detections: 检测结果列表
        images: 图像张量 (S, C, H, W)
        config: 配置参数
        reference_image_idx: 参考图像索引
        transforms_info: 坐标变换信息

    Returns:
        tuple: (对应关系结果, 物体点映射)
    """
    # 输入验证
    if reference_image_idx >= images.shape[0]:
        raise ValueError(
            f"Reference image index {reference_image_idx} out of range for {images.shape[0]} images"
        )

    if len(detections) != images.shape[0]:
        raise ValueError(
            f"Mismatch: {len(detections)} detections vs {images.shape[0]} images"
        )

    return find_correspondences_3d_mapping(
        detections,
        images,
        config,
        reference_image_idx,
        transforms_info,
        image_paths=image_paths,
        target_indices=target_indices,
    )


def find_correspondences_3d_mapping(
    detections: List[Dict],
    images: torch.Tensor,
    config: SKUMatchingConfig,
    reference_image_idx: int = 0,
    transforms_info: Optional[List[ImageTransformBase]] = None,
    image_paths: Optional[List[str]] = None,
    target_indices: Optional[List[int]] = None,
) -> Tuple[Dict[int, List[Dict]], Optional[Dict[int, Dict]]]:
    """基于3D-2D投影的物体匹配算法"""

    try:
        S = images.shape[0]
        _, _, H, W = images.shape
        device = images.device

        # 验证输入参数
        if reference_image_idx >= S:
            raise ValueError(
                f"Reference image index {reference_image_idx} out of range for {S} images"
            )

        # 1. 全局3D场景重建（从缓存加载预重建数据）
        logger.info(f"使用 {config.backend} 后端进行3D场景重建...")

        # 从缓存加载预先重建的数据
        # 路径: Output/<dataset>/<backend>_cache/predictions.npz
        # output_dir格式: Output/<dataset>/output_3dmapping_<backend>/<ref_idx>
        output_path = Path(config.output_dir)
        cache_path = (
            output_path.parent.parent / f"{config.backend}_cache" / "predictions.npz"
        )

        if not cache_path.exists():
            raise FileNotFoundError(
                f"{config.backend.upper()} 缓存文件不存在: {cache_path}"
            )

        cache_key = f"{str(cache_path)}::{str(device)}"
        scene_data = SCENE_CACHE.get(cache_key)

        if scene_data is None:
            with StageTimer("cache_npz_load"):
                data = np.load(cache_path, allow_pickle=True)

            # 验证必需字段
            required_keys = [
                "depth",
                "depth_conf",
                "world_points",
                "world_points_conf",
                "extrinsic",
                "intrinsic",
            ]
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise ValueError(f"DA3缓存缺少字段: {missing}")

            # 提取数据
            depth_np = data["depth"]
            world_np = data["world_points"]
            extr_np = data["extrinsic"]
            intr_np = data["intrinsic"]
            S_cache, H_cache, W_cache = depth_np.shape[:3]

            if S_cache < S:
                raise ValueError(f"DA3缓存帧数({S_cache})少于当前图像数({S})")

            # 帧对齐：根据image_ids重排数据
            image_ids_cache = data.get("image_ids")
            if image_ids_cache is not None and transforms_info is not None:
                try:
                    desired_ids = [int(getattr(t, "image_id")) for t in transforms_info]
                    id_to_idx = {
                        int(img_id): i for i, img_id in enumerate(image_ids_cache)
                    }
                    index_map = np.array([id_to_idx[img_id] for img_id in desired_ids])
                    depth_np, world_np = depth_np[index_map], world_np[index_map]
                    extr_np, intr_np = extr_np[index_map], intr_np[index_map]
                except (AttributeError, KeyError) as e:
                    logger.warning(f"帧对齐失败，使用原始顺序: {e}")

            # 构建scene_data
            _t_scene = time.perf_counter()
            scene_data = {
                "depth": torch.from_numpy(depth_np).to(device),
                "depth_conf": torch.from_numpy(data["depth_conf"]).to(device),
                "world_points": torch.from_numpy(world_np).to(device),
                "world_points_conf": torch.from_numpy(data["world_points_conf"]).to(
                    device
                ),
                "extrinsic": torch.from_numpy(extr_np).to(device),
                "intrinsic": torch.from_numpy(intr_np).to(device),
            }
            StageTimer.record("scene_data_build", time.perf_counter() - _t_scene)
            SCENE_CACHE[cache_key] = scene_data
            logger.info(
                f"加载{config.backend.upper()}缓存: {cache_path} (S={S_cache}, H={H_cache}, W={W_cache})"
            )
        else:
            logger.info(f"复用 {config.backend.upper()} 场景缓存: {cache_path}")

        # 2. 获取参考图像的检出框
        ref_bboxes = extract_bboxes_from_detections(
            [detections[reference_image_idx]], 0, config
        )
        if not ref_bboxes:
            logger.warning(
                f"No bounding boxes found in reference image {reference_image_idx}"
            )
            return {}, None

        if not transforms_info or reference_image_idx >= len(transforms_info):
            raise ValueError("transforms_info missing")

        ref_transform = transforms_info[reference_image_idx]

        sam_masks_by_obj_id: Dict[int, "np.ndarray"] = {}
        with StageTimer("sam3_mask"):
            masks = maybe_run_sam3_for_reference(
                config=config,
                image_paths=image_paths,
                reference_image_idx=reference_image_idx,
                ref_bboxes_xyxy=[b["bbox"] for b in ref_bboxes],
                transform=ref_transform,
                output_mask_space="final",
            )
        _t_mask_post = time.perf_counter()
        if masks is not None:
            for b, m in zip(ref_bboxes, masks):
                sam_masks_by_obj_id[int(b["object_id"])] = m
        StageTimer.record("mask_postprocess", time.perf_counter() - _t_mask_post)

        correspondences = {}
        points_per_object = {}

        # 构建points_per_object用于可视化
        for bbox_info in ref_bboxes:
            obj_id = bbox_info["object_id"]
            model_bbox = ref_transform.map_bbox_to_final(bbox_info["bbox"])
            points_per_object[obj_id] = {
                "bbox": model_bbox,
                "center": [
                    (model_bbox[0] + model_bbox[2]) / 2,
                    (model_bbox[1] + model_bbox[3]) / 2,
                ],
                "confidence": bbox_info["confidence"],
            }

        # 3. 对每个目标图像进行3D-2D投影匹配（添加唯一性约束和3D几何验证）
        target_set = None
        if target_indices is not None:
            target_set = {int(i) for i in target_indices}
        for target_img_idx, target_detection in enumerate(detections):
            if target_set is not None and target_img_idx not in target_set:
                continue
            if target_img_idx == reference_image_idx:
                continue

            target_bboxes = extract_bboxes_from_detections(
                [target_detection], 0, config
            )
            if not target_bboxes or target_img_idx >= len(transforms_info):
                continue

            target_transform = transforms_info[target_img_idx]

            # 存储所有候选匹配，用于后续优化选择
            candidate_matches = []

            # 对参考图像的每个检出框进行匹配
            for ref_bbox_info in ref_bboxes:
                ref_obj_id = ref_bbox_info["object_id"]

                # 从参考图像的检出框采样3D点（使用非重合区域）
                other_ref_bboxes = [
                    other["bbox"]
                    for other in ref_bboxes
                    if other["object_id"] != ref_obj_id
                ]
                with StageTimer("ref_point_sampling"):
                    if int(ref_obj_id) in sam_masks_by_obj_id:
                        points_3d = sample_3d_points_from_mask(
                            scene_data=scene_data,
                            img_idx=reference_image_idx,
                            mask=sam_masks_by_obj_id[int(ref_obj_id)],
                            transform=ref_transform,
                            config=config,
                            mask_space="final",
                            bbox_xyxy=ref_bbox_info["bbox"],
                        )
                    else:
                        points_3d = sample_3d_points_from_non_overlap_regions(
                            scene_data,
                            reference_image_idx,
                            ref_bbox_info["bbox"],
                            ref_transform,
                            config,
                            other_ref_bboxes,
                        )

                if points_3d is None or len(points_3d) < config.min_3d_sample_points:
                    continue

                # 计算参考3D点的统计信息用于几何验证
                ref_3d_center = points_3d.mean(dim=0)  # (3,)
                # 使用参考相机坐标系的Z作为深度（extrinsic为world->camera）
                _t_w2c = time.perf_counter()
                E = scene_data["extrinsic"][reference_image_idx].to(points_3d.device)
                points_cam = transform_world_to_camera(points_3d, E)
                ref_depth_mean = points_cam[:, 2].mean().item()  # 相机坐标系的Z才是深度
                StageTimer.record("world_to_camera", time.perf_counter() - _t_w2c)

                # 投影到目标图像
                with StageTimer("projection_3d_to_2d"):
                    projected_points = project_3d_to_2d(
                        points_3d,
                        scene_data["extrinsic"][target_img_idx],
                        scene_data["intrinsic"][target_img_idx],
                    )

                _t_proj_post = time.perf_counter()
                if len(projected_points) > 0:
                    px = projected_points[:, 0].float().cpu().numpy()
                    py = projected_points[:, 1].float().cpu().numpy()
                    hits = []
                    for bi, binfo in enumerate(target_bboxes):
                        bx1, by1, bx2, by2 = target_transform.map_bbox_to_final(
                            binfo["bbox"]
                        )
                        cnt = int(
                            (
                                (px >= bx1) & (px <= bx2) & (py >= by1) & (py <= by2)
                            ).sum()
                        )
                        h = cnt / max(len(px), 1)
                        if h > 0.1:
                            hits.append((bi, h, cnt))
                    hits.sort(key=lambda x: -x[1])
                    top3 = hits[:3]
                    logger.debug(
                        f"[DIAG] ref{reference_image_idx} obj{ref_obj_id}: 采样={len(points_3d)} 投影={len(projected_points)} Top3框={[(t[0],f'{t[1]:.0%}',t[2]) for t in top3]}"
                    )

                if len(projected_points) < 5:
                    StageTimer.record(
                        "projection_postprocess", time.perf_counter() - _t_proj_post
                    )
                    continue

                # 将目标图像的检出框映射到模型输入坐标
                target_bboxes_model = []
                for bbox_info in target_bboxes:
                    model_bbox = target_transform.map_bbox_to_final(bbox_info["bbox"])
                    bbox_info_copy = dict(bbox_info)
                    bbox_info_copy["bbox"] = model_bbox
                    target_bboxes_model.append(bbox_info_copy)

                # 性能优化：预筛选候选框，只对Top-K个最有希望的框进行昂贵的3D验证
                # 策略：先快速计算所有框的2D投影命中率，然后只对Top-K进行3D采样和验证
                if len(target_bboxes_model) > config.max_3d_validation_candidates:
                    # 快速计算所有框的2D投影命中率（仅GPU向量化操作，无3D采样）
                    candidate_scores = []
                    for idx, bbox_info in enumerate(target_bboxes_model):
                        bbox = bbox_info["bbox"]
                        x1, y1, x2, y2 = bbox

                        # 计算投影点落入框内的数量（GPU并行）
                        points_in_bbox = (
                            (
                                (projected_points[:, 0] >= x1)
                                & (projected_points[:, 0] <= x2)
                                & (projected_points[:, 1] >= y1)
                                & (projected_points[:, 1] <= y2)
                            )
                            .sum()
                            .item()
                        )

                        match_ratio = points_in_bbox / len(projected_points)
                        candidate_scores.append((idx, match_ratio, bbox_info))

                    # 按命中率降序排序，取Top-K
                    candidate_scores.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = [
                        item[2]
                        for item in candidate_scores[
                            : config.max_3d_validation_candidates
                        ]
                    ]

                    logger.debug(
                        f"3D预筛选: {len(target_bboxes_model)}个候选框 → {len(top_candidates)}个进入3D验证 "
                        f"(Top-{len(top_candidates)}命中率: {[f'{s[1]:.2f}' for s in candidate_scores[:len(top_candidates)]]})"
                    )

                    target_bboxes_for_validation = top_candidates
                else:
                    target_bboxes_for_validation = target_bboxes_model
                StageTimer.record(
                    "projection_postprocess", time.perf_counter() - _t_proj_post
                )

                # 找到最匹配的目标框（仅对预筛选后的候选框进行昂贵的3D验证）
                with StageTimer("target_bbox_match"):
                    best_match = find_best_matching_bbox_with_3d_validation(
                        projected_points,
                        target_bboxes_for_validation,
                        config,
                        scene_data,
                        target_img_idx,
                        target_transform,
                        ref_3d_center,
                        ref_depth_mean,
                        ref_points_3d=points_3d,
                    )

                if best_match:
                    # 添加更多3D验证信息
                    best_match["ref_obj_id"] = ref_obj_id
                    best_match["ref_3d_center"] = ref_3d_center
                    best_match["ref_depth_mean"] = ref_depth_mean
                    candidate_matches.append(best_match)

            # 应用唯一性约束：每个目标框只能匹配一个参考框
            final_matches = apply_uniqueness_constraint(candidate_matches)

            if final_matches:
                matched_objects = []
                for match in final_matches:
                    target_bbox_info = match["target_bbox_info"]
                    original_bbox = target_transform.map_bbox_to_original(
                        target_bbox_info["bbox"]
                    )

                    match_result = {
                        "object_id": match["ref_obj_id"],
                        "target_obj_id": target_bbox_info["object_id"],
                        "box": original_bbox,
                        "model_box": target_bbox_info["bbox"],
                        "correspondence_ratio": match["match_ratio"],
                        "matched_points": match["points_in_bbox"],
                        "total_points": match["total_points"],
                        "confidence": target_bbox_info["confidence"],
                        # 新增3D验证信息
                        "3d_distance": match.get("3d_distance", 0.0),
                    }

                    matched_objects.append(match_result)

                correspondences[target_img_idx] = matched_objects

        matched_targets = len(correspondences)
        logger.info(
            f"3D-2D projection complete. Found correspondences in {matched_targets} images."
        )
        return correspondences, points_per_object

    except (RuntimeError, ValueError, KeyError, IndexError) as e:
        logger.error(f"Failed to find 3D-2D projection correspondences: {e}")
        raise
