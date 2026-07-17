"""
SKU匹配系统核心算法模块

包含传统点追踪匹配算法和3D-2D投影匹配算法
"""

import time
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
# VGGT相关导入（路径由 utils/__init__.py 统一配置）
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError as e:
    raise ImportError(f"Failed to import VGGT modules: {e}")

from .config import SKUMatchingConfig
from .data_utils import extract_bboxes_from_detections
from .point_utils import generate_points_from_bboxes
from .geometry_3d import (
    sample_3d_points_from_non_overlap_regions,
    project_3d_to_2d,
    find_best_matching_bbox_with_3d_validation,
    apply_uniqueness_constraint,
    transform_world_to_camera,
)
from .transforms import ImageTransformBase
from .sam3_utils import (
    maybe_run_sam3_for_reference,
    sample_3d_points_from_mask,
    sample_points_from_mask,
)
from .profiling import StageTimer

logger = logging.getLogger(__name__)

# Pi3 场景缓存：避免重复从磁盘加载并拷贝到 device
# key 形如 "<npz_path>::<device>"，value 为包含 depth/world_points 等张量的字典
PI3_SCENE_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}


def _empty_stats(ref_object_id: int, valid_points: int, num_target_bboxes: int) -> Dict:
    """生成空匹配统计结果"""
    return {
        'ref_object_id': ref_object_id, 'valid_points': valid_points,
        'below_min_conf_points': True, 'num_target_bboxes': num_target_bboxes,
        'num_candidates': 0, 'num_below_threshold': 0, 'top_hit_ratio': 0.0, 'produced_matches': 0,
    }


def _bbox_contains(outer: List[float], inner: List[float]) -> bool:
    """检查outer是否严格包含inner（不相等）"""
    return (outer[0] <= inner[0] and outer[1] <= inner[1] and
            outer[2] >= inner[2] and outer[3] >= inner[3] and
            (outer[0], outer[1], outer[2], outer[3]) != (inner[0], inner[1], inner[2], inner[3]))


def _process_single_ref_object(
    ref_object_id: int,
    ref_data: Dict,
    target_bboxes: List[Dict],
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_image_idx: int,
    config: SKUMatchingConfig,
    min_hit_ratio: float = 0.5,
    transforms_info: Optional[List] = None
) -> Tuple[List[Dict], Dict]:
    """处理单个参考对象的匹配计算（用于并行化）
    
    Args:
        ref_object_id: 参考对象ID
        ref_data: 参考对象数据
        target_bboxes: 目标检测框列表
        tracks: 点轨迹张量
        confidence: 置信度张量
        target_image_idx: 目标图像索引
        config: 配置参数
        min_hit_ratio: 最小命中率阈值
        transforms_info: 坐标变换信息
        
    Returns:
        匹配结果列表
    """
    try:
        start_idx, end_idx = ref_data["point_indices"]
        
        # 获取该物体在目标图像中的对应点
        ref_tracks_in_target = tracks[target_image_idx, start_idx:end_idx, :]  # (N_points, 2)
        ref_confidence_in_target = confidence[target_image_idx, start_idx:end_idx]  # (N_points,)
        
        # 过滤置信度高且有效的点
        confident_mask = ref_confidence_in_target > config.confidence_threshold
        valid_points = ref_tracks_in_target[confident_mask]
        valid_points_count = int(valid_points.shape[0]) if valid_points.ndim == 2 else int(valid_points.numel() // 2)

        if valid_points.numel() == 0:
            return [], _empty_stats(ref_object_id, 0, len(target_bboxes))
            
        # 过滤非有限值
        finite_mask = torch.isfinite(valid_points).all(dim=1)
        valid_points = valid_points[finite_mask]
        valid_points_count = int(valid_points.shape[0])

        if len(valid_points) == 0:
            return [], _empty_stats(ref_object_id, 0, len(target_bboxes))
            
        # 检查是否达到最小置信点数要求
        if len(valid_points) < config.min_confident_points:
            logger.debug(f"参考对象 {ref_object_id}: 只有 {len(valid_points)} 个置信点，低于最小值 {config.min_confident_points}")
            return [], _empty_stats(ref_object_id, valid_points_count, len(target_bboxes))
        
        # 收集所有符合条件的匹配（向量化点落框统计）
        all_candidates = []

        top_hit_ratio = 0.0
        if len(target_bboxes) > 0:
            # 组装 boxes 张量 [M, 4]
            boxes_list = [tb['bbox'] for tb in target_bboxes]
            boxes = torch.as_tensor(boxes_list, dtype=valid_points.dtype, device=valid_points.device)

            # 点坐标 [N]
            X = valid_points[:, 0]
            Y = valid_points[:, 1]

            # 框坐标 [M]
            X1 = boxes[:, 0]
            Y1 = boxes[:, 1]
            X2 = boxes[:, 2]
            Y2 = boxes[:, 3]

            # 广播判断 [M, N]
            in_x = (X1[:, None] <= X[None, :]) & (X[None, :] <= X2[:, None])
            in_y = (Y1[:, None] <= Y[None, :]) & (Y[None, :] <= Y2[:, None])
            in_mask = in_x & in_y

            # 每个框命中点数与比例 [M]
            counts = in_mask.sum(dim=1)
            total_pts = max(1, len(valid_points))
            ratios = counts.float() / float(total_pts)

            # 保留满足阈值的框索引
            keep_mask = ratios >= min_hit_ratio
            kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten().tolist()
            below_mask = (ratios > 0) & (ratios < min_hit_ratio)
            num_below_threshold = int(below_mask.sum().item())

            for idx in kept_indices:
                target_bbox_info = target_bboxes[idx]
                vggt_box = boxes[idx].tolist()

                # 将VGGT坐标映射回原图坐标
                if transforms_info and target_image_idx < len(transforms_info):
                    original_bbox = transforms_info[target_image_idx].map_bbox_to_original(vggt_box)
                else:
                    original_bbox = vggt_box

                overlap_ratio = float(ratios[idx].item())
                if overlap_ratio > top_hit_ratio:
                    top_hit_ratio = overlap_ratio
                points_in_bbox = int(counts[idx].item())

                logger.debug(
                    f"目标框 {target_bbox_info['object_id']}: {points_in_bbox}/{total_pts} 点在内 ({overlap_ratio:.3f})"
                )

                match = {
                    'object_id': ref_object_id,
                    'target_obj_id': target_bbox_info['object_id'],
                    'box': original_bbox,
                    'vggt_box': vggt_box,
                    'correspondence_ratio': overlap_ratio,
                    'matched_points': points_in_bbox,
                    'total_points': total_pts,
                    'target_confidence': target_bbox_info.get('confidence', 0.0),
                    'reference_confidence': ref_data['confidence']
                }
                all_candidates.append(match)

        # 去重逻辑 - 移除包含其他框的较大框
        if len(all_candidates) > 1:
            to_remove = set()
            for i in range(len(all_candidates)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(all_candidates)):
                    if j in to_remove:
                        continue
                    bbox_i, bbox_j = all_candidates[i]['vggt_box'], all_candidates[j]['vggt_box']
                    if _bbox_contains(bbox_i, bbox_j):
                        to_remove.add(i)
                        break
                    elif _bbox_contains(bbox_j, bbox_i):
                        to_remove.add(j)
            filtered_candidates = [m for i, m in enumerate(all_candidates) if i not in to_remove]
        else:
            filtered_candidates = all_candidates

        # 按overlap_ratio降序排序，取前2个最好的匹配
        filtered_candidates.sort(key=lambda x: x['correspondence_ratio'], reverse=True)
        matches = filtered_candidates[:2]

        return matches, {
            'ref_object_id': ref_object_id,
            'valid_points': valid_points_count,
            'below_min_conf_points': False,
            'num_target_bboxes': len(target_bboxes),
            'num_candidates': len(all_candidates),
            'num_below_threshold': int(num_below_threshold) if len(target_bboxes) > 0 else 0,
            'top_hit_ratio': float(top_hit_ratio),
            'produced_matches': len(matches),
        }
        
    except (KeyError, IndexError, ValueError, AttributeError) as e:
        logger.error(f"处理参考对象 {ref_object_id} 失败: {e}")
        return [], _empty_stats(ref_object_id, 0, len(target_bboxes))


def find_object_correspondences(
    vggt_model: VGGT,
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
        vggt_model: VGGT模型
        detections: 检测结果列表
        images: 图像张量 (S, C, H, W)
        config: 配置参数
        reference_image_idx: 参考图像索引
        transforms_info: 坐标变换信息

    Returns:
        tuple: (对应关系结果, 物体点映射)
    """
    algorithm_name = config.get_algorithm_name()
    
    # 输入验证
    if reference_image_idx >= images.shape[0]:
        raise ValueError(f"Reference image index {reference_image_idx} out of range for {images.shape[0]} images")
    
    if len(detections) != images.shape[0]:
        raise ValueError(f"Mismatch: {len(detections)} detections vs {images.shape[0]} images")
    
    # 根据配置选择匹配算法
    if config.enable_3d_mapping:
        return find_correspondences_3d_mapping(
            vggt_model,
            detections,
            images,
            config,
            reference_image_idx,
            transforms_info,
            image_paths=image_paths,
            target_indices=target_indices,
        )
    else:
        return find_correspondences_point_tracking(
            vggt_model,
            detections,
            images,
            config,
            reference_image_idx,
            transforms_info,
            image_paths=image_paths,
            target_indices=target_indices,
        )


def find_correspondences_3d_mapping(
    vggt_model: VGGT,
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
            raise ValueError(f"Reference image index {reference_image_idx} out of range for {S} images")

        # 1. 全局3D场景重建（根据backend选择数据源）
        logger.info(f"使用 {config.backend} 后端进行3D场景重建...")

        if config.backend in ("pi3", "da3"):
            # 从缓存加载预先重建的数据
            # 路径: Output/<dataset>/<backend>_cache/predictions.npz
            # output_dir格式: Output/<dataset>/output_3dmapping_<backend>/<ref_idx>
            output_path = Path(config.output_dir)
            cache_path = output_path.parent.parent / f"{config.backend}_cache" / "predictions.npz"

            if not cache_path.exists():
                raise FileNotFoundError(f"{config.backend.upper()} 缓存文件不存在: {cache_path}")

            cache_key = f"{str(cache_path)}::{str(device)}"
            scene_data = PI3_SCENE_CACHE.get(cache_key)

            if scene_data is None:
                with StageTimer("cache_npz_load"):
                    data = np.load(cache_path, allow_pickle=True)

                # 验证必需字段
                required_keys = ["depth", "depth_conf", "world_points", "world_points_conf", "extrinsic", "intrinsic"]
                missing = [k for k in required_keys if k not in data]
                if missing:
                    raise ValueError(f"Pi3缓存缺少字段: {missing}")

                # 提取数据
                depth_np = data["depth"]
                world_np = data["world_points"]
                extr_np = data["extrinsic"]
                intr_np = data["intrinsic"]
                S_cache, H_pi3, W_pi3 = depth_np.shape[:3]

                if S_cache < S:
                    raise ValueError(f"Pi3缓存帧数({S_cache})少于当前图像数({S})")

                # 帧对齐：根据image_ids重排数据
                image_ids_cache = data.get("image_ids")
                if image_ids_cache is not None and transforms_info is not None:
                    try:
                        desired_ids = [int(getattr(t, "image_id")) for t in transforms_info]
                        id_to_idx = {int(img_id): i for i, img_id in enumerate(image_ids_cache)}
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
                    "world_points_conf": torch.from_numpy(data["world_points_conf"]).to(device),
                    "extrinsic": torch.from_numpy(extr_np).to(device),
                    "intrinsic": torch.from_numpy(intr_np).to(device),
                }
                StageTimer.record("scene_data_build", time.perf_counter() - _t_scene)
                PI3_SCENE_CACHE[cache_key] = scene_data
                logger.info(f"加载{config.backend.upper()}缓存: {cache_path} (S={S_cache}, H={H_pi3}, W={W_pi3})")
            else:
                logger.info(f"复用 {config.backend.upper()} 场景缓存: {cache_path}")

        else:  # backend == "vggt"
            # 原有VGGT逻辑
            logger.info("Performing global 3D scene reconstruction...")
            with torch.no_grad():
                predictions = vggt_model(images)  # 不提供query_points

            # 转换姿态编码为相机参数
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                images.shape[-2:]
            )

            scene_data = {
                'depth': predictions["depth"].squeeze(0),  # (S, H, W, 1)
                'depth_conf': predictions["depth_conf"].squeeze(0),  # (S, H, W)
                'world_points': predictions["world_points"].squeeze(0),  # (S, H, W, 3)
                'world_points_conf': predictions["world_points_conf"].squeeze(0),  # (S, H, W)
                'extrinsic': extrinsic.squeeze(0),  # (S, 4, 4)
                'intrinsic': intrinsic.squeeze(0),  # (S, 3, 3)
            }
            logger.info("Global 3D scene reconstruction complete")
        
        # 2. 获取参考图像的检出框
        ref_bboxes = extract_bboxes_from_detections([detections[reference_image_idx]], 0, config)
        if not ref_bboxes:
            logger.warning(f"No bounding boxes found in reference image {reference_image_idx}")
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
            obj_id = bbox_info['object_id']
            vggt_bbox = ref_transform.map_bbox_to_final(bbox_info['bbox'])
            points_per_object[obj_id] = {
                'bbox': vggt_bbox,
                'center': [(vggt_bbox[0] + vggt_bbox[2]) / 2, (vggt_bbox[1] + vggt_bbox[3]) / 2],
                'confidence': bbox_info['confidence']
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
                
            target_bboxes = extract_bboxes_from_detections([target_detection], 0, config)
            if not target_bboxes or target_img_idx >= len(transforms_info):
                continue
                
            target_transform = transforms_info[target_img_idx]
            
            # 存储所有候选匹配，用于后续优化选择
            candidate_matches = []
            
            # 对参考图像的每个检出框进行匹配
            for ref_bbox_info in ref_bboxes:
                ref_obj_id = ref_bbox_info['object_id']
                
                # 从参考图像的检出框采样3D点（使用非重合区域）
                other_ref_bboxes = [other['bbox'] for other in ref_bboxes if other['object_id'] != ref_obj_id]
                with StageTimer("ref_point_sampling"):
                    if int(ref_obj_id) in sam_masks_by_obj_id:
                        points_3d = sample_3d_points_from_mask(
                            scene_data=scene_data,
                            img_idx=reference_image_idx,
                            mask=sam_masks_by_obj_id[int(ref_obj_id)],
                            transform=ref_transform,
                            config=config,
                            mask_space="final",
                            bbox_xyxy=ref_bbox_info['bbox'],
                        )
                    else:
                        points_3d = sample_3d_points_from_non_overlap_regions(
                            scene_data, reference_image_idx, ref_bbox_info['bbox'],
                            ref_transform, config, other_ref_bboxes
                        )
                
                if points_3d is None or len(points_3d) < config.min_3d_sample_points:
                    continue

                # 计算参考3D点的统计信息用于几何验证
                ref_3d_center = points_3d.mean(dim=0)  # (3,)
                # 使用参考相机坐标系的Z作为深度（extrinsic为world->camera）
                _t_w2c = time.perf_counter()
                E = scene_data['extrinsic'][reference_image_idx].to(points_3d.device)
                points_cam = transform_world_to_camera(points_3d, E)
                ref_depth_mean = points_cam[:, 2].mean().item()  # 相机坐标系的Z才是深度
                StageTimer.record("world_to_camera", time.perf_counter() - _t_w2c)

                # 投影到目标图像
                with StageTimer("projection_3d_to_2d"):
                    projected_points = project_3d_to_2d(
                        points_3d,
                        scene_data['extrinsic'][target_img_idx],
                        scene_data['intrinsic'][target_img_idx]
                    )

                _t_proj_post = time.perf_counter()
                if len(projected_points) > 0:
                    px = projected_points[:, 0].float().cpu().numpy()
                    py = projected_points[:, 1].float().cpu().numpy()
                    hits=[]
                    for bi,binfo in enumerate(target_bboxes):
                        bx1,by1,bx2,by2=target_transform.map_bbox_to_final(binfo['bbox'])
                        cnt=int(((px>=bx1)&(px<=bx2)&(py>=by1)&(py<=by2)).sum())
                        h=cnt/max(len(px),1)
                        if h>0.1: hits.append((bi,h,cnt))
                    hits.sort(key=lambda x:-x[1])
                    top3=hits[:3]
                    logger.debug(f"[DIAG] ref{reference_image_idx} obj{ref_obj_id}: 采样={len(points_3d)} 投影={len(projected_points)} Top3框={[(t[0],f'{t[1]:.0%}',t[2]) for t in top3]}")

                if len(projected_points) < 5:
                    StageTimer.record("projection_postprocess", time.perf_counter() - _t_proj_post)
                    continue

                # 将目标图像的检出框映射到VGGT坐标
                target_bboxes_vggt = []
                for bbox_info in target_bboxes:
                    vggt_bbox = target_transform.map_bbox_to_final(bbox_info['bbox'])
                    bbox_info_copy = dict(bbox_info)
                    bbox_info_copy['bbox'] = vggt_bbox
                    target_bboxes_vggt.append(bbox_info_copy)

                # 性能优化：预筛选候选框，只对Top-K个最有希望的框进行昂贵的3D验证
                # 策略：先快速计算所有框的2D投影命中率，然后只对Top-K进行3D采样和验证
                if len(target_bboxes_vggt) > config.max_3d_validation_candidates:
                    # 快速计算所有框的2D投影命中率（仅GPU向量化操作，无3D采样）
                    candidate_scores = []
                    for idx, bbox_info in enumerate(target_bboxes_vggt):
                        bbox = bbox_info['bbox']
                        x1, y1, x2, y2 = bbox

                        # 计算投影点落入框内的数量（GPU并行）
                        points_in_bbox = (
                            (projected_points[:, 0] >= x1) &
                            (projected_points[:, 0] <= x2) &
                            (projected_points[:, 1] >= y1) &
                            (projected_points[:, 1] <= y2)
                        ).sum().item()

                        match_ratio = points_in_bbox / len(projected_points)
                        candidate_scores.append((idx, match_ratio, bbox_info))

                    # 按命中率降序排序，取Top-K
                    candidate_scores.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = [item[2] for item in candidate_scores[:config.max_3d_validation_candidates]]

                    logger.debug(
                        f"3D预筛选: {len(target_bboxes_vggt)}个候选框 → {len(top_candidates)}个进入3D验证 "
                        f"(Top-{len(top_candidates)}命中率: {[f'{s[1]:.2f}' for s in candidate_scores[:len(top_candidates)]]})"
                    )

                    target_bboxes_for_validation = top_candidates
                else:
                    target_bboxes_for_validation = target_bboxes_vggt
                StageTimer.record("projection_postprocess", time.perf_counter() - _t_proj_post)

                # 找到最匹配的目标框（仅对预筛选后的候选框进行昂贵的3D验证）
                with StageTimer("target_bbox_match"):
                    best_match = find_best_matching_bbox_with_3d_validation(
                        projected_points, target_bboxes_for_validation, config,
                        scene_data, target_img_idx, target_transform,
                        ref_3d_center, ref_depth_mean, ref_points_3d=points_3d
                    )
                
                if best_match:
                    # 添加更多3D验证信息
                    best_match['ref_obj_id'] = ref_obj_id
                    best_match['ref_3d_center'] = ref_3d_center
                    best_match['ref_depth_mean'] = ref_depth_mean
                    candidate_matches.append(best_match)
            
            # 应用唯一性约束：每个目标框只能匹配一个参考框
            final_matches = apply_uniqueness_constraint(candidate_matches)
            
            if final_matches:
                matched_objects = []
                for match in final_matches:
                    target_bbox_info = match['target_bbox_info']
                    original_bbox = target_transform.map_bbox_to_original(target_bbox_info['bbox'])
                    
                    match_result = {
                        'object_id': match['ref_obj_id'],
                        'target_obj_id': target_bbox_info['object_id'],
                        'box': original_bbox,
                        'vggt_box': target_bbox_info['bbox'],
                        'correspondence_ratio': match['match_ratio'],
                        'matched_points': match['points_in_bbox'],
                        'total_points': match['total_points'],
                        'confidence': target_bbox_info['confidence'],
                        # 新增3D验证信息
                        '3d_distance': match.get('3d_distance', 0.0),
                    }
                    
                    matched_objects.append(match_result)
                
                correspondences[target_img_idx] = matched_objects

        matched_targets = len(correspondences)
        logger.info(f"3D-2D projection complete. Found correspondences in {matched_targets} images.")
        return correspondences, points_per_object
        
    except (RuntimeError, ValueError, KeyError, IndexError) as e:
        logger.error(f"Failed to find 3D-2D projection correspondences: {e}")
        raise


def find_correspondences_point_tracking(
    vggt_model: VGGT,
    detections: List[Dict],
    images: torch.Tensor,
    config: SKUMatchingConfig,
    reference_image_idx: int = 0,
    transforms_info: Optional[List[ImageTransformBase]] = None,
    image_paths: Optional[List[str]] = None,
    target_indices: Optional[List[int]] = None,
) -> Tuple[Dict[int, List[Dict]], Optional[Dict[int, Dict]]]:
    """基于点追踪的物体匹配算法"""
    
    try:
        S = images.shape[0]
        _, _, H, W = images.shape
        device = images.device
        
        # 验证输入参数
        if reference_image_idx >= S:
            raise ValueError(f"Reference image index {reference_image_idx} out of range for {S} images")
        
        # 1. 从检测结果中提取参考图像的边界框
        logger.info(f"Processing reference image {reference_image_idx}")
        
        if reference_image_idx >= len(detections):
            raise ValueError(f"Reference image index {reference_image_idx} out of range for {len(detections)} detections")
        
        ref_bboxes = extract_bboxes_from_detections(detections, reference_image_idx, config)

        if not ref_bboxes:
            logger.warning(f"No bounding boxes found in reference image {reference_image_idx}")
            return {}, None

        # 2. 使用transforms_info映射坐标
        if transforms_info is None or not (0 <= reference_image_idx < len(transforms_info)):
            logger.warning("transforms_info missing; falling back to VGGT input size for original size.")
            raise ValueError("transforms_info missing")
        else:
            ref_transform = transforms_info[reference_image_idx]
            orig_h = int(ref_transform.orig_height)
            orig_w = int(ref_transform.orig_width)
        logger.info(f"Original image size: {orig_w}x{orig_h}, VGGT input size: {W}x{H}")

        # 3. 映射边界框到模型坐标空间
        mapped_bboxes = []
        for b in ref_bboxes:
            mapped = ref_transform.map_bbox_to_final(b['bbox'])
            b2 = dict(b)
            b2['original_bbox'] = b['bbox']
            b2['bbox'] = mapped
            b2['center'] = [(mapped[0] + mapped[2]) / 2, (mapped[1] + mapped[3]) / 2]
            b2['area'] = max(0.0, (mapped[2] - mapped[0]) * (mapped[3] - mapped[1]))
            mapped_bboxes.append(b2)

        # 4. 生成查询点：启用 SAM3 时从 mask 内采样，否则沿用 bbox 采样
        all_query_points_tensor = None
        points_per_object = None
        masks = maybe_run_sam3_for_reference(
            config=config,
            image_paths=image_paths,
            reference_image_idx=reference_image_idx,
            ref_bboxes_xyxy=[b["bbox"] for b in ref_bboxes],
            transform=ref_transform,
            output_mask_space="final",
        )
        if masks is not None:
            all_pts = []
            points_per_object = {}
            total_points = 0
            for b_orig, b_mapped, m in zip(ref_bboxes, mapped_bboxes, masks):
                obj_id = int(b_orig["object_id"])
                # masks 已经是 final 空间，使用 mapped bbox 坐标
                # 混合采样：SAM3 mask + 高斯加权
                use_gaussian = config.enable_gaussian_sampling and config.enable_gaussian_in_sam3_mask
                pts_final = sample_points_from_mask(
                    m,
                    max_points=int(config.max_points_per_bbox),
                    enable_gaussian=use_gaussian,
                    gaussian_sigma=config.gaussian_sigma,
                    gaussian_truncate=config.gaussian_truncate,
                    bbox_xyxy=b_mapped["bbox"],
                )
                if pts_final.shape[0] == 0:
                    continue
                all_pts.append(pts_final)
                n = int(pts_final.shape[0])
                points_per_object[obj_id] = {
                    "point_indices": (total_points, total_points + n),
                    "bbox": b_mapped["bbox"],
                    "center": b_mapped["center"],
                    "confidence": b_mapped["confidence"],
                    "area": b_mapped["area"],
                    "num_sampled_points": n,
                    "original_bbox": b_orig["bbox"],
                }
                total_points += n

            if all_pts:
                all_query_points_tensor = torch.from_numpy(np.concatenate(all_pts, axis=0)).float()

        if all_query_points_tensor is None:
            ref_bboxes = mapped_bboxes
            all_query_points_tensor, points_per_object = generate_points_from_bboxes(
                ref_bboxes, (H, W), config
            )
        else:
            ref_bboxes = mapped_bboxes
        
        if all_query_points_tensor is None:
            logger.warning("Could not generate query points from bounding boxes.")
            return {}, None

        all_query_points_tensor = all_query_points_tensor.to(device)

        # 5. 重排图像序列：将参考图像移到第一位（VGGT模型要求）
        if reference_image_idx != 0:
            reordered_images = torch.cat([
                images[reference_image_idx:reference_image_idx+1],
                images[:reference_image_idx],
                images[reference_image_idx+1:]
            ], dim=0)
            # 逆映射：新索引 -> 原始索引
            orig_order = [reference_image_idx] + list(range(reference_image_idx)) + list(range(reference_image_idx+1, S))
            reverse_mapping = dict(enumerate(orig_order))
        else:
            # 参考图像已经在位置0，无需重排
            reordered_images = images
            reverse_mapping = {i: i for i in range(S)}

        # 6. 使用VGGT执行点追踪（使用重排后的图像序列）
        start_time = time.time()
        
        with torch.no_grad():
            try:
                predictions = vggt_model(reordered_images.unsqueeze(0), query_points=all_query_points_tensor.unsqueeze(0))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and torch.cuda.is_available():
                    logger.error("CUDA out of memory during tracking. Trying to free cache and fail fast.")
                    torch.cuda.empty_cache()
                raise
        
        tracks = predictions['track'].squeeze(0)      # 点轨迹 (S, N, 2)
        visibility = predictions['vis'].squeeze(0)    # 可见性分数 (S, N)
        confidence = predictions['conf'].squeeze(0)   # 置信度分数 (S, N)
        tracking_time = time.time() - start_time
        logger.info(f"Tracking complete in {tracking_time:.1f}s")

        # 7. 使用基于对应关系的物体匹配逻辑（映射回原始索引）
        object_correspondences = {}
        
        for new_s_idx in range(S):
            # 跳过参考图像（现在在位置0）
            if new_s_idx == 0:
                continue

            # 获取原始图像索引
            orig_s_idx = reverse_mapping[new_s_idx]

            # 使用对应关系匹配函数
            matched_objects = match_objects_by_correspondence(
                tracks=tracks,
                visibility=visibility,
                confidence=confidence,
                points_per_object=points_per_object,
                target_detections=detections[orig_s_idx],  # 使用原始索引获取检测结果
                reference_image_idx=0,  # 在重排后的序列中，参考图像总是在位置0
                target_image_idx=new_s_idx,  # 在重排后序列中的目标图像位置
                config=config,
                transforms_info=transforms_info,
                min_hit_ratio=config.min_hit_ratio
            )
            
            if matched_objects:
                object_correspondences[orig_s_idx] = matched_objects  # 用原始索引存储结果
                logger.info(f"Found {len(matched_objects)} matches in image {orig_s_idx}\n")

        matched_targets = len(object_correspondences)
        matched_pairs = sum(len(v) for v in object_correspondences.values())
        logger.debug(
            f"ref={reference_image_idx} matched_targets={matched_targets} matched_pairs={matched_pairs}"
        )
        logger.info(f"Point tracking complete. Found correspondences in {matched_targets} images.")
        return object_correspondences, points_per_object
        
    except (RuntimeError, ValueError, KeyError, IndexError) as e:
        logger.error(f"Failed to find point tracking correspondences: {e}")
        raise


def match_objects_by_correspondence(
    tracks: torch.Tensor,
    visibility: torch.Tensor, 
    confidence: torch.Tensor,
    points_per_object: Dict[int, Dict],
    target_detections: List[Dict],
    reference_image_idx: int,
    target_image_idx: int,
    config: SKUMatchingConfig,
    transforms_info: Optional[List] = None,
    min_hit_ratio: float = 0.5
) -> List[Dict]:
    """基于点对应关系匹配物体

    Args:
        tracks: 点轨迹 (S, N, 2)
        visibility: 可见性分数 (S, N)
        confidence: 置信度分数 (S, N)
        points_per_object: 参考图像对象点信息
        target_detections: 目标图像检测结果
        reference_image_idx: 参考图像索引
        target_image_idx: 目标图像索引
        config: 配置参数
        transforms_info: 几何变换信息
        min_hit_ratio: 最小命中率阈值，默认0.5(50%)
        
    Returns:
        匹配的物体列表
    """
    # 提取目标图像的检测框
    target_bboxes = extract_bboxes_from_detections([target_detections], 0, config)
    if not target_bboxes:
        logger.warning(f"No bounding boxes found in target image {target_image_idx}")
        return []
    
    if transforms_info and target_image_idx < len(transforms_info):
        target_transform = transforms_info[target_image_idx]
        mapped_target_bboxes = []
        for bbox_info in target_bboxes:
            mapped_bbox = target_transform.map_bbox_to_final(bbox_info['bbox'])
            bbox_info_mapped = dict(bbox_info)
            bbox_info_mapped['original_bbox'] = bbox_info['bbox']
            bbox_info_mapped['bbox'] = mapped_bbox
            mapped_target_bboxes.append(bbox_info_mapped)
        target_bboxes = mapped_target_bboxes
    
    matched_objects = []
    stats_list: List[Dict] = []
    
    # 决定是否使用并行化
    num_ref_objects = len(points_per_object)
    use_parallel = num_ref_objects >= 3  # 至少3个参考对象才启用并行
    max_workers = min(4, num_ref_objects)  # 最多4个线程
    
    if use_parallel:
        logger.info(f"启用参考对象并行匹配: {num_ref_objects} 个对象，{max_workers} 线程")
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有参考对象的处理任务
                future_to_ref_id = {}
                for ref_object_id, ref_data in points_per_object.items():
                    future = executor.submit(
                        _process_single_ref_object,
                        ref_object_id, ref_data, target_bboxes, tracks, confidence,
                        target_image_idx, config, min_hit_ratio, transforms_info
                    )
                    future_to_ref_id[future] = ref_object_id
                
                # 收集结果
                for future in as_completed(future_to_ref_id, timeout=60):
                    ref_object_id = future_to_ref_id[future]
                    try:
                        matches, stats = future.result()
                        matched_objects.extend(matches)
                        stats_list.append(stats)
                    except (TimeoutError, RuntimeError) as e:
                        logger.error(f"并行处理参考对象 {ref_object_id} 失败: {e}")
                        
        except (RuntimeError, TimeoutError, ImportError) as e:
            logger.warning(f"并行处理失败，回退到串行模式: {e}")
            use_parallel = False
    
    if not use_parallel:
        logger.info("使用串行匹配模式")
        # 串行处理（原有逻辑）
        for ref_object_id, ref_data in points_per_object.items():
            matches, stats = _process_single_ref_object(
                ref_object_id, ref_data, target_bboxes, tracks, confidence,
                target_image_idx, config, min_hit_ratio, transforms_info
            )
            matched_objects.extend(matches)
            stats_list.append(stats)
    
    if len(matched_objects) > 0:
        logger.info(f"Finish matching objects between reference image {reference_image_idx} and target image {target_image_idx}")
    
    return matched_objects
