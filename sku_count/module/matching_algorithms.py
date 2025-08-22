"""
SKU匹配系统核心算法模块

包含传统点追踪匹配算法和3D-2D投影匹配算法
"""

import time
import torch
import logging
from typing import Dict, List, Optional, Tuple
# VGGT相关导入
try:
    import sys
    sys.path.insert(0, '../../vggt-main')
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError as e:
    raise ImportError(f"Failed to import VGGT modules: {e}")

from .config import SKUMatchingConfig
from .data_utils import extract_bboxes_from_detections
from .point_utils import generate_points_from_bboxes
from .geometry_3d import (
    sample_3d_points_from_bbox, 
    project_3d_to_2d,
    find_best_matching_bbox_with_3d_validation,
    apply_uniqueness_constraint,
    find_best_matching_bbox
)
from .transforms import VGGTImageTransform

logger = logging.getLogger(__name__)


def find_object_correspondences(
    vggt_model: VGGT,
    detections: List[Dict],
    images: torch.Tensor,
    config: SKUMatchingConfig,
    reference_image_idx: int = 0,
    transforms_info: Optional[List[VGGTImageTransform]] = None,
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
    logger.info(f"Starting {algorithm_name} object correspondence detection...")
    
    # 输入验证
    if reference_image_idx >= images.shape[0]:
        raise ValueError(f"Reference image index {reference_image_idx} out of range for {images.shape[0]} images")
    
    if len(detections) != images.shape[0]:
        raise ValueError(f"Mismatch: {len(detections)} detections vs {images.shape[0]} images")
    
    # 根据配置选择匹配算法
    if config.enable_3d_projection_matching:
        logger.info("Using 3D-2D projection matching algorithm")
        return find_correspondences_3d_projection(
            vggt_model, detections, images, config, reference_image_idx, transforms_info
        )
    else:
        logger.info("Using traditional point tracking matching algorithm")
        return find_correspondences_point_tracking(
            vggt_model, detections, images, config, reference_image_idx, transforms_info
        )


def find_correspondences_3d_projection(
    vggt_model: VGGT,
    detections: List[Dict],
    images: torch.Tensor,
    config: SKUMatchingConfig,
    reference_image_idx: int = 0,
    transforms_info: Optional[List[VGGTImageTransform]] = None,
) -> Tuple[Dict[int, List[Dict]], Optional[Dict[int, Dict]]]:
    """基于3D-2D投影的物体匹配算法"""
    
    try:
        S = images.shape[0]
        _, _, H, W = images.shape
        device = images.device
        
        # 验证输入参数
        if reference_image_idx >= S:
            raise ValueError(f"Reference image index {reference_image_idx} out of range for {S} images")
        
        # 1. 全局3D场景重建（关键：只调用一次VGGT）
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
        for target_img_idx, target_detection in enumerate(detections):
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
                
                # 从参考图像的检出框采样3D点
                points_3d = sample_3d_points_from_bbox(
                    scene_data, reference_image_idx, ref_bbox_info['bbox'], 
                    ref_transform, config
                )
                
                if points_3d is None or len(points_3d) < 10:
                    continue
                
                # 计算参考3D点的统计信息用于几何验证
                ref_3d_center = points_3d.mean(dim=0)  # (3,)
                ref_depth_mean = points_3d[:, 2].mean().item()  # Z坐标作为深度
                
                # 投影到目标图像
                projected_points = project_3d_to_2d(
                    points_3d,
                    scene_data['extrinsic'][target_img_idx],
                    scene_data['intrinsic'][target_img_idx]
                )
                
                if len(projected_points) < 5:
                    continue
                
                # 将目标图像的检出框映射到VGGT坐标
                target_bboxes_vggt = []
                for bbox_info in target_bboxes:
                    vggt_bbox = target_transform.map_bbox_to_final(bbox_info['bbox'])
                    bbox_info_copy = dict(bbox_info)
                    bbox_info_copy['bbox'] = vggt_bbox
                    target_bboxes_vggt.append(bbox_info_copy)
                
                # 找到最匹配的目标框
                best_match = find_best_matching_bbox_with_3d_validation(
                    projected_points, target_bboxes_vggt, config, 
                    scene_data, target_img_idx, target_transform,
                    ref_3d_center, ref_depth_mean
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
                        'target_bbox_id': target_bbox_info['object_id'],
                        'box': original_bbox,
                        'vggt_box': target_bbox_info['bbox'],
                        'correspondence_ratio': match['match_ratio'],
                        'matched_points': match['points_in_bbox'],
                        'total_points': match['total_points'],
                        'confidence': target_bbox_info['confidence'],
                        # 新增3D验证信息
                        '3d_distance': match.get('3d_distance', 0.0),
                        'depth_consistency': match.get('depth_consistency', 0.0)
                    }
                    
                    matched_objects.append(match_result)
                    logger.info(f"3D match: ref {match['ref_obj_id']} → target {target_bbox_info['object_id']} (ratio: {match['match_ratio']:.1%})")
                
                correspondences[target_img_idx] = matched_objects
        
        logger.info(f"3D-2D projection complete. Found correspondences in {len(correspondences)} images.")
        return correspondences, points_per_object
        
    except Exception as e:
        logger.error(f"Failed to find 3D-2D projection correspondences: {e}")
        raise


def find_correspondences_point_tracking(
    vggt_model: VGGT,
    detections: List[Dict],
    images: torch.Tensor,
    config: SKUMatchingConfig,
    reference_image_idx: int = 0,
    transforms_info: Optional[List[VGGTImageTransform]] = None,
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

        # 3. 映射边界框到VGGT坐标空间
        mapped_bboxes = []
        for b in ref_bboxes:
            mapped = ref_transform.map_bbox_to_final(b['bbox'])
            b2 = dict(b)
            b2['original_bbox'] = b['bbox']
            b2['bbox'] = mapped
            b2['center'] = [(mapped[0] + mapped[2]) / 2, (mapped[1] + mapped[3]) / 2]
            b2['area'] = max(0.0, (mapped[2] - mapped[0]) * (mapped[3] - mapped[1]))
            mapped_bboxes.append(b2)
        ref_bboxes = mapped_bboxes
        logger.info("Mapped detection bboxes to preprocessed input coordinates via transforms_info.")

        # 4. 生成查询点
        all_query_points_tensor, points_per_object = generate_points_from_bboxes(
            ref_bboxes, (H, W), config
        )
        
        if all_query_points_tensor is None:
            logger.warning("Could not generate query points from bounding boxes.")
            return {}, None

        logger.info(f"Generated {len(all_query_points_tensor)} query points")
        all_query_points_tensor = all_query_points_tensor.to(device)

        # 5. 使用VGGT执行点追踪
        logger.info("Tracking points with VGGT...")
        start_time = time.time()
        
        with torch.no_grad():
            try:
                predictions = vggt_model(images.unsqueeze(0), query_points=all_query_points_tensor.unsqueeze(0))
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and torch.cuda.is_available():
                    logger.error("CUDA out of memory during tracking. Trying to free cache and fail fast.")
                    torch.cuda.empty_cache()
                raise
        
        tracks = predictions['track'].squeeze(0)      # 点轨迹 (S, N, 2)
        visibility = predictions['vis'].squeeze(0)    # 可见性分数 (S, N)
        tracking_time = time.time() - start_time
        logger.info(f"Tracking complete in {tracking_time:.1f}s")

        # 6. 使用基于对应关系的物体匹配逻辑
        logger.info("Matching objects using correspondence-based logic...")
        object_correspondences = {}
        
        for s_idx in range(S):
            if s_idx == reference_image_idx:
                continue

            # 使用对应关系匹配函数
            matched_objects = match_objects_by_correspondence(
                tracks=tracks,
                visibility=visibility,
                points_per_object=points_per_object,
                target_detections=detections[s_idx],
                reference_image_idx=reference_image_idx,
                target_image_idx=s_idx,
                config=config,
                transforms_info=transforms_info,
                correspondence_threshold=config.correspondence_threshold
            )
            
            if matched_objects:
                object_correspondences[s_idx] = matched_objects
                logger.info(f"Found {len(matched_objects)} matches in image {s_idx}")

        logger.info(f"Point tracking complete. Found correspondences in {len(object_correspondences)} images.")
        return object_correspondences, points_per_object
        
    except Exception as e:
        logger.error(f"Failed to find point tracking correspondences: {e}")
        raise


def match_objects_by_correspondence(
    tracks: torch.Tensor,
    visibility: torch.Tensor, 
    points_per_object: Dict[int, Dict],
    target_detections: List[Dict],
    reference_image_idx: int,
    target_image_idx: int,
    config: SKUMatchingConfig,
    transforms_info: Optional[List] = None,
    correspondence_threshold: float = 0.5
) -> List[Dict]:
    """基于点对应关系匹配物体
    
    Args:
        tracks: 点轨迹 (S, N, 2)
        visibility: 可见性分数 (S, N)
        points_per_object: 参考图像对象点信息
        target_detections: 目标图像检测结果
        reference_image_idx: 参考图像索引
        target_image_idx: 目标图像索引
        config: 配置参数
        transforms_info: 几何变换信息
        correspondence_threshold: 对应关系阈值，默认0.5(50%)
        
    Returns:
        匹配的物体列表
    """
    logger.info(f"Matching objects between reference image {reference_image_idx} and target image {target_image_idx}")
    
    # 提取目标图像的检测框
    target_bboxes = extract_bboxes_from_detections([target_detections], 0, config)
    if not target_bboxes:
        logger.warning(f"No bounding boxes found in target image {target_image_idx}")
        return []
    
    # 如果有变换信息，将目标图像的检测框映射到VGGT输入空间
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
        logger.info(f"Mapped {len(target_bboxes)} target bboxes to VGGT input space")
    
    matched_objects = []
    
    # 遍历参考图像中的每个物体
    for ref_object_id, ref_data in points_per_object.items():
        start_idx, end_idx = ref_data["point_indices"]
        
        # 获取该物体在目标图像中的对应点
        ref_tracks_in_target = tracks[target_image_idx, start_idx:end_idx, :]  # (N_points, 2)
        ref_visibility_in_target = visibility[target_image_idx, start_idx:end_idx]  # (N_points,)
        
        # 过滤可见且有效的点
        visible_mask = ref_visibility_in_target > config.visibility_threshold
        valid_points = ref_tracks_in_target[visible_mask]
        
        if valid_points.numel() == 0:
            continue
            
        # 过滤非有限值
        finite_mask = torch.isfinite(valid_points).all(dim=1)
        valid_points = valid_points[finite_mask]
        
        if len(valid_points) == 0:
            continue
            
        # 检查是否达到最小可见点数要求
        if len(valid_points) < config.min_visible_points:
            logger.debug(f"Reference object {ref_object_id}: Only {len(valid_points)} valid points, below minimum {config.min_visible_points}")
            continue
            
        logger.debug(f"Reference object {ref_object_id}: {len(valid_points)} valid correspondence points")
        
        # 检查每个目标检测框
        best_match = None
        best_overlap_ratio = 0.0
        
        for target_bbox_info in target_bboxes:
            target_bbox = target_bbox_info['bbox']  # [x1, y1, x2, y2]
            
            # 计算有多少对应点落在这个检测框内
            points_in_bbox = 0
            for point in valid_points:
                x, y = point[0].item(), point[1].item()
                if (target_bbox[0] <= x <= target_bbox[2] and 
                    target_bbox[1] <= y <= target_bbox[3]):
                    points_in_bbox += 1
            
            # 计算重叠比例
            overlap_ratio = points_in_bbox / len(valid_points)
            
            logger.debug(f"  Target bbox {target_bbox_info['object_id']}: {points_in_bbox}/{len(valid_points)} points inside ({overlap_ratio:.3f})")
            
            # 如果重叠比例达到阈值且是当前最佳匹配
            if overlap_ratio >= correspondence_threshold and overlap_ratio > best_overlap_ratio:
                # 将VGGT坐标映射回原图坐标
                if transforms_info and target_image_idx < len(transforms_info):
                    original_bbox = transforms_info[target_image_idx].map_bbox_to_original(target_bbox)
                else:
                    original_bbox = target_bbox
                
                best_match = {
                    'object_id': ref_object_id,
                    'target_bbox_id': target_bbox_info['object_id'],
                    'box': original_bbox,
                    'vggt_box': target_bbox,
                    'correspondence_ratio': overlap_ratio,
                    'matched_points': points_in_bbox,
                    'total_points': len(valid_points),
                    'target_confidence': target_bbox_info['confidence'],
                    'reference_confidence': ref_data['confidence']
                }
                best_overlap_ratio = overlap_ratio
        
        # 如果找到匹配，添加到结果中
        if best_match:
            matched_objects.append(best_match)
            logger.info(f"Matched ref {ref_object_id} → target {best_match['target_bbox_id']} (ratio: {best_overlap_ratio:.2f})")
        else:
            logger.debug(f"❌ No match found for reference object {ref_object_id}")
    
    logger.info(f"Found {len(matched_objects)} object matches in target image {target_image_idx}")
    return matched_objects