"""
SKU匹配系统核心算法模块

包含传统点追踪匹配算法和3D-2D投影匹配算法
"""

import time
import torch
import logging
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
    sample_3d_points_from_bbox, 
    project_3d_to_2d,
    find_best_matching_bbox_with_3d_validation,
    apply_uniqueness_constraint
)
from .transforms import VGGTImageTransform

logger = logging.getLogger(__name__)


def _process_single_ref_object(
    ref_object_id: int,
    ref_data: Dict,
    target_bboxes: List[Dict],
    tracks: torch.Tensor,
    confidence: torch.Tensor,
    target_image_idx: int,
    config: SKUMatchingConfig,
    correspondence_threshold: float = 0.5,
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
        correspondence_threshold: 对应关系阈值
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
            return [], {
                'ref_object_id': ref_object_id,
                'valid_points': 0,
                'below_min_conf_points': True,
                'num_target_bboxes': len(target_bboxes),
                'num_candidates': 0,
                'num_below_threshold': 0,
                'top_hit_ratio': 0.0,
                'produced_matches': 0,
            }
            
        # 过滤非有限值
        finite_mask = torch.isfinite(valid_points).all(dim=1)
        valid_points = valid_points[finite_mask]
        valid_points_count = int(valid_points.shape[0])

        if len(valid_points) == 0:
            return [], {
                'ref_object_id': ref_object_id,
                'valid_points': 0,
                'below_min_conf_points': True,
                'num_target_bboxes': len(target_bboxes),
                'num_candidates': 0,
                'num_below_threshold': 0,
                'top_hit_ratio': 0.0,
                'produced_matches': 0,
            }
            
        # 检查是否达到最小置信点数要求
        if len(valid_points) < config.min_confident_points:
            logger.debug(f"参考对象 {ref_object_id}: 只有 {len(valid_points)} 个置信点，低于最小值 {config.min_confident_points}")
            return [], {
                'ref_object_id': ref_object_id,
                'valid_points': valid_points_count,
                'below_min_conf_points': True,
                'num_target_bboxes': len(target_bboxes),
                'num_candidates': 0,
                'num_below_threshold': 0,
                'top_hit_ratio': 0.0,
                'produced_matches': 0,
            }
        
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
            keep_mask = ratios >= correspondence_threshold
            kept_indices = torch.nonzero(keep_mask, as_tuple=False).flatten().tolist()
            below_mask = (ratios > 0) & (ratios < correspondence_threshold)
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
        
        # 去重逻辑 - 如果一个框完全包含另一个框，移除包含者（较大的框）
        if len(all_candidates) > 1:
            to_remove = set()
            for i in range(len(all_candidates)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(all_candidates)):
                    if j in to_remove:
                        continue
                    bbox_i = all_candidates[i]['vggt_box']
                    bbox_j = all_candidates[j]['vggt_box']
                    
                    # 检查i是否完全包含j（i包含j，移除i）
                    if (bbox_i[0] <= bbox_j[0] and bbox_i[1] <= bbox_j[1] and 
                        bbox_i[2] >= bbox_j[2] and bbox_i[3] >= bbox_j[3] and
                        not (bbox_i[0] == bbox_j[0] and bbox_i[1] == bbox_j[1] and 
                             bbox_i[2] == bbox_j[2] and bbox_i[3] == bbox_j[3])):
                        to_remove.add(i)
                        logger.debug(f"移除包含框 {all_candidates[i]['target_obj_id']}，它包含 {all_candidates[j]['target_obj_id']}")
                        break
                    # 检查j是否完全包含i（j包含i，移除j）
                    elif (bbox_j[0] <= bbox_i[0] and bbox_j[1] <= bbox_i[1] and 
                          bbox_j[2] >= bbox_i[2] and bbox_j[3] >= bbox_i[3] and
                          not (bbox_i[0] == bbox_j[0] and bbox_i[1] == bbox_j[1] and 
                               bbox_i[2] == bbox_j[2] and bbox_i[3] == bbox_j[3])):
                        to_remove.add(j)
                        logger.debug(f"移除包含框 {all_candidates[j]['target_obj_id']}，它包含 {all_candidates[i]['target_obj_id']}")
                        break
            
            filtered_candidates = [m for i, m in enumerate(all_candidates) if i not in to_remove]
        else:
            filtered_candidates = all_candidates
        
        # 按overlap_ratio降序排序，取前2个最好的匹配
        filtered_candidates.sort(key=lambda x: x['correspondence_ratio'], reverse=True)
        matches = filtered_candidates[:2]

        for match in matches:
            logger.info(f"🧵 并行匹配: ref {ref_object_id} → target {match['target_obj_id']} (hit rate: {match['correspondence_ratio']:.2f} {match['matched_points']}/{match['total_points']})")
        
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
        
    except Exception as e:
        logger.error(f"处理参考对象 {ref_object_id} 失败: {e}")
        return [], {
            'ref_object_id': ref_object_id,
            'valid_points': 0,
            'below_min_conf_points': True,
            'num_target_bboxes': len(target_bboxes),
            'num_candidates': 0,
            'num_below_threshold': 0,
            'top_hit_ratio': 0.0,
            'produced_matches': 0,
        }


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
    
    # 输入验证
    if reference_image_idx >= images.shape[0]:
        raise ValueError(f"Reference image index {reference_image_idx} out of range for {images.shape[0]} images")
    
    if len(detections) != images.shape[0]:
        raise ValueError(f"Mismatch: {len(detections)} detections vs {images.shape[0]} images")
    
    # 根据配置选择匹配算法
    if config.enable_3d_projection_matching:
        return find_correspondences_3d_projection(
            vggt_model, detections, images, config, reference_image_idx, transforms_info
        )
    else:
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
                other_ref_bboxes = [other['bbox'] for other in ref_bboxes if other['object_id'] != ref_obj_id]
                points_3d = sample_3d_points_from_bbox(
                    scene_data, reference_image_idx, ref_bbox_info['bbox'], 
                    ref_transform, config, other_ref_bboxes
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
                        'target_obj_id': target_bbox_info['object_id'],
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
            # DEBUG 聚合：每个 target 汇总（3D）
            top_hit_ratio = 0.0
            top_hit_ref = None
            if 'final_matches' in locals() and final_matches:
                for m in final_matches:
                    r = float(m.get('match_ratio', 0.0))
                    if r > top_hit_ratio:
                        top_hit_ratio = r
                        top_hit_ref = m.get('ref_obj_id')
            logger.debug(
                f"target={target_img_idx} matched_objs={len(final_matches) if 'final_matches' in locals() and final_matches else 0} "
                f"skipped_low_overlap=0 top_hit={top_hit_ref}(hit={top_hit_ratio:.2f})"
            )
        
        matched_targets = len(correspondences)
        matched_pairs = sum(len(v) for v in correspondences.values())
        logger.debug(
            f"ref={reference_image_idx} matched_targets={matched_targets} matched_pairs={matched_pairs}"
        )
        logger.info(f"3D-2D projection complete. Found correspondences in {matched_targets} images.")
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

        # 4. 生成查询点
        all_query_points_tensor, points_per_object = generate_points_from_bboxes(
            ref_bboxes, (H, W), config
        )
        
        if all_query_points_tensor is None:
            logger.warning("Could not generate query points from bounding boxes.")
            return {}, None

        all_query_points_tensor = all_query_points_tensor.to(device)

        # 5. 重排图像序列：将参考图像移到第一位（VGGT模型要求）
        if reference_image_idx != 0:
            logger.info(f"Reordering images: moving reference image {reference_image_idx} to position 0")
            # 重排图像：[ref_img, img_0, img_1, ..., img_(ref-1), img_(ref+1), ..., img_(S-1)]
            reordered_images = torch.cat([
                images[reference_image_idx:reference_image_idx+1],  # 参考图像
                images[:reference_image_idx],                       # 参考图像之前的图像
                images[reference_image_idx+1:]                      # 参考图像之后的图像
            ], dim=0)
            
            # 创建原始索引到新索引的映射
            index_mapping = {}
            new_idx = 0
            # 参考图像映射到位置0
            index_mapping[reference_image_idx] = new_idx
            new_idx += 1
            # 参考图像之前的图像
            for orig_idx in range(reference_image_idx):
                index_mapping[orig_idx] = new_idx
                new_idx += 1
            # 参考图像之后的图像
            for orig_idx in range(reference_image_idx + 1, S):
                index_mapping[orig_idx] = new_idx
                new_idx += 1
            
            # 创建新索引到原始索引的逆映射
            reverse_mapping = {v: k for k, v in index_mapping.items()}
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
                correspondence_threshold=config.correspondence_threshold
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
        
    except Exception as e:
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
    correspondence_threshold: float = 0.5
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
        correspondence_threshold: 对应关系阈值，默认0.5(50%)
        
    Returns:
        匹配的物体列表
    """
    # 提取目标图像的检测框
    target_bboxes = extract_bboxes_from_detections([target_detections], 0, config)
    if not target_bboxes:
        logger.warning(f"No bounding boxes found in target image {target_image_idx}")
        raise
    
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
                        target_image_idx, config, correspondence_threshold, transforms_info
                    )
                    future_to_ref_id[future] = ref_object_id
                
                # 收集结果
                for future in as_completed(future_to_ref_id, timeout=60):
                    ref_object_id = future_to_ref_id[future]
                    try:
                        matches, stats = future.result()
                        matched_objects.extend(matches)
                        stats_list.append(stats)
                    except Exception as e:
                        logger.error(f"并行处理参考对象 {ref_object_id} 失败: {e}")
                        
        except Exception as e:
            logger.warning(f"并行处理失败，回退到串行模式: {e}")
            use_parallel = False
    
    if not use_parallel:
        logger.info("使用串行匹配模式")
        # 串行处理（原有逻辑）
        for ref_object_id, ref_data in points_per_object.items():
            matches, stats = _process_single_ref_object(
                ref_object_id, ref_data, target_bboxes, tracks, confidence,
                target_image_idx, config, correspondence_threshold, transforms_info
            )
            matched_objects.extend(matches)
            stats_list.append(stats)
    
    # DEBUG 聚合：每个 target 的汇总
    skipped_low_overlap = 0
    for s in stats_list:
        if not s.get('below_min_conf_points', False) and s.get('valid_points', 0) > 0 and s.get('produced_matches', 0) == 0 and s.get('num_below_threshold', 0) > 0:
            skipped_low_overlap += 1

    top_hit_ratio = 0.0
    top_hit_ref = None
    for m in matched_objects:
        r = float(m.get('correspondence_ratio', 0.0))
        if r > top_hit_ratio:
            top_hit_ratio = r
            top_hit_ref = m.get('object_id')
    logger.debug(
        f"target={target_image_idx} matched_objs={len(matched_objects)} skipped_low_overlap={skipped_low_overlap} "
        f"top_hit={top_hit_ref}(hit={top_hit_ratio:.2f})"
    )

    if len(matched_objects) > 0:
        logger.info(f"Finish matching objects between reference image {reference_image_idx} and target image {target_image_idx}")
    
    return matched_objects
