"""
SKU匹配系统3D几何处理模块

包含3D点采样、投影、几何验证等功能
"""

import torch
import logging
from typing import Dict, List, Optional

from .config import SKUMatchingConfig
from .transforms import ImageTransformBase

logger = logging.getLogger(__name__)


def sample_3d_points_from_non_overlap_regions(
    scene_data: Dict, img_idx: int, bbox: List[float], non_overlap_regions: List[List[float]],
    transform: ImageTransformBase, config: SKUMatchingConfig
) -> Optional[torch.Tensor]:
    """从检出框的非重合区域中采样 3D 点
    
    Args:
        scene_data: 场景数据
        img_idx: 图像索引
        bbox: 检出框 [x1, y1, x2, y2]
        non_overlap_regions: 非重合区域列表
        transform: 坐标变换器
        config: 配置参数
        
    Returns:
        3D点张量或None
    """
    device = scene_data['depth'].device
    H, W = scene_data['depth'].shape[1:3]
    
    if not non_overlap_regions:
        logger.debug(f"检出框 {bbox} 没有非重合区域，跳过 3D 采样")
        return None
    
    all_region_points = []
    
    for region in non_overlap_regions:
        # 将原始坐标区域映射到 VGGT 坐标
        vggt_region = transform.map_bbox_to_final(region)
        x1, y1, x2, y2 = [int(c) for c in vggt_region]
        
        # 确保坐标在有效范围内
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        
        if x1 >= x2 or y1 >= y2:
            continue
            
        # 提取该区域的 3D 信息
        depth_region = scene_data['depth'][img_idx, y1:y2, x1:x2, 0]
        depth_conf_region = scene_data['depth_conf'][img_idx, y1:y2, x1:x2]
        world_points_region = scene_data['world_points'][img_idx, y1:y2, x1:x2]
        world_points_conf_region = scene_data['world_points_conf'][img_idx, y1:y2, x1:x2]
        
        # 过滤有效 3D 点
        valid_mask = (
            (depth_conf_region > config.depth_confidence_threshold) &
            (world_points_conf_region > config.point_3d_confidence_threshold) &
            (depth_region > config.min_depth) &
            (depth_region < config.max_depth) &
            torch.isfinite(depth_region) &
            torch.isfinite(world_points_region).all(dim=-1)
        )
        
        if valid_mask.sum() < 3:  # 每个区域至少 3 个点
            continue
        
        # 获取有效的 3D 点
        valid_world_points = world_points_region[valid_mask]
        
        # 随机采样一些点（按区域面积比例分配）
        region_area = (x2 - x1) * (y2 - y1)
        max_points_for_region = max(3, min(len(valid_world_points), 
                                          int(config.max_3d_points_per_bbox * 0.3)))  # 每个区域最多 30% 的点
        
        if len(valid_world_points) > max_points_for_region:
            indices = torch.randperm(len(valid_world_points), device=device)[:max_points_for_region]
            sampled_points = valid_world_points[indices]
        else:
            sampled_points = valid_world_points
        
        all_region_points.append(sampled_points)
    
    if not all_region_points:
        logger.debug(f"检出框 {bbox} 的所有非重合区域都无有效 3D 点")
        return None
    
    # 合并所有区域的 3D 点
    all_points = torch.cat(all_region_points, dim=0)
    
    # 最终采样限制
    if len(all_points) > config.max_3d_points_per_bbox:
        indices = torch.randperm(len(all_points), device=device)[:config.max_3d_points_per_bbox]
        all_points = all_points[indices]
    
    logger.debug(f"从 {len(non_overlap_regions)} 个非重合区域采样了 {len(all_points)} 个 3D 点")
    return all_points


def sample_3d_points_from_bbox(scene_data: Dict, img_idx: int, bbox: List[float],
                               transform: ImageTransformBase, config: SKUMatchingConfig,
                               other_bboxes: Optional[List[List[float]]] = None) -> Optional[torch.Tensor]:
    """
    从检出框中采样3D点 - 使用实际depth值进行精确重建

    Args:
        scene_data: VGGT场景数据（包含depth, depth_conf, world_points等）
        img_idx: 图像索引
        bbox: 检出框坐标 [x1, y1, x2, y2]
        transform: VGGT图像变换
        config: 匹配配置参数
        other_bboxes: 其他检出框列表（保留参数，当前未使用）

    Returns:
        采样的3D点 torch.Tensor (N, 3) 或 None（如果无效点不足）
    """
    vggt_bbox = transform.map_bbox_to_final(bbox)
    x1, y1, x2, y2 = [int(c) for c in vggt_bbox]

    # 确保坐标在有效范围内
    H, W = scene_data['depth'].shape[1:3]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)

    if x1 >= x2 or y1 >= y2:
        return None

    # 提取检出框区域的3D信息
    depth_region = scene_data['depth'][img_idx, y1:y2, x1:x2, 0]
    depth_conf_region = scene_data['depth_conf'][img_idx, y1:y2, x1:x2]
    world_points_region = scene_data['world_points'][img_idx, y1:y2, x1:x2]
    world_points_conf_region = scene_data['world_points_conf'][img_idx, y1:y2, x1:x2]

    # 改进的过滤条件：更严格的质量控制
    valid_mask = (
        (depth_conf_region > config.depth_confidence_threshold) &
        (world_points_conf_region > config.point_3d_confidence_threshold) &
        (depth_region > config.min_depth) &
        (depth_region < config.max_depth) &
        torch.isfinite(depth_region) &  # 确保深度值有限
        torch.isfinite(world_points_region).all(dim=-1)  # 确保3D点有限
    )

    if valid_mask.sum() < 10:
        logger.debug(f"检出框 {bbox} 中有效3D点不足: {valid_mask.sum()}")
        return None

    # 优先使用VGGT的world_points，但用深度图进行验证
    valid_world_points = world_points_region[valid_mask]
    valid_depths = depth_region[valid_mask]

    # 深度一致性检查：需在同一坐标系下比较（相机坐标系Z vs 深度图）
    E = scene_data['extrinsic'][img_idx].to(valid_world_points.device)
    if E.shape == (4, 4):
        R = E[:3, :3]
        t = E[:3, 3]
    elif E.shape == (3, 4):
        R = E[:, :3]
        t = E[:, 3]
    else:
        raise ValueError(f"Unsupported extrinsic matrix shape: {E.shape}")
    points_cam = (R @ valid_world_points.T + t.unsqueeze(1)).T
    camera_depths = points_cam[:, 2]
    depth_diff = torch.abs(camera_depths - valid_depths)
    depth_consistent_mask = depth_diff < config.depth_consistency_threshold

    if depth_consistent_mask.sum() < 5:
        logger.debug(f"检出框 {bbox} 中深度一致的点不足: {depth_consistent_mask.sum()}")
        # 如果深度不一致，使用enhanced_backproject_2d_to_3d重新计算
        device = valid_world_points.device
        reconstructed_points = []

        # 获取有效像素的坐标
        valid_y_indices, valid_x_indices = torch.where(valid_mask)
        for i in range(min(len(valid_y_indices), config.max_3d_points_per_bbox)):
            y_coord = valid_y_indices[i].item() + y1
            x_coord = valid_x_indices[i].item() + x1

            point_3d = enhanced_backproject_2d_to_3d(
                x_coord, y_coord, scene_data, img_idx, config
            )
            if point_3d is not None:
                reconstructed_points.append(point_3d)

        if len(reconstructed_points) < 5:
            return None

        sampled_points = torch.stack(reconstructed_points)

    else:
        # 使用深度一致的world_points
        consistent_points = valid_world_points[depth_consistent_mask]

        # 随机采样指定数量的点
        device = consistent_points.device
        num_points = min(len(consistent_points), config.max_3d_points_per_bbox)
        if len(consistent_points) > num_points:
            indices = torch.randperm(len(consistent_points), device=device)[:num_points]
            sampled_points = consistent_points[indices]
        else:
            sampled_points = consistent_points

    return sampled_points


def enhanced_backproject_2d_to_3d(x: float, y: float, scene_data: Dict, img_idx: int,
                                 config: SKUMatchingConfig) -> Optional[torch.Tensor]:
    """使用深度图增强的2D到3D反投影 - 基于advanced_3d_reconstruction.py优化
    
    Args:
        x, y: 2D图像坐标
        scene_data: 场景数据（包含depth, intrinsic, extrinsic等）
        img_idx: 图像索引
        config: 配置参数
        
    Returns:
        3D点坐标，如果失败则返回None
    """
    device = scene_data['depth'].device
    H, W = scene_data['depth'].shape[1:3]
    
    # 坐标范围检查
    x_int, y_int = int(round(x)), int(round(y))
    if not (0 <= x_int < W and 0 <= y_int < H):
        return None
    
    # 获取深度信息
    depth = scene_data['depth'][img_idx, y_int, x_int, 0].item()
    depth_conf = scene_data['depth_conf'][img_idx, y_int, x_int].item()
    
    # 深度有效性检查
    if (depth_conf < config.depth_confidence_threshold or 
        depth < config.min_depth or 
        depth > config.max_depth):
        return None
    
    # 获取相机参数
    intrinsic = scene_data['intrinsic'][img_idx]  # (3, 3)
    extrinsic = scene_data['extrinsic'][img_idx]  # 应该是(4, 4)，但可能是(3, 4)
    
    # 提取内参
    fx = intrinsic[0, 0].item()
    fy = intrinsic[1, 1].item() 
    cx = intrinsic[0, 2].item()
    cy = intrinsic[1, 2].item()
    
    # 使用实际深度进行反投影
    u = (x - cx) / fx
    v = (y - cy) / fy
    
    # 相机坐标系中的3D点
    point_cam = torch.tensor([u * depth, v * depth, depth], device=device)
    
    # 转换到世界坐标系
    # 处理相机外参矩阵的逆变换
    point_cam_homo = torch.cat([point_cam, torch.ones(1, device=device)])
    
    # 确保extrinsic是4x4矩阵
    if extrinsic.shape == (3, 4):
        # 如果是3x4矩阵，添加[0, 0, 0, 1]行使其成为4x4
        bottom_row = torch.tensor([0., 0., 0., 1.], device=device).unsqueeze(0)
        extrinsic = torch.cat([extrinsic, bottom_row], dim=0)
    elif extrinsic.shape != (4, 4):
        raise ValueError(f"Unsupported extrinsic matrix shape: {extrinsic.shape}")
    
    extrinsic_inv = torch.inverse(extrinsic)
    point_world = (extrinsic_inv @ point_cam_homo)[:3]
    
    return point_world


def project_3d_to_2d(points_3d: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """将3D点投影到2D图像坐标"""
    # 确保所有张量在同一设备上
    device = points_3d.device
    extrinsic = extrinsic.to(device)
    intrinsic = intrinsic.to(device)
    
    # 处理extrinsic矩阵形状
    if extrinsic.shape == (3, 4):
        # 如果是3x4矩阵，直接用于投影
        projection_matrix = intrinsic @ extrinsic
    elif extrinsic.shape == (4, 4):
        # 如果是4x4矩阵，取前3x4部分
        projection_matrix = intrinsic @ extrinsic[:3, :]
    else:
        raise ValueError(f"Unsupported extrinsic matrix shape: {extrinsic.shape}")
    
    # 齐次坐标
    ones = torch.ones(len(points_3d), 1, device=device)
    points_3d_homo = torch.cat([points_3d, ones], dim=1)
    
    # 直接投影到图像坐标
    points_2d_homo = (projection_matrix @ points_3d_homo.T).T
    
    # 过滤在相机前方的点
    valid_depth_mask = points_2d_homo[:, 2] > 0.1
    if not valid_depth_mask.any():
        return torch.empty(0, 2, device=device)
        
    points_2d_homo_valid = points_2d_homo[valid_depth_mask]
    points_2d = points_2d_homo_valid[:, :2] / points_2d_homo_valid[:, 2:3]
    
    return points_2d


def find_best_matching_bbox_with_3d_validation(projected_points: torch.Tensor, target_bboxes: List[Dict],
                                             config: SKUMatchingConfig, scene_data: Dict, target_img_idx: int,
                                             target_transform: ImageTransformBase, ref_3d_center: torch.Tensor,
                                             ref_depth_mean: float) -> Optional[Dict]:
    """找到投影点最多落入的检出框，并进行3D几何验证"""
    if len(projected_points) == 0:
        return None
        
    best_match = None
    best_score = 0.0
    
    for bbox_info in target_bboxes:
        bbox = bbox_info['bbox']
        x1, y1, x2, y2 = bbox
        
        # 1. 基础投影匹配
        points_in_bbox = (
            (projected_points[:, 0] >= x1) & 
            (projected_points[:, 0] <= x2) &
            (projected_points[:, 1] >= y1) & 
            (projected_points[:, 1] <= y2)
        ).sum().item()
        
        match_ratio = points_in_bbox / len(projected_points)
        
        if match_ratio < config.projection_match_threshold:
            continue
        
        # 2. 3D几何验证：从目标检出框采样3D点进行比较
        target_points_3d = sample_3d_points_from_bbox(
            scene_data, target_img_idx, 
            target_transform.map_bbox_to_original(bbox), 
            target_transform, config, None  # 不传递other_bboxes，需要时可以改进
        )
        
        if target_points_3d is None or len(target_points_3d) < 10:
            continue
        
        # 计算目标3D点的统计信息
        target_3d_center = target_points_3d.mean(dim=0)

        # 转换到相机坐标系计算深度
        E_target = scene_data['extrinsic'][target_img_idx].to(target_points_3d.device)
        if E_target.shape == (4, 4):
            R_target = E_target[:3, :3]
            t_target = E_target[:3, 3]
        elif E_target.shape == (3, 4):
            R_target = E_target[:, :3]
            t_target = E_target[:, 3]
        else:
            raise ValueError(f"Unsupported extrinsic matrix shape: {E_target.shape}")
        target_points_cam = (R_target @ target_points_3d.T + t_target.unsqueeze(1)).T
        target_depth_mean = target_points_cam[:, 2].mean().item()

        # 3D距离验证
        spatial_distance = torch.norm(ref_3d_center - target_3d_center).item()
        
        # 深度一致性验证
        depth_diff = abs(ref_depth_mean - target_depth_mean)
        depth_consistency = max(0.0, 1.0 - depth_diff / config.max_depth_difference)
        
        # 组合评分：投影匹配 + 3D几何一致性
        geometry_score = max(0.0, 1.0 - spatial_distance / config.max_3d_distance)
        combined_score = match_ratio * 0.6 + geometry_score * 0.3 + depth_consistency * 0.1
        
        # 严格筛选：必须满足3D几何约束
        if spatial_distance > config.max_3d_distance:
            continue
        if depth_consistency < config.min_depth_consistency:
            continue
            
        if combined_score > best_score:
            best_match = {
                'target_bbox_info': bbox_info,
                'points_in_bbox': points_in_bbox,
                'total_points': len(projected_points),
                'match_ratio': match_ratio,
                '3d_distance': spatial_distance,
                'depth_consistency': depth_consistency,
                'combined_score': combined_score
            }
            best_score = combined_score
            
    return best_match


def apply_uniqueness_constraint(candidate_matches: List[Dict]) -> List[Dict]:
    """应用唯一性约束：每个目标框只能匹配一个参考框（选择最佳匹配）"""
    if not candidate_matches:
        return []
    
    # 按目标框ID分组
    target_groups = {}
    for match in candidate_matches:
        target_id = match['target_bbox_info']['object_id']
        if target_id not in target_groups:
            target_groups[target_id] = []
        target_groups[target_id].append(match)
    
    final_matches = []
    
    for target_id, matches in target_groups.items():
        if len(matches) == 1:
            # 只有一个匹配，直接使用
            final_matches.append(matches[0])
        else:
            # 多个匹配，选择综合评分最高的
            best_match = max(matches, key=lambda x: x['combined_score'])
            final_matches.append(best_match)
            
            # 记录被过滤的匹配
            filtered_matches = [m for m in matches if m != best_match]
            for filtered in filtered_matches:
                ref_obj_id = filtered.get('ref_obj_id', 'unknown')
                filtered_score = filtered.get('combined_score', 0.0)
                best_score = best_match.get('combined_score', 0.0)
                logger.info(f"Filtered duplicate: ref {ref_obj_id} → target {target_id} "
                          f"(score: {filtered_score:.3f} < {best_score:.3f})")
    
    logger.info(f"Applied uniqueness constraint: {len(candidate_matches)} → {len(final_matches)} matches")
    return final_matches
