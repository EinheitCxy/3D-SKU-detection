"""
SKU匹配系统3D几何处理模块

包含3D点采样、投影、几何验证等功能
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Optional

from .config import SKUMatchingConfig
from .transforms import ImageTransformBase

logger = logging.getLogger(__name__)


def _fit_plane_svd(points_3d: torch.Tensor) -> tuple:
    """SVD 拟合平面，返回 (normal: np.ndarray(3,), residual_rms: float)。

    平面法向 = 去中心化后点云协方差矩阵的最小奇异向量；残差 RMS = 点到平面距离的均方根。
    用于共面约束：同一物理面（如货架层板）法向应一致、残差应小。
    """
    pts = points_3d.detach().cpu().numpy().astype(np.float64)
    if pts.shape[0] < 3:
        # 点太少无法稳定拟合平面，返回零向量+大残差表示无效
        return np.zeros(3), float("inf")
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(3), float("inf")
    normal = vh[-1]  # 最小奇异值方向 = 平面法向
    # 归一化法向（SVD 返回的行向量已是单位向量，但保险起见）
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.zeros(3), float("inf")
    normal = normal / norm
    distances = centered @ normal  # 点到平面距离（带符号）
    residual_rms = float(np.sqrt((distances ** 2).mean()))
    return normal, residual_rms


def transform_world_to_camera(points_3d: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
    """将3D点从世界坐标系变换到相机坐标系

    Args:
        points_3d: 世界坐标系中的3D点 (N, 3)
        extrinsic: 外参矩阵，支持(4,4)或(3,4)形状
    Returns:
        相机坐标系中的3D点 (N, 3)，其中[:,2]为深度
    """
    if extrinsic.shape == (4, 4):
        R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    elif extrinsic.shape == (3, 4):
        R, t = extrinsic[:, :3], extrinsic[:, 3]
    else:
        raise ValueError(f"Unsupported extrinsic matrix shape: {extrinsic.shape}")
    return (R @ points_3d.T + t.unsqueeze(1)).T


def transform_camera_to_world(point_cam: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
    """将单个3D点从相机坐标系变换到世界坐标系

    Args:
        point_cam: 相机坐标系中的3D点 (3,)
        extrinsic: 外参矩阵，支持(4,4)或(3,4)形状
    Returns:
        世界坐标系中的3D点 (3,)
    """
    device = point_cam.device
    if extrinsic.shape == (3, 4):
        bottom_row = torch.tensor([0., 0., 0., 1.], device=device).unsqueeze(0)
        extrinsic = torch.cat([extrinsic, bottom_row], dim=0)
    elif extrinsic.shape != (4, 4):
        raise ValueError(f"Unsupported extrinsic matrix shape: {extrinsic.shape}")
    point_cam_homo = torch.cat([point_cam, torch.ones(1, device=device)])
    return (torch.inverse(extrinsic) @ point_cam_homo)[:3]


def sample_3d_points_from_non_overlap_regions_batch(
    scene_data: Dict,
    img_idx: int,
    bboxes_info: List[Dict],
    transform: ImageTransformBase,
    config: SKUMatchingConfig
) -> List[Optional[torch.Tensor]]:
    """
    批量化非重合区域3D点采样（自动计算非重合区域）

    从多个检出框的非重合区域中批量采样3D点。
    模仿 generate_points_from_bboxes 的设计，自动处理非重合区域计算。

    Args:
        scene_data: 场景数据（包含depth, depth_conf, world_points等）
        img_idx: 图像索引
        bboxes_info: 检出框信息列表，每个元素包含 'bbox' 和 'object_id'
        transform: 坐标变换器
        config: 配置参数

    Returns:
        3D点列表，每个元素对应一个bbox的采样结果（可能为None，如果有效点不足10个）
    """
    if not bboxes_info:
        return []

    logger.info(f"批量采样3D点: {len(bboxes_info)} 个检出框 (使用非重合区域采样)...")

    # 分析检出框重合情况（仅用于统计）
    from .bbox_utils import analyze_bbox_overlaps

    if len(bboxes_info) > 1:
        overlap_stats = analyze_bbox_overlaps(bboxes_info, config.overlap_threshold)
        if len(overlap_stats['overlapping_boxes']) > 0:
            logger.info(f"检测到 {len(overlap_stats['overlapping_boxes'])} 个检出框有重合")
        else:
            logger.info("检出框无重合")

    results = []
    successful_count = 0

    for bbox_info in bboxes_info:
        object_id = bbox_info['object_id']
        bbox = bbox_info['bbox']

        # 自动计算非重合区域（传递other_bboxes给单个函数）
        other_bboxes = [other['bbox'] for other in bboxes_info if other['object_id'] != object_id]

        # 调用单个函数处理每个bbox（自动计算非重合区域）
        points_3d = sample_3d_points_from_non_overlap_regions(
            scene_data, img_idx, bbox, transform, config, other_bboxes
        )

        results.append(points_3d)
        if points_3d is not None:
            successful_count += 1

    logger.info(f"批量采样完成: {successful_count}/{len(bboxes_info)} 个检出框成功采样3D点")
    return results


def sample_3d_points_from_non_overlap_regions(
    scene_data: Dict, img_idx: int, bbox: List[float],
    transform: ImageTransformBase, config: SKUMatchingConfig,
    other_bboxes: Optional[List[List[float]]] = None
) -> Optional[torch.Tensor]:
    """从检出框的非重合区域中采样 3D 点 - 增强版（自动计算非重合区域 + 深度一致性检查 + 降级策略）

    Args:
        scene_data: 场景数据
        img_idx: 图像索引
        bbox: 检出框 [x1, y1, x2, y2]
        transform: 坐标变换器
        config: 配置参数
        other_bboxes: 其他检出框列表（可选）。如果提供，将自动计算非重合区域

    Returns:
        3D点张量或None（如果有效点不足10个）
    """
    from .bbox_utils import compute_non_overlap_regions

    device = scene_data['depth'].device
    H, W = scene_data['depth'].shape[1:3]

    # 自动计算非重合区域
    if other_bboxes is not None:
        non_overlap_regions = compute_non_overlap_regions(
            bbox, other_bboxes, config.overlap_threshold
        )
    else:
        # 如果没有提供other_bboxes，使用整个bbox作为单一区域
        non_overlap_regions = [bbox]

    if not non_overlap_regions:
        logger.debug(f"检出框 {bbox} 没有非重合区域，跳过 3D 采样")
        return None

    # 获取相机外参（用于深度一致性检查）
    E = scene_data['extrinsic'][img_idx]

    all_region_points = []
    all_region_coords = []  # 保存像素坐标用于降级策略

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

        # 获取有效的 3D 点和深度
        valid_world_points = world_points_region[valid_mask]
        valid_depths = depth_region[valid_mask]

        # 为有效点准备 2D 像素坐标（与 depth/world_points 同一坐标系，用于高斯加权）
        valid_y_indices, valid_x_indices = torch.where(valid_mask)
        coords_x = valid_x_indices.to(device=device, dtype=torch.float32) + float(x1)
        coords_y = valid_y_indices.to(device=device, dtype=torch.float32) + float(y1)

        # 🔍 深度一致性检查：转换到相机坐标系比较
        points_cam = transform_world_to_camera(valid_world_points.to(E.device), E)
        camera_depths = points_cam[:, 2]
        depth_diff = torch.abs(camera_depths - valid_depths)
        depth_consistent_mask = depth_diff < config.depth_consistency_threshold

        if depth_consistent_mask.sum() >= 3:
            # 使用深度一致的点
            consistent_points = valid_world_points[depth_consistent_mask]
            # 对应的 2D 像素坐标（用于高斯权重）
            consistent_coords_x = coords_x[depth_consistent_mask]
            consistent_coords_y = coords_y[depth_consistent_mask]

            # 按区域面积比例限制每个区域采样上限
            max_points_for_region = max(
                3,
                min(len(consistent_points), int(config.max_3d_points_per_bbox * 0.3))
            )

            if len(consistent_points) > max_points_for_region:
                if config.enable_gaussian_sampling:
                    # 在当前坐标系下，以 bbox 中心为高斯中心进行加权采样
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    rx = max(1.0, (x2 - x1) / 2.0)
                    ry = max(1.0, (y2 - y1) / 2.0)

                    dx_norm = (consistent_coords_x - cx) / rx
                    dy_norm = (consistent_coords_y - cy) / ry
                    distances = torch.sqrt(dx_norm * dx_norm + dy_norm * dy_norm)

                    sigma = float(config.gaussian_sigma)
                    weights = torch.exp(-distances * distances / (2.0 * sigma * sigma))
                    truncate = float(config.gaussian_truncate)
                    weights = torch.where(
                        distances > sigma * truncate,
                        torch.zeros_like(weights),
                        weights,
                    )

                    if weights.sum() <= 1e-6:
                        # 权重退化时回退到均匀采样
                        weights = torch.ones_like(weights)
                    probs = weights / weights.sum()
                    indices = torch.multinomial(probs, max_points_for_region, replacement=False)
                else:
                    # 关闭高斯采样时保持原有的均匀采样行为
                    weights = torch.ones(len(consistent_points), device=device)
                    indices = torch.multinomial(weights, max_points_for_region, replacement=False)

                sampled_points = consistent_points[indices]
            else:
                sampled_points = consistent_points

            all_region_points.append(sampled_points)
        else:
            # 🔄 降级策略：深度不一致时，保存坐标用于后续重建
            valid_y_indices, valid_x_indices = torch.where(valid_mask)
            for i in range(min(len(valid_y_indices), int(config.max_3d_points_per_bbox * 0.3))):
                y_coord = valid_y_indices[i].item() + y1
                x_coord = valid_x_indices[i].item() + x1
                all_region_coords.append((x_coord, y_coord))

    # 如果有深度一致的点，优先使用
    if all_region_points:
        all_points = torch.cat(all_region_points, dim=0)

        # 检查是否满足最小点数要求（min_3d_sample_points）
        if len(all_points) < config.min_3d_sample_points:
            logger.debug(f"检出框 {bbox} 深度一致点不足{config.min_3d_sample_points}个: {len(all_points)}，尝试降级策略")
            # 继续执行降级策略
        else:
            # 最终采样限制
            if len(all_points) > config.max_3d_points_per_bbox:
                # 使用multinomial采样（比randperm快约20%）
                weights = torch.ones(len(all_points), device=device)
                indices = torch.multinomial(weights, config.max_3d_points_per_bbox, replacement=False)
                all_points = all_points[indices]

            logger.debug(f"从 {len(non_overlap_regions)} 个非重合区域采样了 {len(all_points)} 个深度一致的 3D 点")
            return all_points

    # 🔄 降级策略：使用 enhanced_backproject_2d_to_3d 重新计算
    if all_region_coords:
        logger.debug(f"检出框 {bbox} 使用降级策略重建 3D 点")
        reconstructed_points = []

        for x_coord, y_coord in all_region_coords:
            point_3d = enhanced_backproject_2d_to_3d(
                x_coord, y_coord, scene_data, img_idx, config
            )
            if point_3d is not None:
                reconstructed_points.append(point_3d)

        if len(reconstructed_points) < config.min_3d_sample_points:
            logger.debug(f"检出框 {bbox} 降级重建后仍不足{config.min_3d_sample_points}个点: {len(reconstructed_points)}")
            return None

        sampled_points = torch.stack(reconstructed_points)

        # 最终采样限制
        if len(sampled_points) > config.max_3d_points_per_bbox:
            # 使用multinomial采样（比randperm快约20%）
            weights = torch.ones(len(sampled_points), device=device)
            indices = torch.multinomial(weights, config.max_3d_points_per_bbox, replacement=False)
            sampled_points = sampled_points[indices]

        logger.debug(f"从 {len(non_overlap_regions)} 个非重合区域通过降级策略重建了 {len(sampled_points)} 个 3D 点")
        return sampled_points

    logger.debug(f"检出框 {bbox} 的所有非重合区域都无有效 3D 点")
    return None


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
    return transform_camera_to_world(point_cam, extrinsic)


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
                                             ref_depth_mean: float, ref_points_3d: Optional[torch.Tensor] = None) -> Optional[Dict]:
    """找到投影点最多落入的检出框，并进行3D几何验证。

    验证维度：投影命中率 + 3D质心距离 + 平面共面约束（法向对齐）。
    平面约束针对货架场景：同一物理面（层板/商品正面）法向应一致，跨层误匹配法向偏差大。

    返回：最佳匹配 dict（兼容旧调用）。验证过的全部候选也通过返回 dict 的
    'validated_candidates' 字段携带（按 combined_score 降序），供唯一性 fallback 分配复用，
    避免被同 target 框竞争淘汰的 ref 重新采样/验证即可取其次优非冲突框。
    """
    if len(projected_points) == 0:
        return None

    # 预拟合参考点平面（每个 ref 物体只拟合一次，候选验证复用）
    ref_normal, ref_residual = (None, None)
    if ref_points_3d is not None and len(ref_points_3d) >= 3:
        ref_normal, ref_residual = _fit_plane_svd(ref_points_3d)

    best_match = None
    best_score = 0.0
    validated_candidates = []

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
        target_points_3d = sample_3d_points_from_non_overlap_regions(
            scene_data, target_img_idx,
            target_transform.map_bbox_to_original(bbox),
            target_transform, config, None  # 不传递other_bboxes，使用整个bbox
        )

        if target_points_3d is None or len(target_points_3d) < config.min_3d_sample_points:
            continue

        # 计算目标3D点的统计信息
        target_3d_center = target_points_3d.mean(dim=0)

        # 3D距离验证
        spatial_distance = torch.norm(ref_3d_center - target_3d_center).item()

        # 平面共面约束：法向对齐参与评分（不硬拒绝，低法向时 coplanar_score 低，由评分自然淘汰）
        coplanar_score = 0.0
        if ref_normal is not None:
            tgt_normal, tgt_residual = _fit_plane_svd(target_points_3d)
            if np.any(ref_normal) and np.any(tgt_normal):
                normal_alignment = abs(float(np.dot(ref_normal, tgt_normal)))  # |cos|∈[0,1]
                # 残差越小（点越共面）得分越高
                residual_score = max(0.0, 1.0 - tgt_residual / config.max_3d_distance)
                coplanar_score = normal_alignment * residual_score

        # 组合评分：投影命中率 + 3D质心距离 + 平面共面约束（三因素）
        geometry_score = max(0.0, 1.0 - spatial_distance / config.max_3d_distance)
        if ref_normal is not None:
            combined_score = match_ratio * 0.5 + geometry_score * 0.2 + coplanar_score * 0.3
        else:
            # 无参考平面（点太少）时退化为投影+质心两因素
            combined_score = match_ratio * 0.6 + geometry_score * 0.4

        # 严格筛选：必须满足3D几何约束
        if spatial_distance > config.max_3d_distance:
            continue

        candidate = {
            'target_bbox_info': bbox_info,
            'points_in_bbox': points_in_bbox,
            'total_points': len(projected_points),
            'match_ratio': match_ratio,
            '3d_distance': spatial_distance,
            'combined_score': combined_score
        }
        validated_candidates.append(candidate)

        if combined_score > best_score:
            best_match = candidate
            best_score = combined_score

    # 携带全部验证候选（降序），供唯一性 fallback 分配复用
    if best_match is not None:
        validated_candidates.sort(key=lambda c: c['combined_score'], reverse=True)
        best_match['validated_candidates'] = validated_candidates

    return best_match


def apply_uniqueness_constraint(candidate_matches: List[Dict]) -> List[Dict]:
    """应用唯一性约束 + 贪心 fallback 分配。

    每个目标框只能匹配一个参考框（选最高分）。被同 target 框竞争淘汰的 ref 不再直接丢弃，
    而是取其验证候选列表('validated_candidates')中的次优非冲突框--已分配给别 ref 的框跳过。
    救回竞争淘汰漏检(同 ref 多 obj 抢同一框,输者本可匹配次优框却无 fallback)。

    贪心策略:所有 (ref, candidate) 对按 combined_score 全局降序,高分优先占框;
    低分 ref 在其候选链中找首个未被占用 target 框。
    """
    if not candidate_matches:
        return []

    all_pairs = []
    for match in candidate_matches:
        ref_obj_id = match.get('ref_obj_id', 'unknown')
        cands = match.get('validated_candidates')
        if not cands:
            cands = [match]
        for c in cands:
            all_pairs.append((ref_obj_id, c))

    all_pairs.sort(key=lambda rc: rc[1].get('combined_score', 0.0), reverse=True)

    assigned_targets = set()
    assigned_refs = set()
    final_matches = []

    for ref_obj_id, cand in all_pairs:
        if ref_obj_id in assigned_refs:
            continue
        target_id = cand['target_bbox_info']['object_id']
        if target_id in assigned_targets:
            continue
        assigned_refs.add(ref_obj_id)
        assigned_targets.add(target_id)
        m = dict(cand)
        m['ref_obj_id'] = ref_obj_id
        m.pop('validated_candidates', None)
        final_matches.append(m)

    evicted = len(candidate_matches) - len(final_matches)
    logger.debug(
        f"Applied uniqueness constraint: {len(candidate_matches)} candidates -> {len(final_matches)} matches "
        f"(evicted {evicted} refs found no free target)"
    )
    return final_matches
