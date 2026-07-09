"""
SKU匹配系统可视化模块

包含结果可视化和图像处理功能
采用SAM3高质量可视化方案：K-Means LAB色彩空间颜色生成
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans

from .config import SKUMatchingConfig
from .data_utils import extract_bboxes_from_detections

logger = logging.getLogger(__name__)


def generate_perceptually_distinct_colors(n_colors=128, n_samples=5000, seed=42):
    """生成感知上差异最大的颜色集合（基于SAM3方法）

    使用K-Means在LAB色彩空间中聚类，确保生成的颜色在人眼感知上高度区分。
    LAB色彩空间是感知均匀的色彩空间，欧氏距离对应于人眼感知的色差。

    Args:
        n_colors: 要生成的颜色数量
        n_samples: 用于聚类的随机RGB样本数量
        seed: 随机种子，确保颜色生成的可重复性

    Returns:
        np.ndarray: 形状为 (n_colors, 3) 的RGB颜色数组，值范围 [0, 255]
    """
    # Step 1: 生成随机RGB样本
    np.random.seed(seed)
    rgb = np.random.rand(n_samples, 3)

    # Step 2: 转换到LAB色彩空间（感知均匀）
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)

    # Step 3: 在LAB空间中进行K-Means聚类
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=seed)
    kmeans.fit(lab)
    centers_lab = kmeans.cluster_centers_

    # Step 4: 将聚类中心转回RGB空间
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb * 255, 0, 255).astype(np.uint8)

    return colors_rgb


# 全局颜色池（预计算，避免重复生成）
_GLOBAL_COLOR_POOL = generate_perceptually_distinct_colors(n_colors=128, n_samples=5000)


def get_color_from_pool(obj_id: int) -> tuple:
    """从全局颜色池中获取稳定的颜色

    Args:
        obj_id: 对象ID

    Returns:
        tuple: BGR格式的颜色 (B, G, R)
    """
    color_rgb = _GLOBAL_COLOR_POOL[obj_id % len(_GLOBAL_COLOR_POOL)]
    # 转换为OpenCV的BGR格式
    return tuple(map(int, color_rgb[::-1]))


def visualize_results(
    images: torch.Tensor,
    reference_idx: int,
    points_per_object: Dict[int, Dict],
    correspondences: Dict[int, List[Dict]],
    config: SKUMatchingConfig,
    detections: Optional[List[Dict]] = None,
    transforms_info: Optional[List] = None,
) -> None:
    """可视化追踪/匹配结果（使用模型输入尺寸与坐标）

    Args:
        images: 图像张量 (S, C, H, W)，已经过模型预处理
                - VGGT: 裁剪/填充后的 518×518 或批量对齐尺寸
                - Pi3: 依据像素上限 resize 后的统一尺寸
        reference_idx: 参考图像索引
        points_per_object: 物体点映射
        correspondences: 对应关系结果
        config: 配置参数
        detections: 原始检测结果列表
        transforms_info: 图像变换信息列表
    """
    logger.info("Generating visualization for matching results...")

    try:
        # 1. 可视化参考图像和其上的检测框（使用模型输入空间的图像和坐标）
        ref_image_np = (images[reference_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ref_image_bgr = cv2.cvtColor(ref_image_np, cv2.COLOR_RGB2BGR)
        
        overlay = ref_image_bgr.copy()
        colors: Dict[int, List[int]] = {}

        # 绘制参考图像的检测框（使用模型输入坐标）
        for obj_id, data in points_per_object.items():
            # data['bbox'] 已经是在模型输入空间（VGGT/Pi3 预处理后的坐标）
            input_bbox = data["bbox"]

            # 使用LAB空间优化的颜色池（SAM3方法）
            color = get_color_from_pool(obj_id)
            colors[obj_id] = color  # 保存颜色以便在目标图像中复用

            # 转换为整数坐标并确保在当前图像范围内
            h, w = ref_image_bgr.shape[:2]
            x1, y1, x2, y2 = [max(0, int(c)) for c in input_bbox]
            x1, x2 = min(x1, w - 1), min(x2, w)
            y1, y2 = min(y1, h - 1), min(y2, h)

            if x1 < x2 and y1 < y2:  # 有效框
                # 绘制检测框
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # 绘制中心点
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(overlay, (center_x, center_y), 3, color, -1)
                
                # 绘制ID标签 - 只显示ID，字体适中粗细
                label = f"{obj_id}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(
                    overlay,
                    label,
                    (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            else:
                logger.warning(f"Invalid reference bbox {obj_id}")

        ref_output_path = Path(config.output_dir) / "reference_image_with_bboxes.jpg"
        ref_output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        cv2.imwrite(str(ref_output_path), overlay)
        logger.info(f"Saved reference visualization")

        # 2. 可视化每个目标图像上的所有检测框和匹配结果
        for s_idx, matched_boxes in correspondences.items():
            # 使用模型输入空间的图像
            target_image_np = (images[s_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            target_image_bgr = cv2.cvtColor(target_image_np, cv2.COLOR_RGB2BGR)

            # 获取目标图像的所有原始检测框
            if detections and s_idx < len(detections) and transforms_info and s_idx < len(transforms_info):
                # 提取目标图像的所有检测框
                target_bboxes = extract_bboxes_from_detections([detections[s_idx]], 0, config)
                target_transform = transforms_info[s_idx]
                
                # 创建匹配框ID集合，用于区分匹配和未匹配的框
                matched_target_bbox_ids = set()
                for item in matched_boxes:
                    matched_target_bbox_ids.add(item.get('target_obj_id', item.get('target_bbox_id')))
                
                # 绘制所有原始检测框（使用与reference一致的样式）
                h, w = target_image_bgr.shape[:2]
                
                for bbox_info in target_bboxes:
                    # 将原始检测框映射到模型输入坐标
                    input_bbox = target_transform.map_bbox_to_final(bbox_info['bbox'])
                    target_bbox_id = bbox_info['object_id']
                    
                    # 转换为整数坐标
                    x1, y1, x2, y2 = [max(0, int(c)) for c in input_bbox]
                    x1, x2 = min(x1, w-1), min(x2, w)
                    y1, y2 = min(y1, h-1), min(y2, h)
                    
                    if x1 < x2 and y1 < y2:  # 有效框
                        # 区分匹配和未匹配的框，但使用统一的绘制样式
                        if target_bbox_id in matched_target_bbox_ids:
                            # 匹配的框：使用绿色，但采用与reference一致的样式
                            color = (0, 255, 0)
                            cv2.rectangle(target_image_bgr, (x1, y1), (x2, y2), color, 2)
                            
                            # 绘制中心点
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            cv2.circle(target_image_bgr, (center_x, center_y), 3, color, -1)
                            
                            # 绘制ID标签（与reference一致的格式）
                            label = f"{target_bbox_id}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(target_image_bgr, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
                            cv2.putText(target_image_bgr, label, (x1, max(y1 - 5, 10)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            # 未匹配的框：使用灰色细线，不绘制ID和中心点
                            color = (128, 128, 128)
                            cv2.rectangle(target_image_bgr, (x1, y1), (x2, y2), color, 1)  # 细线
            
            # 在匹配的框上添加参考对象的匹配信息
            for item in matched_boxes:
                obj_id = item['object_id']
                
                # 获取模型输入坐标（而不是原图坐标）
                input_box = item.get('vggt_box', [])
                target_bbox_id = item.get('target_obj_id', item.get('target_bbox_id', 'N/A'))
                
                if not input_box or len(input_box) != 4:
                    logger.warning(f"WARNING: 目标图像 {s_idx} 对象 {obj_id}: 无效的模型输入坐标 {input_box}")
                    continue
                
                # 使用与参考图像一致的颜色
                color = colors.get(obj_id, [255, 255, 255])
                
                # 转换为整数坐标并确保在当前图像范围内
                h, w = target_image_bgr.shape[:2]
                x1, y1, x2, y2 = [max(0, int(c)) for c in input_box]
                x1, x2 = min(x1, w-1), min(x2, w)
                y1, y2 = min(y1, h-1), min(y2, h)
                
                if x1 < x2 and y1 < y2:  # 有效框
                    # 绘制参考对象匹配标识（使用与reference一致的样式，但位置稍微偏移）
                    label = f"{obj_id}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(target_image_bgr, (x1, y1-label_h-35), (x1+label_w, y1-25), color, -1)
                    cv2.putText(target_image_bgr, label, (x1, max(y1 - 30, 25)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    logger.warning(f"Invalid target bbox {obj_id} in image {s_idx}")

            output_filename = Path(config.output_dir) / f"target_image_{s_idx}_all_bboxes_and_matches.jpg"
            output_filename.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            cv2.imwrite(str(output_filename), target_image_bgr)
            
    except (cv2.error, FileNotFoundError, PermissionError, KeyError, IndexError) as e:
        logger.error(f"Failed to generate visualization: {e}")
        raise


def draw_bbox_with_id(
    image: np.ndarray,
    bbox: List[float],
    obj_id: int,
    color: tuple,
    thickness: int = 2,
    font_scale: float = 0.7,
    show_center: bool = True
) -> None:
    """在图像上绘制带ID标签的边界框（SAM3增强版）

    增强特性：
    - 更粗的边界框线条（提升清晰度）
    - 白色半透明背景 + 彩色粗体文字（更易读）
    - 可选的中心点标记（提升定位精度）
    - 自动边距padding（避免文字被裁剪）

    Args:
        image: 输入图像 (BGR格式)
        bbox: 边界框坐标 [x1, y1, x2, y2]
        obj_id: 对象ID
        color: 绘制颜色 (B, G, R)
        thickness: 线条粗细
        font_scale: 字体大小
        show_center: 是否显示中心点
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
    x1, x2 = min(x1, w-1), min(x2, w)
    y1, y2 = min(y1, h-1), min(y2, h)

    if x1 < x2 and y1 < y2:  # 有效框
        # 1. 绘制边界框（加粗线条）
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness + 1)

        # 2. 绘制中心点（可选）
        if show_center:
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(image, (center_x, center_y), 4, color, -1)
            cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), 1)  # 白色外圈

        # 3. 绘制ID标签（SAM3风格：白色半透明背景 + 彩色粗体文字）
        label = f"{obj_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, 2)

        # 计算标签位置（添加padding）
        padding = 4
        label_x = x1
        label_y = max(label_h + padding * 2, y1 - 5)

        # 绘制白色半透明背景框
        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (label_x, label_y - label_h - padding),
            (label_x + label_w + padding * 2, label_y + padding),
            (255, 255, 255),
            -1
        )
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)  # 75%透明度

        # 绘制彩色粗体ID文字
        cv2.putText(
            image,
            label,
            (label_x + padding, label_y),
            font,
            font_scale,
            color,
            2,  # 粗体效果
            cv2.LINE_AA  # 抗锯齿
        )


def draw_dashed_bbox(
    image: np.ndarray,
    bbox: List[float],
    color: tuple,
    thickness: int = 2,
    dash_length: int = 10
) -> None:
    """在图像上绘制虚线边界框
    
    Args:
        image: 输入图像 (BGR格式)
        bbox: 边界框坐标 [x1, y1, x2, y2]
        color: 绘制颜色 (B, G, R)
        thickness: 线条粗细
        dash_length: 虚线长度
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
    x1, x2 = min(x1, w-1), min(x2, w)
    y1, y2 = min(y1, h-1), min(y2, h)
    
    if x1 < x2 and y1 < y2:  # 有效框
        # 绘制虚线效果的边界框
        for i in range(0, x2 - x1, dash_length * 2):
            cv2.line(image, (x1 + i, y1), (min(x1 + i + dash_length, x2), y1), color, thickness)
            cv2.line(image, (x1 + i, y2), (min(x1 + i + dash_length, x2), y2), color, thickness)
        for i in range(0, y2 - y1, dash_length * 2):
            cv2.line(image, (x1, y1 + i), (x1, min(y1 + i + dash_length, y2)), color, thickness)
            cv2.line(image, (x2, y1 + i), (x2, min(y1 + i + dash_length, y2)), color, thickness)


def generate_colors_for_objects(object_ids: List[int]) -> Dict[int, tuple]:
    """为对象ID生成稳定的高对比度颜色映射（SAM3方法）

    使用预计算的LAB空间K-Means聚类颜色池，确保：
    - 颜色在感知上高度区分（LAB空间优化）
    - 相同ID始终获得相同颜色（稳定映射）
    - 高性能（从预计算池中直接查询）

    Args:
        object_ids: 对象ID列表

    Returns:
        对象ID到BGR颜色的映射字典
    """
    colors: Dict[int, tuple] = {}
    for obj_id in object_ids:
        colors[obj_id] = get_color_from_pool(obj_id)
    return colors


def save_visualization_summary(
    correspondences: Dict[int, List[Dict]],
    config: SKUMatchingConfig,
    reference_image_idx: int,
    filename: str = "matching_summary.txt"
) -> None:
    """保存匹配结果的文本摘要
    
    Args:
        correspondences: 匹配结果
        config: 配置参数
        filename: 保存文件名
    """
    try:
        summary_path = Path(config.output_dir) / filename
        
        # 确保输出目录存在
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("SKU匹配结果日志\n")
            f.write("=" * 50 + "\n\n")
            
            total_matches = 0
            
            # 按目标图像索引排序处理
            for target_idx in sorted(correspondences.keys()):
                matches = correspondences[target_idx]
                
                # 按参考对象id排序匹配结果
                sorted_matches = sorted(matches, key=lambda x: x['object_id'])
                
                # 写入匹配详情
                for match in sorted_matches:
                    ref_id = match['object_id']
                    target_id = match.get('target_obj_id', match.get('target_bbox_id', 'N/A'))
                    ratio = match['correspondence_ratio']
                    matched_points = match.get('matched_points', 0)
                    total_points = match.get('total_points', 0)
                    
                    f.write(f"Matched ref {ref_id} → target {target_id} (hit ratio: {ratio:.2f} {matched_points}/{total_points})\n")
                
                # 写入分组信息（使用真实的参考图像索引）
                f.write(f"Matching objects between reference image {reference_image_idx} and target image {target_idx}\n")
                f.write(f"Found {len(matches)} matches in image {target_idx}\n\n")
                
                total_matches += len(matches)

            algo_name = config.get_algorithm_name()

            # 写入汇总信息
            f.write(f"{algo_name} complete. Found correspondences in {len(correspondences)} images.\n")
            f.write(f"Found {total_matches} matches across {len(correspondences)} images\n")
        
        logger.info(f"Saved matching summary to {summary_path}")
        
    except (FileNotFoundError, PermissionError, UnicodeEncodeError) as e:
        logger.error(f"Failed to save visualization summary: {e}")
        raise
