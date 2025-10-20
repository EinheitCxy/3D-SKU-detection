"""
VGGT-Detection 帧对齐验证和修复模块

核心功能：
1. 验证 world_points 帧顺序与 detection 对齐
2. 提供不一致时的自动修复策略
3. 生成详细的对齐报告

精简版：仅保留核心对齐逻辑，移除未使用的辅助功能
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class FrameAlignmentError(Exception):
    """帧对齐错误异常"""
    pass


class VGGTDetectionAligner:
    """VGGT输出与检测结果对齐器（精简版）"""

    @staticmethod
    def validate_and_align(
        vggt_data: Dict[str, np.ndarray],
        detections: List[Dict],
        detection_indices: Optional[List[int]] = None,
        vggt_image_ids: Optional[List[int]] = None,
        strict_mode: bool = True
    ) -> Tuple[Dict[str, np.ndarray], List[Dict], Dict[str, Any]]:
        """
        验证并对齐VGGT输出与检测结果

        Args:
            vggt_data: VGGT输出数据，必须包含world_points
            detections: 检测结果列表
            detection_indices: 检测结果对应的图像ID列表（从load_detections获取）
            vggt_image_ids: VGGT处理的图像ID列表
            strict_mode: 严格模式，不一致时报错；False时尝试修复

        Returns:
            (对齐后的vggt_data, 对齐后的detections, 对齐报告)

        Raises:
            FrameAlignmentError: 严格模式下发现不可修复的对齐问题
        """
        logger.info("开始VGGT-Detection帧对齐验证...")

        # 1. 基础检查
        world_points = vggt_data.get("world_points")
        if world_points is None:
            raise ValueError("VGGT数据缺少world_points")

        # 获取VGGT帧数
        if world_points.ndim == 4:  # (S, H, W, 3)
            vggt_frame_count = world_points.shape[0]
        elif world_points.ndim == 3:  # (H, W, 3) - 单帧
            vggt_frame_count = 1
        else:
            raise ValueError(f"world_points维度异常: {world_points.shape}")

        detection_count = len(detections)

        logger.info(f"   VGGT帧数: {vggt_frame_count}")
        logger.info(f"   检测结果数: {detection_count}")

        # 2. 推断图像ID（如果未提供）
        if detection_indices is None:
            detection_indices = list(range(detection_count))
            logger.warning("WARNING: 未提供detection_indices，假设为[0,1,2,...]")

        if vggt_image_ids is None:
            vggt_image_ids = list(range(vggt_frame_count))
            logger.warning("WARNING: 未提供vggt_image_ids，假设为[0,1,2,...]")

        # 3. 对齐分析
        alignment_report = VGGTDetectionAligner._analyze_alignment(
            vggt_image_ids, detection_indices, vggt_frame_count, detection_count
        )

        # 4. 决定处理策略
        if alignment_report["is_perfectly_aligned"]:
            logger.info("帧对齐验证通过，无需修复")
            return vggt_data, detections, alignment_report

        elif not strict_mode:
            logger.warning("WARNING: 检测到帧对齐问题，尝试自动修复...")
            return VGGTDetectionAligner._attempt_repair(
                vggt_data, detections, vggt_image_ids, detection_indices, alignment_report
            )

        else:
            # 严格模式：报错
            error_msg = VGGTDetectionAligner._format_alignment_error(alignment_report)
            raise FrameAlignmentError(error_msg)

    @staticmethod
    def _analyze_alignment(
        vggt_image_ids: List[int],
        detection_indices: List[int],
        vggt_frame_count: int,
        detection_count: int
    ) -> Dict[str, Any]:
        """分析对齐状况"""

        vggt_set = set(vggt_image_ids)
        detection_set = set(detection_indices)

        # 计算交集和差集
        common_ids = vggt_set & detection_set
        vggt_only = vggt_set - detection_set
        detection_only = detection_set - vggt_set

        # 检查顺序一致性
        common_ids_sorted = sorted(common_ids)
        vggt_order_matches = (vggt_image_ids == common_ids_sorted)
        detection_order_matches = (detection_indices == common_ids_sorted)

        is_perfectly_aligned = (
            vggt_frame_count == detection_count and
            vggt_image_ids == detection_indices and
            len(vggt_only) == 0 and
            len(detection_only) == 0
        )

        alignment_report = {
            "is_perfectly_aligned": is_perfectly_aligned,
            "vggt_frame_count": vggt_frame_count,
            "detection_count": detection_count,
            "vggt_image_ids": vggt_image_ids,
            "detection_indices": detection_indices,
            "common_count": len(common_ids),
            "vggt_only_count": len(vggt_only),
            "detection_only_count": len(detection_only),
            "vggt_only_ids": sorted(vggt_only),
            "detection_only_ids": sorted(detection_only),
            "common_ids": common_ids_sorted,
            "vggt_order_matches": vggt_order_matches,
            "detection_order_matches": detection_order_matches,
            "coverage_ratio": len(common_ids) / max(len(vggt_set), len(detection_set)),
        }

        return alignment_report

    @staticmethod
    def _attempt_repair(
        vggt_data: Dict[str, np.ndarray],
        detections: List[Dict],
        vggt_image_ids: List[int],
        detection_indices: List[int],
        alignment_report: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], List[Dict], Dict[str, Any]]:
        """尝试修复对齐问题"""

        common_ids = alignment_report["common_ids"]

        if len(common_ids) == 0:
            raise FrameAlignmentError("无交集图像，无法修复对齐")

        logger.info(f"修复对齐：保留{len(common_ids)}个共同图像")

        # 创建映射表
        vggt_id_to_idx = {img_id: idx for idx, img_id in enumerate(vggt_image_ids)}
        detection_id_to_idx = {img_id: idx for idx, img_id in enumerate(detection_indices)}

        # 构建对齐后的索引
        aligned_vggt_indices = []
        aligned_detection_indices = []
        aligned_image_ids = []

        for img_id in common_ids:
            if img_id in vggt_id_to_idx and img_id in detection_id_to_idx:
                aligned_vggt_indices.append(vggt_id_to_idx[img_id])
                aligned_detection_indices.append(detection_id_to_idx[img_id])
                aligned_image_ids.append(img_id)

        # 重新排列VGGT数据
        aligned_vggt_data = {}
        for key, data in vggt_data.items():
            if isinstance(data, np.ndarray) and data.ndim >= 1:
                # 假设第一个维度是batch/frame维度
                if data.shape[0] == len(vggt_image_ids):
                    aligned_vggt_data[key] = data[aligned_vggt_indices]
                else:
                    # 不是帧维度，直接复制
                    aligned_vggt_data[key] = data
            else:
                aligned_vggt_data[key] = data

        # 重新排列检测数据
        aligned_detections = [detections[i] for i in aligned_detection_indices]

        # 更新报告
        alignment_report.update({
            "repair_applied": True,
            "repaired_frame_count": len(aligned_image_ids),
            "repaired_image_ids": aligned_image_ids,
            "dropped_vggt_frames": len(vggt_image_ids) - len(aligned_image_ids),
            "dropped_detection_frames": len(detections) - len(aligned_image_ids),
        })

        logger.info(f"修复完成：{len(aligned_image_ids)}帧对齐")
        logger.info(f"   丢弃VGGT帧: {alignment_report['dropped_vggt_frames']}")
        logger.info(f"   丢弃检测帧: {alignment_report['dropped_detection_frames']}")

        return aligned_vggt_data, aligned_detections, alignment_report

    @staticmethod
    def _format_alignment_error(alignment_report: Dict[str, Any]) -> str:
        """格式化对齐错误信息"""
        error_lines = [
            "VGGT-Detection帧对齐验证失败！",
            f"VGGT帧数: {alignment_report['vggt_frame_count']}",
            f"检测结果数: {alignment_report['detection_count']}",
            f"共同图像: {alignment_report['common_count']}",
            f"覆盖率: {alignment_report['coverage_ratio']:.2%}",
        ]

        if alignment_report['vggt_only_count'] > 0:
            error_lines.append(f"仅在VGGT中: {alignment_report['vggt_only_ids']}")

        if alignment_report['detection_only_count'] > 0:
            error_lines.append(f"仅在检测中: {alignment_report['detection_only_ids']}")

        error_lines.extend([
            "",
            "解决方案:",
            "1. 设置 strict_mode=False 启用自动修复",
            "2. 检查图像文件名和编号一致性",
            "3. 确保VGGT和检测使用相同的图像集合"
        ])

        return "\n".join(error_lines)