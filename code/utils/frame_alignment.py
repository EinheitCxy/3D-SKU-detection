"""
3D重建-Detection 帧对齐验证和修复模块

核心功能：
1. 验证 world_points 帧顺序与 detection 对齐
2. 提供不一致时的自动修复策略
3. 生成详细的对齐报告

精简版：仅保留核心对齐逻辑，移除未使用的辅助功能
适用于所有3D重建后端
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FrameAlignmentError(Exception):
    """帧对齐错误异常"""

    pass


class ReconstructionDetectionAligner:
    """3D重建输出与检测结果对齐器（精简版）"""

    @staticmethod
    def validate_and_align(
        reconstruction_data: Dict[str, np.ndarray],
        detections: List[Dict],
        detection_indices: Optional[List[int]] = None,
        reconstruction_image_ids: Optional[List[int]] = None,
        strict_mode: bool = True,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict], Dict[str, Any]]:
        """
        验证并对齐3D重建输出与检测结果

        Args:
            reconstruction_data: 3D重建输出数据，必须包含world_points
            detections: 检测结果列表
            detection_indices: 检测结果对应的图像ID列表（从load_detections获取）
            reconstruction_image_ids: 3D重建处理的图像ID列表
            strict_mode: 严格模式，不一致时报错；False时尝试修复

        Returns:
            (对齐后的reconstruction_data, 对齐后的detections, 对齐报告)

        Raises:
            FrameAlignmentError: 严格模式下发现不可修复的对齐问题
        """
        logger.debug("开始3D重建-Detection帧对齐验证...")

        # 1. 基础检查
        world_points = reconstruction_data.get("world_points")
        if world_points is None:
            raise ValueError("3D重建数据缺少world_points")

        # 获取重建帧数
        if world_points.ndim == 4:  # (S, H, W, 3)
            recon_frame_count = world_points.shape[0]
        elif world_points.ndim == 3:  # (H, W, 3) - 单帧
            recon_frame_count = 1
        else:
            raise ValueError(f"world_points维度异常: {world_points.shape}")

        detection_count = len(detections)

        logger.debug(f"   3D重建帧数: {recon_frame_count}")
        logger.debug(f"   检测结果数: {detection_count}")

        # 2. 推断图像ID（如果未提供）
        if detection_indices is None:
            detection_indices = list(range(detection_count))
            logger.warning("WARNING: 未提供detection_indices，假设为[0,1,2,...]")

        if reconstruction_image_ids is None:
            reconstruction_image_ids = list(range(recon_frame_count))
            logger.warning("WARNING: 未提供reconstruction_image_ids，假设为[0,1,2,...]")

        # 3. 对齐分析
        alignment_report = ReconstructionDetectionAligner._analyze_alignment(
            reconstruction_image_ids,
            detection_indices,
            recon_frame_count,
            detection_count,
        )

        # 4. 决定处理策略
        if alignment_report["is_perfectly_aligned"]:
            logger.debug("帧对齐验证通过，无需修复")
            return reconstruction_data, detections, alignment_report

        elif not strict_mode:
            logger.warning("检测到帧对齐问题，尝试自动修复...")
            return ReconstructionDetectionAligner._attempt_repair(
                reconstruction_data,
                detections,
                reconstruction_image_ids,
                detection_indices,
                alignment_report,
            )

        else:
            # 严格模式：报错
            error_msg = ReconstructionDetectionAligner._format_alignment_error(
                alignment_report
            )
            raise FrameAlignmentError(error_msg)

    @staticmethod
    def _analyze_alignment(
        reconstruction_image_ids: List[int],
        detection_indices: List[int],
        recon_frame_count: int,
        detection_count: int,
    ) -> Dict[str, Any]:
        """分析对齐状况"""

        recon_set = set(reconstruction_image_ids)
        detection_set = set(detection_indices)

        # 计算交集和差集
        common_ids = recon_set & detection_set
        recon_only = recon_set - detection_set
        detection_only = detection_set - recon_set

        # 检查顺序一致性
        common_ids_sorted = sorted(common_ids)
        recon_order_matches = reconstruction_image_ids == common_ids_sorted
        detection_order_matches = detection_indices == common_ids_sorted

        is_perfectly_aligned = (
            recon_frame_count == detection_count
            and reconstruction_image_ids == detection_indices
            and len(recon_only) == 0
            and len(detection_only) == 0
        )

        alignment_report = {
            "is_perfectly_aligned": is_perfectly_aligned,
            "recon_frame_count": recon_frame_count,
            "detection_count": detection_count,
            "reconstruction_image_ids": reconstruction_image_ids,
            "detection_indices": detection_indices,
            "common_count": len(common_ids),
            "recon_only_count": len(recon_only),
            "detection_only_count": len(detection_only),
            "recon_only_ids": sorted(recon_only),
            "detection_only_ids": sorted(detection_only),
            "common_ids": common_ids_sorted,
            "recon_order_matches": recon_order_matches,
            "detection_order_matches": detection_order_matches,
            "coverage_ratio": len(common_ids) / max(len(recon_set), len(detection_set)),
        }

        return alignment_report

    @staticmethod
    def _attempt_repair(
        reconstruction_data: Dict[str, np.ndarray],
        detections: List[Dict],
        reconstruction_image_ids: List[int],
        detection_indices: List[int],
        alignment_report: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], List[Dict], Dict[str, Any]]:
        """尝试修复对齐问题"""

        common_ids = alignment_report["common_ids"]

        if len(common_ids) == 0:
            raise FrameAlignmentError("无交集图像，无法修复对齐")

        logger.info(f"修复对齐：保留{len(common_ids)}个共同图像")

        # 创建映射表
        recon_id_to_idx = {
            img_id: idx for idx, img_id in enumerate(reconstruction_image_ids)
        }
        detection_id_to_idx = {
            img_id: idx for idx, img_id in enumerate(detection_indices)
        }

        # 构建对齐后的索引
        aligned_recon_indices = []
        aligned_detection_indices = []
        aligned_image_ids = []

        for img_id in common_ids:
            if img_id in recon_id_to_idx and img_id in detection_id_to_idx:
                aligned_recon_indices.append(recon_id_to_idx[img_id])
                aligned_detection_indices.append(detection_id_to_idx[img_id])
                aligned_image_ids.append(img_id)

        # 重新排列3D重建数据
        aligned_recon_data = {}
        for key, data in reconstruction_data.items():
            if isinstance(data, np.ndarray) and data.ndim >= 1:
                # 假设第一个维度是batch/frame维度
                if data.shape[0] == len(reconstruction_image_ids):
                    aligned_recon_data[key] = data[aligned_recon_indices]
                else:
                    # 不是帧维度，直接复制
                    aligned_recon_data[key] = data
            else:
                aligned_recon_data[key] = data

        # 重新排列检测数据
        aligned_detections = [detections[i] for i in aligned_detection_indices]

        # 更新报告
        alignment_report.update(
            {
                "repair_applied": True,
                "repaired_frame_count": len(aligned_image_ids),
                "repaired_image_ids": aligned_image_ids,
                "dropped_frames": len(reconstruction_image_ids)
                - len(aligned_image_ids),
                "dropped_detection_frames": len(detections) - len(aligned_image_ids),
            }
        )

        logger.info(f"修复完成：{len(aligned_image_ids)}帧对齐")
        logger.debug(f"   丢弃3D重建帧: {alignment_report['dropped_frames']}")
        logger.debug(f"   丢弃检测帧: {alignment_report['dropped_detection_frames']}")

        return aligned_recon_data, aligned_detections, alignment_report

    @staticmethod
    def _format_alignment_error(alignment_report: Dict[str, Any]) -> str:
        """格式化对齐错误信息"""
        error_lines = [
            "3D重建-Detection帧对齐验证失败！",
            f"3D重建帧数: {alignment_report['recon_frame_count']}",
            f"检测结果数: {alignment_report['detection_count']}",
            f"共同图像: {alignment_report['common_count']}",
            f"覆盖率: {alignment_report['coverage_ratio']:.2%}",
        ]

        if alignment_report["recon_only_count"] > 0:
            error_lines.append(f"仅在3D重建中: {alignment_report['recon_only_ids']}")

        if alignment_report["detection_only_count"] > 0:
            error_lines.append(f"仅在检测中: {alignment_report['detection_only_ids']}")

        error_lines.extend(
            [
                "",
                "解决方案:",
                "1. 设置 strict_mode=False 启用自动修复",
                "2. 检查图像文件名和编号一致性",
                "3. 确保3D重建和检测使用相同的图像集合",
            ]
        )

        return "\n".join(error_lines)
