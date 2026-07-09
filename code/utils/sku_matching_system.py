"""
SKU匹配系统主模块

封装完整的SKU匹配流程，提供高级接口
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional
from pathlib import Path
from contextlib import nullcontext

# VGGT相关导入（路径由 utils/__init__.py 统一配置）
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except ImportError as e:
    raise ImportError(f"Failed to import VGGT modules: {e}")

from .config import SKUMatchingConfig
from .data_utils import save_correspondences_json, load_detections
from .transforms import build_transforms
from .matching_algorithms import find_object_correspondences
from .visualization import visualize_results, save_visualization_summary

logger = logging.getLogger(__name__)


class SKUMatchingSystem:
    """SKU匹配系统类
    
    封装完整的SKU匹配流程，包括：
    - 模型初始化和配置管理
    - 图像和检测数据加载
    - 坐标变换处理
    - 匹配算法执行
    - 结果可视化和保存
    """
    
    def __init__(self, config: Optional[SKUMatchingConfig] = None):
        """初始化SKU匹配系统。
        
        Args:
            config: 配置参数，为None则使用默认配置
        """
        self.config = config or SKUMatchingConfig()
        self.vggt_model = None
        self._is_initialized = False
        
    def initialize(self) -> None:
        """初始化模型和系统组件。"""
        if self._is_initialized:
            logger.info("System already initialized")
            return

        logger.info(f"Initializing SKU matching system (backend: {self.config.backend})...")

        try:
            # 设置随机种子（如果指定）
            if self.config.seed is not None:
                self._set_random_seeds()

            # 根据 backend 条件加载模型
            if self.config.backend in ("pi3", "da3"):
                # Pi3/DA3 后端：不需要加载 VGGT 模型（直接从缓存加载 3D 数据）
                logger.info(f"{self.config.backend.upper()} 后端：跳过 VGGT 模型加载，将从 {self.config.backend}_cache 加载预重建数据")
                self.vggt_model = None
            else:
                # VGGT 后端：加载 VGGT 模型用于实时 3D 重建
                logger.info("VGGT 后端：加载 VGGT 模型...")
                self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.config.device).eval()
                logger.info("VGGT 模型加载完成")

            self._is_initialized = True
            logger.info("System initialization complete")

        except (ImportError, RuntimeError, FileNotFoundError) as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def _set_random_seeds(self) -> None:
        """设置随机种子确保结果可复现。"""
        try:
            import random
            random.seed(self.config.seed)
        except ImportError:
            pass
            
        import numpy as np
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def process_images(
        self, 
        image_folder: str, 
        detection_dir: str, 
        reference_image_idx: int = 0, 
        max_images: Optional[int] = 50
    ) -> Dict[int, List[Dict]]:
        """处理图像文件夹并执行SKU匹配
        
        Args:
            image_folder: 图像文件夹路径
            detection_dir: 检测结果目录路径，包含按数字命名的JSON文件
            reference_image_idx: 参考图像索引
            max_images: 最大处理图像数量
            
        Returns:
            对应关系结果字典
        """
        if not self._is_initialized:
            self.initialize()
            
        try:
            # 1. 加载和预处理数据
            logger.info("Loading images and detection results...")
            image_paths, detections = self._load_data(image_folder, detection_dir, max_images)
            
            if not image_paths:
                raise ValueError(f"No valid images found in {image_folder}")
            
            logger.info(f"Loaded {len(image_paths)} images with {len(detections)} detection files")

            # 2. 预处理图像和构建 transforms（使用 config 的衍生属性，消除参数冗余）
            if self.config.model_type == "pi3":
                # Pi3: 先加载图像（动态尺寸），再根据实际尺寸构建 transforms
                # 注意：必须使用 image_paths 列表，而不是直接扫描 image_folder，以确保与帧对齐后的检测结果一致
                from pi3.utils.basic import load_images_as_tensor
                images = load_images_as_tensor(
                    image_paths,
                    PIXEL_LIMIT=self.config.transform_kwargs["pixel_limit"]
                )
                images = images.to(self.config.device)

                # Pi3 返回动态尺寸 [N, C, H, W]，需要从实际尺寸构建 transforms
                _, _, H_actual, W_actual = images.shape
                logger.info(f"Pi3 加载图像: {len(image_paths)} 张, 实际尺寸: ({W_actual}, {H_actual})")

                # 使用实际尺寸构建 transforms（确保与 Pi3 重建时一致）
                transforms_info = build_transforms(
                    image_paths,
                    model_type=self.config.model_type,
                    **self.config.transform_kwargs
                )
            else:
                # VGGT: 先构建 transforms（固定 518×518），再加载图像
                transforms_info = build_transforms(
                    image_paths,
                    model_type=self.config.model_type,
                    **self.config.transform_kwargs
                )
                images = load_and_preprocess_images(image_paths, mode=self.config.preprocess_mode).to(self.config.device)
            
            # 4. 执行匹配算法
            correspondences, points_per_object = self._run_matching(
                images, detections, reference_image_idx, transforms_info, image_paths
            )
            
            # 5. 后处理和可视化
            self._post_process_results(
                correspondences, points_per_object, images, 
                detections, reference_image_idx, transforms_info, image_paths
            )
            
            return correspondences
            
        except (RuntimeError, ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to process images: {e}")
            raise
    
    def _load_data(
        self,
        image_folder: str,
        detection_dir: str,
        max_images: Optional[int]
    ) -> tuple[List[str], List[Dict]]:
        """
        加载图像和检测数据，并使用ReconstructionDetectionAligner进行严格对齐验证

        该方法确保图像编号与检测编号严格对应，并自动处理顺序不一致的情况。
        """
        from utils.frame_alignment import ReconstructionDetectionAligner

        image_folder_path = Path(image_folder)
        if not image_folder_path.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        # 1) 收集图像（按数字排序）
        image_files = []
        for f in os.listdir(image_folder):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    file_number = int(Path(f).stem)
                    image_files.append((file_number, str(image_folder_path / f)))
                except ValueError:
                    logger.warning(f"Skipping non-numeric image file: {f}")
                    continue
        image_files.sort(key=lambda x: x[0])

        if not image_files:
            raise ValueError(f"No valid image files found in {image_folder}")

        # 2) 加载检测（带索引映射）
        detections_with_numbers = load_detections(detection_dir, return_index_map=True)
        if not detections_with_numbers:
            raise ValueError(f"No valid detection files found in {detection_dir}")

        # 分离索引和数据
        detection_indices = [item[0] for item in detections_with_numbers]
        detections = [item[1] for item in detections_with_numbers]
        image_ids = [num for num, _ in image_files]

        logger.info(f"Found {len(image_files)} images and {len(detections)} detection files")
        logger.info(f"Image IDs: {image_ids}")
        logger.info(f"Detection IDs: {detection_indices}")

        # 3) 使用VGGTDetectionAligner进行帧对齐验证
        # 构建占位vggt_data（仅用于对齐验证，不需要真实的world_points）
        import numpy as np
        dummy_reconstruction_data = {
            "world_points": np.zeros((len(image_ids), 1, 1, 3), dtype=np.float32)
        }

        # 严格模式=False：允许自动修复顺序不一致和缺失图像
        _, aligned_detections, alignment_report = ReconstructionDetectionAligner.validate_and_align(
            reconstruction_data=dummy_reconstruction_data,
            detections=detections,
            detection_indices=detection_indices,
            reconstruction_image_ids=image_ids,
            strict_mode=False  # 允许自动修复
        )

        # 4) 根据对齐结果重新排列图像路径
        aligned_image_ids = alignment_report.get('repaired_image_ids', alignment_report['common_ids'])

        # 构建image_id到路径的映射
        image_id_to_path = {num: path for num, path in image_files}

        # 按对齐后的ID顺序构建图像路径列表
        aligned_image_paths = []
        for img_id in aligned_image_ids:
            if img_id in image_id_to_path:
                aligned_image_paths.append(image_id_to_path[img_id])
            else:
                logger.warning(f"Missing image for ID {img_id} after alignment")

        # 验证对齐结果
        if len(aligned_image_paths) != len(aligned_detections):
            raise ValueError(
                f"Alignment failed: {len(aligned_image_paths)} images vs {len(aligned_detections)} detections"
            )

        # 5) 应用max_images限制
        if max_images is not None:
            aligned_image_paths = aligned_image_paths[:max_images]
            aligned_detections = aligned_detections[:max_images]

        # 6) 记录对齐统计信息
        if alignment_report.get('repair_applied'):
            logger.info(f"Alignment repaired: {alignment_report['repaired_frame_count']} frames aligned")
            logger.info(f"   Dropped images: {alignment_report.get('dropped_vggt_frames', 0)}")
            logger.info(f"   Dropped detections: {alignment_report.get('dropped_detection_frames', 0)}")
        # else:
        #     logger.info(f"Perfect alignment: {len(aligned_image_paths)} frames")

        return aligned_image_paths, aligned_detections
    
    def _run_matching(
        self, 
        images: torch.Tensor, 
        detections: List[Dict], 
        reference_image_idx: int,
        transforms_info: List,
        image_paths: List[str],
    ) -> tuple[Dict[int, List[Dict]], Optional[Dict[int, Dict]]]:
        """运行匹配算法"""
        import time
        start_time = time.time()
        
        algorithm_name = self.config.get_algorithm_name()
        logger.info(f"Running {algorithm_name} algorithm...")
        
        # 设置自动混合精度
        use_amp = (
            self.config.use_autocast
            and torch.cuda.is_available()
            and (self.config.dtype in (torch.float16, torch.bfloat16))
            and (isinstance(self.config.device, str) and self.config.device.startswith("cuda"))
        )
        
        target_indices = None
        if self.config.enable_3d_mapping:
            pairing = getattr(self.config, "pairing_3d", "all")
            logger.info(f"[CONFIG] pairing_3d = '{pairing}'")
            if pairing == "next":
                total_images = len(detections)
                if total_images >= 2:
                    next_idx = (reference_image_idx + 1) % total_images
                    target_indices = [next_idx]
                    logger.info(
                        f"3D pairing=next: ref={reference_image_idx} -> target={next_idx} (total={total_images})"
                    )
                else:
                    target_indices = []
                    logger.info("3D pairing=next: only 1 image available, no targets to match")

        amp_ctx = torch.amp.autocast('cuda', dtype=self.config.dtype) if use_amp else nullcontext()
        
        try:
            with amp_ctx:
                correspondences, points_per_object = find_object_correspondences(
                    self.vggt_model,
                    detections,
                    images,
                    self.config,
                    reference_image_idx=reference_image_idx,
                    transforms_info=transforms_info,
                    image_paths=image_paths,
                    target_indices=target_indices,
                )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and torch.cuda.is_available():
                logger.error(f"CUDA out of memory during {algorithm_name}. Try reducing max_images or max_points_per_bbox")
                torch.cuda.empty_cache()
            raise
        
        end_time = time.time()
        processing_time = end_time - start_time
        total_matches = sum(len(matches) for matches in correspondences.values())
        
        logger.info(f"{algorithm_name} completed in {processing_time:.1f}s")
        logger.info(f"Found {total_matches} matches across {len(correspondences)} images")
        
        return correspondences, points_per_object
    
    def _post_process_results(
        self,
        correspondences: Dict[int, List[Dict]],
        points_per_object: Optional[Dict[int, Dict]],
        images: torch.Tensor,
        detections: List[Dict],
        reference_image_idx: int,
        transforms_info: List,
        image_paths: List[str]
    ) -> None:
        """后处理结果：可视化和保存"""
        if correspondences:
            visualize_results(
                images, reference_image_idx, points_per_object,
                correspondences, self.config, detections, transforms_info
            )

        # 始终保存可视化摘要（即使没有匹配结果）
        save_visualization_summary(correspondences, self.config, reference_image_idx)

        # 保存JSON结果（如果启用）
        if self.config.save_json and correspondences:
            # 计算机器可解析的 per-image 统计信息
            per_image_summary: Dict[int, Dict[str, float]] = {}
            total_matches = 0
            for target_idx, objs in correspondences.items():
                match_count = len(objs)
                total_matches += match_count
                if match_count > 0:
                    avg_hit = sum(o.get("correspondence_ratio", 0.0) for o in objs) / match_count
                    avg_match_frac = sum(
                        (float(o.get("matched_points", 0)) / max(1.0, float(o.get("total_points", 1))))
                        for o in objs
                    ) / match_count
                else:
                    avg_hit = 0.0
                    avg_match_frac = 0.0
                per_image_summary[int(target_idx)] = {
                    "match_count": match_count,
                    "avg_correspondence_ratio": float(avg_hit),
                    "avg_matched_fraction": float(avg_match_frac),
                }

            meta = {
                "image_paths": image_paths,
                "reference_image_idx": reference_image_idx,
                "algorithm": "3D mapping" if self.config.enable_3d_mapping else "point tracking",
                "config": {
                    "confidence_threshold": self.config.confidence_threshold,
                    "min_confident_points": self.config.min_confident_points,
                    "max_points_per_bbox": self.config.max_points_per_bbox,
                    "max_bboxes": self.config.max_bboxes,
                    "enable_3d_mapping": self.config.enable_3d_mapping,
                },
                "summary": {
                    "total_matches": int(total_matches),
                    "images_with_matches": int(len(correspondences)),
                    "per_image": per_image_summary,
                },
            }
            save_correspondences_json(correspondences, points_per_object, self.config, meta)

        if not correspondences:
            logger.warning(f"No object correspondences found for reference image {reference_image_idx}")
    
    def _print_results_summary(self, correspondences: Dict[int, List[Dict]]) -> None:
        """打印结果摘要"""
        total_matches = sum(len(matches) for matches in correspondences.values())
        algorithm_name = "3D Mapping" if self.config.enable_3d_mapping else "Point Tracking"
        
        logger.info(f"\n=== {algorithm_name} Algorithm Results Summary ===")
        logger.info(f"Total matches found: {total_matches}")
        logger.info(f"Images with matches: {len(correspondences)}")
        
        for target_idx, found_objects in correspondences.items():
            logger.info(f"\nTarget Image {target_idx}: {len(found_objects)} matches")
            
            for obj in found_objects:
                correspondence_ratio = obj.get('correspondence_ratio', 0.0)
                matched_points = obj.get('matched_points', 0)
                total_points = obj.get('total_points', 0)
                target_obj_id = obj.get('target_obj_id', 'N/A')
                
                # 基本信息
                info_str = (
                    f"  - Ref {obj['object_id']} → Target {target_obj_id}: "
                    f"ratio={correspondence_ratio:.3f} ({matched_points}/{total_points})"
                )
                
                # 3D算法的额外信息
                if self.config.enable_3d_mapping:
                    distance_3d = obj.get('3d_distance', 0.0)
                    depth_consistency = obj.get('depth_consistency', 0.0)
                    info_str += f", 3D_dist={distance_3d:.3f}m, depth_cons={depth_consistency:.3f}"
                
                logger.info(info_str)
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            "initialized": self._is_initialized,
            "device": self.config.device,
            "algorithm": "3D mapping" if self.config.enable_3d_mapping else "point tracking",
            "model_loaded": self.vggt_model is not None,
            "output_directory": self.config.output_dir
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.vggt_model is not None:
            del self.vggt_model
            self.vggt_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_initialized = False
        logger.info("System cleanup complete")
