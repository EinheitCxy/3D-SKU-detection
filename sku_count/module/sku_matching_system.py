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

# VGGT相关导入
try:
    import sys
    sys.path.insert(0, '../../vggt-main')
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except ImportError as e:
    raise ImportError(f"Failed to import VGGT modules: {e}")

from .config import SKUMatchingConfig
from .data_utils import save_correspondences_json, load_detections
from .transforms import build_vggt_transforms
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
            
        logger.info("Initializing SKU matching system...")
        
        try:
            # 设置随机种子（如果指定）
            if self.config.seed is not None:
                self._set_random_seeds()
            
            # 加载VGGT模型
            self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.config.device).eval()
            
            self._is_initialized = True
            logger.info("System initialization complete")
            
        except Exception as e:
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
        max_images: Optional[int] = 20
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
            
            # 2. 构建坐标变换信息
            transforms_info = build_vggt_transforms(image_paths, target_size=518)
            
            # 3. 预处理图像
            images = load_and_preprocess_images(image_paths, mode="crop").to(self.config.device)
            
            # 4. 执行匹配算法
            correspondences, points_per_object = self._run_matching(
                images, detections, reference_image_idx, transforms_info
            )
            
            # 5. 后处理和可视化
            self._post_process_results(
                correspondences, points_per_object, images, 
                detections, reference_image_idx, transforms_info, image_paths
            )
            
            return correspondences
            
        except Exception as e:
            logger.error(f"Failed to process images: {e}")
            raise
    
    def _load_data(
        self, 
        image_folder: str, 
        detection_dir: str, 
        max_images: Optional[int]
    ) -> tuple[List[str], List[Dict]]:
        """加载图像和检测数据（解析委托给 data_utils.load_detections）"""
        image_folder_path = Path(image_folder)
        if not image_folder_path.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")

        # 1) 收集图像（按数字）
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

        # 2) 加载检测（一次性）
        detections_with_numbers = load_detections(detection_dir, return_index_map=True)
        if not detections_with_numbers:
            raise ValueError(f"No valid detection files found in {detection_dir}")
        det_map = {num: det for num, det in detections_with_numbers}

        # 3) 对齐图像与检测
        matched_files = []
        for file_number, image_path in image_files:
            if file_number in det_map:
                matched_files.append((file_number, image_path))
            else:
                logger.info(f"Skipping image {file_number}.jpg - no corresponding detection file")

        if max_images is not None:
            matched_files = matched_files[:max_images]

        if not matched_files:
            raise ValueError("No images matched with detection files after alignment")

        valid_image_paths = [path for _, path in matched_files]
        detections = [det_map[num] for num, _ in matched_files]

        return valid_image_paths, detections
    
    def _run_matching(
        self, 
        images: torch.Tensor, 
        detections: List[Dict], 
        reference_image_idx: int,
        transforms_info: List
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
            and isinstance(self.config.dtype, torch.dtype)
            and (isinstance(self.config.device, str) and self.config.device.startswith("cuda"))
        )
        
        amp_ctx = torch.amp.autocast('cuda', dtype=self.config.dtype) if use_amp else nullcontext()
        
        try:
            with amp_ctx:
                correspondences, points_per_object = find_object_correspondences(
                    self.vggt_model,
                    detections,
                    images,
                    self.config,
                    reference_image_idx=reference_image_idx,
                    transforms_info=transforms_info
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
            # 生成可视化
            logger.info("Generating visualization...")
            visualize_results(
                images, reference_image_idx, points_per_object, 
                correspondences, self.config, detections, transforms_info
            )
            
            # 保存可视化摘要
            save_visualization_summary(correspondences, self.config)
            
            # # 打印结果摘要
            # self._print_results_summary(correspondences)
            
            # 保存JSON结果（如果启用）
            if self.config.save_json:
                meta = {
                    "image_paths": image_paths,
                    "reference_image_idx": reference_image_idx,
                    "algorithm": "3D-2D projection" if self.config.enable_3d_projection_matching else "point tracking",
                    "config": {
                        "visibility_threshold": self.config.visibility_threshold,
                        "min_visible_points": self.config.min_visible_points,
                        "max_points_per_bbox": self.config.max_points_per_bbox,
                        "max_bboxes": self.config.max_bboxes,
                        "enable_3d_projection_matching": self.config.enable_3d_projection_matching,
                    },
                }
                save_correspondences_json(correspondences, points_per_object, self.config, meta)
        else:
            logger.warning("No object correspondences found")
    
    def _print_results_summary(self, correspondences: Dict[int, List[Dict]]) -> None:
        """打印结果摘要"""
        total_matches = sum(len(matches) for matches in correspondences.values())
        algorithm_name = "3D-2D Projection" if self.config.enable_3d_projection_matching else "Point Tracking"
        
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
                if self.config.enable_3d_projection_matching:
                    distance_3d = obj.get('3d_distance', 0.0)
                    depth_consistency = obj.get('depth_consistency', 0.0)
                    info_str += f", 3D_dist={distance_3d:.3f}m, depth_cons={depth_consistency:.3f}"
                
                logger.info(info_str)
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            "initialized": self._is_initialized,
            "device": self.config.device,
            "algorithm": "3D-2D projection" if self.config.enable_3d_projection_matching else "point tracking",
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
