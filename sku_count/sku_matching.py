# TODO: integrate sam 2 model before sampling, after substracting bboxs
# https://huggingface.co/facebook/sam2.1-hiera-large

import torch
import numpy as np
import cv2
import os
import json
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext

# 尝试导入 VGGT 相关模块
try:
    import sys
    # 添加VGGT模块路径
    sys.path.insert(0, '../vggt-main')
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

except ImportError as e:
    raise e

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler('sku_matching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SKUMatchingConfig:
    """SKU匹配配置参数"""
    max_points_per_bbox: int = 50  # 每个检测框最大采样点数
    visibility_threshold: float = 0.8
    min_visible_points: int = 8
    max_bboxes: int = 100
    device: str = "cuda"
    dtype: torch.dtype = None
    output_dir: str = "output"
    det_conf_threshold: float = 0.0  # 检测置信度阈值
    min_bbox_area: float = 100.0      # 忽略极小框
    correspondence_threshold: float = 0.5  # 对应关系阈值（50%以上点匹配）
    max_total_points: int = 5000     # 全局最大采样点数上限（控制内存/速度）
    seed: Optional[int] = 42         # 全流程随机种子（可复现实验）
    use_autocast: bool = True        # 仅在CUDA上启用autocast
    save_json: bool = False          # 是否将结果保存为JSON
    json_filename: str = "correspondences.json"  # 结果保存文件名
    
    # 新增：3D-2D投影匹配相关参数
    use_3d_projection_matching: bool = False  # 是否使用3D-2D投影匹配
    depth_confidence_threshold: float = 0.1   # 深度置信度阈值
    world_points_confidence_threshold: float = 0.1  # 3D点置信度阈值
    min_depth: float = 0.1            # 最小深度值
    max_depth: float = 10.0           # 最大深度值
    points_per_bbox_3d: int = 50      # 每个检出框采样的3D点数
    projection_match_threshold: float = 0.7  # 投影匹配阈值（提高到70%）
    
    # 新增：3D几何验证参数
    max_3d_distance: float = 1.0      # 最大3D空间距离阈值（米）
    max_depth_difference: float = 2.0  # 最大深度差异容忍（米）
    min_depth_consistency: float = 0.3  # 最小深度一致性阈值
    
    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

# --- Helper Functions ---

def load_detections(detection_dir: str) -> List[Dict]:
    """加载检测结果文件
    
    Args:
        detection_dir: 检测结果目录路径，包含按数字命名的JSON文件(1.json, 2.json, ...)
        
    Returns:
        检测结果列表，按文件名数字顺序排列
        
    Raises:
        FileNotFoundError: 目录不存在时抛出
        ValueError: 文件格式不正确时抛出
    """
    detection_path = Path(detection_dir)
    if not detection_path.exists():
        raise FileNotFoundError(f"Detection directory not found: {detection_dir}")
    
    if not detection_path.is_dir():
        raise ValueError(f"Path is not a directory: {detection_dir}")
    
    try:
        # 获取所有JSON文件并按数字顺序排序
        json_files = []
        for file_path in detection_path.glob("*.json"):
            try:
                # 尝试提取文件名中的数字
                file_number = int(file_path.stem)
                json_files.append((file_number, file_path))
            except ValueError:
                logger.warning(f"Skipping non-numeric JSON file: {file_path.name}")
                continue
        
        # 按数字顺序排序
        json_files.sort(key=lambda x: x[0])
        
        if not json_files:
            raise ValueError(f"No valid JSON files found in {detection_dir}")
        
        detections = []
        for file_number, file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_detections = json.load(f)
                
                # 每个文件包含一个列表，取第一个元素
                if isinstance(file_detections, list) and len(file_detections) > 0:
                    detections.append(file_detections[0])
                elif isinstance(file_detections, dict):
                    detections.append(file_detections)
                else:
                    logger.warning(f"Invalid format in file {file_path.name}, skipping")
                    continue
                    
            except Exception as e:
                logger.error(f"Failed to load detection from {file_path.name}: {e}")
                raise
        
        logger.info(f"Loaded {len(detections)} detection results from {len(json_files)} files in {detection_dir}")
        return detections
        
    except Exception as e:
        logger.error(f"Failed to load detections from directory {detection_dir}: {e}")
        raise

def extract_bboxes_from_detections(detections: List[Dict], image_idx: int, config: SKUMatchingConfig) -> List[Dict]:
    """从检测结果中提取边界框
    
    Args:
        detections: 检测结果列表
        image_idx: 图像索引
        config: 配置参数
        
    Returns:
        边界框列表
    """
    if image_idx >= len(detections):
        logger.warning(f"Image index {image_idx} out of range for {len(detections)} detections")
        raise
    
    detection_data = detections[image_idx]
    if 'objects' not in detection_data:
        logger.warning(f"No objects found in detection data for image {image_idx}")
        raise
    
    bboxes = []
    for obj_idx, obj in enumerate(detection_data['objects']):
        if 'position' in obj:
            x1, y1, x2, y2 = obj['position']
            confidence = obj.get('confidences', {}).get('det', 0.0)
            if confidence < config.det_conf_threshold:
                continue
            area = max(0.0, (x2 - x1) * (y2 - y1))
            if area < config.min_bbox_area:
                continue
            bbox_info = {
                'bbox': [x1, y1, x2, y2],
                'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                'confidence': confidence,
                'object_id': obj_idx,
                'area': area
            }
            bboxes.append(bbox_info)
    
    # 按面积排序并限制数量
    bboxes.sort(key=lambda x: x['area'], reverse=True)
    if len(bboxes) > config.max_bboxes:
        logger.warning(f"Limiting to top {config.max_bboxes} bboxes (found {len(bboxes)})")
        bboxes = bboxes[:config.max_bboxes]
    
    logger.info(f"Extracted {len(bboxes)} bounding boxes from image {image_idx}")
    return bboxes

class VGGTImageTransform:
    """修复版本的VGGT图像变换类，完全对齐load_and_preprocess_images(crop)的实现
    
    关键修复点：
    1. 正确的裁剪offset计算 - 修复了裁剪坐标系映射错误
    2. 精确的坐标映射逻辑 - 按VGGT实际变换顺序处理
    3. 批量填充的正确处理 - 修正为左上角对齐，而非居中对齐
    """

    def __init__(self, orig_width: int, orig_height: int, target_size: int = 518):
        """初始化变换参数，严格按照VGGT预处理逻辑"""
        self.orig_width = int(orig_width)
        self.orig_height = int(orig_height)
        self.target_size = int(target_size)
        
        # 步骤1：固定宽度为target_size，高度按比例调整并取整到14的倍数
        # 完全复制VGGT的计算逻辑：round(height * (new_width / width) / 14) * 14
        self.proc_width = self.target_size  # 固定518
        self.proc_height = int(round(self.orig_height * (self.proc_width / self.orig_width) / 14) * 14)
        
        # 计算缩放比例
        self.scale_x = self.proc_width / self.orig_width
        self.scale_y = self.proc_height / self.orig_height
        
        # 步骤2：处理高度裁剪（如果proc_height > target_size）
        # 修复：不再使用负offset，而是直接记录裁剪起始位置
        self.crop_applied = False
        self.crop_start_y = 0
        
        if self.proc_height > self.target_size:
            self.crop_applied = True
            self.crop_start_y = (self.proc_height - self.target_size) // 2
            self.final_height = self.target_size
        else:
            self.final_height = self.proc_height
            
        self.final_width = self.proc_width
        
        # 步骤3：初始化批量填充参数（会在apply_batch_padding中设置）
        self.batch_pad_left = 0
        self.batch_pad_top = 0
        self.padded_width = self.final_width
        self.padded_height = self.final_height

    def apply_batch_padding(self, max_width: int, max_height: int) -> None:
        """应用批量填充，对齐到最大尺寸
        
        【关键修正】: 根据VGGT源代码分析，在crop模式下，批处理填充是居中对齐的：
        
        VGGT源代码（第210-213行）：
        h_padding = max_height - img.shape[1]
        w_padding = max_width - img.shape[2]
        pad_top = h_padding // 2
        pad_left = w_padding // 2
        
        这是导致坐标错乱的根本原因！
        """
        # 计算填充量
        h_padding = max_height - self.final_height
        w_padding = max_width - self.final_width
        
        # 居中填充（与VGGT源代码保持一致）
        self.batch_pad_left = w_padding // 2 if w_padding > 0 else 0
        self.batch_pad_top = h_padding // 2 if h_padding > 0 else 0
            
        # 更新最终画布尺寸
        self.padded_width = max_width
        self.padded_height = max_height

    def map_xy_to_final(self, x: float, y: float) -> Tuple[float, float]:
        """将原图坐标映射到VGGT最终输入坐标
        
        修复后的变换顺序：
        1. 原图 -> 缩放后坐标
        2. 缩放后 -> 裁剪后坐标（如果有裁剪）
        3. 裁剪后 -> 批量填充后坐标
        """
        # 步骤1：应用缩放
        x_scaled = x * self.scale_x
        y_scaled = y * self.scale_y
        
        # 步骤2：应用裁剪（如果有）
        if self.crop_applied:
            y_cropped = y_scaled - self.crop_start_y
            # 裁剪后的坐标需要在有效范围内
            y_cropped = max(0.0, min(y_cropped, self.final_height - 1))
        else:
            y_cropped = y_scaled
            
        x_cropped = x_scaled
        x_cropped = max(0.0, min(x_cropped, self.final_width - 1))
        
        # 步骤3：应用批量填充
        x_final = x_cropped + self.batch_pad_left
        y_final = y_cropped + self.batch_pad_top
        
        # 确保在最终图像范围内
        x_final = max(0.0, min(x_final, self.padded_width - 1))
        y_final = max(0.0, min(y_final, self.padded_height - 1))
        
        return x_final, y_final

    def map_points_to_final(self, points):
        """批量映射点坐标到VGGT输入空间
        points: 形如 (..., 2) 的 numpy 数组或 torch 张量。返回同类型同形状。
        """
        is_torch = torch.is_tensor(points)
        
        if is_torch:
            # PyTorch tensor处理
            result = torch.zeros_like(points)
            flat_points = points.view(-1, 2)
            
            for i in range(flat_points.shape[0]):
                x, y = flat_points[i, 0].item(), flat_points[i, 1].item()
                xf, yf = self.map_xy_to_final(x, y)
                result.view(-1, 2)[i, 0] = xf
                result.view(-1, 2)[i, 1] = yf
                
            return result
        else:
            # NumPy数组处理  
            result = np.zeros_like(points)
            flat_points = points.reshape(-1, 2)
            
            for i, (x, y) in enumerate(flat_points):
                xf, yf = self.map_xy_to_final(x, y)
                result.flat[i*2] = xf
                result.flat[i*2+1] = yf
                
            return result.reshape(points.shape)

    def map_bbox_to_final(self, bbox: List[float]) -> List[float]:
        """将原图边界框映射到VGGT输入坐标"""
        x1, y1 = self.map_xy_to_final(bbox[0], bbox[1])
        x2, y2 = self.map_xy_to_final(bbox[2], bbox[3])
        return [x1, y1, x2, y2]

    # -------- 模型输入(final) -> 原图 映射 --------
    def map_xy_to_original(self, xp: float, yp: float) -> Tuple[float, float]:
        """将VGGT最终输入坐标映射回原图坐标
        
        修复后的变换顺序（逆向）：
        1. 批量填充后 -> 裁剪后坐标
        2. 裁剪后 -> 缩放后坐标（如果有裁剪）
        3. 缩放后 -> 原图坐标
        """
        # 步骤1：移除批量填充
        x_cropped = xp - self.batch_pad_left
        y_cropped = yp - self.batch_pad_top
        
        # 步骤2：移除裁剪（如果有）
        if self.crop_applied:
            y_scaled = y_cropped + self.crop_start_y
        else:
            y_scaled = y_cropped
            
        x_scaled = x_cropped
        
        # 步骤3：移除缩放
        x_orig = x_scaled / self.scale_x if self.scale_x != 0 else 0.0
        y_orig = y_scaled / self.scale_y if self.scale_y != 0 else 0.0
        
        # 确保在原图范围内
        x_orig = max(0.0, min(x_orig, self.orig_width - 1))
        y_orig = max(0.0, min(y_orig, self.orig_height - 1))
        
        return x_orig, y_orig

    def map_points_to_original(self, points):
        """批量映射点坐标回原图空间"""
        is_torch = torch.is_tensor(points)
        
        if is_torch:
            # PyTorch tensor处理
            result = torch.zeros_like(points)
            flat_points = points.view(-1, 2)
            
            for i in range(flat_points.shape[0]):
                x, y = flat_points[i, 0].item(), flat_points[i, 1].item()
                xo, yo = self.map_xy_to_original(x, y)
                result.view(-1, 2)[i, 0] = xo
                result.view(-1, 2)[i, 1] = yo
                
            return result
        else:
            # NumPy数组处理
            result = np.zeros_like(points)
            flat_points = points.reshape(-1, 2)
            
            for i, (x, y) in enumerate(flat_points):
                xo, yo = self.map_xy_to_original(x, y)
                result.flat[i*2] = xo
                result.flat[i*2+1] = yo
                
            return result.reshape(points.shape)

    def map_bbox_to_original(self, bbox: List[float]) -> List[float]:
        """将VGGT输入边界框映射回原图坐标"""
        x1, y1 = self.map_xy_to_original(bbox[0], bbox[1])
        x2, y2 = self.map_xy_to_original(bbox[2], bbox[3])
        return [x1, y1, x2, y2]

    def get_transform_info(self) -> dict:
        """返回变换信息用于调试"""
        return {
            "original_size": (self.orig_width, self.orig_height),
            "processed_size": (self.proc_width, self.proc_height),
            "final_size": (self.final_width, self.final_height),
            "padded_size": (self.padded_width, self.padded_height),
            "scales": (self.scale_x, self.scale_y),
            "crop_applied": self.crop_applied,
            "crop_start_y": self.crop_start_y,
            "batch_padding": (self.batch_pad_left, self.batch_pad_top)
        }


def build_vggt_transforms(image_paths: List[str], target_size: int = 518) -> List[VGGTImageTransform]:
    """构建修复版本的VGGT变换列表，完全对齐load_and_preprocess_images(“crop”)的实现
    
    修复要点：
    1. 正确处理裁剪坐标映射
    2. 精确的批量填充计算
    3. 与VGGT实际预处理的完全一致性
    
    Args:
        image_paths: 图像路径列表
        target_size: 目标尺寸，默认518
        
    Returns:
        修复后的变换对象列表
    """
    from PIL import Image as _Image

    transforms: List[VGGTImageTransform] = []
    
    # 第一步：为每个图像创建变换对象
    for p in image_paths:
        img = _Image.open(p).convert("RGB")
        w, h = img.size
        transforms.append(VGGTImageTransform(w, h, target_size=target_size))

    # 第二步：计算批次最大尺寸并应用填充
    max_w = max(t.final_width for t in transforms)
    max_h = max(t.final_height for t in transforms)
    for t in transforms:
        t.apply_batch_padding(max_w, max_h)

    return transforms

def generate_points_from_bboxes(bboxes: List[Dict], image_shape: Tuple[int, int], config: SKUMatchingConfig) -> Tuple[Optional[torch.Tensor], Dict[int, Dict]]:
    """在检测框内随机采样点作为查询点
    
    Args:
        bboxes: 检测框列表
        image_shape: 图像形状 (height, width)
        config: 配置参数
        
    Returns:
        tuple: (查询点张量, 物体点映射字典)
    """
    logger.info(f"Generating query points from {len(bboxes)} bounding boxes...")
    
    all_query_points = []
    points_per_object = {}
    
    height, width = image_shape
    total_points = 0
    
    try:
        for bbox_info in bboxes:
            x1, y1, x2, y2 = bbox_info['bbox']
            object_id = bbox_info['object_id']
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # 在检测框内生成网格点
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            
            # 计算需要的网格密度
            area = bbox_width * bbox_height
            if area == 0:
                logger.warning(f"Invalid bbox area for object {object_id}, skipping")
                continue
            
            # 基于面积计算采样点数，但不超过最大值
            desired_points = min(int(np.sqrt(area) * 2), config.max_points_per_bbox)
            if desired_points < 5:  # 至少5个点
                desired_points = 5
            
            # 计算网格大小
            grid_cols = max(1, int(np.sqrt(desired_points * bbox_width / bbox_height)))
            grid_rows = max(1, int(np.sqrt(desired_points * bbox_height / bbox_width)))
            
            # 生成网格点
            x_points = np.linspace(x1, x2, grid_cols)
            y_points = np.linspace(y1, y2, grid_rows)
            
            # 创建网格
            xx, yy = np.meshgrid(x_points, y_points)
            xx = xx.flatten()
            yy = yy.flatten()
            
            # 随机选择点以达到 desired_points 数量
            if len(xx) > desired_points:
                indices = np.random.choice(len(xx), desired_points, replace=False)
                xx = xx[indices]
                yy = yy[indices]
            
            # 使用numpy操作提高性能
            points = np.stack([xx, yy], axis=-1).astype(np.float32)

            # 全局点数上限控制
            remaining_allowance = config.max_total_points - total_points
            if remaining_allowance <= 0:
                logger.warning("Reached max_total_points limit; stopping further point sampling.")
                break
            if len(points) > remaining_allowance:
                sel_idx = np.random.choice(len(points), remaining_allowance, replace=False)
                points = points[sel_idx]

            all_query_points.append(points)

            num_points = len(points)
            start_idx = total_points
            points_per_object[object_id] = {
                "point_indices": (start_idx, start_idx + num_points),
                "bbox": bbox_info['bbox'],
                "center": bbox_info['center'],
                "confidence": bbox_info['confidence'],
                "area": bbox_info['area'],
                "num_sampled_points": num_points,
                "original_bbox": bbox_info.get('original_bbox', bbox_info['bbox'])
            }
            total_points += num_points
            
        if not all_query_points:
            logger.warning("No query points generated from bounding boxes")
            return None, None
            
        # 一次性转换为torch tensor，减少内存碎片
        all_points_array = np.concatenate(all_query_points, axis=0)
        query_points_tensor = torch.from_numpy(all_points_array).float()
        
        logger.info(f"Generated {total_points} query points from {len(points_per_object)} objects")
        return query_points_tensor, points_per_object
        
    except Exception as e:
        logger.error(f"Failed to generate points from bounding boxes: {e}")
        raise

def visualize_results(images: torch.Tensor, reference_idx: int, points_per_object: Dict[int, Dict], 
                     correspondences: Dict[int, List[Dict]], config: SKUMatchingConfig,
                     detections: List[Dict] = None, transforms_info: List = None) -> None:
    """可视化追踪结果 - 使用VGGT尺寸的图像和坐标
    
    Args:
        images: 图像张量 (S, C, H, W) - VGGT处理后的518x518图像
        reference_idx: 参考图像索引
        points_per_object: 物体点映射
        correspondences: 对应关系结果
        config: 配置参数
        detections: 原始检测结果列表
        transforms_info: 图像变换信息列表
    """
    logger.info("Generating visualization...")
    
    try:
        # 1. 可视化参考图像和其上的检测框（使用VGGT图像和坐标）
        ref_image_np = (images[reference_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        ref_image_bgr = cv2.cvtColor(ref_image_np, cv2.COLOR_RGB2BGR)
        
        # 【关键调试】检查图像实际尺寸与变换预期是否一致
        actual_h, actual_w = ref_image_bgr.shape[:2]
        if transforms_info and reference_idx < len(transforms_info):
            expected_h, expected_w = transforms_info[reference_idx].padded_height, transforms_info[reference_idx].padded_width
            logger.info(f"参考图像尺寸检查: 实际{actual_w}x{actual_h} vs 预期{expected_w}x{expected_h}")
            if (actual_w, actual_h) != (expected_w, expected_h):
                logger.warning(f"⚠️ 图像尺寸不匹配！这可能是坐标错乱的原因")
        
        overlay = ref_image_bgr.copy()
        colors = {}
        
        # 绘制参考图像的检测框（使用VGGT坐标）
        for obj_id, data in points_per_object.items():
            # 使用VGGT处理后的坐标
            vggt_bbox = data['bbox']  # 这已经是VGGT输入空间的坐标
            
            # 生成稳定的颜色
            np.random.seed(obj_id)  # 确保颜色一致
            colors[obj_id] = np.random.randint(50, 255, (3,)).tolist()
            color = colors[obj_id]
            
            # 转换为整数坐标并确保在VGGT图像范围内
            h, w = ref_image_bgr.shape[:2]  # 应该是518x518
            x1, y1, x2, y2 = [max(0, int(c)) for c in vggt_bbox]
            x1, x2 = min(x1, w-1), min(x2, w)
            y1, y2 = min(y1, h-1), min(y2, h)
            
            if x1 < x2 and y1 < y2:  # 有效框
                # 绘制检测框
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                
                # 绘制中心点
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(overlay, (center_x, center_y), 3, color, -1)
                
                # 绘制ID标签 - 只显示ID，字体适中粗细
                label = f"{obj_id}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
                cv2.putText(overlay, label, (x1, max(y1 - 5, 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                pass
            else:
                logger.warning(f"Invalid reference bbox {obj_id}")
            
        ref_output_path = Path(config.output_dir) / "reference_image_with_bboxes.jpg"
        cv2.imwrite(str(ref_output_path), overlay)
        logger.info(f"Saved reference visualization")

        # 2. 可视化每个目标图像上的所有检测框和匹配结果
        for s_idx, matched_boxes in correspondences.items():
            # 使用VGGT处理后的图像
            target_image_np = (images[s_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            target_image_bgr = cv2.cvtColor(target_image_np, cv2.COLOR_RGB2BGR)
            
            # 【关键调试】检查目标图像尺寸
            actual_h, actual_w = target_image_bgr.shape[:2]
            if transforms_info and s_idx < len(transforms_info):
                expected_h, expected_w = transforms_info[s_idx].padded_height, transforms_info[s_idx].padded_width
                logger.info(f"目标图像{s_idx}尺寸检查: 实际{actual_w}x{actual_h} vs 预期{expected_w}x{expected_h}")
                if (actual_w, actual_h) != (expected_w, expected_h):
                    logger.warning(f"⚠️ 目标图像{s_idx}尺寸不匹配！")
            
            # 获取目标图像的所有原始检测框
            if detections and s_idx < len(detections) and transforms_info and s_idx < len(transforms_info):
                # 提取目标图像的所有检测框
                target_bboxes = extract_bboxes_from_detections([detections[s_idx]], 0, config)
                target_transform = transforms_info[s_idx]
                
                # 创建匹配框ID集合，用于区分匹配和未匹配的框
                matched_target_bbox_ids = set()
                for item in matched_boxes:
                    matched_target_bbox_ids.add(item.get('target_bbox_id'))
                
                # 绘制所有原始检测框
                h, w = target_image_bgr.shape[:2]  # 应该是518x518
                
                for bbox_info in target_bboxes:
                    # 将原始检测框映射到VGGT坐标
                    vggt_bbox = target_transform.map_bbox_to_final(bbox_info['bbox'])
                    target_bbox_id = bbox_info['object_id']
                    
                    # 转换为整数坐标
                    x1, y1, x2, y2 = [max(0, int(c)) for c in vggt_bbox]
                    x1, x2 = min(x1, w-1), min(x2, w)
                    y1, y2 = min(y1, h-1), min(y2, h)
                    
                    if x1 < x2 and y1 < y2:  # 有效框
                        # 区分匹配和未匹配的框
                        if target_bbox_id in matched_target_bbox_ids:
                            # 匹配的框：使用绿色粗线
                            cv2.rectangle(target_image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(target_image_bgr, f"{target_bbox_id}", 
                                       (x1, max(y1 - 25, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            # 未匹配的框：使用灰色细线
                            cv2.rectangle(target_image_bgr, (x1, y1), (x2, y2), (128, 128, 128), 2)
                            cv2.putText(target_image_bgr, f"{target_bbox_id}", 
                                       (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            
            # 在匹配的框上添加详细的匹配信 息
            for item in matched_boxes:
                obj_id = item['object_id']
                
                # 获取VGGT坐标（而不是原图坐标）
                vggt_box = item.get('vggt_box', [])
                target_bbox_id = item.get('target_bbox_id', 'N/A')
                
                if not vggt_box or len(vggt_box) != 4:
                    logger.warning(f"⚠️ 目标图像 {s_idx} 对象 {obj_id}: 无效的VGGT坐标 {vggt_box}")
                    continue
                
                # 使用预定义的颜色（与参考图像一致）
                color = colors.get(obj_id, [255, 255, 255])
                
                # 转换为整数坐标并确保在VGGT图像范围内
                h, w = target_image_bgr.shape[:2]  # 应该是518x518
                x1, y1, x2, y2 = [max(0, int(c)) for c in vggt_box]
                x1, x2 = min(x1, w-1), min(x2, w)
                y1, y2 = min(y1, h-1), min(y2, h)
                
                if x1 < x2 and y1 < y2:  # 有效框
                    # 绘制匹配标签 - 只显示参考ID，字体适中粗细
                    label = f"{obj_id}"
                    cv2.putText(target_image_bgr, label, (x1, max(y1 - 40, 25)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # 绘制参考框的轮廓（使用参考框的颜色，虚线效果）
                    for i in range(0, x2 - x1, 10):  # 虚线效果
                        cv2.line(target_image_bgr, (x1 + i, y1), (min(x1 + i + 5, x2), y1), color, 2)
                        cv2.line(target_image_bgr, (x1 + i, y2), (min(x1 + i + 5, x2), y2), color, 2)
                    for i in range(0, y2 - y1, 10):
                        cv2.line(target_image_bgr, (x1, y1 + i), (x1, min(y1 + i + 5, y2)), color, 2)
                        cv2.line(target_image_bgr, (x2, y1 + i), (x2, min(y1 + i + 5, y2)), color, 2)
                    
                    pass
                else:
                    logger.warning(f"Invalid target bbox {obj_id} in image {s_idx}")
                
            output_filename = Path(config.output_dir) / f"target_image_{s_idx}_all_bboxes_and_matches.jpg"
            cv2.imwrite(str(output_filename), target_image_bgr)
            logger.info(f"Saved target image {s_idx} visualization")
            
    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
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


def save_correspondences_json(
    correspondences: Dict[int, List[Dict]],
    points_per_object: Optional[Dict[int, Dict]],
    config: SKUMatchingConfig,
    meta: Optional[Dict] = None,
) -> Path:
    """将匹配结果保存为 JSON 文件
    
    Args:
        correspondences: 匹配结果
        points_per_object: 参考图像对象点信息
        config: 配置
        meta: 可选元数据（如图像路径列表、时间戳等）
    Returns:
        保存文件路径
    """
    try:
        result = {
            "correspondences": correspondences,
            "reference_points": points_per_object if points_per_object is not None else {},
            "meta": meta or {},
        }
        out_path = Path(config.output_dir) / config.json_filename
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved correspondences JSON to '{out_path}'")
        return out_path
    except Exception as e:
        logger.error(f"Failed to save correspondences JSON: {e}")
        raise

# --- Main Function ---

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
        
    Returns:
        tuple: (对应关系结果, 物体点映射)
    """
    logger.info("Starting object correspondence detection...")
    
    # 根据配置选择匹配算法
    if config.use_3d_projection_matching:
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

def sample_3d_points_from_bbox(scene_data: Dict, img_idx: int, bbox: List[float], 
                               transform: VGGTImageTransform, config: SKUMatchingConfig) -> Optional[torch.Tensor]:
    """从检出框中采样3D点"""
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
    
    # 过滤高置信度的3D点
    valid_mask = (
        (depth_conf_region > config.depth_confidence_threshold) &
        (world_points_conf_region > config.world_points_confidence_threshold) &
        (depth_region > config.min_depth) &
        (depth_region < config.max_depth)
    )
    
    if valid_mask.sum() < 10:
        return None
        
    valid_world_points = world_points_region[valid_mask]
    
    # 随机采样指定数量的点 - 确保在正确的设备上
    device = valid_world_points.device
    num_points = min(len(valid_world_points), config.points_per_bbox_3d)
    if len(valid_world_points) > num_points:
        indices = torch.randperm(len(valid_world_points), device=device)[:num_points]
        sampled_points = valid_world_points[indices]
    else:
        sampled_points = valid_world_points
        
    return sampled_points

def project_3d_to_2d(points_3d: torch.Tensor, extrinsic: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """将3D点投影到2D图像坐标"""
    # 确保所有张量在同一设备上
    device = points_3d.device
    extrinsic = extrinsic.to(device)
    intrinsic = intrinsic.to(device)
    
    # 齐次坐标 - 确保ones张量在正确的设备上
    ones = torch.ones(len(points_3d), 1, device=device)
    points_3d_homo = torch.cat([points_3d, ones], dim=1)
    
    # 世界坐标 → 相机坐标
    points_cam = (extrinsic @ points_3d_homo.T).T[:, :3]
    
    # 过滤在相机前方的点
    valid_depth_mask = points_cam[:, 2] > 0.1
    if not valid_depth_mask.any():
        return torch.empty(0, 2, device=device)
        
    points_cam_valid = points_cam[valid_depth_mask]
    
    # 相机坐标 → 图像坐标
    points_2d_homo = (intrinsic @ points_cam_valid.T).T
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    return points_2d

def find_best_matching_bbox_with_3d_validation(projected_points: torch.Tensor, target_bboxes: List[Dict], 
                                             config: SKUMatchingConfig, scene_data: Dict, target_img_idx: int,
                                             target_transform: VGGTImageTransform, ref_3d_center: torch.Tensor, 
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
            target_transform, config
        )
        
        if target_points_3d is None or len(target_points_3d) < 10:
            continue
        
        # 计算目标3D点的统计信息
        target_3d_center = target_points_3d.mean(dim=0)
        target_depth_mean = target_points_3d[:, 2].mean().item()
        
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
                logger.info(f"⚠️  Filtered duplicate: Reference SKU {filtered['ref_obj_id']} → Target bbox {target_id} "
                          f"(score: {filtered['combined_score']:.3f} < {best_match['combined_score']:.3f})")
    
    logger.info(f"Applied uniqueness constraint: {len(candidate_matches)} → {len(final_matches)} matches")
    return final_matches

def find_best_matching_bbox(projected_points: torch.Tensor, target_bboxes: List[Dict], 
                           config: SKUMatchingConfig) -> Optional[Dict]:
    """原有的简单投影匹配函数（保持向后兼容）"""
    if len(projected_points) == 0:
        return None
        
    best_match = None
    best_ratio = 0.0
    
    for bbox_info in target_bboxes:
        bbox = bbox_info['bbox']
        x1, y1, x2, y2 = bbox
        
        # 统计有多少投影点落在这个框内
        points_in_bbox = (
            (projected_points[:, 0] >= x1) & 
            (projected_points[:, 0] <= x2) &
            (projected_points[:, 1] >= y1) & 
            (projected_points[:, 1] <= y2)
        ).sum().item()
        
        match_ratio = points_in_bbox / len(projected_points)
        
        if match_ratio > config.projection_match_threshold and match_ratio > best_ratio:
            best_match = {
                'target_bbox_info': bbox_info,
                'points_in_bbox': points_in_bbox,
                'total_points': len(projected_points),
                'match_ratio': match_ratio
            }
            best_ratio = match_ratio
            
    return best_match

def find_correspondences_point_tracking(
    vggt_model: VGGT,
    detections: List[Dict],
    images: torch.Tensor,
    config: SKUMatchingConfig,
    reference_image_idx: int = 0,
    transforms_info: Optional[List[VGGTImageTransform]] = None,
) -> Tuple[Dict[int, List[Dict]], Optional[Dict[int, Dict]]]:
    """原有的基于点跟踪的物体匹配算法"""
    
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

        # 2. 原图尺寸：优先使用 transforms_info（VGGTImageTransform），否则回退到 VGGT 输入尺寸
        if transforms_info is None or not (0 <= reference_image_idx < len(transforms_info)):
            logger.warning("transforms_info missing; falling back to VGGT input size for original size.")
            raise ValueError("transforms_info missing")
        else:
            ref_transform = transforms_info[reference_image_idx]
            orig_h = int(ref_transform.orig_height)
            orig_w = int(ref_transform.orig_width)
        logger.info(f"Original image size: {orig_w}x{orig_h}, VGGT input size: {W}x{H}")

        # 3. 为每个边界框生成查询点（严格按 load_and_preprocess_images 的几何映射对齐）
        
        # 标准路径：使用 transforms_info（源自 PIL.Image.size）进行严格对齐
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

        all_query_points_tensor, points_per_object = generate_points_from_bboxes(
            ref_bboxes, (H, W), config
        )
        
        if all_query_points_tensor is None:
            logger.warning("Could not generate query points from bounding boxes.")
            return {}, None

        logger.info(f"Generated {len(all_query_points_tensor)} query points")
        all_query_points_tensor = all_query_points_tensor.to(device)

        # 4. 使用 VGGT 执行点追踪
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

        # 5. 使用新的基于对应关系的物体匹配逻辑
        logger.info("Matching objects using correspondence-based logic...")
        object_correspondences = {}
        
        for s_idx in range(S):
            if s_idx == reference_image_idx:
                continue

            # 使用新的匹配函数
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


# --- Execution Example ---

class SKUMatchingSystem:
    """SKU匹配系统类"""
    
    def __init__(self, config: SKUMatchingConfig = None):
        """初始化SKU匹配系统
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        self.config = config or SKUMatchingConfig()
        self.vggt_model = None
        self._is_initialized = False
        
    def initialize(self) -> None:
        """初始化模型"""
        if self._is_initialized:
            logger.info("Models already initialized")
            return
            
        logger.info("Initializing SKU matching system...")
        
        try:
            if self.config.seed is not None:
                try:
                    import random
                    random.seed(self.config.seed)
                except Exception:
                    pass
                np.random.seed(self.config.seed)
                torch.manual_seed(self.config.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.config.seed)

            self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.config.device).eval()
            logger.info("VGGT model loaded successfully")
            
            self._is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def process_images(self, image_folder: str, detection_dir: str, 
                      reference_image_idx: int = 0, max_images: int = 20) -> Dict[int, List[Dict]]:
        """处理图像文件夹
        
        Args:
            image_folder: 图像文件夹路径
            detection_dir: 检测结果目录路径，包含按数字命名的JSON文件
            reference_image_idx: 参考图像索引
            max_images: 最大处理图像数量
            
        Returns:
            对应关系结果
        """
        if not self._is_initialized:
            self.initialize()
            
        try:
            logger.info("Loading images and detection results...")
            
            image_folder_path = Path(image_folder)
            if not image_folder_path.exists():
                raise FileNotFoundError(f"Image folder not found: {image_folder}")
                
            # 修复：按数字順序加载图像和检测结果，确保按文件名匹配
            image_files = []
            for f in os.listdir(image_folder):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    try:
                        # 提取文件名中的数字
                        file_number = int(Path(f).stem)
                        image_files.append((file_number, str(image_folder_path / f)))
                    except ValueError:
                        logger.warning(f"Skipping non-numeric image file: {f}")
                        continue
            
            # 按数字順序排序
            image_files.sort(key=lambda x: x[0])
            
            # 只保留有对应检测结果的图像
            available_detection_numbers = set()
            for detection_file in Path(detection_dir).glob("*.json"):
                try:
                    detection_number = int(detection_file.stem)
                    available_detection_numbers.add(detection_number)
                except ValueError:
                    continue
            
            # 过滤出有对应检测结果的图像
            matched_files = []
            for file_number, image_path in image_files:
                if file_number in available_detection_numbers:
                    matched_files.append((file_number, image_path))
                else:
                    logger.info(f"Skipping image {file_number}.jpg - no corresponding detection file")
            
            image_paths = [path for _, path in matched_files]
            image_numbers = [num for num, _ in matched_files]
            
            # 按照图像的数字順序加载对应的检测结果
            detections = []
            for img_number in image_numbers:
                detection_file = Path(detection_dir) / f"{img_number}.json"
                if detection_file.exists():
                    with open(detection_file, 'r', encoding='utf-8') as f:
                        file_detections = json.load(f)
                    if isinstance(file_detections, list) and len(file_detections) > 0:
                        detections.append(file_detections[0])
                    elif isinstance(file_detections, dict):
                        detections.append(file_detections)
                    else:
                        logger.error(f"Invalid detection format in {detection_file}")
                        raise ValueError(f"Invalid detection format in {detection_file}")
                else:
                    logger.error(f"Detection file not found: {detection_file}")
                    raise FileNotFoundError(f"Detection file not found: {detection_file}")
            
            if not image_paths:
                raise ValueError(f"No images found in {image_folder}")
            
            if max_images is not None:
                image_paths = image_paths[:max_images]
                image_numbers = image_numbers[:max_images]
                detections = detections[:max_images]
            
            logger.info(f"Loaded {len(image_paths)} images with {len(detections)} detection files")
            
            transforms_info = build_vggt_transforms(image_paths, target_size=518)
            images = load_and_preprocess_images(image_paths, mode="crop").to(self.config.device)
            
            # 运行物体对应流程
            logger.info("Running object correspondence detection...")
            use_amp = (
                self.config.use_autocast
                and torch.cuda.is_available()
                and isinstance(self.config.dtype, torch.dtype)
                and (isinstance(self.config.device, str) and self.config.device.startswith("cuda"))
            )
            amp_ctx = torch.cuda.amp.autocast(dtype=self.config.dtype) if use_amp else nullcontext()
            with amp_ctx:
                correspondences, points_map = find_object_correspondences(
                    self.vggt_model,
                    detections,
                    images,
                    self.config,
                    reference_image_idx=reference_image_idx,
                    transforms_info=transforms_info
                )
            
            # 可视化结果
            if correspondences:
                logger.info("Generating visualization...")
                visualize_results(images, reference_image_idx, points_map, correspondences, self.config,
                                 detections, transforms_info)
                
                # 打印结果摘要
                self._print_results_summary(correspondences)

                # 可选保存 JSON
                if self.config.save_json:
                    meta = {
                        "image_paths": image_paths,
                        "reference_image_idx": reference_image_idx,
                        "config": {
                            "visibility_threshold": self.config.visibility_threshold,
                            "min_visible_points": self.config.min_visible_points,
                            "max_points_per_bbox": self.config.max_points_per_bbox,
                            "max_bboxes": self.config.max_bboxes,
                        },
                    }
                    save_correspondences_json(correspondences, points_map, self.config, meta)
            else:
                logger.warning("No object correspondences found")
            
            # 可选：返回时附带 transforms_info 以便上游复用（不改变现有返回结构）
            return correspondences
            
        except Exception as e:
            logger.error(f"Failed to process images: {e}")
            raise
    
    def _print_results_summary(self, correspondences: Dict[int, List[Dict]]) -> None:
        """打印结果摘要"""
        logger.info("\n=== Object Correspondences Summary (Based on 50% Threshold) ===")
        for target_idx, found_objects in correspondences.items():
            logger.info(f"\nTarget Image {target_idx}:")
            for obj in found_objects:
                correspondence_ratio = obj.get('correspondence_ratio', 0.0)
                matched_points = obj.get('matched_points', 0)
                total_points = obj.get('total_points', 0)
                target_bbox_id = obj.get('target_bbox_id', 'N/A')               
                
                original_box = [round(c, 2) for c in obj['box']]
                vggt_box = [round(c, 2) for c in obj.get('vggt_box', [])]
                
                logger.info(f"  - Reference ID {obj['object_id']} → Target bbox {target_bbox_id}: "
                         f"original_box={original_box}, vggt_box={vggt_box}, "
                         f"ratio={correspondence_ratio:.3f} ({matched_points}/{total_points})")

if __name__ == '__main__':
    # 示例使用 - 展示两种匹配算法
    try:
        print("=== SKU匹配系统示例 ===\n")
        
        # ===== 示例1: 使用传统的点跟踪匹配算法 =====
        print("1. 使用传统的点跟踪匹配算法:")
        config_traditional = SKUMatchingConfig(
            max_points_per_bbox=100,
            visibility_threshold=0.7,
            min_visible_points=10,
            output_dir="output_results_traditional",
            use_3d_projection_matching=False  # 使用传统算法
        )
        
        sku_system_traditional = SKUMatchingSystem(config_traditional)
        
        correspondences_traditional = sku_system_traditional.process_images(
            image_folder="../imdata/total",
            detection_dir="../imdata/detections_results",
            reference_image_idx=0,
            max_images=13
        )
        
        print(f"传统算法找到 {sum(len(matches) for matches in correspondences_traditional.values())} 个匹配\n")
        
        # ===== 示例2: 使用新的3D-2D投影匹配算法 =====
        print("2. 使用新的3D-2D投影匹配算法（增强3D验证）:")
        config_3d = SKUMatchingConfig(
            output_dir="output_results_3d_projection",
            use_3d_projection_matching=True,  # 使用新的3D-2D投影算法
            # 3D相关参数（更严格的筛选）
            depth_confidence_threshold=0.15,
            world_points_confidence_threshold=0.15,
            min_depth=0.1,
            max_depth=10.0,
            points_per_bbox_3d=50,
            projection_match_threshold=0.7,  # 提高投影匹配阈值到70%
            # 3D几何验证参数
            max_3d_distance=1.0,            # 最大1米距离差异
            max_depth_difference=2.0,       # 最大2米深度差异
            min_depth_consistency=0.3       # 最小30%深度一致性
        )
        
        sku_system_3d = SKUMatchingSystem(config_3d)
        
        correspondences_3d = sku_system_3d.process_images(
            image_folder="../imdata/total",
            detection_dir="../imdata/detections_results",
            reference_image_idx=0,
            max_images=13
        )
        
        print(f"3D-2D投影算法找到 {sum(len(matches) for matches in correspondences_3d.values())} 个匹配\n")
        
        print("=== 处理完成! ===")
        print("可以查看不同输出目录中的可视化结果:")
        print("- output_results_traditional/ (传统点跟踪算法结果)")
        print("- output_results_3d_projection/ (3D-2D投影算法结果)")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
