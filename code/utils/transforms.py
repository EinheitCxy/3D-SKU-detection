"""
SKU匹配系统坐标变换模块

包含VGGT图像变换类和坐标映射功能
"""

import numpy as np
import torch
from typing import List, Tuple
from PIL import Image


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

    def _clamp_coord(self, value: float, min_val: float, max_val: float) -> float:
        """辅助函数: 将坐标限制在有效范围内"""
        return max(min_val, min(value, max_val))

    def map_xy_to_final(self, x: float, y: float) -> Tuple[float, float]:
        """将原图坐标映射到VGGT最终输入坐标

        变换顺序: 缩放 → 裁剪 → 填充
        """
        # 步骤1：缩放
        x_scaled, y_scaled = x * self.scale_x, y * self.scale_y

        # 步骤2：裁剪
        y_cropped = y_scaled - self.crop_start_y if self.crop_applied else y_scaled
        x_cropped = self._clamp_coord(x_scaled, 0.0, self.final_width - 1)
        y_cropped = self._clamp_coord(y_cropped, 0.0, self.final_height - 1)

        # 步骤3：填充
        x_final = self._clamp_coord(x_cropped + self.batch_pad_left, 0.0, self.padded_width - 1)
        y_final = self._clamp_coord(y_cropped + self.batch_pad_top, 0.0, self.padded_height - 1)

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

        逆变换顺序: 移除填充 → 移除裁剪 → 移除缩放
        """
        # 步骤1：移除填充
        x_cropped, y_cropped = xp - self.batch_pad_left, yp - self.batch_pad_top

        # 步骤2：移除裁剪
        y_scaled = y_cropped + self.crop_start_y if self.crop_applied else y_cropped

        # 步骤3：移除缩放
        x_orig = x_cropped / self.scale_x if self.scale_x != 0 else 0.0
        y_orig = y_scaled / self.scale_y if self.scale_y != 0 else 0.0

        # 限制在原图范围内
        x_orig = self._clamp_coord(x_orig, 0.0, self.orig_width - 1)
        y_orig = self._clamp_coord(y_orig, 0.0, self.orig_height - 1)

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
    """构建修复版本的VGGT变换列表，完全对齐load_and_preprocess_images("crop")的实现
    
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
    transforms: List[VGGTImageTransform] = []
    
    # 第一步：为每个图像创建变换对象
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        transforms.append(VGGTImageTransform(w, h, target_size=target_size))

    # 第二步：计算批次最大尺寸并应用填充
    max_w = max(t.final_width for t in transforms)
    max_h = max(t.final_height for t in transforms)
    for t in transforms:
        t.apply_batch_padding(max_w, max_h)

    return transforms