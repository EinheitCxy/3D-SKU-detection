"""
SKU匹配系统坐标变换模块

包含VGGT图像变换类和坐标映射功能
"""

import math
import numpy as np
import torch
from typing import List, Tuple, Optional, Sequence, Dict, Any
from pathlib import Path
from PIL import Image
from abc import ABC, abstractmethod


class ImageTransformBase(ABC):
    """图像变换抽象基类，定义统一接口"""

    @abstractmethod
    def map_xy_to_final(self, x: float, y: float) -> Tuple[float, float]:
        """将原图坐标映射到模型输入坐标"""
        pass

    @abstractmethod
    def map_xy_to_original(self, x: float, y: float) -> Tuple[float, float]:
        """将模型输入坐标映射回原图坐标"""
        pass

    def map_bbox_to_final(self, bbox: List[float]) -> List[float]:
        """将原图边界框映射到模型输入坐标"""
        x1, y1 = self.map_xy_to_final(bbox[0], bbox[1])
        x2, y2 = self.map_xy_to_final(bbox[2], bbox[3])
        return [x1, y1, x2, y2]

    def map_bbox_to_original(self, bbox: List[float]) -> List[float]:
        """将模型输入边界框映射回原图坐标"""
        x1, y1 = self.map_xy_to_original(bbox[0], bbox[1])
        x2, y2 = self.map_xy_to_original(bbox[2], bbox[3])
        return [x1, y1, x2, y2]


class Pi3ImageTransform(ImageTransformBase):
    """Pi3图像变换类，用于原图坐标与Pi3 resize后坐标的映射

    Pi3特点：
    1. 保持宽高比的等比例缩放
    2. 无裁剪操作
    3. 尺寸对齐到14的倍数（DINOv2 patch size）
    4. 批次内所有图片统一尺寸

    ⚠️ 警告：仅用于Pi3模型！不要与VGGT模型混用！
    """

    def __init__(self, orig_width: int, orig_height: int, target_width: int, target_height: int):
        """初始化Pi3变换参数

        Args:
            orig_width: 原图宽度
            orig_height: 原图高度
            target_width: Pi3 resize后的宽度（14的倍数）
            target_height: Pi3 resize后的高度（14的倍数）
        """
        self.orig_width = int(orig_width)
        self.orig_height = int(orig_height)
        self.target_width = int(target_width)
        self.target_height = int(target_height)

        # 计算缩放比例
        self.scale_x = self.target_width / self.orig_width if self.orig_width > 0 else 1.0
        self.scale_y = self.target_height / self.orig_height if self.orig_height > 0 else 1.0

    def map_xy_to_final(self, x: float, y: float) -> Tuple[float, float]:
        """将原图坐标映射到Pi3 resize后的坐标"""
        return x * self.scale_x, y * self.scale_y

    def map_xy_to_original(self, x: float, y: float) -> Tuple[float, float]:
        """将Pi3 resize后的坐标映射回原图坐标"""
        return x / self.scale_x, y / self.scale_y

    # 继承基类的 map_bbox_to_final 和 map_bbox_to_original

    def map_bbox_to_resized(self, bbox: List[float]) -> List[float]:
        """别名方法：兼容旧代码"""
        return self.map_bbox_to_final(bbox)

    def map_xy_to_resized(self, x: float, y: float) -> Tuple[float, float]:
        """别名方法：兼容旧代码"""
        return self.map_xy_to_final(x, y)

    def map_points_to_resized(self, points):
        """批量映射点坐标到Pi3 resize后的空间

        Args:
            points: (..., 2) numpy数组或torch张量

        Returns:
            同类型同形状的映射后坐标
        """
        is_torch = torch.is_tensor(points)

        if is_torch:
            result = points.clone()
            result[..., 0] *= self.scale_x
            result[..., 1] *= self.scale_y
            return result
        else:
            result = points.copy()
            result[..., 0] *= self.scale_x
            result[..., 1] *= self.scale_y
            return result

    def map_points_to_original(self, points):
        """批量映射点坐标回原图空间

        Args:
            points: (..., 2) numpy数组或torch张量

        Returns:
            同类型同形状的映射后坐标
        """
        is_torch = torch.is_tensor(points)

        if is_torch:
            result = points.clone()
            result[..., 0] /= self.scale_x
            result[..., 1] /= self.scale_y
            return result
        else:
            result = points.copy()
            result[..., 0] /= self.scale_x
            result[..., 1] /= self.scale_y
            return result

    def get_transform_info(self) -> dict:
        """返回变换信息用于调试"""
        return {
            "original_size": (self.orig_width, self.orig_height),
            "target_size": (self.target_width, self.target_height),
            "scales": (self.scale_x, self.scale_y),
        }


def build_pi3_transforms(
    image_paths: List[str],
    pixel_limit: int = 255000
) -> List[Pi3ImageTransform]:
    """构建Pi3变换列表，复用Pi3的resize逻辑

    ⚠️ 警告：返回的Transform仅用于Pi3模型！不要传给VGGT模型！

    Args:
        image_paths: 图像路径列表
        pixel_limit: 像素数限制，默认255000（与Pi3一致）

    Returns:
        Pi3变换对象列表
    """
    if not image_paths:
        return []

    # 基于第一张图片计算统一的目标尺寸
    first_img = Image.open(image_paths[0]).convert("RGB")
    W_orig, H_orig = first_img.size

    # Pi3的resize逻辑
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1.0
    W_target = W_orig * scale
    H_target = H_orig * scale

    # 调整到14的倍数
    k = round(W_target / 14)
    m = round(H_target / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1

    TARGET_W = max(1, k) * 14
    TARGET_H = max(1, m) * 14

    # 为每张图片创建变换对象（统一目标尺寸）
    transforms = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        t = Pi3ImageTransform(w, h, TARGET_W, TARGET_H)

        # 尝试从文件名解析数值 ID，便于与 Pi3 缓存中的 image_ids 对齐
        try:
            t.image_id = int(Path(img_path).stem)
        except (ValueError, TypeError):
            # 非纯数字文件名时，保留为 None，不强制要求
            t.image_id = None

        transforms.append(t)

    return transforms


class VGGTImageTransform(ImageTransformBase):
    """修复版本的VGGT图像变换类，完全对齐load_and_preprocess_images(crop)的实现

    关键修复点：
    1. 正确的裁剪offset计算 - 修复了裁剪坐标系映射错误
    2. 精确的坐标映射逻辑 - 按VGGT实际变换顺序处理
    3. 批量填充的正确处理 - 修正为左上角对齐，而非居中对齐

    ⚠️ 警告：仅用于VGGT模型！不要与Pi3模型混用！
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

    # 继承基类的 map_bbox_to_final

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

    # 继承基类的 map_bbox_to_original

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

    ⚠️ 警告：返回的Transform仅用于VGGT模型！不要传给Pi3模型！

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


def build_transforms(
    image_paths: List[str],
    model_type: str = "vggt",
    **kwargs
) -> List[ImageTransformBase]:
    """自动根据模型类型构建对应的Transform（统一入口）

    Args:
        image_paths: 图像路径列表
        model_type: 模型类型，"vggt" 或 "pi3"
        **kwargs: 传递给具体构建函数的参数
            - target_size: VGGT目标尺寸（默认518）
            - pixel_limit: Pi3像素限制（默认255000）

    Returns:
        Transform列表（统一基类类型）

    Raises:
        ValueError: 不支持的模型类型

    Examples:
        >>> # VGGT模型
        >>> transforms = build_transforms(image_paths, model_type="vggt")
        >>> # Pi3模型
        >>> transforms = build_transforms(image_paths, model_type="pi3")
    """
    model_type = model_type.lower()

    if model_type in ("vggt", "dust3r"):
        target_size = kwargs.get("target_size", 518)
        return build_vggt_transforms(image_paths, target_size=target_size)
    elif model_type == "pi3":
        pixel_limit = kwargs.get("pixel_limit", 255000)
        return build_pi3_transforms(image_paths, pixel_limit=pixel_limit)
    else:
        raise ValueError(
            f"不支持的模型类型: {model_type}。"
            f"支持的类型: 'vggt', 'pi3'"
        )


# ========== Lightweight adapters for transforms.json ==========

class JSONTransformAdapter:
    """A lightweight adapter built from transforms.json minimal fields.

    It re-implements only the forward mapping used by the viewer: map_xy_to_final.
    """

    __slots__ = ("sx", "sy", "crop_start_y", "pad_left", "pad_top", "pw", "ph")

    def __init__(
        self,
        sx: float,
        sy: float,
        crop_start_y: int,
        pad_left: int,
        pad_top: int,
        padded_width: int,
        padded_height: int,
    ) -> None:
        self.sx = float(sx)
        self.sy = float(sy)
        self.crop_start_y = int(crop_start_y)
        self.pad_left = int(pad_left)
        self.pad_top = int(pad_top)
        self.pw = int(padded_width)
        self.ph = int(padded_height)

    def map_xy_to_final(self, x: float, y: float) -> Tuple[float, float]:
        xs = x * self.sx
        ys = y * self.sy
        if self.crop_start_y > 0:
            ys -= self.crop_start_y
        # apply batch-centered padding
        xf = xs + self.pad_left
        yf = ys + self.pad_top
        # clamp to canvas
        if self.pw > 0:
            xf = 0.0 if xf < 0.0 else (float(self.pw - 1) if xf > self.pw - 1 else xf)
        if self.ph > 0:
            yf = 0.0 if yf < 0.0 else (float(self.ph - 1) if yf > self.ph - 1 else yf)
        return float(xf), float(yf)


def build_transforms_from_json(
    json_path: str | Path,
    aligned_image_ids: Sequence[int],
) -> Optional[List[JSONTransformAdapter]]:
    """Load transforms.json and return adapters ordered by aligned_image_ids.

    Returns None when file missing, malformed, or any image id is absent.
    """
    p = Path(json_path)
    if not p.exists():
        return None

    import json

    with p.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    padded_size = data.get("padded_size")
    if not padded_size or len(padded_size) != 2:
        return None
    pw, ph = int(padded_size[0]), int(padded_size[1])

    frames = data.get("frames") or []
    if not isinstance(frames, list) or len(frames) == 0:
        return None

    by_img: Dict[int, JSONTransformAdapter] = {}
    for fr in frames:
        try:
            img_id = int(fr["image_id"])
            sx, sy = fr["scales"][0], fr["scales"][1]
            crop_start_y = int(fr.get("crop_start_y", 0))
            pad_left, pad_top = fr.get("batch_padding", [0, 0])
            by_img[img_id] = JSONTransformAdapter(sx, sy, crop_start_y, int(pad_left), int(pad_top), pw, ph)
        except (KeyError, ValueError, TypeError) as e:
            # Skip malformed frame entries (missing fields or invalid types)
            logger.debug(f"Skipping malformed frame entry: {e}")
            continue

    adapters: List[JSONTransformAdapter] = []
    for img_id in aligned_image_ids:
        a = by_img.get(int(img_id))
        if a is None:
            return None
        adapters.append(a)

    return adapters
