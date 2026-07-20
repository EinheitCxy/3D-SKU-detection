"""
SKU匹配系统坐标变换模块

包含图像变换类和坐标映射功能（等比例缩放，用于模型输入坐标映射）
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image


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


class ResizeImageTransform(ImageTransformBase):
    """等比例缩放图像变换类，用于原图坐标与 resize 后模型输入坐标的映射

    特点：
    1. 保持宽高比的等比例缩放
    2. 无裁剪操作
    3. 批次内所有图片统一尺寸
    """

    def __init__(
        self, orig_width: int, orig_height: int, target_width: int, target_height: int
    ):
        """初始化变换参数

        Args:
            orig_width: 原图宽度
            orig_height: 原图高度
            target_width: resize后的宽度
            target_height: resize后的高度
        """
        self.orig_width = int(orig_width)
        self.orig_height = int(orig_height)
        self.target_width = int(target_width)
        self.target_height = int(target_height)

        # 计算缩放比例
        self.scale_x = (
            self.target_width / self.orig_width if self.orig_width > 0 else 1.0
        )
        self.scale_y = (
            self.target_height / self.orig_height if self.orig_height > 0 else 1.0
        )

    def map_xy_to_final(self, x: float, y: float) -> Tuple[float, float]:
        """将原图坐标映射到 resize 后的坐标"""
        return x * self.scale_x, y * self.scale_y

    def map_xy_to_original(self, x: float, y: float) -> Tuple[float, float]:
        """将 resize 后的坐标映射回原图坐标"""
        return x / self.scale_x, y / self.scale_y

    # 继承基类的 map_bbox_to_final 和 map_bbox_to_original

    def map_bbox_to_resized(self, bbox: List[float]) -> List[float]:
        """别名方法：兼容旧代码"""
        return self.map_bbox_to_final(bbox)

    def map_xy_to_resized(self, x: float, y: float) -> Tuple[float, float]:
        """别名方法：兼容旧代码"""
        return self.map_xy_to_final(x, y)

    def map_points_to_resized(self, points):
        """批量映射点坐标到 resize 后的空间

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
        """返回变换信息"""
        return {
            "original_size": (self.orig_width, self.orig_height),
            "target_size": (self.target_width, self.target_height),
            "scales": (self.scale_x, self.scale_y),
        }


def build_da3_transforms(
    image_paths: List[str], process_res: int = 504
) -> List[ResizeImageTransform]:
    """构建 DA3 变换列表，目标尺寸用 DA3 upper_bound_resize 算法从 process_res 派生。

    与 da3_runner.py 的 model.inference(process_res=, process_res_method="upper_bound_resize")
    逐像素一致（算法源自 DA3 input_processor._resize_longest_side）。

    Args:
        image_paths: 图像路径列表
        process_res: DA3 处理分辨率（默认504，须与 da3_runner 的 --process_res 一致）

    Returns:
        变换对象列表（target 尺寸 = DA3 cache 尺寸）
    """
    if not image_paths:
        return []

    # PIL 懒读取：仅需 (W,H) 尺寸，不 convert("RGB") 触发完整解码
    W_orig, H_orig = Image.open(image_paths[0]).size

    # DA3 upper_bound_resize 算法：长边缩到 process_res，短边按比例（无 14 对齐）
    longest = max(W_orig, H_orig)
    scale = process_res / float(longest) if longest > 0 else 1.0
    TARGET_W = max(1, int(round(W_orig * scale)))
    TARGET_H = max(1, int(round(H_orig * scale)))

    transforms = []
    for img_path in image_paths:
        # PIL 懒读取尺寸，不 convert("RGB")（ResizeImageTransform 只用 w/h 数值）
        w, h = Image.open(img_path).size
        t = ResizeImageTransform(w, h, TARGET_W, TARGET_H)
        try:
            t.image_id = int(Path(img_path).stem)
        except (ValueError, TypeError):
            t.image_id = None
        transforms.append(t)

    return transforms


def build_transforms(
    image_paths: List[str], model_type: str = "da3", **kwargs
) -> List[ImageTransformBase]:
    """根据模型类型构建对应的 Transform（统一入口，仅支持 da3）

    Args:
        image_paths: 图像路径列表
        model_type: 模型类型，当前仅支持 "da3"
        **kwargs: 传递给具体构建函数的参数
            - process_res: DA3 处理分辨率（默认504）

    Returns:
        Transform列表（统一基类类型）

    Raises:
        ValueError: 不支持的模型类型

    Examples:
        >>> transforms = build_transforms(image_paths, model_type="da3")
    """
    model_type = model_type.lower()

    if model_type == "da3":
        process_res = kwargs.get("process_res", 504)
        return build_da3_transforms(image_paths, process_res=process_res)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}。" f"仅支持 da3")
