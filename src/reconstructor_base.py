"""
通用 3D 重建器抽象基类。

为不同后端（VGGT、Pi3、Dust3r 等）提供统一的模板方法：
- 设备/精度选择与日志
- 模型加载、图像加载、推理与导出 GLB 的接口
- 可选的 predictions 缓存保存钩子

子类至少需要实现：
- load_model()
- load_images(input_dir)
- run_inference(images)
- export_glb(predictions, output_path, *, conf_thres, show_cam, **kwargs)
- 如需写入缓存：save_predictions_cache(...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import gc
import os
import time

import torch

from utils.config import get_optimal_device_config


logger = logging.getLogger(__name__)


class ReconstructorBase:
    """3D 重建器抽象基类。

    子类应实现模型相关的方法；本类提供统一的 `reconstruct_from_directory` 模板流程。
    """

    def __init__(self, device: Optional[str] = None, model_path: Optional[str] = None, backend_name: str = "vggt") -> None:
        optimal_device, optimal_dtype = get_optimal_device_config(verbose=True)
        if device is None:
            self.device = str(optimal_device)
            self.dtype = optimal_dtype if str(optimal_device).startswith("cuda") else torch.float32
        else:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，回退 CPU")
                self.device = "cpu"
                self.dtype = torch.float32
            elif device == "cpu":
                self.device = "cpu"
                self.dtype = torch.float32
            else:
                self.device = device
                self.dtype = optimal_dtype if str(optimal_device).startswith("cuda") else torch.float32

        self.model_path = model_path
        self.model: Any = None
        self.backend_name = backend_name  # 用于生成cache目录名称：vggt_cache 或 pi3_cache

    # ---- 子类需实现的方法 ----
    def load_model(self) -> None:
        raise NotImplementedError

    def load_images(self, input_dir: str) -> Any:
        raise NotImplementedError

    def run_inference(self, images: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def export_glb(
        self,
        predictions: Dict[str, Any],
        output_path: Path,
        *,
        conf_thres: float = 50.0,
        show_cam: bool = True,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    # 可选：由子类实现，保存 predictions 缓存（viewer 兼容）
    def save_predictions_cache(
        self,
        predictions: Dict[str, Any],
        images: Any,
        out_dir: Path,
        *,
        image_names: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        pass

    # ---- 通用工具 ----
    @staticmethod
    def extract_image_ids(image_names: List[str]) -> List[int]:
        """从文件名中提取数字 ID；若失败则按顺序编号。

        兼容 "1.jpg", "IMG_0003.png" 等格式。
        """
        import re

        ids: List[int] = []
        for name in image_names:
            m = re.search(r"(\d+)", Path(name).stem)
            if m:
                ids.append(int(m.group(1)))
            else:
                ids.append(len(ids))
        return ids

    # ---- 模板方法 ----
    def reconstruct_from_directory(
        self,
        *,
        input_dir: str,
        output_path: str,
        conf_thres: float = 50.0,
        show_cam: bool = True,
        save_predictions: bool = True,
        **kwargs: Any,
    ) -> Path:
        """执行完整重建流程：加载 → 推理 → (可选)缓存 → 导出 GLB。

        子类如需自定义流程，可覆盖本方法；否则实现各抽象接口即可。
        """
        total_t0 = time.time()
        if self.model is None:
            t0 = time.time()
            self.load_model()
            logger.info(f"模型加载耗时: {time.time() - t0:.2f}s")

        t0 = time.time()
        images = self.load_images(input_dir)
        logger.info(f"图像加载/预处理耗时: {time.time() - t0:.2f}s")

        # 记录输入顺序（日志用，按文件名数字数值序避免≥10图时字典序错位如1,10,11,2）
        import re as _re
        try:
            image_names = sorted(
                [p for p in os.listdir(input_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))],
                key=lambda p: int(_re.search(r"(\d+)", os.path.splitext(p)[0]).group(1)) if _re.search(r"(\d+)", os.path.splitext(p)[0]) else p
            )
            logger.info(f"处理图片: {image_names}")
        except Exception:
            image_names = None

        t0 = time.time()
        predictions = self.run_inference(images)
        logger.info(f"推理耗时: {time.time() - t0:.2f}s")

        out_path = Path(output_path)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_predictions:
            try:
                t0 = time.time()
                self.save_predictions_cache(
                    predictions, images, out_dir, image_names=image_names, input_dir=input_dir, **kwargs
                )
                logger.info(f"缓存保存耗时: {time.time() - t0:.2f}s")
            except Exception as e:  # noqa: BLE001 - 容忍缓存失败不影响 GLB
                logger.warning(f"保存预测缓存失败（不影响GLB导出）：{e}")

        # 导出 GLB
        t0 = time.time()
        self.export_glb(
            predictions,
            out_path,
            conf_thres=conf_thres,
            show_cam=show_cam,
            **kwargs,
        )
        logger.info(f"GLB导出耗时: {time.time() - t0:.2f}s")

        # 资源清理（不抛异常）
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        logger.info(f"总流程耗时: {time.time() - total_t0:.2f}s")
        return out_path


# ---- 后端注册表 ----
# 新增后端只需：实现 ReconstructorBase 子类 + 用 @register_reconstructor("<name>") 装饰，
# 然后在 src/__init__.py import 该子类（触发注册）。无需改动 main.py / config.py 的 if/else。
RECONSTRUCTOR_REGISTRY: Dict[str, type] = {}


def register_reconstructor(name: str):
    """装饰器：注册 3D 重建后端类。"""

    def decorator(cls: type) -> type:
        if name in RECONSTRUCTOR_REGISTRY and RECONSTRUCTOR_REGISTRY[name] is not cls:
            logger.debug(
                f"重建后端 '{name}' 已注册，覆盖: "
                f"{RECONSTRUCTOR_REGISTRY[name].__name__} -> {cls.__name__}"
            )
        RECONSTRUCTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_reconstructor(name: str) -> type:
    """按名称查表返回重建后端类；未注册则列出可用后端。"""
    key = name.lower()
    if key not in RECONSTRUCTOR_REGISTRY:
        available = ", ".join(sorted(RECONSTRUCTOR_REGISTRY.keys())) or "(无)"
        raise ValueError(f"未知重建后端: {name}. 已注册: {available}")
    return RECONSTRUCTOR_REGISTRY[key]
