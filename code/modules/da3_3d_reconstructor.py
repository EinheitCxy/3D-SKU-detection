#!/usr/bin/env python3
"""
基于 Depth-Anything-3 (DA3) 的3D重建器

功能：
1. 从目录加载图像
2. 使用 DA3 模型推理，得到深度图与相机位姿
3. 从深度图 + 内外参计算 world_points（与 Pi3 缓存格式兼容）
4. 导出 GLB（调用 DA3 内置 export）
5. 保存 da3_cache/predictions.npz（供 SKU 匹配使用）

使用：
  uv run python -m modules.da3_3d_reconstructor --input_dir <images_dir> --output_file <out.glb>
"""

from __future__ import annotations

import os
import sys
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# 路径注入：DA3 源码位于仓库根目录/Depth-Anything-3/src
THIS_DIR = Path(__file__).resolve().parent   # code/modules
CODE_ROOT = THIS_DIR.parent                  # code/
REPO_ROOT = CODE_ROOT.parent                 # 3D_Recognization/
DA3_SRC = REPO_ROOT / "Depth-Anything-3" / "src"

if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from .reconstructor_base import ReconstructorBase, register_reconstructor  # noqa: E402


# ---- 工具函数 ----

def _depth_to_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    """将深度图转为世界坐标系 3D 点云。

    Args:
        depth:      (N, H, W)   深度图（相机前向 z 值）
        intrinsics: (N, 3, 3)   相机内参 K
        extrinsics: (N, 4, 4)   W2C 外参矩阵

    Returns:
        world_points: (N, H, W, 3)  世界坐标系点云
    """
    N, H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    # (H, W, 3) 齐次像素坐标
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)  # (H, W, 3)
    pixels_flat = pixels.reshape(-1, 3)  # (H*W, 3)

    world_points = np.zeros((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        K = intrinsics[i]           # (3, 3)
        W2C = extrinsics[i]         # (4, 4)
        C2W = np.linalg.inv(W2C)    # (4, 4)
        K_inv = np.linalg.inv(K)    # (3, 3)

        d = depth[i].reshape(-1)    # (H*W,)
        # 相机坐标系下 3D 点
        p_cam = (K_inv @ pixels_flat.T).T * d[:, None]   # (H*W, 3)
        # 齐次坐标
        p_cam_h = np.concatenate(
            [p_cam, np.ones((len(p_cam), 1), dtype=np.float32)], axis=-1
        )  # (H*W, 4)
        # 变换到世界坐标系
        p_world = (C2W @ p_cam_h.T).T[:, :3]            # (H*W, 3)
        world_points[i] = p_world.reshape(H, W, 3)
    return world_points


# ---- DA3 重建器 ----

@register_reconstructor("da3")
class DA33DReconstructor(ReconstructorBase):
    """Depth-Anything-3 3D重建器（与 Pi3 接口兼容）。

    推理后将结果缓存到 da3_cache/predictions.npz，
    缓存格式与 pi3_cache 完全一致，无需修改下游 SKU 匹配代码。
    """

    # HuggingFace 默认模型；可通过 model_path 覆盖
    DEFAULT_HF_REPO = "depth-anything/DA3NESTED-GIANT-LARGE"

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        model_name: str = "da3nested-giant-large",
    ) -> None:
        super().__init__(device=device, model_path=model_path, backend_name="da3")
        self.model_name = model_name
        # 延迟导入验证
        try:
            import depth_anything_3  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"无法导入 depth_anything_3，请检查路径: {DA3_SRC}。错误: {e}"
            )

    # ---- 模型加载 ----

    def load_model(self) -> None:
        """加载 DA3 模型（本地路径 / HuggingFace Hub）。"""
        from depth_anything_3.api import DepthAnything3

        repo_or_path = self.model_path or self.DEFAULT_HF_REPO
        logger.info(f"加载 DA3 模型: {repo_or_path} ...")
        t0 = time.time()
        self.model = DepthAnything3.from_pretrained(repo_or_path).to(self.device)
        self.model.eval()
        logger.info(f"DA3 模型加载完成，用时 {time.time() - t0:.2f}s")

    # ---- 图像加载 ----

    def load_images(self, input_dir: str) -> List[str]:
        """返回目录下排好序的图片路径列表（DA3 接受路径列表）。"""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            str(p) for p in Path(input_dir).iterdir()
            if p.suffix.lower() in exts
        )
        if not paths:
            raise ValueError(f"目录中未找到图片: {input_dir}")
        logger.info(f"DA3 加载 {len(paths)} 张图片")
        return paths

    # ---- 推理 ----

    def run_inference(self, image_paths: List[str]) -> Dict[str, Any]:
        """运行 DA3 推理，返回包含 depth/extrinsics/intrinsics/conf/images 的字典。

        Returns:
            pred 字典，键名与 Pi3 缓存兼容:
            - depth          : np.ndarray (N, H, W)
            - extrinsic      : np.ndarray (N, 4, 4)  W2C
            - intrinsic      : np.ndarray (N, 3, 3)
            - world_points   : np.ndarray (N, H, W, 3)
            - conf           : np.ndarray (N, H, W)  置信度（可选）
            - images         : np.ndarray (N, H, W, 3) uint8
            - _prediction    : Prediction             原始 DA3 预测对象（供 export_glb 使用）
        """
        logger.info("运行 DA3 推理...")
        t0 = time.time()

        from PIL import Image

        pil_images = [Image.open(p).convert("RGB") for p in image_paths]

        prediction = self.model.inference(
            pil_images,
            process_res=504,
            process_res_method="upper_bound_resize",
        )

        logger.info(f"DA3 推理完成，用时 {time.time() - t0:.2f}s")

        depth: np.ndarray = prediction.depth         # (N, H, W)
        extrinsics: np.ndarray = prediction.extrinsics  # (N, 4, 4) or None
        intrinsics: np.ndarray = prediction.intrinsics  # (N, 3, 3) or None
        conf: Optional[np.ndarray] = prediction.conf    # (N, H, W) or None
        proc_imgs: Optional[np.ndarray] = prediction.processed_images  # (N, H, W, 3)

        N, H, W = depth.shape
        logger.info(f"DA3 输出: N={N}, H={H}, W={W}")

        # 计算 world_points
        if extrinsics is not None and intrinsics is not None:
            logger.info("从深度图 + 内外参计算 world_points ...")
            world_points = _depth_to_world_points(depth, intrinsics, extrinsics)
        else:
            logger.warning("DA3 未返回相机参数，world_points 置零（匹配功能受限）")
            world_points = np.zeros((N, H, W, 3), dtype=np.float32)

        # 构建 images (N, H, W, 3) uint8
        if proc_imgs is not None:
            images_np = proc_imgs.astype(np.uint8) if proc_imgs.dtype != np.uint8 else proc_imgs
        else:
            # 回退：从文件重读缩放到推理尺寸
            from PIL import Image as PILImage
            images_list = []
            for p in image_paths:
                img = PILImage.open(p).convert("RGB").resize((W, H))
                images_list.append(np.array(img))
            images_np = np.stack(images_list, axis=0).astype(np.uint8)

        pred: Dict[str, Any] = {
            "depth": depth.astype(np.float32),
            "depth_conf": conf.astype(np.float32) if conf is not None else np.ones((N, H, W), dtype=np.float32),
            "world_points": world_points.astype(np.float32),
            "world_points_conf": conf.astype(np.float32) if conf is not None else np.ones((N, H, W), dtype=np.float32),
            "extrinsic": extrinsics.astype(np.float32) if extrinsics is not None else None,
            "intrinsic": intrinsics.astype(np.float32) if intrinsics is not None else None,
            "conf": conf.astype(np.float32) if conf is not None else None,
            "images": images_np,
            "_prediction": prediction,   # 保留原始对象供 export_glb 使用
            "_image_paths": image_paths,
        }
        return pred

    # ---- 导出 GLB ----

    def export_glb(
        self,
        pred: Dict[str, Any],
        output_path: Path,
        *,
        conf_thres: float = 50.0,
        show_cam: bool = True,
        **_: Any,
    ) -> None:
        """调用 DA3 内置 GLB 导出器。"""
        from depth_anything_3.utils.export import export as da3_export

        prediction = pred.get("_prediction")
        if prediction is None:
            logger.warning("DA3 export_glb: 缺少 _prediction 对象，跳过 GLB 导出")
            return

        out_dir = output_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # conf_thres 百分比 → DA3 的 conf_thresh_percentile
        da3_export(
            prediction,
            export_format="glb",
            export_dir=str(out_dir),
            glb={
                "conf_thresh_percentile": conf_thres,
                "show_cameras": show_cam,
                "num_max_points": 1_000_000,
            },
        )
        # DA3 默认保存到 <export_dir>/exports/glb/scene.glb，重命名到期望路径
        default_glb = out_dir / "exports" / "glb" / "scene.glb"
        if default_glb.exists() and default_glb != output_path:
            import shutil
            shutil.move(str(default_glb), str(output_path))
            logger.info(f"GLB 导出成功: {output_path}")
        elif output_path.exists():
            logger.info(f"GLB 已存在: {output_path}")
        else:
            logger.warning(f"GLB 导出可能失败，未找到文件: {output_path}")

    # ---- 保存缓存 ----

    def save_predictions_cache(
        self,
        predictions: Dict[str, Any],
        images: Any,
        out_dir: Path,
        *,
        image_names: Optional[List[str]] = None,
        input_dir: Optional[str] = None,
        **_: Any,
    ) -> None:
        """保存 da3_cache/predictions.npz，格式与 Pi3 缓存完全兼容。"""
        cache_dir = out_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "predictions.npz"

        save_kwargs: Dict[str, Any] = {}

        for key in ("depth", "depth_conf", "world_points", "world_points_conf", "images"):
            val = predictions.get(key)
            if val is not None and isinstance(val, np.ndarray):
                save_kwargs[key] = val.astype(np.float32 if key != "images" else np.uint8, copy=False)

        for key in ("extrinsic", "intrinsic"):
            val = predictions.get(key)
            if val is not None and isinstance(val, np.ndarray):
                save_kwargs[key] = val.astype(np.float32, copy=False)

        # conf → 作为 conf 字段（供 bbox 采样时使用）
        conf_val = predictions.get("conf")
        if conf_val is not None and isinstance(conf_val, np.ndarray):
            save_kwargs["conf"] = conf_val.astype(np.float32, copy=False)

        # image_ids
        if image_names is not None:
            image_ids = self.extract_image_ids(image_names)
            save_kwargs["image_ids"] = np.asarray(image_ids, dtype=np.int32)

        np.savez_compressed(cache_path, **save_kwargs)
        logger.info(f"保存 DA3 预测缓存: {cache_path}")


# ---- CLI 入口 ----

def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DA3 3D重建器 CLI")
    parser.add_argument("--input_dir", required=True, help="图片目录")
    parser.add_argument("--output_file", default="reconstruction_da3.glb", help="输出 GLB 路径")
    parser.add_argument("--model_path", default=None, help="DA3 模型本地路径（默认从 HuggingFace 加载）")
    parser.add_argument("--conf_thres", type=float, default=50.0)
    parser.add_argument("--no_cam", action="store_true")
    args = parser.parse_args()

    recon = DA33DReconstructor(model_path=args.model_path)
    recon.reconstruct_from_directory(
        input_dir=args.input_dir,
        output_path=args.output_file,
        conf_thres=args.conf_thres,
        show_cam=not args.no_cam,
    )


if __name__ == "__main__":
    _cli()
