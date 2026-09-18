#!/usr/bin/env python3
"""
基于 MapAnything 的 3D 重建与 GLB 导出器

MapAnything（map-anything/，CC-BY-NC 4.0）多视角几何模型：单次前向输出每视角
米制世界系点云 pts3d、深度 depth_z、OpenCV c2w 位姿与内部分辨率内参。
本适配器产出与 da3 相同的 schema-v3 predictions.npz 缓存契约（含
source_image_sizes + source_to_processed_affine 完整 (2,3) affine），匹配阶段走
da3 式完整 affine 消费路径（model_type="mapanything"）。

图像几何：mapanything.utils.image.load_images(resize_mode="fixed_mapping") 按平均
宽高比选 RESOLUTION_MAPPINGS[518] 桶（如 3:4 → 392×518），先等比缩放
（max(tw/W, th/H)+1e-8）再居中裁剪 → affine 含平移分量（utils/transforms.py
的 build_mapanything_transforms / mapanything_frame_affine 逐像素复现同一算法）。

使用：
  uv run python -m src.mapanything_3d_reconstructor --input_dir <images_dir> --output_file <out.glb>
"""

from __future__ import annotations

import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    import sys as _sys
    _h = logging.StreamHandler(_sys.stdout)
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# 路径注入：vendored map-anything 和 root utils
THIS_DIR = Path(__file__).resolve().parent  # src/
REPO_ROOT = THIS_DIR.parent
MAPANYTHING_ROOT = REPO_ROOT / "map-anything"
DEFAULT_MAPANYTHING_MODEL_DIR = Path(
    "/home/xingyu/.cache/modelscope/models/facebook--map-anything/snapshots/master"
)

if str(MAPANYTHING_ROOT) not in sys.path:
    sys.path.insert(0, str(MAPANYTHING_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.transforms import (
    mapanything_frame_affine,
    mapanything_oriented_size,
    mapanything_target_size,
)

from .reconstructor_base import ReconstructorBase, register_reconstructor
from .pi3_3d_reconstructor import _save_predictions_npz


def _numeric_sort_key(name: str):
    """与 reconstructor_base.reconstruct_from_directory 的 image_names 排序一致。"""
    m = re.search(r"(\d+)", os.path.splitext(name)[0])
    return (0, int(m.group(1))) if m else (1, name)


@register_reconstructor("mapanything")
class MapAnything3DReconstructor(ReconstructorBase):
    """MapAnything 3D 重建器（缓存契约同 da3 schema-v3，匹配侧走完整 affine 路径）。"""

    def __init__(self, device: str | None = None, model_path: str | None = None) -> None:
        super().__init__(device=device, model_path=model_path, backend_name="mapanything")
        self._orig_sizes: List[Tuple[int, int]] = []  # EXIF 方向校正后的每帧 (W, H)

    # ---- 加载 ----
    def load_model(self) -> None:
        """加载 MapAnything 模型（默认 modelscope 快照目录，避免 HF 网络）。"""
        from mapanything.models import MapAnything

        src = self.model_path or str(DEFAULT_MAPANYTHING_MODEL_DIR)
        logger.info(f"加载 MapAnything 模型: {src}")
        self.model = MapAnything.from_pretrained(src).to(self.device).eval()

    def load_images(self, input_dir: str) -> List[Dict[str, Any]]:
        """按文件名数值序加载 views（顺序与基类 image_names 一致），并记录 cache 几何。"""
        from mapanything.utils.image import load_images as ma_load_images

        names = sorted(
            (p for p in os.listdir(input_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))),
            key=_numeric_sort_key,
        )
        paths = [str(Path(input_dir) / n) for n in names]
        orig_sizes = [mapanything_oriented_size(p) for p in paths]
        expected_target = mapanything_target_size(orig_sizes)

        views = ma_load_images(paths)
        if len(views) != len(paths):
            raise ValueError(f"MapAnything 加载帧数({len(views)})与输入({len(paths)})不一致")

        # 校验 vendored fixed_mapping 桶与本仓库几何复现一致（防漂移）
        for idx, view in enumerate(views):
            true_h, true_w = (int(v) for v in view["true_shape"][0])
            if (true_w, true_h) != tuple(expected_target):
                raise ValueError(
                    f"MapAnything 第{idx}帧输出尺寸({true_w},{true_h})"
                    f"与本地桶算法{expected_target}不一致"
                )

        self._orig_sizes = orig_sizes
        return views

    # ---- 推理 ----
    def run_inference(self, views: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行 MapAnything 推理，输出后处理为 da3 缓存契约所需的字段。"""
        logger.info("运行 MapAnything 推理...")
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
            )

        # outputs: 每视角一个 dict，张量带 batch=1 维；按视角维拼接
        pts3d = torch.cat([o["pts3d"] for o in outputs], dim=0)  # (N,H,W,3) world, metric
        pts3d_cam = torch.cat([o["pts3d_cam"] for o in outputs], dim=0)
        depth_z = torch.cat([o["depth_z"] for o in outputs], dim=0)  # (N,H,W,1) metric
        camera_poses = torch.cat([o["camera_poses"] for o in outputs], dim=0)  # (N,4,4) c2w
        intrinsics = torch.cat([o["intrinsics"] for o in outputs], dim=0)  # (N,3,3)
        conf = torch.cat([o["conf"] for o in outputs], dim=0)  # (N,H,W) 原样，越高越好
        mask = torch.cat([o["mask"] for o in outputs], dim=0)  # (N,H,W,1) bool
        images = torch.cat([o["img_no_norm"] for o in outputs], dim=0)  # (N,H,W,3) [0,1]

        # masked 像素几何已被 mapanything 清零；conf 头未过 mask，这里补齐语义：
        # 无几何的像素置信度置 0（where 选择而非乘法，NaN 安全）
        conf = torch.where(mask[..., 0], conf, torch.zeros_like(conf))

        pred: Dict[str, Any] = {
            "points": pts3d,  # world_points
            "local_points": pts3d_cam,
            "depth": depth_z,
            "extrinsic": torch.linalg.inv(camera_poses),  # c2w -> w2c (N,4,4)
            "intrinsic": intrinsics,
            "camera_poses": camera_poses,  # GLB 导出用
            "conf": conf,
            "depth_conf": conf,
            "world_points_conf": conf,
            "images": images,
        }

        elapsed = time.time() - t0
        logger.info(f"MapAnything 推理完成，用时 {elapsed:.2f}s；返回键: {list(pred.keys())}")
        return pred

    # ---- 导出 ----
    def prepare_export_data(
        self, predictions: Dict[str, Any], images: List[Dict[str, Any]]
    ) -> tuple[Dict[str, Any], torch.Tensor]:
        """每个张量仅拷贝到 CPU 一次；images 直接取 pred['images']。"""
        cpu_predictions = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in predictions.items()
        }
        return cpu_predictions, cpu_predictions["images"]

    def export_glb(
        self, pred: Dict[str, Any], output_path: Path, *, conf_thres: float = 50.0, show_cam: bool = True
    ) -> None:
        """将 MapAnything 预测结果导出为 GLB 文件（src.pi3_glb_export，无 gradio 依赖）。"""
        from .pi3_glb_export import predictions_to_glb

        pred_np: Dict[str, Any] = {}
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy()
                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr[0]
                pred_np[k] = arr
            else:
                pred_np[k] = v

        out_dir = output_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        scene = predictions_to_glb(
            pred_np,
            conf_thres=conf_thres,
            filter_by_frames="all",
            show_cam=show_cam,
        )
        scene.export(file_obj=str(output_path))
        logger.info(f"GLB导出成功: {output_path}")

    # ---- 缓存 ----
    def save_predictions_cache(
        self,
        predictions: Dict[str, Any],
        images: torch.Tensor,
        out_dir: Path,
        *,
        image_names: Optional[List[str]] = None,
        input_dir: Optional[str] = None,
        **_: Any,
    ) -> None:
        """保存 MapAnything 预测缓存（da3 式 schema-v3：完整 source→processed affine）。"""
        if image_names is None:
            raise ValueError("MapAnything cache save requires image_names")
        image_ids = self.extract_image_ids(image_names)

        cache_dir = out_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 目标尺寸以模型实际输出为准（load_images 已校验与本地桶算法一致）
        n, h, w = images.shape[:3]
        if len(self._orig_sizes) != n:
            raise ValueError(f"orig sizes 数({len(self._orig_sizes)})与帧数({n})不一致")

        source_image_sizes = np.asarray(self._orig_sizes, dtype=np.int64)  # (N,2) (W,H)
        affines = np.stack(
            [mapanything_frame_affine(ow, oh, w, h) for ow, oh in self._orig_sizes]
        )  # (N,2,3) float64

        extra_arrays = {
            "source_image_sizes": source_image_sizes,
            "source_to_processed_affine": affines,
            "cache_schema_version": np.array(3, dtype=np.int64),
            "affine_convention": np.array(["pixel_center_v1"], dtype=object),
            "is_metric": np.array(1, dtype=np.int64),
            "scale_factor": np.array(1.0, dtype=np.float64),
        }
        cache_path = cache_dir / "predictions.npz"
        _save_predictions_npz(
            predictions,
            images,
            cache_path,
            image_ids=image_ids,
            source_model="mapanything",
            extra_arrays=extra_arrays,
        )


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="MapAnything 3D重建GLB导出器")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output_file", type=str, required=True, help="输出GLB文件路径")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="计算设备")
    parser.add_argument("--model_path", type=str, default=None,
                        help="MapAnything权重目录（默认 modelscope 快照）")
    parser.add_argument("--conf_thres", type=float, default=50.0, help="置信度阈值(0-100)")
    parser.add_argument("--no_show_cam", action="store_true", help="GLB中不显示相机")
    args = parser.parse_args()

    recon = MapAnything3DReconstructor(device=args.device, model_path=args.model_path)
    try:
        path = recon.reconstruct_from_directory(
            input_dir=args.input_dir,
            output_path=args.output_file,
            conf_thres=args.conf_thres,
            show_cam=(not args.no_show_cam),
            save_predictions=True,
        )
        logger.info(f"成功: {path}")
        return 0
    except Exception as e:
        logger.error(f"失败: {e}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
