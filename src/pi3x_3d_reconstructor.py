#!/usr/bin/env python3
"""
基于 Pi3X 的 3D 重建与 GLB 导出器

Pi3X 是 Pi3 的增强版（卷积头减少网格伪影、可选位姿/内参/深度条件注入、
连续置信度、近似米制尺度）。本适配器产出与 Pi3 相同的 predictions.npz
缓存契约（world_points/depth/extrinsic(w2c)/intrinsic/image_ids/conf），
匹配阶段复用 pi3 数据路径（model_type="pi3"）。

与 Pi3 适配器的差异：
- 模型：pi3.models.pi3x.Pi3X，默认从本地 runtime/models/pi3x 加载（避免 HF 网络）
- 图像加载：显式按文件名数值序传入路径列表（避免字典序导致的帧错位）
- 深度/距离阈值：近似米制（匹配侧 for_3d_mapping 的 pi3x 分支按米制标定）

使用：
  uv run python -m src.pi3x_3d_reconstructor --input_dir <images_dir> --output_file <out.glb>
"""

from __future__ import annotations

import os
import re
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    import sys as _sys
    _h = logging.StreamHandler(_sys.stdout)
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# 路径注入：vendored Pi3（含 pi3x）和 root utils
THIS_DIR = Path(__file__).resolve().parent  # src/
REPO_ROOT = THIS_DIR.parent
PI3_ROOT = REPO_ROOT / "Pi3"
DEFAULT_PI3X_MODEL_DIR = REPO_ROOT / "runtime" / "models" / "pi3x"

if str(PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(PI3_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .reconstructor_base import ReconstructorBase, register_reconstructor
from .pi3_3d_reconstructor import (
    _estimate_intrinsics_from_local_points,
    _save_predictions_npz,
)


def _numeric_sort_key(filename: str):
    stem = os.path.splitext(filename)[0]
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else filename


@register_reconstructor("pi3x")
class Pi3X3DReconstructor(ReconstructorBase):
    """Pi3X 3D 重建器（缓存契约与 Pi3 一致，供匹配阶段复用 pi3 路径）。"""

    def __init__(self, device: str | None = None, model_path: str | None = None) -> None:
        super().__init__(device=device, model_path=model_path, backend_name="pi3x")
        from pi3.models.pi3x import Pi3X  # noqa: F401

    # ---- 加载 ----
    def load_model(self) -> None:
        """加载 Pi3X 模型（默认本地 runtime/models/pi3x，避免 HF 下载）。"""
        from pi3.models.pi3x import Pi3X

        src = self.model_path or str(DEFAULT_PI3X_MODEL_DIR)
        p = Path(src)
        logger.info(f"加载 Pi3X 模型: {src}")
        if p.is_dir():
            self.model = Pi3X.from_pretrained(str(p)).to(self.device).eval()
        elif p.is_file() and p.suffix == ".safetensors":
            from safetensors.torch import load_file

            self.model = Pi3X()
            self.model.load_state_dict(load_file(str(p)))
            self.model = self.model.to(self.device).eval()
        else:
            # 形如 "yyfz233/Pi3X" 的 HF repo id（需网络）
            self.model = Pi3X.from_pretrained(src).to(self.device).eval()

    def load_images(self, input_dir: str) -> torch.Tensor:
        """按文件名数值序加载图像 (N,3,H,W)，与基类 image_names 顺序一致。"""
        from pi3.utils.basic import load_images_as_tensor

        names = sorted(
            [p for p in os.listdir(input_dir) if p.lower().endswith((".jpg", ".jpeg", ".png"))],
            key=_numeric_sort_key,
        )
        paths = [str(Path(input_dir) / n) for n in names]
        imgs = load_images_as_tensor(paths, verbose=False)
        return imgs.to(self.device)

    # ---- 推理 ----
    def run_inference(self, images_nchw: torch.Tensor) -> Dict[str, Any]:
        """运行 Pi3X 推理，输出后处理为 pi3 缓存契约所需的字段。"""
        x = images_nchw[None]
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        logger.info("运行 Pi3X 推理...")
        t0 = time.time()
        with torch.no_grad():
            amp_ctx = (
                torch.amp.autocast("cuda", dtype=dtype)
                if self.device.startswith("cuda") and torch.cuda.is_available()
                else torch.cuda.amp.autocast(enabled=False)
            )
            with amp_ctx:
                pred: Dict[str, torch.Tensor] = self.model(x)

        from pi3.utils.geometry import depth_edge

        # Pi3X 输出 keys: points/local_points/rays/conf(logits)/camera_poses(c2w)/metric
        # 置信度后处理：sigmoid + 深度边缘抑制（同 pi3）
        pred["conf"] = torch.sigmoid(pred["conf"])
        edge = depth_edge(pred["local_points"][..., 2], rtol=0.03)
        pred["conf"][edge] = 0.0

        # Extrinsic：由 C2W 的 camera_poses 反求 W2C
        pred["extrinsic"] = torch.linalg.inv(pred["camera_poses"])

        # Depth / depth_conf / world_points_conf
        pred["depth"] = pred["local_points"][..., 2:3]
        depth_conf = pred["conf"]
        if depth_conf.ndim == 5 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]
        pred["depth_conf"] = depth_conf
        pred["world_points_conf"] = depth_conf

        # Intrinsic：基于 local_points 最小二乘拟合（复用 pi3 实现）
        pred["intrinsic"] = _estimate_intrinsics_from_local_points(
            pred["local_points"],
            conf=pred["conf"],
            max_points_per_view=50000,
        )

        # 存储原始图像 (BNHWC，0-1范围) 以兼容 viewer
        pred["images"] = x.permute(0, 1, 3, 4, 2)

        elapsed = time.time() - t0
        logger.info(f"Pi3X 推理完成，用时 {elapsed:.2f}s；返回键: {list(pred.keys())}")
        return pred

    # ---- 导出 ----
    def prepare_export_data(
        self, predictions: Dict[str, Any], images: torch.Tensor
    ) -> tuple[Dict[str, Any], torch.Tensor]:
        """每个独立张量仅拷贝到 CPU 一次，已知切片和别名在 CPU 上派生。"""
        derived_keys = {"depth", "depth_conf", "world_points_conf", "images"}
        cpu_predictions = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in predictions.items()
            if key not in derived_keys
        }
        cpu_images = images.detach().cpu()
        cpu_predictions["images"] = cpu_images[None].permute(0, 1, 3, 4, 2)
        cpu_predictions["depth"] = cpu_predictions["local_points"][..., 2:3]
        conf = cpu_predictions["conf"]
        depth_conf = conf[..., 0] if conf.ndim == 5 and conf.shape[-1] == 1 else conf
        cpu_predictions["depth_conf"] = depth_conf
        cpu_predictions["world_points_conf"] = depth_conf
        if "metric" in cpu_predictions:
            logger.info(f"metric scale: {cpu_predictions['metric'].float().numpy()}")
        return cpu_predictions, cpu_images

    def export_glb(self, pred: Dict[str, Any], output_path: Path, *, conf_thres: float = 50.0, show_cam: bool = True) -> None:
        """将 Pi3X 预测结果导出为 GLB 文件（src.pi3_glb_export，无 gradio 依赖）。"""
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
        """保存 Pi3X 预测缓存（与 pi3_cache 同契约，写到 pi3x_cache/）。"""
        image_ids: Optional[List[int]] = None
        image_paths: List[str] = []
        if image_names is not None:
            image_ids = self.extract_image_ids(image_names)
            if input_dir is not None:
                image_paths = [str(Path(input_dir) / n) for n in image_names]

        cache_dir = out_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1) transforms.json（pi3 式：仅缩放，无裁剪/填充）
        if image_paths:
            from utils.transforms import build_transforms
            transforms = build_transforms(image_paths, model_type="pi3", pixel_limit=255000)
            first_info = transforms[0].get_transform_info()
            tw, th = int(first_info["target_size"][0]), int(first_info["target_size"][1])
            tf_path = cache_dir / "transforms.json"

            frames = []
            for idx, (name, t) in enumerate(zip(image_names or [], transforms)):
                info = t.get_transform_info()
                sx, sy = float(info["scales"][0]), float(info["scales"][1])
                frames.append({
                    "frame_idx": int(idx),
                    "image_id": int(image_ids[idx]) if image_ids else int(idx),
                    "source_path": str(Path(input_dir or "") / name) if input_dir and name else "",
                    "scales": [sx, sy],
                    "crop_start_y": 0,
                    "batch_padding": [0, 0],
                })

            import json
            payload = {
                "target_size": [tw, th],
                "padded_size": [tw, th],
                "frames": frames,
            }
            with tf_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        # 2) predictions.npz
        cache_path = cache_dir / "predictions.npz"
        _save_predictions_npz(predictions, images, cache_path, image_ids=image_ids, source_model="pi3x")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Pi3X 3D重建GLB导出器")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output_file", type=str, required=True, help="输出GLB文件路径")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="计算设备")
    parser.add_argument("--model_path", type=str, default=None, help="Pi3X权重路径（目录或.safetensors，默认 runtime/models/pi3x）")
    parser.add_argument("--conf_thres", type=float, default=50.0, help="置信度阈值(0-100)")
    parser.add_argument("--no_show_cam", action="store_true", help="GLB中不显示相机")
    args = parser.parse_args()

    recon = Pi3X3DReconstructor(device=args.device, model_path=args.model_path)
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
