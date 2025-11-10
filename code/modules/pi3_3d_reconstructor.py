#!/usr/bin/env python3
"""
基于 PI3 的3D重建与GLB导出器

目标：仅替换重建（reconstruct）环节，SKU匹配仍使用VGGT。

功能：
1. 从目录加载图像
2. 使用 Pi3 模型推理，得到稠密点云与相机位姿
3. 导出 GLB（支持可选相机可视化）
4. 兼容 viewer 的 predictions 缓存（保存为 vggt_cache/predictions.npz，包含 world_points 与 images）

使用：
  uv run python -m modules.pi3_3d_reconstructor --input_dir <images_dir> --output_file <out.glb>
"""

from __future__ import annotations

import os
import sys
import glob
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    import sys as _sys
    _h = logging.StreamHandler(_sys.stdout)
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# 路径注入：确保可以导入 Pi3（仓库根目录/Pi3）和 code/utils
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]  # code/modules -> code -> 仓库根目录
PI3_ROOT = REPO_ROOT / "Pi3"
if str(PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(PI3_ROOT))
CODE_ROOT = THIS_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# utils 中的通用设备与dtype选择
from .reconstructor_base import ReconstructorBase


def _to_nhwc_uint8(imgs: torch.Tensor | np.ndarray) -> np.ndarray:
    """将图像张量/数组转为 NHWC uint8（范围[0,255]）"""
    if isinstance(imgs, torch.Tensor):
        arr = imgs.detach().cpu().numpy()
    else:
        arr = imgs
    # 支持 (N,3,H,W) 或 (N,H,W,3)
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    # 归一化到 uint8
    if arr.dtype != np.uint8:
        maxv = float(arr.max()) if arr.size > 0 else 1.0
        scale = 255.0 if maxv <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr


def _save_predictions_npz(
    pred: Dict[str, Any],
    image_tensor: torch.Tensor,
    out_dir: Path,
    *,
    image_ids: Optional[List[int]] = None,
) -> Path:
    """保存与 viewer 兼容的 predictions 缓存。

    - 保存位置：<out_dir>/vggt_cache/predictions.npz
    - 最低要求键：world_points (S,H,W,3), images (S,H,W,3 uint8)
      注：将 Pi3 的 'points' 重命名为 'world_points' 以保持兼容。
    """
    cache_dir = out_dir / "vggt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "predictions.npz"

    points = pred.get("points")  # 期望 (N,H,W,3)
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if points is None:
        raise ValueError("Pi3 predictions missing 'points'")

    imgs_nhwc = _to_nhwc_uint8(image_tensor)

    # Pi3 输出通常是 (N,H,W,3)。若包含 batch 维且为单batch，去掉batch
    if points.ndim == 5 and points.shape[0] == 1:
        points = points[0]

    # 置信度（用于后续 gid 分配中的 bbox 采样）
    conf = pred.get("conf")
    if isinstance(conf, torch.Tensor):
        conf_np = conf.detach().cpu().numpy()
    else:
        conf_np = conf
    if conf_np is not None and conf_np.ndim >= 4 and conf_np.shape[0] == 1:
        # 去掉batch维
        conf_np = conf_np[0]
    if conf_np is not None and conf_np.ndim == 4 and conf_np.shape[-1] == 1:
        # (N,H,W,1) -> (N,H,W)
        conf_np = conf_np[..., 0]

    save_kwargs: Dict[str, Any] = {
        "world_points": points.astype(np.float32, copy=False),
        "images": imgs_nhwc,
    }
    if conf_np is not None:
        save_kwargs["conf"] = conf_np.astype(np.float32, copy=False)
    if image_ids is not None:
        save_kwargs["image_ids"] = np.asarray(image_ids, dtype=np.int32)
    # 附加相机位姿（若存在）
    cam = pred.get("camera_poses")
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
    if cam is not None:
        if cam.ndim >= 3 and cam.shape[0] == 1:
            cam = cam[0]
        save_kwargs["camera_poses"] = cam
    # 标注来源模型，便于下游判断
    save_kwargs["source_model"] = np.array(["pi3"], dtype=object)

    np.savez_compressed(cache_path, **save_kwargs)
    logger.info(f"保存Pi3预测缓存: {cache_path}")
    return cache_path


@dataclass
class _ExportOptions:
    conf_thres: float = 50.0  # 百分比 0-100
    show_cam: bool = True


class PI33DReconstructor(ReconstructorBase):
    """Pi3 3D重建器（仅用于重建，可无缝替换现有 VGGT 重建流程）。"""

    def __init__(self, device: str | None = None, model_path: str | None = None) -> None:
        super().__init__(device=device, model_path=model_path)

        # 延迟导入，以便路径注入生效
        try:
            from pi3.models.pi3 import Pi3  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                f"无法导入 Pi3 包，请检查子模块或路径。Pi3路径: {PI3_ROOT}. 错误: {e}"
            )

    # ---- 加载 ----
    def load_model(self) -> None:
        """加载 Pi3 模型（支持 from_pretrained 或本地 ckpt）。"""
        logger.info("加载 Pi3 模型...")
        from pi3.models.pi3 import Pi3
        if self.model_path:
            self.model = Pi3().to(self.device).eval()
            if self.model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(self.model_path)
            else:
                weight = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(weight)
            logger.info(f"已从本地权重加载: {self.model_path}")
        else:
            # 需 huggingface-hub 支持
            self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device).eval()
            logger.info("已从 HuggingFace 加载预训练 Pi3 模型")

    def load_images(self, input_dir: str) -> torch.Tensor:
        """使用 Pi3 自带工具加载图像 (N,3,H,W)"""
        from pi3.utils.basic import load_images_as_tensor
        # 仅目录输入；视频输入不在本模块范围
        imgs = load_images_as_tensor(input_dir, interval=1)
        return imgs.to(self.device)

    # ---- 推理 ----
    def run_inference(self, images_nchw: torch.Tensor) -> Dict[str, Any]:
        """运行 Pi3 推理，返回包含 points/Conf/camera_poses/images 的字典。"""
        # Pi3 需要 batch 维
        x = images_nchw[None]
        # dtype: Amp 在 Ampere 及以上默认 bfloat16；否则 float16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        logger.info("运行 Pi3 推理...")
        t0 = time.time()
        with torch.no_grad():
            amp_ctx = torch.amp.autocast('cuda', dtype=dtype) if self.device.startswith("cuda") and torch.cuda.is_available() else torch.cuda.amp.autocast(enabled=False)
            with amp_ctx:
                pred: Dict[str, torch.Tensor] = self.model(x)
        # 后处理：置信度与图像备份
        from pi3.utils.geometry import depth_edge
        pred['conf'] = torch.sigmoid(pred['conf'])
        edge = depth_edge(pred['local_points'][..., 2], rtol=0.03)
        pred['conf'][edge] = 0.0
        # 为简洁不返回 local_points
        if 'local_points' in pred:
            del pred['local_points']
        pred['images'] = x.permute(0, 1, 3, 4, 2)  # BNCHW->BNHWC（0-1范围）

        # 转 numpy 但保留原始 torch 供 GLB 导出使用
        elapsed = time.time() - t0
        logger.info(f"Pi3 推理完成，用时 {elapsed:.2f}s")
        return pred  # tensors（带batch维）

    # ---- 导出 ----
    def export_glb(self, pred: Dict[str, Any], output_path: Path, *, conf_thres: float = 50.0, show_cam: bool = True) -> None:
        """将 Pi3 预测结果导出为 GLB 文件。"""
        try:
            # 直接复用 Pi3/demo_gradio.py 中的 predictions_to_glb（不修改vendor，仅导入）
            from demo_gradio import predictions_to_glb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(f"无法从 Pi3/demo_gradio 导入 predictions_to_glb: {e}")

        # 将 torch.Tensor 转为 numpy 以用于 trimesh
        pred_np: Dict[str, Any] = {}
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy()
                # 去 batch 维
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

    # ---- 主流程 ----
    # 由基类统一流程调用
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
        """保存 Pi3 预测缓存（viewer 兼容）。"""
        image_ids: Optional[List[int]] = None
        image_paths: List[str] = []
        if image_names is not None:
            image_ids = self.extract_image_ids(image_names)
            if input_dir is not None:
                image_paths = [str(Path(input_dir) / n) for n in image_names]

        # 1) 保存 transforms.json（Pi3 仅包含缩放，无裁剪/填充）
        try:
            if image_paths:
                from utils.transforms import build_transforms
                transforms = build_transforms(image_paths, model_type="pi3", pixel_limit=255000)
                # 取统一目标尺寸
                first_info = transforms[0].get_transform_info()
                tw, th = int(first_info["target_size"][0]), int(first_info["target_size"][1])
                vggt_cache_dir = out_dir / "vggt_cache"
                vggt_cache_dir.mkdir(parents=True, exist_ok=True)
                tf_path = vggt_cache_dir / "transforms.json"

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
        except Exception as e:
            logger.warning(f"保存 Pi3 transforms.json 失败（不影响GLB/NPZ）：{e}")

        # 2) 保存 predictions.npz
        _save_predictions_npz(predictions, images, out_dir, image_ids=image_ids)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Pi3 3D重建GLB导出器")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录（包含图像文件）")
    parser.add_argument("--output_file", type=str, required=True, help="输出GLB文件路径")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="计算设备")
    parser.add_argument("--model_path", type=str, default=None, help="Pi3权重路径（可选）")
    parser.add_argument("--conf_thres", type=float, default=50.0, help="置信度阈值(0-100)")
    parser.add_argument("--no_show_cam", action="store_true", help="GLB中不显示相机")
    args = parser.parse_args()

    recon = PI33DReconstructor(device=args.device, model_path=args.model_path)
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
