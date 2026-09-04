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
  uv run python -m src.da3_3d_reconstructor --input_dir <images_dir> --output_file <out.glb>
"""

from __future__ import annotations

import logging
import os
import sys
import time
import zipfile
import gc
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
THIS_DIR = Path(__file__).resolve().parent  # src/
REPO_ROOT = THIS_DIR.parent  # 3D_Recognization/
DA3_SRC = REPO_ROOT / "Depth-Anything-3" / "src"

if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .reconstructor_base import ReconstructorBase, register_reconstructor  # noqa: E402

# ---- DA3 重建器 ----


@register_reconstructor("da3")
class DA33DReconstructor(ReconstructorBase):
    """Depth-Anything-3 3D重建器（与 Pi3 接口兼容）。

    通过 subprocess 调用根 .venv 运行 DA3 推理。DA3/SAM3 源码保留在仓库中，由子进程
    从仓库路径导入；结果缓存到 da3_cache/predictions.npz。
    """

    # HuggingFace 默认模型；可通过 model_path 覆盖
    DEFAULT_HF_REPO = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    # 统一宿主环境包含 DA3/SAM3 的运行时依赖。
    DEFAULT_DA3_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
    DA3_RUNNER = THIS_DIR / "da3_runner.py"

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        model_name: str = "da3nested-giant-large",
    ) -> None:
        super().__init__(device=device, model_path=model_path, backend_name="da3")
        self.model_name = model_name
        self.da3_venv_python = Path(
            os.environ.get("DA3_VENV_PYTHON", self.DEFAULT_DA3_VENV_PYTHON)
        ).expanduser()
        # subprocess 模式：父进程不加载 DA3，仅校验统一环境与 runner 可用。
        if not self.da3_venv_python.is_file():
            raise FileNotFoundError(
                f"DA3 Python 不存在: {self.da3_venv_python}。"
                "请设置 DA3_VENV_PYTHON 指向已有的统一宿主环境 Python。"
            )
        if not self.DA3_RUNNER.exists():
            raise FileNotFoundError(f"DA3 runner 脚本不存在: {self.DA3_RUNNER}")
        self._active_cache_dir: Optional[Path] = None

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
        """Run DA3 reconstruction while treating cache publication as mandatory."""
        if not save_predictions:
            raise ValueError("DA3 reconstruction requires save_predictions=True")
        self._active_cache_dir = Path(output_path).parent
        try:
            if self.model is None:
                self.load_model()
            images = self.load_images(input_dir)
            predictions = self.run_inference(images)
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if save_predictions:
                self.save_predictions_cache(
                    predictions, images, out_path.parent, input_dir=input_dir, **kwargs
                )
            self.export_glb(
                predictions,
                out_path,
                conf_thres=conf_thres,
                show_cam=show_cam,
                **kwargs,
            )
            return out_path
        finally:
            self._active_cache_dir = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # ---- 模型加载（subprocess 模式下为 no-op，真实加载在子进程） ----

    def load_model(self) -> None:
        """subprocess 模式：父进程不加载模型，仅标记就绪。"""
        self.model = True  # 占位，满足基类模板的 self.model is None 检查
        logger.info(f"DA3 subprocess 模式就绪: python={self.da3_venv_python}")

    # ---- 图像加载 ----

    def load_images(self, input_dir: str) -> List[str]:
        """返回目录下排好序的图片路径列表（DA3 接受路径列表）。"""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(
            str(p) for p in Path(input_dir).iterdir() if p.suffix.lower() in exts
        )
        if not paths:
            raise ValueError(f"目录中未找到图片: {input_dir}")
        logger.info(f"DA3 加载 {len(paths)} 张图片")
        return paths

    # ---- 推理（subprocess 调 da3_runner.py，在 DA3 venv 中跑） ----

    def run_inference(self, image_paths: List[str]) -> Dict[str, Any]:
        """subprocess 调统一宿主环境运行 da3_runner.py，返回与 Pi3 缓存兼容的 pred 字典。

        子进程直接写出最终 cache 同目录的 partial；父进程只验证后原子发布。
        """
        import subprocess

        if not image_paths:
            raise ValueError("run_inference: image_paths 为空")

        if self._active_cache_dir is None:
            raise RuntimeError(
                "DA3 inference requires reconstruct_from_directory cache context"
            )
        cache_dir = self._active_cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "predictions.npz"
        tmp_npz = cache_path.with_name(f"{cache_path.name}.partial")
        tmp_npz.unlink(missing_ok=True)
        input_dir = str(Path(image_paths[0]).parent)

        cmd = [
            str(self.da3_venv_python),
            str(self.DA3_RUNNER),
            "--input_dir",
            input_dir,
            "--output_npz",
            str(tmp_npz),
            "--model_path",
            self.model_path or self.DEFAULT_HF_REPO,
            "--device",
            self.device,
        ]
        logger.info(f"DA3 subprocess: {' '.join(cmd)}")
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"DA3 runner 失败 (exit={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
        except Exception:
            tmp_npz.unlink(missing_ok=True)
            raise
        if proc.stdout:
            logger.info(proc.stdout.strip())
        logger.info(f"DA3 推理完成，用时 {time.time() - t0:.2f}s")

        pred: Dict[str, Any] = {"_npz_path": tmp_npz}
        pred["_image_paths"] = image_paths
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
        """subprocess 模式跳过 GLB（无 _prediction 对象；SKU matching 仅需 npz 缓存）。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"DA3 export_glb: subprocess 模式跳过 GLB（期望路径 {output_path}）"
        )

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
        """Validate and atomically publish DA3 runner's complete schema-v3 cache."""
        cache_dir = out_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "predictions.npz"
        partial_path = Path(predictions.get("_npz_path", ""))
        expected_partial = cache_path.with_name(f"{cache_path.name}.partial")
        if partial_path != expected_partial:
            raise ValueError(
                "DA3 cache publication requires the sibling predictions.npz.partial"
            )
        try:
            _validate_da3_runner_cache(partial_path)
            os.replace(partial_path, cache_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise
        logger.info(f"原子发布 DA3 预测缓存: {cache_path}")


_MATCHER_ARRAY_KEYS = {
    "depth",
    "depth_conf",
    "world_points",
    "world_points_conf",
    "extrinsic",
    "intrinsic",
    "images",
    "image_ids",
    "source_image_sizes",
    "source_to_processed_affine",
}
_SCHEMA_V3_KEYS = {
    "cache_schema_version",
    "source_model",
    "source_image_sha256",
    "affine_convention",
    "preprocess_resolution",
    "preprocess_method",
    "is_metric",
    "scale_factor",
    "frame_alignment_sorted_indices",
    "frame_alignment_map_keys",
    "frame_alignment_map_values",
}
_EXACT_DTYPES = {
    "depth": np.dtype(np.float32),
    "depth_conf": np.dtype(np.float32),
    "world_points": np.dtype(np.float32),
    "world_points_conf": np.dtype(np.float32),
    "extrinsic": np.dtype(np.float32),
    "intrinsic": np.dtype(np.float32),
    "images": np.dtype(np.uint8),
    "image_ids": np.dtype(np.int32),
    "source_image_sizes": np.dtype(np.int32),
    "source_to_processed_affine": np.dtype(np.float32),
    "cache_schema_version": np.dtype(np.int32),
    "preprocess_resolution": np.dtype(np.int32),
    "is_metric": np.dtype(np.int32),
    "scale_factor": np.dtype(np.float32),
    "frame_alignment_sorted_indices": np.dtype(np.intp),
    "frame_alignment_map_keys": np.dtype(np.int32),
    "frame_alignment_map_values": np.dtype(np.int32),
}
_UNICODE_DTYPE_KEYS = {
    "source_model",
    "source_image_sha256",
    "affine_convention",
    "preprocess_method",
}


def _npy_header(archive: zipfile.ZipFile, key: str) -> tuple[tuple[int, ...], np.dtype]:
    with archive.open(f"{key}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        elif version == (3, 0):
            shape, _, dtype = np.lib.format.read_array_header_3_0(stream)
        else:
            raise ValueError(f"unsupported npy version for {key}: {version}")
    return tuple(shape), np.dtype(dtype)


def _validate_da3_runner_cache(path: Path) -> None:
    """Validate headers plus compact metric metadata without expanding dense arrays."""
    if not path.is_file():
        raise ValueError(f"DA3 runner partial cache is missing: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            keys = {
                Path(name).stem for name in archive.namelist() if name.endswith(".npy")
            }
            missing = sorted((_MATCHER_ARRAY_KEYS | _SCHEMA_V3_KEYS) - keys)
            if missing:
                raise ValueError(f"DA3 runner cache missing fields: {missing}")
            headers = {key: _npy_header(archive, key) for key in keys}
    except (OSError, zipfile.BadZipFile, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"invalid DA3 runner cache: {path}") from exc

    for key, expected_dtype in _EXACT_DTYPES.items():
        actual_dtype = headers[key][1]
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"DA3 runner cache {key} dtype {actual_dtype} != {expected_dtype}"
            )
    for key in _UNICODE_DTYPE_KEYS:
        if headers[key][1].kind != "U":
            raise ValueError(f"DA3 runner cache {key} dtype must be Unicode")

    depth_shape, _ = headers["depth"]
    if len(depth_shape) != 4 or depth_shape[-1] != 1:
        raise ValueError("DA3 runner cache depth must have shape (N,H,W,1)")
    n, height, width, _ = depth_shape
    if min(n, height, width) <= 0:
        raise ValueError("DA3 runner cache depth dimensions must be positive")
    expected_shapes = {
        "depth_conf": (n, height, width),
        "world_points": (n, height, width, 3),
        "world_points_conf": (n, height, width),
        "images": (n, height, width, 3),
        "image_ids": (n,),
        "source_image_sizes": (n, 2),
        "source_to_processed_affine": (n, 2, 3),
        "intrinsic": (n, 3, 3),
        "source_image_sha256": (n,),
        "frame_alignment_sorted_indices": (n,),
        "frame_alignment_map_keys": (n,),
        "frame_alignment_map_values": (n,),
    }
    for key, expected_shape in expected_shapes.items():
        if headers[key][0] != expected_shape:
            raise ValueError(
                f"DA3 runner cache {key} shape {headers[key][0]} != {expected_shape}"
            )
    extrinsic_shape = headers["extrinsic"][0]
    if extrinsic_shape not in {(n, 3, 4), (n, 4, 4)}:
        raise ValueError(
            "DA3 runner cache extrinsic must have shape (N,3,4) or (N,4,4)"
        )
    for key in (
        "cache_schema_version",
        "source_model",
        "affine_convention",
        "preprocess_resolution",
        "preprocess_method",
        "is_metric",
        "scale_factor",
    ):
        if headers[key][0] != ():
            raise ValueError(f"DA3 runner cache {key} must be scalar")

    with np.load(path, allow_pickle=False) as cache:
        if int(cache["cache_schema_version"]) != 3:
            raise ValueError("DA3 runner cache_schema_version must be 3")
        if int(cache["is_metric"]) != 1:
            raise ValueError("DA3 runner cache is_metric must be 1")


# ---- CLI 入口 ----


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="DA3 3D重建器 CLI")
    parser.add_argument("--input_dir", required=True, help="图片目录")
    parser.add_argument(
        "--output_file", default="reconstruction_da3.glb", help="输出 GLB 路径"
    )
    parser.add_argument(
        "--model_path", default=None, help="DA3 模型本地路径（默认从 HuggingFace 加载）"
    )
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
