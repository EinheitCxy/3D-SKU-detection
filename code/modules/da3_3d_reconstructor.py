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


# ---- DA3 重建器 ----

@register_reconstructor("da3")
class DA33DReconstructor(ReconstructorBase):
    """Depth-Anything-3 3D重建器（与 Pi3 接口兼容）。

    通过 subprocess 调用 Depth-Anything-3/.venv 运行 DA3 推理（DA3 依赖 numpy<2 与
    code/ 的 numpy>=2 冲突，故隔离在 DA3 自带 venv 中），结果缓存到 da3_cache/predictions.npz，
    格式与 pi3_cache 完全一致，无需修改下游 SKU 匹配代码。
    """

    # HuggingFace 默认模型；可通过 model_path 覆盖
    DEFAULT_HF_REPO = "depth-anything/DA3NESTED-GIANT-LARGE"
    # DA3 自带 venv（含 numpy<2 + omegaconf/addict/e3nn 等 DA3 依赖）
    DEFAULT_DA3_VENV_PYTHON = (
        REPO_ROOT / "Depth-Anything-3" / ".venv" / "bin" / "python"
    )
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
        # subprocess 模式：不在父进程加载 DA3（依赖隔离在 DA3 venv），仅校验 venv 与 runner 可用
        if not self.da3_venv_python.is_file():
            raise FileNotFoundError(
                f"DA3 Python 不存在: {self.da3_venv_python}。"
                "请设置 DA3_VENV_PYTHON 指向已有的 Depth-Anything-3/.venv/bin/python。"
            )
        if not self.DA3_RUNNER.exists():
            raise FileNotFoundError(f"DA3 runner 脚本不存在: {self.DA3_RUNNER}")

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
            str(p) for p in Path(input_dir).iterdir()
            if p.suffix.lower() in exts
        )
        if not paths:
            raise ValueError(f"目录中未找到图片: {input_dir}")
        logger.info(f"DA3 加载 {len(paths)} 张图片")
        return paths

    # ---- 推理（subprocess 调 da3_runner.py，在 DA3 venv 中跑） ----

    def run_inference(self, image_paths: List[str]) -> Dict[str, Any]:
        """subprocess 调 DA3 venv 运行 da3_runner.py，返回与 Pi3 缓存兼容的 pred 字典。

        子进程直接写出 da3_cache/predictions.npz（含正确 shape），父进程读回返回。
        """
        import subprocess

        if not image_paths:
            raise ValueError("run_inference: image_paths 为空")

        # 缓存路径：约定 Output/<dataset>/da3_cache/predictions.npz
        # 由基类模板传入的 out_dir 决定；run_inference 无 out_dir，故用临时目录，
        # save_predictions_cache 会把它移到最终 da3_cache/。这里写到 tmp。
        import tempfile

        tmp_npz = Path(tempfile.mkdtemp(prefix="da3_inf_")) / "predictions.npz"
        input_dir = str(Path(image_paths[0]).parent)

        cmd = [
            str(self.da3_venv_python),
            str(self.DA3_RUNNER),
            "--input_dir", input_dir,
            "--output_npz", str(tmp_npz),
            "--model_path", self.model_path or self.DEFAULT_HF_REPO,
            "--device", self.device,
        ]
        logger.info(f"DA3 subprocess: {' '.join(cmd)}")
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"DA3 runner 失败 (exit={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        if proc.stdout:
            logger.info(proc.stdout.strip())
        logger.info(f"DA3 推理完成，用时 {time.time() - t0:.2f}s")

        data = np.load(tmp_npz, allow_pickle=True)
        pred: Dict[str, Any] = {k: data[k] for k in data.files}
        pred["_npz_path"] = tmp_npz  # 供 save_predictions_cache 直接复用
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
        logger.info(f"DA3 export_glb: subprocess 模式跳过 GLB（期望路径 {output_path}）")

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

        for key in (
            "depth",
            "depth_conf",
            "world_points",
            "world_points_conf",
            "images",
            "source_to_processed_affine",
        ):
            val = predictions.get(key)
            if val is not None and isinstance(val, np.ndarray):
                save_kwargs[key] = val.astype(np.float32 if key != "images" else np.uint8, copy=False)

        source_image_sizes = predictions.get("source_image_sizes")
        if source_image_sizes is not None and isinstance(source_image_sizes, np.ndarray):
            save_kwargs["source_image_sizes"] = source_image_sizes.astype(
                np.int32, copy=False
            )

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

        # source_model（优先从 da3_runner 写入的 pred 透传，否则用默认）
        src = predictions.get("source_model")
        if src is not None and isinstance(src, np.ndarray):
            save_kwargs["source_model"] = src
        else:
            save_kwargs["source_model"] = np.array(["depth-anything/DA3NESTED-GIANT-LARGE"], dtype=object)

        # 帧对齐索引（对齐 pi3 schema，从 da3_runner 透传）
        for fa_key in ("frame_alignment_sorted_indices", "frame_alignment_map_keys", "frame_alignment_map_values"):
            fa_val = predictions.get(fa_key)
            if fa_val is not None and isinstance(fa_val, np.ndarray):
                save_kwargs[fa_key] = fa_val

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
