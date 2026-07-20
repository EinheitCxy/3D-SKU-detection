"""路径解析辅助：定位 da3 重建缓存目录。

da3 是当前唯一重建后端，缓存为 `<save_root>/<dataset>/da3_cache/predictions.npz`
（含 depth/world_points/extrinsic(N,3,4 w2c)/intrinsic/image_ids/source_model 等；
无 transforms.json、不产 GLB）。本模块容忍调用方传入数据集根目录或 da3_cache 本身。
"""

from __future__ import annotations

from pathlib import Path


def _resolve_da3_cache_dir(base: Path) -> Path:
    """容忍 base 指向数据集根或 da3_cache 本身，返回 da3_cache 目录。

    - base 本身就是 da3_cache -> 原样返回；
    - base/da3_cache 存在 -> 返回该子目录；
    - 否则返回 base（由调用方检查 predictions.npz 是否存在）。
    """
    if base.name == "da3_cache":
        return base
    cand = base / "da3_cache"
    return cand if cand.exists() else base


def resolve_predictions_npz(base: Path) -> Path:
    """返回 da3_cache/predictions.npz 路径（base 为数据集根或 da3_cache 本身）。"""
    return _resolve_da3_cache_dir(base) / "predictions.npz"
