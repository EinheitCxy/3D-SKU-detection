from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.global_id_mapper import GlobalIDMapper

from .datasource import (
    load_points_from_glb,
    load_points_from_npz,
    normalize_colors_to_uint8,
)
from .id_assign import assign_global_ids_to_points
from .indexer import build_global_object_index
from .paths import resolve_predictions_npz
from .types import ViewerArtifacts, ViewerConfig

logger = logging.getLogger(__name__)


class CacheValidator:
    CACHE_VERSION = "2.0"

    @staticmethod
    def compute_file_hash(file_path: Path, method: str = "mtime") -> str:
        if method == "mtime":
            stat = file_path.stat()
            return f"{stat.st_mtime:.6f}_{stat.st_size}"
        raise ValueError("Only mtime method supported in viewer.cache")

    @staticmethod
    def create_metadata(
        config: ViewerConfig, statistics: Dict[str, Any]
    ) -> Dict[str, Any]:
        source_path = _resolve_point_source_path(config)
        return {
            "version": CacheValidator.CACHE_VERSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "config": {
                "downsample_ratio": config.downsample_ratio,
                "points_source": config.points_source,
            },
            "source_files": {
                "global_mapping.json": {
                    "path": str(config.global_mapping),
                    "hash": CacheValidator.compute_file_hash(config.global_mapping),
                    "mtime": datetime.fromtimestamp(
                        config.global_mapping.stat().st_mtime
                    ).isoformat(),
                },
                "point_source": {
                    "path": str(source_path),
                    "hash": CacheValidator.compute_file_hash(source_path),
                    "mtime": datetime.fromtimestamp(
                        source_path.stat().st_mtime
                    ).isoformat(),
                },
            },
            "statistics": statistics,
        }

    @staticmethod
    def is_cache_valid(config: ViewerConfig) -> bool:
        meta = config.cache_dir / "cache_metadata.json"
        if not meta.exists():
            return False
        try:
            with meta.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        if data.get("version") != CacheValidator.CACHE_VERSION:
            return False
        cfg = data.get("config", {})
        if cfg.get("downsample_ratio") != config.downsample_ratio:
            return False
        if cfg.get("points_source") != config.points_source:
            return False
        try:
            gm_hash = data["source_files"]["global_mapping.json"]["hash"]
            src_hash = data["source_files"]["point_source"]["hash"]
            if gm_hash != CacheValidator.compute_file_hash(config.global_mapping):
                return False
            source_path = _resolve_point_source_path(config)
            if src_hash != CacheValidator.compute_file_hash(source_path):
                return False
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, KeyError):
            return False
        return True


def _downsample(
    points: np.ndarray, colors: Optional[np.ndarray], ratio: float, seed: int = 42
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if ratio >= 1.0:
        return points, colors
    n = max(1, int(len(points) * ratio))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=n, replace=False)
    pts = points[idx]
    cols = colors[idx] if colors is not None else None
    return pts, cols


def _resolve_point_source_path(config: ViewerConfig) -> Path:
    """解析实际用于加载点云的源文件路径（GLB 或 da3_cache/predictions.npz）。

    优先级：
    1. config.reconstruction 显式指向 .npz -> 该 npz；
    2. points_source == 'predictions' -> da3_cache/predictions.npz（base 为
       reconstruction.parent 或 cache_dir.parent）；
    3. config.reconstruction 指向存在的 GLB -> 该 GLB；
    4. 回退：cache_dir.parent 下的 da3_cache/predictions.npz。
    """
    recon = config.reconstruction
    if recon is not None and recon.suffix == ".npz" and recon.exists():
        return recon
    if config.points_source == "predictions":
        base = recon.parent if recon is not None else config.cache_dir.parent
        return resolve_predictions_npz(base)
    if recon is not None and recon.exists():
        return recon
    # da3 无 GLB：回退到 cache_dir.parent 下的 da3_cache
    return resolve_predictions_npz(config.cache_dir.parent)


def build_cache(config: ViewerConfig) -> ViewerArtifacts:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    if (not config.force_rebuild) and CacheValidator.is_cache_valid(config):
        logger.info("使用现有缓存")
        return ViewerArtifacts(
            pcd_cache_path=config.cache_dir / "pcd_gid.npz",
            index_cache_path=config.cache_dir / "global_object_index.json",
            metadata_path=config.cache_dir / "cache_metadata.json",
        )

    logger.info("开始构建缓存…")
    t0 = time.time()

    # 1) 加载点云：da3 默认走 da3_cache/predictions.npz，GLB 可选
    source_path = _resolve_point_source_path(config)
    if source_path.suffix == ".npz":
        points, colors = load_points_from_npz(source_path)
    else:
        try:
            points, colors = load_points_from_glb(source_path)
        except (FileNotFoundError, ValueError, KeyError) as e:
            logger.warning(f"加载GLB失败，回退da3_cache: {e}")
            pred_path = resolve_predictions_npz(config.cache_dir.parent)
            points, colors = load_points_from_npz(pred_path)

    # 2) 下采样
    points, colors = _downsample(points, colors, config.downsample_ratio)
    colors = normalize_colors_to_uint8(colors)

    # 3) 分配 gid/conf/frame
    gids, confs, frames = assign_global_ids_to_points(
        points=points,
        reconstruction_path=config.reconstruction,
        global_mapping_path=config.global_mapping,
        image_dir=config.image_dir,
        detection_dir=config.detection_dir,
    )

    # 4) 保存 pcd_gid.npz
    pcd_cache_path = config.cache_dir / "pcd_gid.npz"
    np.savez_compressed(
        pcd_cache_path,
        points=points,
        colors=colors,
        global_ids=gids,
        confidences=confs,
        frame_indices=frames,
    )

    # 5) 保存索引
    mapper = GlobalIDMapper(str(config.global_mapping))
    index = build_global_object_index(mapper)
    index_cache_path = config.cache_dir / "global_object_index.json"
    with index_cache_path.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # 6) 保存元数据
    md = CacheValidator.create_metadata(
        config,
        statistics={
            "total_points": int(len(points)),
            "points_with_gid": int(np.sum(gids >= 0)),
            "unique_global_ids": int(len(np.unique(gids[gids >= 0]))),
            "build_time_seconds": float(time.time() - t0),
        },
    )
    metadata_path = config.cache_dir / "cache_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(md, f, ensure_ascii=False, indent=2)

    logger.info("缓存构建完成")
    return ViewerArtifacts(pcd_cache_path, index_cache_path, metadata_path)
