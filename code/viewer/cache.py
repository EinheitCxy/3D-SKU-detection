from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.global_id_mapper import GlobalIDMapper
from .types import ViewerConfig, ViewerArtifacts
from .datasource import load_points_from_glb, load_points_from_npz, normalize_colors_to_uint8
from .id_assign import assign_global_ids_to_points
from .indexer import build_global_object_index

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
    def create_metadata(config: ViewerConfig, statistics: Dict[str, Any]) -> Dict[str, Any]:
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
                    "mtime": datetime.fromtimestamp(config.global_mapping.stat().st_mtime).isoformat(),
                },
                "reconstruction.glb": {
                    "path": str(config.reconstruction),
                    "hash": CacheValidator.compute_file_hash(config.reconstruction),
                    "mtime": datetime.fromtimestamp(config.reconstruction.stat().st_mtime).isoformat(),
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
            rc_hash = data["source_files"]["reconstruction.glb"]["hash"]
            if gm_hash != CacheValidator.compute_file_hash(config.global_mapping):
                return False
            if rc_hash != CacheValidator.compute_file_hash(config.reconstruction):
                return False
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        return True


def _downsample(points: np.ndarray, colors: Optional[np.ndarray], ratio: float, seed: int = 42) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if ratio >= 1.0:
        return points, colors
    n = max(1, int(len(points) * ratio))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=n, replace=False)
    pts = points[idx]
    cols = colors[idx] if colors is not None else None
    return pts, cols


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

    # 1) 加载点云
    if config.points_source == "predictions":
        pred_path = config.reconstruction.parent / "vggt_cache" / "predictions.npz"
        try:
            points, colors = load_points_from_npz(pred_path)
        except (FileNotFoundError, ValueError, KeyError) as e:
            logger.warning(f"加载NPZ失败，回退GLB: {e}")
            points, colors = load_points_from_glb(config.reconstruction)
    else:
        points, colors = load_points_from_glb(config.reconstruction)

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

