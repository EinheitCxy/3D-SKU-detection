from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ViewerConfig:
    global_mapping: Path
    reconstruction: Path
    image_dir: Path
    detection_dir: Optional[Path]
    cache_dir: Path
    downsample_ratio: float = 1.0
    points_source: str = "glb"  # 'glb' | 'predictions'
    force_rebuild: bool = False


@dataclass
class ViewerArtifacts:
    pcd_cache_path: Path
    index_cache_path: Path
    metadata_path: Path


@dataclass
class ViewerRuntimeConfig:
    port: int = 8080
    default_conf_percentile: float = 5.0
    default_point_size: float = 0.0006
    rotate_model_default: bool = False
    hide_unknown_default: bool = True
    show_cameras_default: bool = True
    show_pick_sphere_default: bool = False

