from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_points_from_glb(glb_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required. Install: pip install trimesh")

    logger.info(f"Loading GLB file: {glb_path}")
    scene = trimesh.load(str(glb_path))

    points_list = []
    colors_list = []

    if hasattr(scene, "geometry"):
        for _, geom in scene.geometry.items():
            if hasattr(geom, "vertices"):
                points_list.append(geom.vertices)
                if hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
                    colors_list.append(geom.visual.vertex_colors[:, :3] / 255.0)
                else:
                    colors_list.append(
                        np.ones((len(geom.vertices), 3), dtype=np.float32) * 0.8
                    )
    elif hasattr(scene, "vertices"):
        points_list.append(scene.vertices)
        if hasattr(scene, "visual") and hasattr(scene.visual, "vertex_colors"):
            colors_list.append(scene.visual.vertex_colors[:, :3] / 255.0)
        else:
            colors_list.append(
                np.ones((len(scene.vertices), 3), dtype=np.float32) * 0.8
            )

    if not points_list:
        raise ValueError("No point cloud data found in GLB file")

    points = np.vstack(points_list).astype(np.float32, copy=False)
    colors = np.vstack(colors_list).astype(np.float32, copy=False)
    logger.info(f"Loaded {len(points)} points from GLB")
    return points, colors


def load_points_from_npz(pred_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions.npz not found: {pred_path}")
    logger.info(f"Loading points from predictions: {pred_path}")
    data = np.load(pred_path, allow_pickle=True)
    wp = data["world_points"]  # (S,H,W,3) or (H,W,3)
    if wp.ndim == 3:
        wp = wp[np.newaxis, ...]
    S, H, W, _ = wp.shape
    points = wp.reshape(-1, 3).astype(np.float32, copy=False)

    colors: Optional[np.ndarray]
    if "images" in data:
        imgs = data["images"]
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            imgs = imgs.transpose(0, 2, 3, 1)
        colors = imgs.reshape(-1, 3)
        if colors.dtype != np.uint8:
            maxv = float(colors.max()) if colors.size > 0 else 1.0
            scale = 255.0 if maxv <= 1.0 else 1.0
            colors = np.clip(colors * scale, 0, 255).astype(np.uint8)
    else:
        colors = (np.ones((points.shape[0], 3), dtype=np.float32) * 204).astype(
            np.uint8
        )

    logger.info(f"Loaded {len(points)} points from predictions (S={S}, H={H}, W={W})")
    return points, colors


def normalize_colors_to_uint8(colors: np.ndarray) -> np.ndarray:
    try:
        if colors.dtype == np.uint8:
            return colors
        col = colors
        maxv = float(np.nanmax(col)) if np.size(col) > 0 else 1.0
        if maxv <= 1.0:
            col = col * 255.0
        return np.clip(col, 0, 255).astype(np.uint8)
    except (ValueError, AttributeError, TypeError):
        return colors.astype(np.uint8, copy=False)
