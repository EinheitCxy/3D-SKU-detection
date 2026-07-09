"""
Viewer launcher module under modules/.

Provides thin wrappers to build viewer cache and start the Viser UI,
so that main.py can simply import and call these APIs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from viewer.types import ViewerConfig, ViewerRuntimeConfig, ViewerArtifacts
from viewer.cache import build_cache
from viewer.runtime import ViserViewer


def build_viewer_cache(
    *,
    global_mapping: str,
    reconstruction: str,
    image_dir: str,
    detection_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    downsample_ratio: float = 1.0,
    points_source: str = "glb",
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """Build viewer cache and return artifact paths.

    Returns dict with keys: pcd_cache_path, index_cache_path, metadata_path, cache_dir
    """
    out_dir = Path(cache_dir) if cache_dir else (Path.cwd() / "viewer_cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ViewerConfig(
        global_mapping=Path(global_mapping),
        reconstruction=Path(reconstruction),
        image_dir=Path(image_dir),
        detection_dir=(Path(detection_dir) if detection_dir else None),
        cache_dir=out_dir,
        downsample_ratio=downsample_ratio,
        points_source=points_source,
        force_rebuild=force_rebuild,
    )
    artifacts = build_cache(cfg)
    return {
        "pcd_cache_path": str(artifacts.pcd_cache_path),
        "index_cache_path": str(artifacts.index_cache_path),
        "metadata_path": str(artifacts.metadata_path),
        "cache_dir": str(out_dir),
    }


def start_viewer(
    *,
    pcd_cache_path: str,
    index_cache_path: str,
    image_dir: str,
    global_mapping: str,
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Start interactive viewer from prepared artifacts."""
    import threading
    import time as _time
    import webbrowser
    artifacts = ViewerArtifacts(
        pcd_cache_path=Path(pcd_cache_path),
        index_cache_path=Path(index_cache_path),
        metadata_path=Path(index_cache_path).parent / "cache_metadata.json",
    )
    runtime = ViewerRuntimeConfig(port=int(port))
    viewer = ViserViewer(artifacts, runtime, image_dir=Path(image_dir), global_mapping=Path(global_mapping))
    if open_browser:
        def _open():
            _time.sleep(0.5)
            try:
                webbrowser.open(f"http://127.0.0.1:{int(port)}")
            except (OSError, RuntimeError):
                pass
        threading.Thread(target=_open, daemon=True).start()
    viewer.start()


def run_viewer(
    *,
    global_mapping: str,
    reconstruction: str,
    image_dir: str,
    detection_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    downsample_ratio: float = 1.0,
    points_source: str = "glb",
    port: int = 8080,
    force_rebuild: bool = False,
    open_browser: bool = True,
) -> None:
    """Build and run the viewer in one call."""
    res = build_viewer_cache(
        global_mapping=global_mapping,
        reconstruction=reconstruction,
        image_dir=image_dir,
        detection_dir=detection_dir,
        cache_dir=cache_dir,
        downsample_ratio=downsample_ratio,
        points_source=points_source,
        force_rebuild=force_rebuild,
    )
    start_viewer(
        pcd_cache_path=res["pcd_cache_path"],
        index_cache_path=res["index_cache_path"],
        image_dir=image_dir,
        global_mapping=global_mapping,
        port=port,
        open_browser=open_browser,
    )
