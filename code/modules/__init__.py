"""
Modules package for 3D SKU Detection system

This package contains core modules that were moved from the parent directory.
"""

from .da3_3d_reconstructor import DA33DReconstructor
from .deduplicate_detections import (
    DatasetPaths,
    deduplicate_sequence,
    resolve_dataset_paths,
)
from .draw_detection_boxes import main as draw_detection_boxes_main
from .improved_sku_analyzer import ImprovedSKUCountAnalyzer

# Import main entry points from modules
from .inference import main as inference_main
from .reconstructor_base import (
    RECONSTRUCTOR_REGISTRY,
    ReconstructorBase,
    get_reconstructor,
    register_reconstructor,
)
from .viewer_runner import build_viewer_cache, run_viewer

__all__ = [
    "inference_main",
    "draw_detection_boxes_main",
    "ImprovedSKUCountAnalyzer",
    "DatasetPaths",
    "resolve_dataset_paths",
    "deduplicate_sequence",
    "run_viewer",
    "build_viewer_cache",
    "ReconstructorBase",
    "register_reconstructor",
    "get_reconstructor",
    "RECONSTRUCTOR_REGISTRY",
    "DA33DReconstructor",
]
