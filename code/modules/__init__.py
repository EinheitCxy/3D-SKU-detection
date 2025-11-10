"""
Modules package for 3D SKU Detection system

This package contains core modules that were moved from the parent directory.
"""

# Import main entry points from modules
from .inference import main as inference_main
from .draw_detection_boxes import main as draw_detection_boxes_main
from .improved_sku_analyzer import ImprovedSKUCountAnalyzer
from .deduplicate_detections import DatasetPaths, resolve_dataset_paths, deduplicate_sequence
from .viewer_runner import run_viewer, build_viewer_cache
from .reconstructor_base import ReconstructorBase
from .vggt_3d_reconstructor import VGGT3DReconstructor
from .pi3_3d_reconstructor import PI33DReconstructor

__all__ = [
    'inference_main',
    'draw_detection_boxes_main',
    'ImprovedSKUCountAnalyzer',
    'DatasetPaths',
    'resolve_dataset_paths',
    'deduplicate_sequence',
    'run_viewer',
    'build_viewer_cache',
    'ReconstructorBase',
    'VGGT3DReconstructor',
    'PI33DReconstructor',
]
