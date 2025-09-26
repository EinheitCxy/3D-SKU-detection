"""
Modules package for 3D SKU Detection system

This package contains core modules that were moved from the parent directory.
"""

# Import main entry points from modules
from .inference import main as inference_main
from .draw_detection_boxes import main as draw_detection_boxes_main  
from .improved_sku_analyzer import ImprovedSKUCountAnalyzer
from .deduplicate_detections import DatasetPaths, resolve_dataset_paths, deduplicate_sequence

__all__ = [
    'inference_main',
    'draw_detection_boxes_main', 
    'ImprovedSKUCountAnalyzer',
    'DatasetPaths',
    'resolve_dataset_paths', 
    'deduplicate_sequence'
]