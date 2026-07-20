"""
SKU匹配系统工具模块

包含SKU匹配系统的所有核心功能模块
"""

from .config import (
    SKUMatchingConfig,
    build_matching_config_from_yaml,
    extract_main_settings,
    extract_reconstruction_settings,
    load_yaml_config,
)
from .data_utils import (
    extract_bboxes_from_detections,
    load_detections,
    save_correspondences_json,
)
from .geometry_3d import (
    apply_uniqueness_constraint,
    find_best_matching_bbox_with_3d_validation,
    project_3d_to_2d,
    sample_3d_points_from_non_overlap_regions,
    sample_3d_points_from_non_overlap_regions_batch,
    transform_camera_to_world,
    transform_world_to_camera,
)
from .matching_algorithms import find_object_correspondences
from .point_utils import generate_points_from_bboxes
from .sku_matching_system import SKUMatchingSystem
from .transforms import ImageTransformBase, ResizeImageTransform, build_transforms

try:
    from .visualization import save_visualization_summary, visualize_results

    _viz_available = True
except ImportError:
    _viz_available = False

# 基础导出
__all__ = [
    "SKUMatchingConfig",
    "load_detections",
    "extract_bboxes_from_detections",
    "save_correspondences_json",
    "ResizeImageTransform",
    "build_transforms",
    "ImageTransformBase",
    "generate_points_from_bboxes",
    "sample_3d_points_from_non_overlap_regions",
    "sample_3d_points_from_non_overlap_regions_batch",
    "project_3d_to_2d",
    "find_best_matching_bbox_with_3d_validation",
    "apply_uniqueness_constraint",
    "transform_world_to_camera",
    "transform_camera_to_world",
    "find_object_correspondences",
    "SKUMatchingSystem",
]

if _viz_available:
    __all__.extend(["visualize_results", "save_visualization_summary"])


def check_dependencies():
    """检查依赖模块可用性"""
    return {"visualization": _viz_available}


# Export config helpers
__all__.extend(
    [
        "load_yaml_config",
        "build_matching_config_from_yaml",
        "extract_main_settings",
        "extract_reconstruction_settings",
    ]
)
