"""
SKU匹配系统工具模块

包含SKU匹配系统的所有核心功能模块
"""

# 基础模块 - 不依赖VGGT
from .config import SKUMatchingConfig, DEFAULT_TRADITIONAL_CONFIG, DEFAULT_3D_PROJECTION_CONFIG
from .data_utils import load_detections, extract_bboxes_from_detections, save_correspondences_json
from .transforms import VGGTImageTransform, build_vggt_transforms
from .point_utils import generate_points_from_bboxes
from .geometry_3d import (
    sample_3d_points_from_bbox, 
    project_3d_to_2d,
    find_best_matching_bbox_with_3d_validation,
    apply_uniqueness_constraint,
    find_best_matching_bbox
)

# 延迟导入VGGT相关模块
def _import_vggt_modules():
    """延迟导入依赖VGGT的模块"""
    try:
        from .matching_algorithms import find_object_correspondences, match_objects_by_correspondence
        from .sku_matching_system import SKUMatchingSystem
        return True
    except ImportError:
        return False

# 尝试导入VGGT模块
_vggt_available = _import_vggt_modules()

if _vggt_available:
    from .matching_algorithms import find_object_correspondences, match_objects_by_correspondence
    from .sku_matching_system import SKUMatchingSystem

try:
    from .visualization import visualize_results, save_visualization_summary
    _viz_available = True
except ImportError:
    _viz_available = False

# 基础导出
__all__ = [
    'SKUMatchingConfig',
    'DEFAULT_TRADITIONAL_CONFIG', 
    'DEFAULT_3D_PROJECTION_CONFIG',
    'load_detections',
    'extract_bboxes_from_detections',
    'save_correspondences_json',
    'VGGTImageTransform',
    'build_vggt_transforms',
    'generate_points_from_bboxes',
    'sample_3d_points_from_bbox',
    'project_3d_to_2d',
    'find_best_matching_bbox_with_3d_validation',
    'apply_uniqueness_constraint',
    'find_best_matching_bbox',
]

if _vggt_available:
    __all__.extend([
        'find_object_correspondences',
        'match_objects_by_correspondence',
        'SKUMatchingSystem'
    ])

if _viz_available:
    __all__.extend([
        'visualize_results',
        'save_visualization_summary'
    ])

def check_dependencies():
    """检查依赖模块可用性"""
    return {
        'vggt_modules': _vggt_available,
        'visualization': _viz_available
    }