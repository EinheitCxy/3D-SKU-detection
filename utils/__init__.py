"""
SKU匹配系统工具模块

包含SKU匹配系统的所有核心功能模块
"""

# 统一配置 VGGT 路径，其他模块无需自行注入 sys.path
import sys
from pathlib import Path


def _resolve_vggt_root() -> Path:
    """定位 vggt-main 目录（相对本包位置进行多级回溯）。"""
    here = Path(__file__).resolve()
    candidates = [here.parents[1] / "vggt-main"]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


VGGT_ROOT = _resolve_vggt_root()
if VGGT_ROOT.exists():
    vg = str(VGGT_ROOT)
    if vg not in sys.path:
        sys.path.insert(0, vg)

# 基础模块 - 不依赖VGGT
from .config import (
    SKUMatchingConfig,
    load_yaml_config,
    build_matching_config_from_yaml,
    extract_main_settings,
    extract_reconstruction_settings,
)
from .data_utils import load_detections, extract_bboxes_from_detections, save_correspondences_json
from .transforms import VGGTImageTransform, Pi3ImageTransform, ImageTransformBase, build_transforms
from .point_utils import generate_points_from_bboxes
from .geometry_3d import (
    sample_3d_points_from_non_overlap_regions,
    sample_3d_points_from_non_overlap_regions_batch,
    project_3d_to_2d,
    find_best_matching_bbox_with_3d_validation,
    apply_uniqueness_constraint,
    transform_world_to_camera,
    transform_camera_to_world,
)

# DA3/Pi3 cache backends share these modules but do not require retired VGGT
# source at import time. VGGT-only imports remain inside their backend branches.
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
    'load_detections',
    'extract_bboxes_from_detections',
    'save_correspondences_json',
    'VGGTImageTransform',
    'build_transforms',
    'ImageTransformBase',
    'Pi3ImageTransform',
    'generate_points_from_bboxes',
    'sample_3d_points_from_non_overlap_regions',
    'sample_3d_points_from_non_overlap_regions_batch',
    'project_3d_to_2d',
    'find_best_matching_bbox_with_3d_validation',
    'apply_uniqueness_constraint',
    'transform_world_to_camera',
    'transform_camera_to_world',
]

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
        'vggt_modules': VGGT_ROOT.exists(),
        'visualization': _viz_available
    }


def get_vggt_root() -> Path:
    """返回 vggt-main 根目录（若未找到，返回预期路径）。"""
    return VGGT_ROOT

# Export config helpers
__all__.extend([
    'load_yaml_config',
    'build_matching_config_from_yaml',
    'extract_main_settings',
    'extract_reconstruction_settings',
])
