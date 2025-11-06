import os
import torch
import logging
from dataclasses import dataclass
from typing import Optional, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def get_optimal_device_config(verbose: bool = True):
    """
    根据GPU架构自动选择最优数据类型。
    
    Args:
        verbose: 是否输出详细信息
    Returns:
        tuple: (device, dtype) 设备和数据类型
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        capability = torch.cuda.get_device_capability(0)
        
        # 根据GPU能力选择数据类型
        if capability[0] >= 8:  # A100, H100等
            dtype = torch.bfloat16
        else:  # V100, RTX 30系列等  
            dtype = torch.float16
            
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}, 使用{dtype}")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        if verbose:
            logger.info("使用CPU计算")
    
    return device, dtype


# ============ YAML config helpers ============


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file with PyYAML into a Python dict.

    Args:
        path: Path to the YAML file
    Returns:
        Dict with config content (empty dict if file is empty)
    Raises:
        FileNotFoundError: when path does not exist
        ImportError: when PyYAML is not available
        ValueError: when loaded content is not a mapping
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ImportError("PyYAML is required to load YAML configs. Install 'pyyaml'.") from e

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping (dict)")
    logger.info(f"Loaded YAML config: {p}")
    return data


def build_matching_config_from_yaml(path: str | Path, algorithm: str | None = None) -> "SKUMatchingConfig":
    """Build SKUMatchingConfig from a YAML file.

    Expected schema (example):
    matching:
      algorithm: point_tracking  # or '3d'/'3d_projection'
      device: cuda
      max_points_per_bbox: 50
      confidence_threshold: 0.5
      min_confident_points: 7
      correspondence_threshold: 0.5
      output_dir: ./Output/floor_display2
      save_json: true

    Args:
        path: YAML path
        algorithm: optional override of matching.algorithm
    Returns:
        SKUMatchingConfig
    """
    data = load_yaml_config(path)
    section = data.get("matching", data)

    algo = (algorithm or section.get("algorithm") or "point_tracking").lower()
    if algo in ("3d", "3d_projection", "projection", "3d-2d"):
        base = DEFAULT_3D_PROJECTION_CONFIG.copy()
    else:
        base = DEFAULT_POINT_TRACKING_CONFIG.copy()

    # Map allowed fields into dataclass
    cfg_dict: Dict[str, Any] = {
        **base,
    }
    for key in (
        "device",
        "max_points_per_bbox",
        "confidence_threshold",
        "min_confident_points",
        "correspondence_threshold",
        "seed",
        "save_json",
        "output_dir",
        "max_bboxes",
        "max_total_points",
        "min_bbox_area",
        # 3D-specific (safe to include; unused for PT)
        "depth_confidence_threshold",
        "point_3d_confidence_threshold",
        "min_depth",
        "max_depth",
        "max_3d_points_per_bbox",
        "projection_match_threshold",
        "max_3d_distance",
        "max_depth_difference",
        "min_depth_consistency",
        "enable_3d_projection_matching",
    ):
        if key in section:
            cfg_dict[key] = section[key]

    # Construct dataclass; device/dtype resolved in __post_init__
    return SKUMatchingConfig(**cfg_dict)  # type: ignore[arg-type]


def extract_main_settings(data_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    """Extract top-level main settings from YAML or dict.

    Returns a flat dict with keys commonly used by main.py: dataset, mode, algorithm,
    reference_idx, max_images, device, save_json, save_root.
    """
    data = load_yaml_config(data_or_path) if not isinstance(data_or_path, dict) else data_or_path
    main = data.get("main", data)
    out: Dict[str, Any] = {}
    for k in (
        "dataset",
        "mode",
        "algorithm",
        "reference_idx",
        "max_images",
        "device",
        "save_json",
        "save_root",
    ):
        if k in main:
            out[k] = main[k]
    return out


def extract_reconstruction_settings(data_or_path: Dict[str, Any] | str | Path) -> Dict[str, Any]:
    """Extract reconstruction settings section as a dict.

    Keys may include: device, conf_thres, output (filename), model_path, show_cam,
    mask_black_bg, mask_white_bg, mask_sky.
    """
    data = load_yaml_config(data_or_path) if not isinstance(data_or_path, dict) else data_or_path
    rec = data.get("reconstruction", data)
    out: Dict[str, Any] = {}
    for k in (
        "backend",
        "device",
        "conf_thres",
        "output",
        "model_path",
        "show_cam",
        "mask_black_bg",
        "mask_white_bg",
        "mask_sky",
    ):
        if k in rec:
            out[k] = rec[k]
    return out


@dataclass
class SKUMatchingConfig:
    """SKU匹配配置参数"""
    
    # === 核心检测参数 ===
    detection_confidence_threshold: float = 0.0  # 检测置信度阈值
    min_bbox_area: float = 10.0                  # 最小边界框面积
    max_bboxes: int = 500                        # 最大检测框数量
    
    # === 点采样参数 ===
    max_points_per_bbox: int = 70                # 每个2D检测框最大采样点数
    max_3d_points_per_bbox: int = 70             # 每个3D检测框最大采样点数
    max_total_points: int = 100000               # 全局最大采样点数上限
    
    # === 非重合区域采样参数 ===
    enable_non_overlap_sampling: bool = True     # 是否启用非重合区域采样
    overlap_threshold: float = 0.1               # 检出框重合阈值
    min_non_overlap_area: float = 20.0           # 非重合区域最小面积
    
    # === 匹配阈值参数 ===
    confidence_threshold: float = 0.5            # 点追踪置信度阈值
    min_confident_points: int = 7                # 最小置信点数
    correspondence_threshold: float = 0.5         # 2D对应关系阈值
    projection_match_threshold: float = 0.7       # 3D投影匹配阈值
    
    # === 3D重建参数 ===
    enable_3d_projection_matching: bool = False  # 是否启用3D投影匹配
    depth_confidence_threshold: float = 0.1      # 深度置信度阈值  
    point_3d_confidence_threshold: float = 0.1   # 3D点置信度阈值
    min_depth: float = 0.1                       # 最小深度值
    max_depth: float = 10.0                      # 最大深度值
    
    # === 3D几何验证参数 ===
    max_3d_distance: float = 1.0                 # 最大3D空间距离阈值(米)
    max_depth_difference: float = 2.0            # 最大深度差异容忍(米)  
    min_depth_consistency: float = 0.3           # 最小深度一致性阈值
    
    # === 系统配置参数 ===
    device: str = "auto"                         # 计算设备选择
    dtype: torch.dtype = None                    # 数据类型
    use_autocast: bool = True                    # 是否使用混合精度
    seed: Optional[int] = 42                     # 随机种子
    
    # === 输出配置参数 ===
    output_dir: str = ""                   # 输出目录
    save_json: bool = False                      # 是否保存JSON结果
    json_filename: str = "correspondences.json"  # JSON文件名
    
    def __post_init__(self):
        # 智能设备选择 - 替换原有简单逻辑
        if self.device == "auto" or self.dtype is None:
            optimal_device, optimal_dtype = get_optimal_device_config(verbose=True)
            if self.device == "auto":
                self.device = str(optimal_device)
            if self.dtype is None:
                self.dtype = optimal_dtype
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        self._validate_config()
    
    def _validate_config(self):
        if self.max_points_per_bbox <= 0:
            raise ValueError(f"max_points_per_bbox must be positive, got {self.max_points_per_bbox}")
        
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0,1], got {self.confidence_threshold}")
        
        if self.min_confident_points > self.max_points_per_bbox:
            raise ValueError(f"min_confident_points ({self.min_confident_points}) cannot exceed max_points_per_bbox ({self.max_points_per_bbox})")
        
        if self.min_bbox_area <= 0:
            raise ValueError(f"min_bbox_area must be positive, got {self.min_bbox_area}")
        
        # 3D相关参数验证
        if self.enable_3d_projection_matching:
            if self.min_depth >= self.max_depth:
                raise ValueError(f"min_depth ({self.min_depth}) must be less than max_depth ({self.max_depth})")
            
            if not (0.0 <= self.projection_match_threshold <= 1.0):
                raise ValueError(f"projection_match_threshold must be in [0,1], got {self.projection_match_threshold}")
            
            if self.max_3d_distance <= 0:
                raise ValueError(f"max_3d_distance must be positive, got {self.max_3d_distance}")
        
        # 非重合区域采样参数验证
        if self.overlap_threshold < 0 or self.overlap_threshold > 1:
            raise ValueError(f"overlap_threshold must be in [0,1], got {self.overlap_threshold}")
        
        if self.min_non_overlap_area <= 0:
            raise ValueError(f"min_non_overlap_area must be positive, got {self.min_non_overlap_area}")
        
        # 性能相关验证
        if self.max_total_points <= 0:
            raise ValueError(f"max_total_points must be positive, got {self.max_total_points}")
        
        if self.max_total_points < self.max_points_per_bbox:
            import warnings
            warnings.warn(f"max_total_points ({self.max_total_points}) is less than max_points_per_bbox ({self.max_points_per_bbox}), may limit performance")
    
    def get_algorithm_name(self) -> str:
        return "3D-2D Projection" if self.enable_3d_projection_matching else "Point Tracking"
    
    def to_dict(self) -> dict:
        """将配置转为字典格式"""
        return {
            'algorithm': self.get_algorithm_name(),
            'max_points_per_bbox': self.max_points_per_bbox,
            'confidence_threshold': self.confidence_threshold,
            'min_confident_points': self.min_confident_points,
            'min_bbox_area': self.min_bbox_area,
            'output_dir': self.output_dir,
            'enable_3d_projection_matching': self.enable_3d_projection_matching,
            'device': self.device,
            'seed': self.seed
        }


DEFAULT_POINT_TRACKING_CONFIG = {
    "max_points_per_bbox": 80,
    "max_bboxes": 200,
    "max_total_points": 20000,
    "confidence_threshold": 0.5,
    "min_confident_points": 7,
    "min_bbox_area": 10.0,
    "output_dir": "",
    "enable_3d_projection_matching": False
}

DEFAULT_3D_PROJECTION_CONFIG = {
    "max_bboxes": 500,
    "max_total_points": 100000,
    "confidence_threshold": 0.5,
    "min_confident_points": 7,
    "min_bbox_area": 10.0,
    "output_dir": "output_3d_projection",
    "enable_3d_projection_matching": True,
    "depth_confidence_threshold": 0.15,
    "point_3d_confidence_threshold": 0.15,
    "min_depth": 0.1,
    "max_depth": 10.0,
    "max_3d_points_per_bbox": 70,
    "projection_match_threshold": 0.7,
    "max_3d_distance": 1.0,
    "max_depth_difference": 2.0,
    "min_depth_consistency": 0.3
}
