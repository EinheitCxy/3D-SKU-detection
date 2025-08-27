import os
import torch
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def get_optimal_device_config(verbose: bool = True):
    """智能设备选择。
    
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
            logger.info(f"🚀 GPU: {gpu_name}, 使用{dtype}")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        if verbose:
            logger.info("🖥️ 使用CPU计算")
    
    return device, dtype


@dataclass
class SKUMatchingConfig:
    """SKU匹配配置参数"""
    
    # === 核心检测参数 ===
    detection_confidence_threshold: float = 0.0  # 检测置信度阈值
    min_bbox_area: float = 10.0                  # 最小边界框面积
    max_bboxes: int = 500                        # 最大检测框数量
    
    # === 点采样参数 ===
    max_points_per_bbox: int = 100               # 每个2D检测框最大采样点数
    max_3d_points_per_bbox: int = 50             # 每个3D检测框最大采样点数
    max_total_points: int = 100000               # 全局最大采样点数上限
    
    # === 匹配阈值参数 ===
    confidence_threshold: float = 0.5            # 点追踪置信度阈值
    min_confident_points: int = 10               # 最小置信点数
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
    output_dir: str = "output"                   # 输出目录
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
        elif self.dtype is None:
            # 回退到原有逻辑
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        # 创建输出目录
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
    "max_points_per_bbox": 200,
    "max_bboxes": 500,
    "max_total_points": 100000,
    "confidence_threshold": 0.5,
    "min_confident_points": 10,
    "min_bbox_area": 10.0,
    "output_dir": "output_point_tracking",
    "enable_3d_projection_matching": False
}

DEFAULT_3D_PROJECTION_CONFIG = {
    "max_bboxes": 500,
    "max_total_points": 100000,
    "confidence_threshold": 0.5,
    "min_confident_points": 10,
    "min_bbox_area": 10.0,
    "output_dir": "output_3d_projection",
    "enable_3d_projection_matching": True,
    "depth_confidence_threshold": 0.15,
    "point_3d_confidence_threshold": 0.15,
    "min_depth": 0.1,
    "max_depth": 10.0,
    "max_3d_points_per_bbox": 50,
    "projection_match_threshold": 0.7,
    "max_3d_distance": 1.0,
    "max_depth_difference": 2.0,
    "min_depth_consistency": 0.3
}