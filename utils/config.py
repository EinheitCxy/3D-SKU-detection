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


def build_matching_config_from_yaml(path: str | Path, algorithm: str | None = None, backend: str | None = None) -> "SKUMatchingConfig":
    """Build SKUMatchingConfig from a YAML file.

    Args:
        path: YAML 配置路径
        algorithm: 可选算法覆盖
        backend: 可选后端覆盖，优先于 YAML 的 inference.backend（使 CLI --match_backend 对 da3 生效 backend-aware 阈值）

    Expected schema (example):
    matching:
      algorithm: point_tracking  # or '3d'/'3d_mapping'
      device: cuda
      max_points_per_bbox: 50
      confidence_threshold: 0.5
      min_confident_points: 7
      min_hit_ratio: 0.5
      output_dir: Output/floor_display2
      save_json: true

    Args:
        path: YAML path
        algorithm: optional override of matching.algorithm
    Notes:
        - YAML 仅作为覆盖项（overrides）；未提供的字段使用 `SKUMatchingConfig.for_point_tracking()`
          / `SKUMatchingConfig.for_3d_mapping()` 的默认值。

    Returns:
        SKUMatchingConfig
    """
    data = load_yaml_config(path)
    # Support both:
    # - dedicated matching configs: {"matching": {...}}
    # - project-wide config.yaml used by main.py: {"inference": {...}}
    if "matching" in data and isinstance(data.get("matching"), dict):
        section = data["matching"]
    elif "inference" in data and isinstance(data.get("inference"), dict):
        section = data["inference"]
    else:
        section = data

    algo = (algorithm or section.get("algorithm") or "point_tracking").lower()

    # Map allowed fields into overrides for dataclass builders
    overrides: Dict[str, Any] = {}
    for key in (
        "backend",
        "device",
        "max_points_per_bbox",
        "confidence_threshold",
        "min_confident_points",
        "min_hit_ratio",
        "seed",
        "save_json",
        "output_dir",
        "max_bboxes",
        "max_total_points",
        "min_bbox_area",
        # Optional SAM3-guided sampling (minimal knobs)
        "enable_sam3_mask_sampling",
        "sam3_checkpoint_path",
        "sam3_use_self_exemplar",
        "sam3_self_exemplar_threshold",
        "sam3_device",
        "sam3_max_queries_per_forward",
        "sam3_batch_image_size",
        "sam3_max_batch_size",
        "sam3_max_dets_per_query",
        "sam3_min_cuda_free_gb",
        # 3D-specific (safe to include; unused for PT)
        "depth_confidence_threshold",
        "point_3d_confidence_threshold",
        "min_depth",
        "max_depth",
        "max_3d_points_per_bbox",
        "projection_match_threshold",
        "max_3d_distance",
        "plane_normal_alignment_threshold",
        "enable_3d_mapping",
        "pairing_3d",
        "min_3d_sample_points",
    ):
        if key in section:
            overrides[key] = section[key]

    if algo in ("3d", "3d_mapping", "mapping", "3d-2d"):
        # backend 优先级：显式传入参数 > YAML override > 默认 pi3
        backend = backend if backend is not None else overrides.pop("backend", "pi3")
        overrides.pop("backend", None)  # 避免重复传 key
        return SKUMatchingConfig.for_3d_mapping(backend=backend, **overrides)
    return SKUMatchingConfig.for_point_tracking(**overrides)


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

    # === 3D重建后端选择 ===
    backend: str = "vggt"  # 3D重建模型后端 (vggt/pi3)

    # === 核心检测参数 ===
    detection_confidence_threshold: float = 0.0  # 检测置信度阈值
    min_bbox_area: float = 10.0                  # 最小边界框面积
    max_bboxes: int = 500                        # 最大检测框数量
    
    # === 点采样参数 ===
    max_points_per_bbox: int = 70                # 每个2D检测框最大采样点数
    max_3d_points_per_bbox: int = 70             # 每个3D检测框最大采样点数
    max_total_points: int = 100000               # 全局最大采样点数上限

    # === SAM3 mask 引导采样（可选）===
    # 目标：在 bbox 点采样前先用 SAM3 预测 mask，再从 mask 内采样点
    enable_sam3_mask_sampling: bool = False      # 是否启用 SAM3 mask 引导采样
    sam3_checkpoint_path: Optional[str] = None   # sam3.pt 本地路径（禁用 HF 下载）
    sam3_use_self_exemplar: bool = False         # 是否使用self-exemplar分割（每个bbox作为自己的visual exemplar）
    sam3_self_exemplar_threshold: float = 0.5    # self-exemplar检测阈值
    sam3_device: str = "auto"                   # SAM3单独设备: auto/cuda/cpu
    sam3_max_queries_per_forward: int = 16       # self-exemplar时每次forward的最大query数（降低显存峰值）
    sam3_batch_image_size: int = 1008            # batch API resize尺寸（默认1008，显存紧张可降到768/640）
    sam3_max_batch_size: int = 5                  # self-exemplar每批最多bbox数(单ref一次forward的batch上限,大值=少forward overhead)
    sam3_max_dets_per_query: int = 8             # batch API每个query保留的最大候选数（越小越省显存）
    sam3_min_cuda_free_gb: float = 10.0          # sam3_device=auto 时启用 CUDA 的最小可用显存(GB)

    # === 高斯分布采样参数 ===
    enable_gaussian_sampling: bool = True        # 是否启用高斯分布采样（中心密集，向外正态递减）
    gaussian_sigma: float = 0.3                  # 高斯分布标准差（相对于bbox半径，0.2-0.4推荐，越小中心越集中）
    gaussian_truncate: float = 3.0               # 高斯分布截断倍数（超过sigma*truncate的点权重接近0）
    enable_gaussian_in_sam3_mask: bool = True    # SAM3 mask内是否也应用高斯加权采样（混合采样模式）

    # === 非重合区域采样参数 ===
    enable_non_overlap_sampling: bool = True     # 是否启用非重合区域采样
    overlap_threshold: float = 0.1               # 检出框重合阈值
    min_non_overlap_area: float = 20.0           # 非重合区域最小面积
    
    # === 匹配阈值参数 ===
    confidence_threshold: float = 0.5            # 点追踪置信度阈值
    min_confident_points: int = 5                # 最小置信点数
    min_3d_sample_points: int = 10             # 3D采样最少有效点数(不足则跳过该物体)，货架小bbox可调低
    min_hit_ratio: float = 0.4         # 最小命中率阈值
    projection_match_threshold: float = 0.3       # 3D投影匹配阈值

    # === 3D重建参数 ===
    enable_3d_mapping: bool = False  # 是否启用3D映射匹配
    pairing_3d: str = "all"                      # 3D匹配配对策略: all/next
    depth_confidence_threshold: float = 0.05     # 深度置信度阈值
    point_3d_confidence_threshold: float = 0.05  # 3D点置信度阈值
    min_depth: float = -1.0                      # 最小深度值
    max_depth: float = 7.0                       # 最大深度值（基于场景深度2.9米）

    # === 3D几何验证参数 ===（基于实际场景：宽2.4m×高1.5m×深2.9m）
    max_3d_distance: float = 0.8                 # 最大3D空间距离阈值(米)
    depth_consistency_threshold: float = 0.5     # 深度一致性阈值(米)，用于3D点采样端cache自洽性验证
    plane_normal_alignment_threshold: float = 0.2  # 平面法向对齐阈值(|cos|夹角)，低于此值(夹角>78°)判非同面拒绝；放宽因环绕多视角下同平面法向估计有视角偏转

    # === 3D验证性能优化参数 ===
    max_3d_validation_candidates: int = 5        # 最大3D验证候选框数量（预筛选Top-K，减少昂贵的3D采样）

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

        # 不在初始化时创建目录，而是在实际写入文件时创建
        # if self.output_dir:
        #     os.makedirs(self.output_dir, exist_ok=True)

        self._validate_config()
    
    # === Backend 衍生属性（自动推导，减少参数传递）===
    @property
    def model_type(self) -> str:
        """根据 backend 自动推导 model_type

        Returns:
            "da3" / "pi3" / "vggt"
        """
        if self.backend == "da3":
            return "da3"
        if self.backend == "pi3":
            return "pi3"
        return "vggt"

    @property
    def transform_kwargs(self) -> dict:
        """根据 backend 自动推导 transform 构建参数

        Returns:
            DA3: {"process_res": 504}（upper_bound_resize 算法派生目标尺寸）
            Pi3: {"pixel_limit": 255000}
            VGGT: {"target_size": 518}
        """
        if self.backend == "da3":
            return {"process_res": 504}
        if self.backend == "pi3":
            return {"pixel_limit": 255000}
        return {"target_size": 518}

    @property
    def preprocess_mode(self) -> str:
        """根据 backend 自动推导图像预处理模式

        Returns:
            Pi3/DA3: "resize" (等比例缩放)
            VGGT: "crop" (裁剪+填充)
        """
        return "resize" if self.backend in ("pi3", "da3") else "crop"

    def _validate_config(self):
        # Backend验证
        if self.backend not in ("vggt", "pi3", "da3"):
            raise ValueError(f"backend must be 'vggt', 'pi3', or 'da3', got {self.backend}")

        if self.max_points_per_bbox <= 0:
            raise ValueError(f"max_points_per_bbox must be positive, got {self.max_points_per_bbox}")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(f"confidence_threshold must be in [0,1], got {self.confidence_threshold}")
        
        if self.min_confident_points > self.max_points_per_bbox:
            raise ValueError(f"min_confident_points ({self.min_confident_points}) cannot exceed max_points_per_bbox ({self.max_points_per_bbox})")
        
        if self.min_bbox_area <= 0:
            raise ValueError(f"min_bbox_area must be positive, got {self.min_bbox_area}")
        
        # 3D相关参数验证
        if self.enable_3d_mapping:
            if self.pairing_3d not in ("all", "next"):
                raise ValueError(f"pairing_3d must be 'all' or 'next', got {self.pairing_3d}")
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
        return "3D Mapping" if self.enable_3d_mapping else "Point Tracking"
    
    def to_dict(self) -> dict:
        """将配置转为字典格式"""
        return {
            'algorithm': self.get_algorithm_name(),
            'max_points_per_bbox': self.max_points_per_bbox,
            'confidence_threshold': self.confidence_threshold,
            'min_confident_points': self.min_confident_points,
            'min_bbox_area': self.min_bbox_area,
            'output_dir': self.output_dir,
            'enable_3d_mapping': self.enable_3d_mapping,
            'pairing_3d': self.pairing_3d,
            'device': self.device,
            'seed': self.seed
        }

    @classmethod
    def for_point_tracking(cls, **overrides) -> 'SKUMatchingConfig':
        """创建点追踪算法的默认配置（仅覆盖算法特定参数，其他使用类默认值）

        Args:
            **overrides: 覆盖默认值的参数

        Returns:
            配置好的SKUMatchingConfig实例
        """
        # 仅覆盖点追踪算法特定的参数
        algorithm_specific = {
            "max_points_per_bbox": 80,
            "max_bboxes": 200,
            "max_total_points": 20000,
            "min_confident_points": 7,
            "enable_3d_mapping": False
        }
        algorithm_specific.update(overrides)
        return cls(**algorithm_specific)

    @classmethod
    def for_3d_mapping(cls, backend: str = "pi3", **overrides) -> 'SKUMatchingConfig':
        """创建3D映射算法的默认配置（仅覆盖算法特定参数，其他使用类默认值）

        Args:
            backend: 3D重建后端，da3 用 metric 深度+原始 conf 需独立阈值；pi3/vggt 用相对深度+sigmoid conf
            **overrides: 覆盖默认值的参数（YAML/CLI 仍可覆盖 backend 设的值）

        Returns:
            配置好的SKUMatchingConfig实例
        """
        # 共通 3D 算法参数
        algorithm_specific = {
            "max_bboxes": 500,
            "max_total_points": 100000,
            "min_confident_points": 5,
            "output_dir": "output_3dmapping",
            "enable_3d_mapping": True,
        }
        if backend == "da3":
            # da3: metric 米制深度 + 原始 conf[1,6] 未归一化，需独立标定
            algorithm_specific.update({
                "min_depth": 0.3,                       # 米: 30cm 近场下限
                "max_depth": 8.0,                       # 米: 货架1-5m + 过道纵深
                "depth_confidence_threshold": 1.5,     # da3 conf[1,6], 1.5 过滤低端噪声
                "point_3d_confidence_threshold": 1.5,
                "max_3d_distance": 0.5,                 # 米: 同物体跨视角中心应<0.1m, 0.5 容采样抖动
                "depth_consistency_threshold": 0.3,     # 米: 采样端cache自洽性检查容差
            })
        else:
            # pi3/vggt: 相对深度 + sigmoid conf（已标定，保持原值）
            algorithm_specific.update({
                "min_depth": 0.1,
                "max_depth": 3.0,
            })
        algorithm_specific["backend"] = backend
        algorithm_specific.update(overrides)
        return cls(**algorithm_specific)
