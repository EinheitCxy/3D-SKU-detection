"""
Interactive 3D Viewer - Viser交互式3D可视化系统

使用方式：
uv run interactive_3d_viewer.py \
--global-mapping Output/floor_display2/dedup_detections/global_mapping.json \
--reconstruction Output/floor_display2/reconstruction.glb \
--image-dir ../imdata/floor_display2/images \
--detection-dir ../imdata/floor_display2/detections_results \
[--downsample 0.1] \
[--cache-dir <dir>]

架构设计：
1. **数据加载层**：从GLB加载点云 + 从global_mapping.json加载全局ID映射
2. **ID分配策略**：
   - 优先：从 VGGT 中间输出加载真实点云归属（<dataset>/vggt_cache/points_with_gid.npz）
   - 降级：均匀分配全局ID（仅用于演示）
3. **Viser交互层**：点云拾取、全局ID显示、置信度过滤
4. **缓存优化**：自动构建 pcd_gid.npz + global_object_index.json 缓存

技术栈:
- Viser: 3D交互可视化（支持点拾取、相机控制）
- NumPy: 点云数据处理
- scipy.spatial.cKDTree: 最近邻搜索（点拾取 + ID匹配）
- trimesh: GLB/PLY文件加载

VGGT中间输出格式（可选，用于真实ID分配）:
- 路径: <dataset>/vggt_cache/points_with_gid.npz
- 内容: {
    "points": (N, 3),           # 点云坐标
    "global_ids": (N,),          # 每个点的全局ID
    "confidences": (N,),         # 置信度（可选）
  }

数据产物格式（自动生成）:
- pcd_gid.npz: {
    "points": (N, 3),           # 所有点坐标
    "colors": (N, 3),            # RGB颜色
    "global_ids": (N,),          # 每个点的全局ID
    "confidences": (N,),         # 置信度
  }
- global_object_index.json: {
    "1": {"images": [1, 2], "objects": [0, 20], "active": 2, "removed": 5},
    ...
  }

"""

import sys
import json
import logging
import argparse
import time
import colorsys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

# Rich progress bar
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logging.warning("rich not available, progress bars will be disabled. Install: pip install rich")

# 确保可以导入本地模块
CODE_DIR = Path(__file__).parent
VGGT_DIR = CODE_DIR.parent / "vggt-main"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(VGGT_DIR) not in sys.path:
    sys.path.insert(0, str(VGGT_DIR))

from utils.global_id_mapper import GlobalIDMapper
from utils.kdtree_utils import build_kdtree, nearest_neighbor_mapping

# 延迟导入viser（避免启动时加载）
viser = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU加速配置
class GPUConfig:
    """GPU加速配置与环境检测"""
    def __init__(self):
        self.use_cupy = self._check_cupy()
        self.use_faiss_gpu = self._check_faiss_gpu()
        self.device_name = self._get_device_name()

        if self.use_cupy or self.use_faiss_gpu:
            logger.info(f"GPU加速已启用: {self.device_name}")
        else:
            logger.info("GPU加速不可用，使用CPU模式")

    def _check_cupy(self) -> bool:
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return True
        except Exception:
            return False

    def _check_faiss_gpu(self) -> bool:
        try:
            import faiss
            return faiss.get_num_gpus() > 0
        except Exception:
            return False

    def _get_device_name(self) -> str:
        if self.use_cupy:
            try:
                import cupy as cp
                return cp.cuda.Device(0).name.decode()
            except Exception:
                pass
        return "CPU"

GPU_CONFIG = GPUConfig()


# 缓存版本控制
class CacheValidator:
    """缓存版本控制与失效检测"""

    CACHE_VERSION = "2.0"

    @staticmethod
    def compute_file_hash(file_path: Path, method: str = 'mtime') -> str:
        """计算文件哈希（快速模式使用mtime）"""
        if method == 'mtime':
            stat = file_path.stat()
            return f"{stat.st_mtime:.6f}_{stat.st_size}"
        elif method == 'md5':
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        else:
            raise ValueError(f"Unknown hash method: {method}")

    @staticmethod
    def create_metadata(
        global_mapping_path: Path,
        reconstruction_path: Path,
        config: Dict[str, Any],
        statistics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建缓存元数据"""
        return {
            "version": CacheValidator.CACHE_VERSION,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "config": config,
            "source_files": {
                "global_mapping.json": {
                    "path": str(global_mapping_path),
                    "hash": CacheValidator.compute_file_hash(global_mapping_path),
                    "mtime": datetime.fromtimestamp(global_mapping_path.stat().st_mtime).isoformat()
                },
                "reconstruction.glb": {
                    "path": str(reconstruction_path),
                    "hash": CacheValidator.compute_file_hash(reconstruction_path),
                    "mtime": datetime.fromtimestamp(reconstruction_path.stat().st_mtime).isoformat()
                }
            },
            "statistics": statistics
        }

    @staticmethod
    def is_cache_valid(
        cache_dir: Path,
        global_mapping_path: Path,
        reconstruction_path: Path,
        config: Dict[str, Any]
    ) -> bool:
        """检查缓存是否有效"""
        metadata_path = cache_dir / "cache_metadata.json"

        if not metadata_path.exists():
            logger.info("缓存元数据不存在，需要重建")
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"读取缓存元数据失败: {e}")
            return False

        # 检查版本兼容性
        if metadata.get('version') != CacheValidator.CACHE_VERSION:
            logger.info(f"缓存版本不兼容 ({metadata.get('version')} vs {CacheValidator.CACHE_VERSION})")
            return False

        # 检查配置参数
        if metadata.get('config', {}).get('downsample_ratio') != config.get('downsample_ratio'):
            logger.info("下采样参数变化，需要重建")
            return False
        if metadata.get('config', {}).get('points_source') != config.get('points_source'):
            logger.info("点云来源变化，需要重建")
            return False

        # 检查源文件变更（快速模式：仅检查hash）
        try:
            current_mapping_hash = CacheValidator.compute_file_hash(global_mapping_path)
            current_recon_hash = CacheValidator.compute_file_hash(reconstruction_path)

            cached_mapping_hash = metadata['source_files']['global_mapping.json']['hash']
            cached_recon_hash = metadata['source_files']['reconstruction.glb']['hash']

            if current_mapping_hash != cached_mapping_hash:
                logger.info("global_mapping.json 已更新")
                return False

            if current_recon_hash != cached_recon_hash:
                logger.info("reconstruction.glb 已更新")
                return False

        except Exception as e:
            logger.warning(f"源文件检查失败: {e}")
            return False

        logger.info("缓存有效，将使用现有缓存")
        return True


class DataCacheBuilder:
    """
    数据缓存构建器

    负责从VGGT重建结果和global_mapping.json生成优化的缓存数据：
    - pcd_gid.npz: 点云+全局ID
    - global_object_index.json: 全局ID索引
    """

    def __init__(
        self,
        global_mapping_path: str,
        reconstruction_path: str,
        output_dir: str,
        image_dir: Optional[str] = None,
        detection_dir: Optional[str] = None,
        downsample_ratio: float = 0.1,
        points_source: str = "glb",
        seed: Optional[int] = 42,
        enable_progress: bool = True,
        enable_gpu: bool = True,
    ):
        """
        初始化DataCacheBuilder

        Args:
            global_mapping_path: global_mapping.json 文件路径
            reconstruction_path: GLB/PLY 3D重建文件路径
            output_dir: 缓存输出目录
            image_dir: 原始图像目录
            detection_dir: 检测结果目录（可选，默认自动推断）
            downsample_ratio: 下采样比例（0-1），用于减少点云规模
            seed: 随机种子
            enable_progress: 是否启用进度条
            enable_gpu: 是否启用GPU加速
        """
        self.global_mapping_path = Path(global_mapping_path)
        self.reconstruction_path = Path(reconstruction_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downsample_ratio = downsample_ratio
        self.points_source = points_source  # 'glb' | 'predictions'
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.detection_dir = Path(detection_dir) if detection_dir is not None else None
        self._rng = np.random.default_rng(seed)
        self.enable_progress = enable_progress and RICH_AVAILABLE
        self.enable_gpu = enable_gpu and (GPU_CONFIG.use_cupy or GPU_CONFIG.use_faiss_gpu)

        self.mapper = GlobalIDMapper(str(self.global_mapping_path))

        logger.info(f"DataCacheBuilder initialized:")
        logger.info(f"  - Global mapping: {self.global_mapping_path}")
        logger.info(f"  - Reconstruction: {self.reconstruction_path}")
        logger.info(f"  - Output directory: {self.output_dir}")
        logger.info(f"  - Detection directory: {self.detection_dir or 'Auto-inferred'}")
        logger.info(f"  - Downsample ratio: {downsample_ratio}")
        logger.info(f"  - Points source: {self.points_source}")
        logger.info(f"  - RNG seed: {seed}")
        logger.info(f"  - Progress bar: {self.enable_progress}")
        logger.info(f"  - GPU acceleration: {self.enable_gpu}")
        if self.image_dir is not None:
            logger.info(f"  - Image directory: {self.image_dir}")

    def load_point_cloud_from_glb(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        从GLB文件加载点云和颜色

        Returns:
            (points, colors): 点云坐标 (N, 3) 和颜色 (N, 3) 或 None
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required. Install: pip install trimesh")

        logger.info(f"Loading GLB file: {self.reconstruction_path}")
        scene = trimesh.load(str(self.reconstruction_path))

        points_list = []
        colors_list = []

        if isinstance(scene, trimesh.Scene):
            for name, geom in scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    points_list.append(geom.vertices)
                    # 尝试提取颜色
                    if hasattr(geom, 'visual') and hasattr(geom.visual, 'vertex_colors'):
                        colors_list.append(geom.visual.vertex_colors[:, :3] / 255.0)
                    else:
                        colors_list.append(np.ones((len(geom.vertices), 3)) * 0.8)
        elif hasattr(scene, 'vertices'):
            points_list.append(scene.vertices)
            if hasattr(scene, 'visual') and hasattr(scene.visual, 'vertex_colors'):
                colors_list.append(scene.visual.vertex_colors[:, :3] / 255.0)
            else:
                colors_list.append(np.ones((len(scene.vertices), 3)) * 0.8)

        if not points_list:
            raise ValueError("No point cloud data found in GLB file")

        points = np.vstack(points_list)
        colors = np.vstack(colors_list) if colors_list else None

        logger.info(f"Loaded {len(points)} points from GLB")
        return points, colors

    def load_point_cloud_from_predictions(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        从 predictions.npz 直接加载点云与颜色（提高几何精度，避免GLB量化影响）

        Returns:
            (points, colors): 点云坐标 (N,3) 和颜色 (N,3)
        """
        dataset_root = self.reconstruction_path.parent
        pred_path = dataset_root / "vggt_cache" / "predictions.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"找不到 predictions.npz: {pred_path}。可改用 --points-source glb 或先运行重建导出。")

        logger.info(f"从 predictions.npz 加载点云: {pred_path}")
        data = np.load(pred_path, allow_pickle=True)

        # world_points: (S,H,W,3) 或 (H,W,3)
        wp = data['world_points']
        if wp.ndim == 3:
            wp = wp[np.newaxis, ...]
        S, H, W, _ = wp.shape
        points = wp.reshape(-1, 3)

        # 颜色来自 images: (S,3,H,W) 或 (S,H,W,3)
        if 'images' in data:
            imgs = data['images']
            if imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = imgs.transpose(0, 2, 3, 1)  # CHW->HWC
            colors = imgs.reshape(-1, 3)
        else:
            colors = np.ones((points.shape[0], 3), dtype=np.float32) * 0.8

        logger.info(f"Loaded {len(points)} points from predictions (S={S}, H={H}, W={W})")
        return points, colors

    def assign_global_ids_to_points(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        优先从VGGT缓存(points_with_gid.npz)加载真实全局ID；否则从predictions.npz在线计算

        工作流程：
        1) 快速路径：<dataset>/vggt_cache/points_with_gid.npz
           - 直接将预先打好gid的3D点与GLB点云做最近邻映射，得到(gid, conf, frame_idx)
        2) 回退路径：<dataset>/vggt_cache/predictions.npz
           - 通过检测bbox→3D提取→最近邻映射的方式在线计算

        Args:
            points: 点云坐标 (N, 3)

        Returns:
            (global_ids, confidences, frame_indices): 全局ID数组 (N,), 置信度 (N,), 帧索引 (N,)

        Raises:
            FileNotFoundError: 当无任何可用VGGT缓存时抛出
        """
        dataset_root = self.reconstruction_path.parent
        vggt_cache_dir = dataset_root / "vggt_cache"

        # 1) 优先：预计算的 points_with_gid.npz
        pvg_path = vggt_cache_dir / "points_with_gid.npz"
        if pvg_path.exists():
            logger.info(f"使用预计算VGGT点云: {pvg_path}")
            try:
                data = np.load(pvg_path)
                pre_points = data["points"]  # (M,3)
                pre_gids = data["global_ids"]  # (M,)
                pre_confs = data["confidences"] if "confidences" in data else np.ones(len(pre_points), dtype=np.float32)
                pre_frame_indices = data["frame_indices"] if "frame_indices" in data else None

                # 最近邻映射至GLB点
                distances, indices = nearest_neighbor_mapping(pre_points, points, k=1)
                final_gids = pre_gids[indices].astype(np.int32)
                final_confs = pre_confs[indices].astype(np.float32)

                # 映射帧索引（如果存在）
                if pre_frame_indices is not None:
                    final_frame_indices = pre_frame_indices[indices].astype(np.int32)
                else:
                    # 降级：尝试从predictions.npz生成
                    logger.warning("points_with_gid.npz缺少frame_indices，尝试从predictions.npz生成")
                    final_frame_indices = self._generate_frame_indices_from_predictions(points)

                logger.info(
                    f"完成快速映射: {len(np.unique(final_gids))} 个唯一ID，平均距离={float(np.mean(distances)):.4f}"
                )
                return final_gids, final_confs, final_frame_indices
            except Exception as e:
                logger.warning(f"读取 {pvg_path} 失败，将回退在线计算: {e}")

        # 2) 回退：在线计算（需要 predictions.npz）
        predictions_cache = vggt_cache_dir / "predictions.npz"
        if not predictions_cache.exists():
            error_msg = (
                f"VGGT中间输出不存在: {predictions_cache}\n\n"
                f"解决方案：\n"
                f"在 main.py 的 run_reconstruction() 中，VGGT重建会自动保存。\n"
                f"确保 vggt_3d_reconstructor.py 的 reconstruct_from_directory() 中：\n"
                f"  save_predictions=True (默认开启)\n\n"
                f"手动保存示例（在重建后）：\n"
                f"  vggt_cache_dir = output_dir / 'vggt_cache'\n"
                f"  vggt_cache_dir.mkdir(exist_ok=True)\n"
                f"  np.savez_compressed(\n"
                f"      vggt_cache_dir / 'predictions.npz',\n"
                f"      world_points=predictions['world_points_from_depth'],  # (B,H,W,3)\n"
                f"      depth=predictions['depth'],                           # (B,H,W)\n"
                f"      conf=predictions['conf'],                             # (B,H,W)\n"
                f"  )\n\n"
            )
            raise FileNotFoundError(error_msg)

        logger.info(f"使用VGGT predictions: {predictions_cache}")
        logger.info("实时计算点云全局ID归属...")
        return self._compute_gids_from_predictions(
            predictions_cache,
            dataset_root,
            points
        )

    def _compute_gids_from_predictions(
        self,
        predictions_path: Path,
        dataset_root: Path,
        target_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从VGGT predictions实时计算点云全局ID归属

        核心算法：
        1. 加载 predictions (world_points, depth, conf, images)
        2. 加载 global_mapping.json + detections
        3. 对每个3D点，找到其在2D图像中的投影位置
        4. 根据投影位置判断属于哪个bbox
        5. 从 global_mapping 查询该bbox的全局ID

        Args:
            predictions_path: VGGT predictions缓存路径
            dataset_root: 数据集根目录
            target_points: 目标点云（GLB中的点）

        Returns:
            (global_ids, confidences, frame_indices)
        """
        logger.info("加载VGGT predictions...")
        data = np.load(predictions_path, allow_pickle=True)

        world_points = data['world_points']  # (B,H,W,3) 或 (H,W,3)
        conf = data['conf']                  # (B,H,W) 或 (H,W)

        # 展平点云
        if world_points.ndim == 4:  # (B,H,W,3)
            B, H, W, _ = world_points.shape
            world_points_flat = world_points.reshape(-1, 3)
            conf_flat = conf.reshape(-1)
        else:  # (H,W,3)
            H, W, _ = world_points.shape
            world_points_flat = world_points.reshape(-1, 3)
            conf_flat = conf.reshape(-1)

        logger.info(f"   世界坐标点云: {world_points_flat.shape}")
        logger.info(f"   目标点云: {target_points.shape}")

        # 工具：强制要求路径存在（不再设置回退）
        def _require_path(p: Path, desc: str) -> Path:
            if p is None:
                raise FileNotFoundError(f"缺少必需路径: {desc}")
            if not p.exists():
                raise FileNotFoundError(f"{desc} 不存在: {p}")
            logger.info(f"使用 {desc}: {p}")
            return p

        # 加载 global_mapping（必须通过 --global-mapping 指定且存在）
        global_mapping_path = _require_path(self.global_mapping_path, "global_mapping.json (--global-mapping)")

        with global_mapping_path.open('r') as f:
            global_mapping_data = json.load(f)

        logger.info(f"加载 global_mapping: {len(global_mapping_data)} 个全局ID")

        # 加载检测结果目录（必须通过 --detection-dir 指定且存在）
        if self.detection_dir is None:
            raise FileNotFoundError("缺少必需参数 --detection-dir（检测结果目录）")
        detection_dir = _require_path(self.detection_dir, "检测结果目录 (--detection-dir)")

        from utils.data_utils import load_detections
        detections_with_indices = load_detections(str(detection_dir), return_index_map=True)

        # 分离索引和数据
        detection_indices = [item[0] for item in detections_with_indices]  # 文件编号
        detections = [item[1] for item in detections_with_indices]  # 检测数据

        logger.info(f"加载检测结果: {len(detections)} 张图片")
        logger.info(f"   图像索引范围: {min(detection_indices)} - {max(detection_indices)}")

        # 关键：帧对齐验证（严格模式，有问题直接报错）
        logger.info("验证VGGT-Detection帧对齐...")
        from utils.frame_alignment import VGGTDetectionAligner

        # 尝试从NPZ中读取图像ID
        vggt_image_ids = None
        if "image_ids" in data:
            vggt_image_ids = data["image_ids"].tolist()
            logger.info(f"从NPZ读取VGGT图像ID: {vggt_image_ids}")
        else:
            logger.warning("NPZ文件缺少image_ids，假设按顺序对应")
            # 假设VGGT按顺序处理
            vggt_frame_count = world_points.shape[0] if world_points.ndim == 4 else 1
            vggt_image_ids = list(range(vggt_frame_count))

        # 帧对齐验证（严格模式）
        aligned_vggt_data, aligned_detections, alignment_report = VGGTDetectionAligner.validate_and_align(
            vggt_data={"world_points": world_points, "conf": conf},
            detections=detections,
            detection_indices=detection_indices,
            vggt_image_ids=vggt_image_ids,
            strict_mode=True  # 严格模式：有问题直接报错
        )

        logger.info("帧对齐验证通过")

        # 变换将在计算出 aligned_image_ids 后优先从 transforms.json 载入
        vggt_transforms = None

        # 构建反向索引：(image_id, object_id) -> global_id
        reverse_mapping = {}
        for gid_str, instances in global_mapping_data.items():
            for inst in instances:
                key = (inst['image_id'], inst['object_id'])
                reverse_mapping[key] = int(gid_str)

        logger.info(f"构建反向索引: {len(reverse_mapping)} 个实例")

        # 提取对齐后的真实image_ids
        aligned_image_ids = alignment_report.get('repaired_image_ids') or alignment_report.get('common_ids')
        if aligned_image_ids is None:
            # 完美对齐情况，使用原始detection_indices
            aligned_image_ids = detection_indices[:len(aligned_detections)]

        logger.info(f"对齐后的图像ID: {aligned_image_ids}")

        # 优先从 vggt_cache/transforms.json 加载VGGT变换，失败则回退到自动构建
        try:
            from utils.transforms import build_transforms_from_json
            transforms_path = dataset_root / "vggt_cache" / "transforms.json"
            vggt_transforms = build_transforms_from_json(str(transforms_path), aligned_image_ids)
            if vggt_transforms is not None:
                logger.info(f"已从 transforms.json 加载变换，共 {len(vggt_transforms)} 张")
            else:
                logger.info("未找到或不完整的 transforms.json，准备回退到自动构建变换")
        except Exception as e:
            logger.warning(f"加载 transforms.json 失败，将回退到自动构建：{e}")

        if vggt_transforms is None:
            try:
                from utils.transforms import build_vggt_transforms

                search_dir = None
                if self.image_dir is not None and self.image_dir.exists():
                    search_dir = self.image_dir
                elif (dataset_root / "images").exists():
                    search_dir = dataset_root / "images"

                if search_dir is not None:
                    image_paths, found_all = self._find_image_paths(aligned_image_ids, search_dir)
                    if found_all and image_paths:
                        vggt_transforms = build_vggt_transforms(image_paths, target_size=518)
                        logger.info(f"构建VGGT裁剪对齐变换，共 {len(vggt_transforms)} 张")
                    else:
                        logger.warning("无法为所有对齐后的图像构建路径，跳过裁剪对齐变换")
                else:
                    logger.warning("无法定位图像目录，跳过裁剪对齐。可通过 --image-dir 指定。")
            except Exception as e:
                logger.warning(f"构建VGGT裁剪对齐变换失败，将降级为直接像素对齐：{e}")

        # 高效方法：从2D检测框直接提取3D点（O(K×A)复杂度）
        logger.info("从检测框直接提取3D点并分配全局ID...")
        from utils.bbox_3d_extractor import extract_3d_from_bboxes

        # 使用高效掩码索引方法（传入真实image_ids）
        extracted_points, extracted_gids, extracted_confs = extract_3d_from_bboxes(
            world_points=world_points,
            world_points_conf=conf,
            detections=aligned_detections,
            reverse_mapping=reverse_mapping,
            image_ids=aligned_image_ids,  # **关键修复**：传入真实image_ids
            conf_threshold=0.1,
            vggt_transforms=vggt_transforms,
        )

        logger.info(f"高效提取完成: {len(extracted_points)} 个3D点, {len(np.unique(extracted_gids))} 个唯一ID")

        # 生成frame_indices（参考demo_viser.py）
        if world_points.ndim == 4:
            B, H, W, _ = world_points.shape
        else:
            B, H, W = 1, *world_points.shape[:2]

        # 为每个VGGT点生成帧索引标签
        source_frame_ids = np.repeat(np.arange(B), H * W)  # (B*H*W,)

        # 匹配到target_points（GLB点云）
        if len(extracted_points) == 0:
            logger.warning("未提取到任何3D点，使用占位数据")
            return (
                np.zeros(len(target_points), dtype=np.int32),
                np.zeros(len(target_points), dtype=np.float32),
                np.zeros(len(target_points), dtype=np.int32)
            )

        # 使用KDTree将提取的点映射到GLB点云（仅需一次）
        distances, indices = nearest_neighbor_mapping(extracted_points, target_points, k=1)

        # 分配global_id和置信度（加入距离鲁棒性：超出阈值视为无匹配）
        final_gids = np.full(len(target_points), -1, dtype=np.int32)
        final_confs = np.zeros(len(target_points), dtype=np.float32)

        # 自适应阈值：使用更鲁棒的方式计算阈值，避免中位数为0
        try:
            med = float(np.median(distances)) if len(distances) > 0 else 0.0
            p90 = float(np.percentile(distances, 90)) if len(distances) > 0 else 0.0
            thr = max(med * 3.0, p90 * 1.5, 0.01)
            valid = distances <= thr
            final_gids[valid] = extracted_gids[indices[valid]]
            final_confs[valid] = extracted_confs[indices[valid]]
            logger.info(f"全局ID映射: {int(np.sum(valid))}/{len(target_points)} 点有效")
        except Exception as e:
            logger.warning(f"距离阈值计算失败: {e}")
            final_gids = extracted_gids[indices]
            final_confs = extracted_confs[indices]

        # 为target_points分配frame_indices（用VGGT构建树，在GLB上查询）
        world_points_flat = world_points.reshape(-1, 3)
        distances_frame, glb_to_vggt_indices = nearest_neighbor_mapping(
            world_points_flat,  # source: VGGT点云（用于构建KDTree）
            target_points,       # target: GLB点云（用于查询）
            k=1
        )

        # 根据VGGT索引获取frame_id
        final_frame_indices = source_frame_ids[glb_to_vggt_indices].astype(np.int32)

        # 统计帧分布
        unique_mapped_frames = np.unique(final_frame_indices)
        logger.info(f"完成点云映射: {len(np.unique(final_gids[final_gids >= 0]))} 个唯一ID, {len(unique_mapped_frames)} 个帧")

        return final_gids, final_confs, final_frame_indices

    def _find_image_paths(
        self,
        detection_indices: List[int],
        search_dir: Path
    ) -> Tuple[List[str], bool]:
        """
        根据检测索引查找对应的图像文件路径

        Args:
            detection_indices: 检测文件编号列表
            search_dir: 图像搜索目录

        Returns:
            (image_paths, found_all): 图像路径列表和是否找到所有图像的标志
        """
        exts = [".JPG", ".JPEG", ".PNG", ".jpg", ".jpeg", ".png"]
        image_paths: List[str] = []
        found_all = True

        for num in detection_indices:
            found = False
            for ext in exts:
                p = search_dir / f"{num}{ext}"
                if p.exists():
                    image_paths.append(str(p))
                    found = True
                    break
            if not found:
                found_all = False
                logger.warning(f"找不到与检测编号 {num} 对应的图像文件（目录: {search_dir}）")
                break

        return image_paths, found_all

    def build_global_object_index(self) -> Dict[str, Any]:
        """
        构建全局ID索引（用于快速查询）

        Returns:
            索引字典
        """
        index = {}
        for gid in self.mapper.get_all_global_ids():
            instances = self.mapper.get_instances(gid)
            active = [inst for inst in instances if not inst.removed]

            index[str(gid)] = {
                "images": self.mapper.get_images_for_id(gid),
                "objects": self.mapper.get_object_ids_for_id(gid),
                "active_count": len(active),
                "removed_count": len(instances) - len(active),
                "total_count": len(instances),
                # 更细粒度：完整实例清单，便于可视化侧展示
                "instances": [inst.to_dict() for inst in instances],
            }

        return index

    def _downsample_points(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray],
        ratio: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        点云下采样（支持GPU加速）

        Args:
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3) 或 None
            ratio: 下采样比例 (0-1)

        Returns:
            (downsampled_points, downsampled_colors)
        """
        if ratio >= 1.0:
            return points, colors

        n_samples = int(len(points) * ratio)
        n_samples = max(1, min(n_samples, len(points)))

        # GPU加速采样
        if self.enable_gpu and GPU_CONFIG.use_cupy:
            try:
                import cupy as cp
                points_gpu = cp.asarray(points)
                indices_gpu = cp.random.choice(len(points_gpu), size=n_samples, replace=False)
                indices = cp.asnumpy(indices_gpu)
                logger.info(f"使用GPU加速下采样: {len(points)} → {n_samples} 点")
            except Exception as e:
                logger.warning(f"GPU下采样失败，回退到CPU: {e}")
                indices = self._rng.choice(len(points), size=n_samples, replace=False)
        else:
            indices = self._rng.choice(len(points), size=n_samples, replace=False)

        downsampled_points = points[indices]
        downsampled_colors = colors[indices] if colors is not None else None

        return downsampled_points, downsampled_colors

    def build_cache(self, progress: Optional[Any] = None) -> Tuple[Path, Path]:
        """
        构建完整的数据缓存（支持进度条）

        Args:
            progress: Rich Progress 对象（可选）

        Returns:
            (pcd_cache_path, index_cache_path)
        """
        build_start_time = time.time()

        # 创建任务进度
        task_id = None
        if progress is not None:
            task_id = progress.add_task("[cyan]构建缓存...", total=100)

        logger.info("Starting cache build process...")

        # 1. 加载点云 (20%)
        if self.points_source == "predictions":
            # 优先NPZ，不存在则回退GLB
            if progress and task_id is not None:
                progress.update(task_id, description="[cyan]加载NPZ点云...", advance=0)
            try:
                points, colors = self.load_point_cloud_from_predictions()
            except Exception as e:
                logger.warning(f"加载NPZ失败，将回退到GLB: {e}")
                if progress and task_id is not None:
                    progress.update(task_id, description="[cyan]加载GLB点云...", advance=0)
                points, colors = self.load_point_cloud_from_glb()
        else:
            if progress and task_id is not None:
                progress.update(task_id, description="[cyan]加载GLB点云...", advance=0)
            points, colors = self.load_point_cloud_from_glb()

        if progress and task_id is not None:
            progress.update(task_id, advance=20)

        # 2. 下采样 (10%)
        if self.downsample_ratio < 1.0:
            if progress and task_id is not None:
                progress.update(task_id, description=f"[cyan]点云下采样 ({self.downsample_ratio*100:.0f}%)...")

            points, colors = self._downsample_points(points, colors, self.downsample_ratio)
            logger.info(f"Downsampled to {len(points)} points ({self.downsample_ratio*100:.1f}%)")

        if progress and task_id is not None:
            progress.update(task_id, advance=10)

        # 3. 分配全局ID和帧索引 (60%)
        if progress and task_id is not None:
            progress.update(task_id, description="[cyan]分配全局ID和帧索引...")

        global_ids, confidences, frame_indices = self.assign_global_ids_to_points(points)

        if progress and task_id is not None:
            progress.update(task_id, advance=60)

        # 4. 保存点云缓存 (5%)
        if progress and task_id is not None:
            progress.update(task_id, description="[cyan]保存点云缓存...")

        pcd_cache_path = self.output_dir / "pcd_gid.npz"
        np.savez_compressed(
            pcd_cache_path,
            points=points,
            colors=colors if colors is not None else np.ones((len(points), 3)) * 0.8,
            global_ids=global_ids,
            confidences=confidences,
            frame_indices=frame_indices,  # 新增：保存帧索引
        )
        logger.info(f"Saved point cloud cache: {pcd_cache_path}")

        if progress and task_id is not None:
            progress.update(task_id, advance=5)

        # 5. 构建并保存索引 (5%)
        if progress and task_id is not None:
            progress.update(task_id, description="[cyan]构建全局ID索引...")

        index = self.build_global_object_index()
        index_cache_path = self.output_dir / "global_object_index.json"
        with index_cache_path.open('w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved index cache: {index_cache_path}")

        if progress and task_id is not None:
            progress.update(task_id, advance=5)

        # 6. 保存元数据
        build_time = time.time() - build_start_time
        metadata = CacheValidator.create_metadata(
            global_mapping_path=self.global_mapping_path,
            reconstruction_path=self.reconstruction_path,
            config={
                'downsample_ratio': self.downsample_ratio,
                'points_source': self.points_source,
                'gpu_enabled': self.enable_gpu,
            },
            statistics={
                'total_points': len(points),
                'points_with_gid': int(np.sum(global_ids >= 0)),
                'unique_global_ids': len(np.unique(global_ids[global_ids >= 0])),
                'build_time_seconds': build_time,
            }
        )

        metadata_path = self.output_dir / "cache_metadata.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved cache metadata: {metadata_path}")

        logger.info(f"Cache build complete! (耗时: {build_time:.1f}s)")
        return pcd_cache_path, index_cache_path


class ViserInteractive3DViewer:
    """
    Viser交互式3D可视化系统

    基于viser实现点云可视化、点拾取、全局ID显示
    """

    def __init__(
        self,
        pcd_cache_path: str,
        index_cache_path: str,
        image_dir: str,
        global_mapping_path: str,
        port: int = 8080,
    ):
        """
        初始化ViserInteractive3DViewer

        Args:
            pcd_cache_path: pcd_gid.npz 缓存路径
            index_cache_path: global_object_index.json 缓存路径
            image_dir: 原始图像目录
            global_mapping_path: global_mapping.json 路径
            port: Viser服务端口
        """
        self.pcd_cache_path = Path(pcd_cache_path)
        self.index_cache_path = Path(index_cache_path)
        self.image_dir = Path(image_dir)
        self.global_mapping_path = Path(global_mapping_path)
        self.port = port

        # Viser服务器（延迟初始化）
        self.server = None

        # KD-Tree用于点拾取（延迟构建）
        self.kdtree = None
        self.points_centered = None

        # 拾取状态
        self.pick_mode_enabled = False
        self.last_picked_gid = None

        # 加载数据（必须在初始化变量之后）
        self.load_cache()
        self.mapper = GlobalIDMapper(str(self.global_mapping_path))

        logger.info(f"ViserInteractive3DViewer initialized on port {port}")

    def load_cache(self) -> None:
        """加载缓存数据"""
        logger.info(f"Loading point cloud cache: {self.pcd_cache_path}")
        data = np.load(self.pcd_cache_path)
        self.points = data['points']
        self.colors = data['colors']
        self.global_ids = data['global_ids']
        self.confidences = data['confidences']
        self.frame_indices = data['frame_indices'] if 'frame_indices' in data else None

        logger.info(f"Loaded {len(self.points)} points")

        if self.frame_indices is not None:
            unique_frames = len(np.unique(self.frame_indices))
            logger.info(f"Loaded frame_indices: {unique_frames} unique frames")
        else:
            logger.warning("Cache缺少frame_indices，帧选择器功能将禁用")

        # 颜色格式标准化为uint8（viser期望0-255），兼容float[0,1]
        try:
            if self.colors.dtype != np.uint8:
                col = self.colors
                # 兼容float [0,1] 或 [0,255]
                maxv = float(np.nanmax(col)) if np.size(col) > 0 else 1.0
                if maxv <= 1.0:
                    col = (col * 255.0)
                col = np.clip(col, 0, 255).astype(np.uint8)
                self.colors = col
                logger.info("Converted colors to uint8 for viewer")
        except Exception as e:
            logger.warning(f"Failed to normalize color dtype: {e}")

        # 计算点云中心并缓存中心化后的坐标（避免在启动服务器时重复计算）
        self.scene_center = np.mean(self.points, axis=0)
        self.points_centered = self.points - self.scene_center
        logger.info(f"Computed scene center: {self.scene_center}")

        logger.info(f"Loading index cache: {self.index_cache_path}")
        with self.index_cache_path.open('r', encoding='utf-8') as f:
            self.index = json.load(f)
        logger.info(f"Loaded {len(self.index)} global IDs in index")

    def build_kdtree_for_picking(self, points: np.ndarray) -> None:
        """构建KD-Tree用于快速点拾取（使用统一工具函数）"""
        self.kdtree = build_kdtree(points)

    def pick_point_by_sphere(
        self,
        click_position: np.ndarray,
        radius: float = 0.05
    ) -> Optional[int]:
        """
        通过球形区域拾取点（优化版：单次查询）

        策略：先找最近点，再验证是否在半径内

        Args:
            click_position: 点击位置 (3D坐标)
            radius: 拾取半径

        Returns:
            拾取到的点的索引，如果半径内无点则返回None
        """
        if self.kdtree is None:
            logger.warning("KD-Tree not built yet")
            return None

        # 单次查询：找到最近的点及其距离
        distance, nearest_idx = self.kdtree.query(click_position, k=1)

        # 确保返回标量（scipy可能返回0维数组）
        distance = float(distance) if hasattr(distance, '__float__') else distance
        nearest_idx = int(nearest_idx) if hasattr(nearest_idx, '__int__') else nearest_idx

        logger.info(f"Nearest point: idx={nearest_idx}, distance={distance:.4f}, radius={radius:.4f}")

        # 验证是否在半径内
        if distance > radius:
            logger.warning(f"Point NOT picked: distance {distance:.4f} > radius {radius:.4f}")
            logger.warning(f"Suggestion: Increase 'Pick Radius' slider to at least {distance * 1.2:.2f}")
            return None

        logger.info(f"Successfully picked point {nearest_idx} at distance {distance:.4f}")
        return nearest_idx

    def get_global_id_from_point_index(self, point_idx: int) -> Optional[str]:
        """
        根据点索引获取全局ID

        Args:
            point_idx: 点索引

        Returns:
            全局ID字符串，如果无效则返回None
        """
        if point_idx is None or point_idx >= len(self.global_ids):
            return None

        gid = int(self.global_ids[point_idx])
        return str(gid)

    def _load_camera_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]]:
        """
        从predictions.npz加载相机数据（参考demo_viser.py）

        Returns:
            (extrinsics, intrinsics, images, image_ids) 或 None（如果数据不可用）
            - extrinsics: (S, 3, 4) 相机外参（cam2world）
            - intrinsics: (S, 3, 3) 相机内参
            - images: (S, H, W, 3) 输入图像，uint8格式
            - image_ids: List[int] 图像ID列表
        """
        # 定位 predictions.npz：优先从缓存元数据推断，其次从global_mapping路径推断
        cache_dir = self.pcd_cache_path.parent
        metadata_path = cache_dir / "cache_metadata.json"
        predictions_path = None
        dataset_root = None

        try:
            if metadata_path.exists():
                with metadata_path.open('r', encoding='utf-8') as f:
                    meta = json.load(f)
                recon_entry = meta.get('source_files', {}).get('reconstruction.glb', {})
                recon_path = recon_entry.get('path')
                if recon_path:
                    dataset_root = Path(recon_path).parent
                    predictions_path = dataset_root / "vggt_cache" / "predictions.npz"
        except Exception as e:
            logger.warning(f"Failed to parse cache metadata for predictions path: {e}")

        # 回退：基于global_mapping路径推断（.../Output/<scene>/dedup_detections/global_mapping.json → Output/<scene>/vggt_cache/predictions.npz）
        if predictions_path is None:
            try:
                gm = self.global_mapping_path.resolve()
                dataset_root = gm.parent.parent  # dedup_detections 的父目录
                predictions_path = dataset_root / "vggt_cache" / "predictions.npz"
            except Exception:
                pass

        # 最后的回退（兼容旧逻辑）：从cache目录上溯两级
        if predictions_path is None:
            dataset_root = self.pcd_cache_path.parent.parent
            predictions_path = dataset_root / "vggt_cache" / "predictions.npz"

        if not predictions_path.exists():
            logger.warning(f"相机数据不可用: {predictions_path} 不存在")
            logger.warning("相机可视化功能将被禁用")
            logger.warning("解决方案: 运行main.py重建时会自动保存predictions.npz")
            return None

        try:
            logger.info(f"加载相机数据: {predictions_path}")
            data = np.load(predictions_path)

            # 检查必需字段
            required_fields = ['extrinsic', 'intrinsic', 'images']
            missing_fields = [f for f in required_fields if f not in data]
            if missing_fields:
                logger.warning(f"predictions.npz缺少字段: {missing_fields}")
                logger.warning("相机可视化功能将被禁用")
                return None

            extrinsics = data['extrinsic']  # (S, 3, 4)
            intrinsics = data['intrinsic']  # (S, 3, 3)
            images = data['images']         # (S, 3, H, W) 或 (S, H, W, 3)
            image_ids = data['image_ids'].tolist() if 'image_ids' in data else None

            # 转换images格式: (S, 3, H, W) → (S, H, W, 3)
            if images.ndim == 4 and images.shape[1] == 3:
                images = images.transpose(0, 2, 3, 1)  # CHW → HWC

            # 转换为uint8（如果是[0,1]浮点数）
            if images.dtype == np.float32 or images.dtype == np.float64:
                images = (images * 255).astype(np.uint8)

            S = extrinsics.shape[0]

            # 生成默认image_ids
            if image_ids is None:
                image_ids = list(range(1, S + 1))
                logger.warning(f"predictions.npz缺少image_ids，使用默认值: {image_ids}")

            # 验证数据一致性
            if extrinsics.shape[0] != S or intrinsics.shape[0] != S or images.shape[0] != S:
                logger.warning(f"相机数据维度不一致: extrinsics={extrinsics.shape}, "
                              f"intrinsics={intrinsics.shape}, images={images.shape}")
                return None

            # 转换外参为cam2world（如果是world2cam）
            # VGGT输出的是world2cam，需要转换为cam2world（参考demo_viser.py line 102-104）
            from vggt.utils.geometry import closed_form_inverse_se3
            cam2world_4x4 = closed_form_inverse_se3(extrinsics)  # (S, 4, 4)
            cam2world_3x4 = cam2world_4x4[:, :3, :]  # (S, 3, 4)

            # 场景中心化对齐（参考demo_viser.py line 109）
            cam2world_3x4[..., -1] -= self.scene_center

            logger.info(f"成功加载相机数据: {S}个相机")
            logger.info(f"   图像分辨率: {images.shape[1]}x{images.shape[2]}")
            logger.info(f"   Image IDs: {image_ids}")

            return cam2world_3x4, intrinsics, images, image_ids

        except Exception as e:
            logger.warning(f"加载相机数据失败: {e}")
            logger.warning("相机可视化功能将被禁用")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _add_camera_visualization(
        self,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        images: np.ndarray,
        image_ids: List[int]
    ) -> Tuple[List, List]:
        """
        添加相机位姿可视化（Frames + Frustums）

        参考demo_viser.py的visualize_frames函数实现

        Args:
            extrinsics: (S, 3, 4) 相机外参（cam2world，已中心化）
            intrinsics: (S, 3, 3) 相机内参
            images: (S, H, W, 3) 输入图像，uint8
            image_ids: List[int] 图像ID列表

        Returns:
            (frames, frustums): 相机框架和视锥列表
        """
        import viser.transforms as viser_tf

        frames = []
        frustums = []
        S = extrinsics.shape[0]

        logger.info(f"添加{S}个相机可视化...")

        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]  # (3, 4)
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # 1. 添加坐标轴（Frame Axis）
            frame_axis = self.server.scene.add_frame(
                f"camera_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,        # 参考demo_viser: 0.05，这里放大一点便于观察
                axes_radius=0.005,
                origin_radius=0.005,
            )
            frames.append(frame_axis)

            # 2. 添加相机视锥（Frustum）
            img = images[img_id]  # (H, W, 3) uint8
            h, w = img.shape[:2]

            # 从内参计算FOV（参考demo_viser.py line 189-190）
            intrinsic = intrinsics[img_id]
            fy = intrinsic[1, 1]  # 焦距
            fov = 2 * np.arctan2(h / 2, fy)

            frustum = self.server.scene.add_camera_frustum(
                f"camera_{img_id}/frustum",
                fov=float(fov),
                aspect=w / h,
                scale=0.1,  # 视锥大小
                image=img,
                line_width=1.0,
            )
            frustums.append(frustum)

            # 3. 添加Frustum点击跳转视角回调（参考demo_viser.py line 157-162）
            # 使用闭包捕获frame_axis
            @frustum.on_click
            def _(event, frame=frame_axis):
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        logger.info(f"相机可视化完成: {len(frames)}个frames, {len(frustums)}个frustums")
        return frames, frustums

    def format_global_id_info(self, gid_str: str) -> str:
        """
        格式化全局ID信息为Markdown文本（支持自动换行）

        Args:
            gid_str: 全局ID字符串

        Returns:
            格式化的Markdown信息文本
        """
        if gid_str not in self.index:
            return f"**Global ID {gid_str}**: No info available"

        info = self.index[gid_str]

        # 构建 Markdown 格式的输出
        md_lines = [
            f"### Global ID: {gid_str}",
            "",
            f"**出现的图片 (Images)**:  ",
            f"{', '.join(map(str, info['images']))}",
            "",
            f"**包含的物体 (Objects)**:  ",
            f"{', '.join(map(str, info['objects'][:20]))}{'...' if len(info['objects']) > 20 else ''}",
            "",
            f"**状态统计**:  ",
            f"- Active: {info['active_count']}  ",
            f"- Removed: {info['removed_count']}",
            "",
        ]
        return "\n".join(md_lines)


    def start_viser_server(self) -> None:
        """
        启动Viser可视化服务器

        服务器会阻塞当前线程直到用户按 Ctrl+C
        如需后台运行，请在命令行使用 & 或 nohup:
          uv run interactive_3d_viewer.py ... &
          nohup uv run interactive_3d_viewer.py ... > viewer.log 2>&1 &
        """
        global viser
        if viser is None:
            import viser as viser_module
            viser = viser_module

        logger.info(f"Starting Viser server on port {self.port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=self.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # 使用缓存的中心化点云（已在load_cache中计算）
        points_centered = self.points_centered

        # 为所有连接的客户端设置初始相机位置（斜45度角查看，以便看到3D深度）
        # 计算合适的相机距离（基于点云最大跨度）
        points_span = np.max(points_centered, axis=0) - np.min(points_centered, axis=0)
        max_span = np.max(points_span)
        camera_distance = max_span * 2.0  # 相机距离为最大跨度的2倍

        # 设置相机初始位置：斜上方45度角
        # 注意：position参数在viser中表示相机的世界坐标位置
        initial_camera_position = (
            float(camera_distance * 0.7),  # X轴偏移
            float(camera_distance * 0.5),  # Y轴偏移
            float(camera_distance * 0.7)   # Z轴偏移
        )

        # 依据场景尺寸设置拾取半径范围（避免默认半径过大遮挡点云）
        safe_span = float(max(max_span, 1e-6))
        # 默认非常小，避免遮挡：默认 0.05% 跨度，范围 0.02%–2%
        pick_radius_min = max(safe_span * 0.0002, 1e-9)   # 0.02% 场景跨度
        pick_radius_max = safe_span * 0.02                 # 2% 场景跨度
        pick_radius_init = safe_span * 0.0005              # 0.05% 场景跨度
        pick_radius_step = max((pick_radius_max - pick_radius_min) / 100.0, 1e-6)

        # 为所有客户端设置初始视角
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            """新客户端连接时设置初始相机位置"""
            try:
                client.camera.position = initial_camera_position
                logger.info(f"Client connected. Set initial camera position: {initial_camera_position}")
            except Exception as e:
                logger.warning(f"Failed to set initial camera position: {e}")

        # ===== 旋转控制（Yaw/Pitch/Roll） & 初始旋转 =====
        def _deg2rad(x: float) -> float:
            return float(np.deg2rad(x))

        def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
            """Z(=yaw) · Y(=pitch) · X(=roll) 旋转顺序，单位：度"""
            cy, sy = np.cos(_deg2rad(yaw_deg)), np.sin(_deg2rad(yaw_deg))
            cp, sp = np.cos(_deg2rad(pitch_deg)), np.sin(_deg2rad(pitch_deg))
            cr, sr = np.cos(_deg2rad(roll_deg)), np.sin(_deg2rad(roll_deg))
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
            return (Rz @ Ry @ Rx)

        def _apply_rot(P: np.ndarray, Rm: np.ndarray) -> np.ndarray:
            # P: (N,3), Rm: (3,3) → (N,3)
            return (Rm @ P.T).T

        R_cur = np.eye(3, dtype=float)
        rotated_points = points_centered  # 初始无旋转

        # 构建KD-Tree用于点拾取（基于当前旋转后的点）
        self.build_kdtree_for_picking(rotated_points)

        # 使用原始RGB颜色（从GLB文件加载的真实颜色）
        # 不再根据Global ID重新着色
        colors_original = self.colors  # 已经是uint8格式 (N, 3)

        # 保存unique_gids用于dropdown选项（过滤无效ID，如 -1）
        try:
            valid_mask_ids = self.global_ids >= 0
            unique_gids = np.unique(self.global_ids[valid_mask_ids]) if np.any(valid_mask_ids) else np.array([], dtype=int)
        except Exception:
            unique_gids = np.unique(self.global_ids)

        # 为显示层添加稳定的随机采样掩码（固定种子，保证不同筛选下采样一致）
        rng = np.random.default_rng(42)
        display_rand = rng.random(len(self.points))  # [0,1) 每点一个随机数

        # ========== GUI Controls（参考demo_viser.py的简洁设计）==========

        # 主要控件（始终可见，参考demo_viser只保留核心功能）
        gui_conf_threshold = self.server.gui.add_slider(
            "Confidence %", min=0, max=100, step=0.1, initial_value=15
        )
        # 显示相机位置（世界坐标）
        try:
            cam_init = (
                float(initial_camera_position[0]),
                float(initial_camera_position[1]),
                float(initial_camera_position[2]),
            )
            cam_init_str = f"{cam_init[0]:.3f}, {cam_init[1]:.3f}, {cam_init[2]:.3f}"
        except Exception:
            cam_init_str = "0.000, 0.000, 0.000"
        gui_camera_pos = self.server.gui.add_text("Camera (world)", initial_value=cam_init_str)

        gui_point_size = self.server.gui.add_slider(
            "Point Size",
            min=0.0001,
            max=0.005,
            step=0.0001,
            initial_value=0.0006,  # 参考demo_viser使用0.001
        )

        gui_selected_id = self.server.gui.add_dropdown(
            "Show Global ID",
            options=["All"] + [str(gid) for gid in sorted(unique_gids)],
            initial_value="All"
        )

        # Selected ID Info - 使用多个text字段代替markdown（解决刷新问题）
        with self.server.gui.add_folder("📋 Selected ID Info", expand_by_default=True):
            gui_info_gid = self.server.gui.add_text(
                "Global ID",
                initial_value="(None)",
                disabled=True
            )
            gui_info_images = self.server.gui.add_text(
                "Images",
                initial_value="-",
                disabled=True
            )
            gui_info_objects = self.server.gui.add_text(
                "Objects (first 10)",
                initial_value="-",
                disabled=True
            )
            gui_info_status = self.server.gui.add_text(
                "Status (Active/Removed)",
                initial_value="-",
                disabled=True
            )
            self.server.gui.add_markdown(
                "*Tip: Enable Pick Mode and click on point cloud, or select from dropdown above.*"
            )

        # Point Picking（折叠，包含拾取相关控件）
        with self.server.gui.add_folder("🔍 Point Picking"):
            gui_pick_mode = self.server.gui.add_checkbox(
                "Enable Pick Mode",
                initial_value=False
            )

            gui_pick_radius = self.server.gui.add_slider(
                "Pick Radius",
                min=float(pick_radius_min),
                max=float(pick_radius_max),
                step=float(pick_radius_step),
                initial_value=float(pick_radius_init)
            )

            gui_show_pick_sphere = self.server.gui.add_checkbox(
                "Show Pick Sphere",
                initial_value=False
            )

        # Advanced Options（折叠，默认关闭）
        with self.server.gui.add_folder("⚙️ Advanced Options", expand_by_default=False):
            gui_sampling = self.server.gui.add_slider(
                "Display Sample %",
                min=10,
                max=100,
                step=10,
                initial_value=100,
            )

            gui_rotate_mode = self.server.gui.add_checkbox(
                "Rotate Model (Shift+Drag)",
                initial_value=False
            )

            gui_hide_unknown = self.server.gui.add_checkbox(
                "Hide Unknown IDs (-1)",
                initial_value=True
            )

            # 帮助信息
            self.server.gui.add_markdown(
                """**Interaction Guide**

- **Left-drag**: Rotate view
- **Right-drag**: Pan view
- **Scroll**: Zoom in/out
- **Shift+Drag**: Rotate model (when enabled)
                """
            )

        # ========== 相机可视化 ==========
        # 加载相机数据并创建可视化（参考demo_viser.py）
        camera_data = self._load_camera_data()
        camera_frames = []
        camera_frustums = []

        if camera_data is not None:
            extrinsics, intrinsics, images, image_ids = camera_data
            camera_frames, camera_frustums = self._add_camera_visualization(
                extrinsics, intrinsics, images, image_ids
            )

            # 添加相机显示开关（参考demo_viser.py line 115）
            gui_show_cameras = self.server.gui.add_checkbox(
                "Show Cameras",
                initial_value=True
            )

            @gui_show_cameras.on_update
            def _(_):
                """切换相机可见性（参考demo_viser.py line 227-233）"""
                for frame in camera_frames:
                    frame.visible = gui_show_cameras.value
                for frustum in camera_frustums:
                    frustum.visible = gui_show_cameras.value
        else:
            logger.info("相机可视化未启用（predictions.npz不可用）")

        # ========== 帧选择器 ==========
        # 添加帧选择器GUI（参考demo_viser.py line 122-124）
        gui_frame_selector = None
        if self.frame_indices is not None:
            unique_frame_ids = sorted(np.unique(self.frame_indices).tolist())
            num_frames = len(unique_frame_ids)

            gui_frame_selector = self.server.gui.add_dropdown(
                "Show Points from Frame",
                options=["All"] + [str(fid) for fid in unique_frame_ids],
                initial_value="All"
            )
            logger.info(f"帧选择器已启用: {num_frames}个帧可选")
        else:
            logger.info("帧选择器未启用（frame_indices不可用）")

        # 初始点云（占位，统一走 update_point_cloud 刷新）
        point_cloud = self.server.scene.add_point_cloud(
            name="sku_pcd",
            points=rotated_points[:1],
            colors=colors_original[:1],
            point_size=gui_point_size.value,
            point_shape="circle",
        )

        # 拾取球体指示器（初始隐藏）
        # 注意：viser的add_icosphere不支持opacity参数，使用默认透明度
        pick_sphere = self.server.scene.add_icosphere(
            name="pick_sphere",
            radius=gui_pick_radius.value,
            color=(255, 255, 0),  # 黄色
            position=(0.0, 0.0, 0.0),
            visible=False,
        )

        # ========== 智能点拾取系统 ==========
        # 用户体验：启用Pick Mode后，直接在3D视图中点击点云即可查询Global ID

        # 辅助函数：更新ID信息显示
        def update_id_info_display(gid_str: str) -> None:
            """更新Selected ID Info的所有字段"""
            if gid_str not in self.index:
                gui_info_gid.value = f"ID {gid_str} (No data)"
                gui_info_images.value = "-"
                gui_info_objects.value = "-"
                gui_info_status.value = "-"
                return

            info = self.index[gid_str]
            gui_info_gid.value = gid_str
            gui_info_images.value = ', '.join(map(str, info['images']))
            gui_info_objects.value = ', '.join(map(str, info['objects'][:10])) + ('...' if len(info['objects']) > 10 else '')
            gui_info_status.value = f"{info['active_count']} / {info['removed_count']}"

        def clear_id_info_display() -> None:
            """清空ID信息显示"""
            gui_info_gid.value = "(None)"
            gui_info_images.value = "-"
            gui_info_objects.value = "-"
            gui_info_status.value = "-"

        # 统一的点拾取处理函数
        def handle_point_pick(click_pos: np.ndarray, source: str = "click") -> None:
            """
            统一的点拾取处理逻辑

            Args:
                click_pos: 点击位置（已中心化的坐标）
                source: 触发来源（"click" 或 "manual"）
            """
            if not gui_pick_mode.value:
                # Pick Mode关闭时，不覆盖已选中的信息
                return

            # 显示拾取球体指示器
            pick_sphere.position = tuple(float(x) for x in click_pos)
            pick_sphere.radius = gui_pick_radius.value
            pick_sphere.visible = bool(gui_show_pick_sphere.value)

            # 执行点拾取
            point_idx = self.pick_point_by_sphere(click_pos, radius=gui_pick_radius.value)

            if point_idx is not None:
                gid_str = self.get_global_id_from_point_index(point_idx)
                if gid_str is None or gid_str.strip() == "":
                    clear_id_info_display()
                    return

                # 过滤无效/未知ID（如 -1 或不在索引中）
                try:
                    gid_int = int(gid_str)
                except Exception:
                    gid_int = -1

                if gid_int < 0:
                    clear_id_info_display()
                    return

                self.last_picked_gid = gid_str

                # 更新信息显示
                logger.info(f"Picked Global ID: {gid_str}")
                update_id_info_display(gid_str)

                # 自动选择该全局ID并高亮显示
                gui_selected_id.value = gid_str
                update_point_cloud()
            else:
                clear_id_info_display()

        # 统一的射线拾取逻辑（用于 click / down / pointer_down 事件）
        def _try_ray_pick(event: viser.ScenePointerEvent) -> None:
            if not gui_pick_mode.value:
                return
            # 右键/中键保留给平移/默认交互
            try:
                btn = None
                if hasattr(event, 'button') and event.button is not None:
                    btn = str(event.button).lower()
                elif hasattr(event, 'mouse_button') and event.mouse_button is not None:
                    btn = str(event.mouse_button).lower()
                if btn and (('right' in btn) or ('middle' in btn)):
                    return
            except Exception:
                pass
            # 如果正在进行模型旋转（需要按住Shift且开启开关），不要触发拾取
            if gui_rotate_mode.value and _shift_held(event):
                return

            ray_origin = np.array(event.ray_origin)
            ray_direction = np.array(event.ray_direction) if hasattr(event, 'ray_direction') and event.ray_direction is not None else None

            try:
                gui_camera_pos.value = f"{ray_origin[0]:.3f}, {ray_origin[1]:.3f}, {ray_origin[2]:.3f}"
            except Exception:
                pass

            if ray_direction is None:
                logger.warning("No ray_direction available; skip pick")
                return

            # 射线追踪采样
            ray_dir_norm = ray_direction / (np.linalg.norm(ray_direction) + 1e-10)
            depth_estimate = np.linalg.norm(ray_origin)

            best_point_idx = None
            best_distance = float('inf')

            for t in np.linspace(0, depth_estimate * 2, 20):
                ray_point = ray_origin + ray_dir_norm * t
                if self.kdtree is not None:
                    distance, nearest_idx = self.kdtree.query(ray_point, k=1)
                    if distance < best_distance and distance < gui_pick_radius.value:
                        best_distance = distance
                        best_point_idx = int(nearest_idx)

            if best_point_idx is not None:
                click_pos = rotated_points[best_point_idx]
                handle_point_pick(click_pos, source="Ray-Tracing")
            else:
                clear_id_info_display()

        # 注册场景点击事件（以及 pointer_down 作为兼容兜底）
        @self.server.scene.on_pointer_event(event_type="click")
        def handle_scene_click(event: viser.ScenePointerEvent) -> None:
            _try_ray_pick(event)

        # 尝试注册pointer_down事件（某些viser版本支持）
        try:
            @self.server.scene.on_pointer_event(event_type="pointer_down")
            def handle_scene_pointer_down(event: viser.ScenePointerEvent) -> None:
                _try_ray_pick(event)
        except Exception:
            pass  # 不支持pointer_down，使用默认click事件即可

        # 直接绑定到点云对象的点击事件（更可靠的拾取入口）
        try:
            @point_cloud.on_click
            def _on_pcd_click(event: viser.ScenePointerEvent) -> None:
                logger.info("Point cloud clicked; attempting pick")
                _try_ray_pick(event)
        except Exception:
            pass

        # ========== Shift+拖拽旋转（Arcball） ==========
        arcball_active = False
        arcball_prev = None  # type: Optional[np.ndarray]

        def _ray_sphere_intersect(o: np.ndarray, d: np.ndarray, R: float) -> Optional[np.ndarray]:
            b = 2.0 * float(np.dot(o, d))
            c = float(np.dot(o, o) - R * R)
            disc = b * b - 4.0 * c
            if disc < 0:
                return None
            sqrt_disc = float(np.sqrt(disc))
            t1 = (-b - sqrt_disc) / 2.0
            t2 = (-b + sqrt_disc) / 2.0
            t = t1 if t1 > 0 else (t2 if t2 > 0 else None)
            if t is None:
                return None
            return o + t * d

        def _rot_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a_n = a / (np.linalg.norm(a) + 1e-9)
            b_n = b / (np.linalg.norm(b) + 1e-9)
            v = np.cross(a_n, b_n)
            s = np.linalg.norm(v)
            c = float(np.dot(a_n, b_n))
            if s < 1e-9:
                if c > 0:
                    return np.eye(3)
                axis = np.array([1.0, 0.0, 0.0]) if abs(a_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                v = np.cross(a_n, axis)
                v = v / (np.linalg.norm(v) + 1e-9)
                K = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]], dtype=float)
                return np.eye(3) + 2 * K @ K
            vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]], dtype=float)
            return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

        def _shift_held(evt: viser.ScenePointerEvent) -> bool:
            try:
                if hasattr(evt, 'shift') and bool(evt.shift):
                    return True
                if hasattr(evt, 'shift_key') and bool(evt.shift_key):
                    return True
                if hasattr(evt, 'modifiers') and evt.modifiers is not None:
                    mods = str(evt.modifiers).lower()
                    return ('shift' in mods) or ('mod.shift' in mods)
            except Exception:
                return False
            return False

        # Pointer down → 捕获初始点
        try:
            @self.server.scene.on_pointer_event(event_type="down")
            def _on_down(evt: viser.ScenePointerEvent) -> None:
                nonlocal arcball_active, arcball_prev
                # 更新相机位置显示
                try:
                    o = np.array(evt.ray_origin)
                    gui_camera_pos.value = f"{o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}"
                except Exception:
                    pass
                # 右/中键交给默认相机平移
                try:
                    btn = None
                    if hasattr(evt, 'button') and evt.button is not None:
                        btn = str(evt.button).lower()
                    elif hasattr(evt, 'mouse_button') and evt.mouse_button is not None:
                        btn = str(evt.mouse_button).lower()
                    if btn and (('right' in btn) or ('middle' in btn)):
                        return
                except Exception:
                    pass
                # 仅当开启开关且按住Shift时才启用模型旋转，不干扰默认相机拖拽
                if not (gui_rotate_mode.value and _shift_held(evt)):
                    return
                o_world = np.array(evt.ray_origin)
                d_world = np.array(evt.ray_direction)
                o = o_world - self.scene_center
                d = d_world / (np.linalg.norm(d_world) + 1e-9)
                R_sphere = float(np.linalg.norm(o)) * 0.6
                p = _ray_sphere_intersect(o, d, R_sphere)
                if p is None:
                    return
                arcball_active = True
                arcball_prev = p
        except Exception:
            pass

        # 兼容: 有些版本的 viser 使用 pointer_down/move/up 事件名
        try:
            @self.server.scene.on_pointer_event(event_type="pointer_down")
            def _on_down_pd(evt: viser.ScenePointerEvent) -> None:
                nonlocal arcball_active, arcball_prev
                try:
                    o = np.array(evt.ray_origin)
                    gui_camera_pos.value = f"{o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}"
                except Exception:
                    pass
                try:
                    btn = None
                    if hasattr(evt, 'button') and evt.button is not None:
                        btn = str(evt.button).lower()
                    elif hasattr(evt, 'mouse_button') and evt.mouse_button is not None:
                        btn = str(evt.mouse_button).lower()
                    if btn and (('right' in btn) or ('middle' in btn)):
                        return
                except Exception:
                    pass
                if not (gui_rotate_mode.value and _shift_held(evt)):
                    return
                o_world = np.array(evt.ray_origin)
                d_world = np.array(evt.ray_direction)
                o = o_world - self.scene_center
                d = d_world / (np.linalg.norm(d_world) + 1e-9)
                R_sphere = float(np.linalg.norm(o)) * 0.6
                p = _ray_sphere_intersect(o, d, R_sphere)
                if p is None:
                    return
                arcball_active = True
                arcball_prev = p
        except Exception:
            pass

        # Pointer move → 增量旋转
        try:
            @self.server.scene.on_pointer_event(event_type="move")
            def _on_move(evt: viser.ScenePointerEvent) -> None:
                nonlocal R_cur, rotated_points, arcball_active, arcball_prev
                try:
                    o = np.array(evt.ray_origin)
                    gui_camera_pos.value = f"{o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}"
                except Exception:
                    pass
                # 右/中键交给默认相机平移
                try:
                    btn = None
                    if hasattr(evt, 'button') and evt.button is not None:
                        btn = str(evt.button).lower()
                    elif hasattr(evt, 'mouse_button') and evt.mouse_button is not None:
                        btn = str(evt.mouse_button).lower()
                    if btn and (('right' in btn) or ('middle' in btn)):
                        return
                except Exception:
                    pass
                if not arcball_active:
                    return
                if not (gui_rotate_mode.value and _shift_held(evt)):
                    return
                o_world = np.array(evt.ray_origin)
                d_world = np.array(evt.ray_direction)
                o = o_world - self.scene_center
                d = d_world / (np.linalg.norm(d_world) + 1e-9)
                R_sphere = float(np.linalg.norm(o)) * 0.6
                p = _ray_sphere_intersect(o, d, R_sphere)
                if p is None or arcball_prev is None:
                    return
                R_delta = _rot_from_to(arcball_prev, p)
                R_cur = R_delta @ R_cur
                rotated_points = _apply_rot(points_centered, R_cur)
                self.build_kdtree_for_picking(rotated_points)
                update_point_cloud()
                arcball_prev = p
        except Exception:
            pass

        try:
            @self.server.scene.on_pointer_event(event_type="pointer_move")
            def _on_move_pm(evt: viser.ScenePointerEvent) -> None:
                nonlocal R_cur, rotated_points, arcball_active, arcball_prev
                try:
                    o = np.array(evt.ray_origin)
                    gui_camera_pos.value = f"{o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}"
                except Exception:
                    pass
                try:
                    btn = None
                    if hasattr(evt, 'button') and evt.button is not None:
                        btn = str(evt.button).lower()
                    elif hasattr(evt, 'mouse_button') and evt.mouse_button is not None:
                        btn = str(evt.mouse_button).lower()
                    if btn and (('right' in btn) or ('middle' in btn)):
                        return
                except Exception:
                    pass
                if not arcball_active:
                    return
                if not (gui_rotate_mode.value and _shift_held(evt)):
                    return
                o_world = np.array(evt.ray_origin)
                d_world = np.array(evt.ray_direction)
                o = o_world - self.scene_center
                d = d_world / (np.linalg.norm(d_world) + 1e-9)
                R_sphere = float(np.linalg.norm(o)) * 0.6
                p = _ray_sphere_intersect(o, d, R_sphere)
                if p is None or arcball_prev is None:
                    return
                R_delta = _rot_from_to(arcball_prev, p)
                R_cur = R_delta @ R_cur
                rotated_points = _apply_rot(points_centered, R_cur)
                self.build_kdtree_for_picking(rotated_points)
                update_point_cloud()
                arcball_prev = p
        except Exception:
            pass

        # Pointer up → 结束旋转
        try:
            @self.server.scene.on_pointer_event(event_type="up")
            def _on_up(_: viser.ScenePointerEvent) -> None:
                nonlocal arcball_active, arcball_prev
                arcball_active = False
                arcball_prev = None
        except Exception:
            pass

        try:
            @self.server.scene.on_pointer_event(event_type="pointer_up")
            def _on_up_pu(_: viser.ScenePointerEvent) -> None:
                nonlocal arcball_active, arcball_prev
                arcball_active = False
                arcball_prev = None
        except Exception:
            pass

        @gui_pick_mode.on_update
        def _(_) -> None:
            """拾取模式切换"""
            # 进入拾取模式时，先隐藏拾取球，等待用户点击再显示
            if not gui_pick_mode.value:
                pick_sphere.visible = False

        @gui_pick_radius.on_update
        def _(_) -> None:
            """拾取半径更新"""
            pick_sphere.radius = gui_pick_radius.value

        @gui_show_pick_sphere.on_update
        def _(_) -> None:
            """显示/隐藏拾取球体"""
            if not gui_show_pick_sphere.value:
                pick_sphere.visible = False


        def update_point_cloud() -> None:
            """更新点云显示（参考demo_viser.py line 199-217）"""
            # 1. 置信度过滤（百分位数方式）
            threshold_val = np.percentile(self.confidences, gui_conf_threshold.value)
            conf_mask = self.confidences >= threshold_val

            # 2. Global ID过滤
            if gui_selected_id.value == "All":
                id_mask = np.ones(len(self.global_ids), dtype=bool)
            else:
                selected_gid = int(gui_selected_id.value)
                id_mask = self.global_ids == selected_gid

            # 3. 帧过滤（参考demo_viser.py line 209-213）
            if gui_frame_selector is not None and gui_frame_selector.value != "All":
                selected_frame = int(gui_frame_selector.value)
                frame_mask = self.frame_indices == selected_frame
            else:
                frame_mask = np.ones(len(self.global_ids), dtype=bool)

            # 4. 未知ID过滤（可选）
            if 'gui_hide_unknown' in locals() and gui_hide_unknown.value:
                known_mask = self.global_ids >= 0
            else:
                known_mask = np.ones(len(self.global_ids), dtype=bool)

            # 5. 采样率掩码（稳定随机）
            sample_mask = display_rand <= (gui_sampling.value / 100.0)

            # 6. 组合所有掩码（参考demo_viser.py line 215）
            combined_mask = conf_mask & id_mask & frame_mask & known_mask & sample_mask

            # 7. 更新点云（参考demo_viser.py line 216-217）
            point_cloud.points = rotated_points[combined_mask]
            point_cloud.colors = colors_original[combined_mask]

        # 初始化一次显示（替代上面的初始掩码计算）
        update_point_cloud()

        # ========== GUI回调函数 ==========
        @gui_conf_threshold.on_update
        def _(_) -> None:
            update_point_cloud()

        @gui_selected_id.on_update
        def _(_) -> None:
            update_point_cloud()
            # 更新信息文本
            if gui_selected_id.value != "All":
                logger.info(f"[Dropdown] Selected ID changed to: {gui_selected_id.value}")
                update_id_info_display(gui_selected_id.value)
            else:
                logger.info("[Dropdown] Reset to show all IDs")
                clear_id_info_display()

        @gui_sampling.on_update
        def _(_) -> None:
            update_point_cloud()

        # 未知ID过滤回调
        try:
            @gui_hide_unknown.on_update
            def _(_) -> None:
                update_point_cloud()
        except Exception:
            pass

        @gui_point_size.on_update
        def _(_) -> None:
            point_cloud.point_size = gui_point_size.value

        # 帧选择器回调（参考demo_viser.py line 223-225）
        if gui_frame_selector is not None:
            @gui_frame_selector.on_update
            def _(_) -> None:
                update_point_cloud()

        logger.info("Viser server started successfully")
        logger.info(f"Open http://localhost:{self.port} in your browser")
        logger.info("Press Ctrl+C to stop the server")

        # 保持服务器运行，直到用户按 Ctrl+C
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("\nServer stopped by user")
            logger.info("Goodbye!")


def main():
    """主程序入口 - 直接从GLB+JSON启动可视化"""
    parser = argparse.ArgumentParser(
        description="3D SKU 交互式可视化系统（基于Viser）",
        epilog="""
        示例用法（所有路径在同一数据集目录下）:
        cd code
        uv run python interactive_3d_viewer.py \\
            --global-mapping Output/floor_display2/dedup_detections/global_mapping.json \\
            --reconstruction Output/floor_display2/reconstruction.glb \\
            --image-dir ../imdata/floor_display2/images \\
            --detection-dir ../imdata/floor_display2/detections_results
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 核心参数（必需）
    parser.add_argument('--global-mapping', type=str, required=True,
                        help='global_mapping.json 路径（如 code/output_dedup/floor_display2/global_mapping.json）')
    parser.add_argument('--reconstruction', type=str, required=True,
                        help='GLB/PLY 3D重建文件路径')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='原始图像目录（如 ../imdata/floor_display2/images）')

    # 可选参数
    parser.add_argument('--detection-dir', type=str, default=None,
                        help='检测结果目录（如 ../imdata/floor_display2/detections_results，默认：自动推断）')
    parser.add_argument('--port', type=int, default=8080,
                        help='Viser服务端口 (默认: 8080)')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='点云下采样比例 0-1 (默认: 1.0 无下采样，可设为0.3-0.5以加快加载速度)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='缓存目录（默认: 临时目录，退出后自动清理）')
    parser.add_argument('--points-source', type=str, default='glb', choices=['glb', 'predictions'],
                        help='点云来源：glb（默认）或 predictions（更高几何精度）。')
    parser.add_argument('--no-progress', action='store_true',
                        help='禁用进度条')
    parser.add_argument('--no-gpu', action='store_true',
                        help='禁用GPU加速')
    parser.add_argument('--force-rebuild', action='store_true',
                        help='强制重建缓存（忽略现有缓存）')

    args = parser.parse_args()

    # 自动构建缓存 + 启动可视化（一步到位）
    import tempfile

    # 确定缓存目录
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"使用指定缓存目录: {cache_dir}")
    else:
        cache_dir = Path(tempfile.mkdtemp(prefix="3d_viewer_cache_"))
        logger.info(f"使用临时缓存目录: {cache_dir}")

    # 检查缓存有效性
    cache_valid = False
    if not args.force_rebuild:
        cache_valid = CacheValidator.is_cache_valid(
            cache_dir=cache_dir,
            global_mapping_path=Path(args.global_mapping),
            reconstruction_path=Path(args.reconstruction),
            config={'downsample_ratio': args.downsample, 'points_source': args.points_source}
        )

    if cache_valid:
        logger.info("使用现有缓存")
        pcd_path = cache_dir / "pcd_gid.npz"
        index_path = cache_dir / "global_object_index.json"
    else:
        # Step 1: 构建缓存
        logger.info("=" * 60)
        logger.info("步骤 1/2: 从GLB文件构建缓存...")
        logger.info("=" * 60)

        builder = DataCacheBuilder(
            global_mapping_path=args.global_mapping,
            reconstruction_path=args.reconstruction,
            output_dir=str(cache_dir),
            image_dir=args.image_dir,
            detection_dir=args.detection_dir,  # 新增：传递检测目录参数
            downsample_ratio=args.downsample,
            points_source=args.points_source,
            enable_progress=not args.no_progress,
            enable_gpu=not args.no_gpu,
        )

        # 使用进度条
        if builder.enable_progress and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                pcd_path, index_path = builder.build_cache(progress=progress)
        else:
            pcd_path, index_path = builder.build_cache()

    # Step 2: 启动可视化
    logger.info("=" * 60)
    logger.info("步骤 2/2: 启动Viser可视化...")
    logger.info("=" * 60)
    viewer = ViserInteractive3DViewer(
        pcd_cache_path=str(pcd_path),
        index_cache_path=str(index_path),
        image_dir=args.image_dir,
        global_mapping_path=args.global_mapping,
        port=args.port,
    )
    viewer.start_viser_server()


if __name__ == "__main__":
    main()
