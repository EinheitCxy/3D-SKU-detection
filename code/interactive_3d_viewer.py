"""
Interactive 3D Viewer - Viser交互式3D可视化系统

使用方式：
  uv run interactive_3d_viewer.py \
    --global-mapping <global_mapping.json> \
    --reconstruction <reconstruction.glb> \
    --image-dir <images_folder> \
    [--port 8080] \
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
        downsample_ratio: float = 0.1,
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
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self._rng = np.random.default_rng(seed)
        self.enable_progress = enable_progress and RICH_AVAILABLE
        self.enable_gpu = enable_gpu and (GPU_CONFIG.use_cupy or GPU_CONFIG.use_faiss_gpu)

        self.mapper = GlobalIDMapper(str(self.global_mapping_path))

        logger.info(f"DataCacheBuilder initialized:")
        logger.info(f"  - Global mapping: {self.global_mapping_path}")
        logger.info(f"  - Reconstruction: {self.reconstruction_path}")
        logger.info(f"  - Output directory: {self.output_dir}")
        logger.info(f"  - Downsample ratio: {downsample_ratio}")
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

    def assign_global_ids_to_points(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        优先从VGGT缓存(points_with_gid.npz)加载真实全局ID；否则从predictions.npz在线计算

        工作流程：
        1) 快速路径：<dataset>/vggt_cache/points_with_gid.npz
           - 直接将预先打好gid的3D点与GLB点云做最近邻映射，得到(gid, conf)
        2) 回退路径：<dataset>/vggt_cache/predictions.npz
           - 通过检测bbox→3D提取→最近邻映射的方式在线计算

        Args:
            points: 点云坐标 (N, 3)

        Returns:
            (global_ids, confidences): 全局ID数组 (N,) 和置信度 (N,)

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

                # 最近邻映射至GLB点
                distances, indices = nearest_neighbor_mapping(pre_points, points, k=1)
                final_gids = pre_gids[indices].astype(np.int32)
                final_confs = pre_confs[indices].astype(np.float32)

                logger.info(
                    f"完成快速映射: {len(np.unique(final_gids))} 个唯一ID，平均距离={float(np.mean(distances)):.4f}"
                )
                return final_gids, final_confs
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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            (global_ids, confidences)
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

        # 加载 global_mapping 和 detections
        global_mapping_path = dataset_root / "global_mapping.json"
        if not global_mapping_path.exists():
            logger.error(f"global_mapping.json 不存在: {global_mapping_path}")
            logger.error("请先运行去重流程生成 global_mapping.json")
            raise FileNotFoundError(f"global_mapping.json not found: {global_mapping_path}")

        with global_mapping_path.open('r') as f:
            global_mapping_data = json.load(f)

        logger.info(f"加载 global_mapping: {len(global_mapping_data)} 个全局ID")

        # 加载检测结果（带索引映射）
        detection_dir = dataset_root / "detections_results"
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

        # 构建 VGGT 裁剪/填充对齐的坐标变换（可选）
        vggt_transforms = None
        try:
            from utils.transforms import build_vggt_transforms

            search_dir = None
            if self.image_dir is not None and self.image_dir.exists():
                search_dir = self.image_dir
            elif (dataset_root / "images").exists():
                search_dir = dataset_root / "images"

            if search_dir is not None:
                image_paths, found_all = self._find_image_paths(detection_indices, search_dir)

                if found_all and image_paths:
                    vggt_transforms = build_vggt_transforms(image_paths, target_size=518)
                    logger.info(f"构建VGGT裁剪对齐变换，共 {len(vggt_transforms)} 张")
            else:
                logger.warning("无法定位图像目录，跳过裁剪对齐。可通过 --image-dir 指定。")
        except Exception as e:
            logger.warning(f"构建VGGT裁剪对齐变换失败，将降级为直接像素对齐：{e}")

        # 构建反向索引：(image_id, object_id) -> global_id
        reverse_mapping = {}
        for gid_str, instances in global_mapping_data.items():
            for inst in instances:
                key = (inst['image_id'], inst['object_id'])
                reverse_mapping[key] = int(gid_str)

        logger.info(f"构建反向索引: {len(reverse_mapping)} 个实例")

        # 高效方法：从2D检测框直接提取3D点（O(K×A)复杂度）
        logger.info("从检测框直接提取3D点并分配全局ID...")
        from utils.bbox_3d_extractor import extract_3d_from_bboxes

        # 使用高效掩码索引方法
        extracted_points, extracted_gids, extracted_confs = extract_3d_from_bboxes(
            world_points=world_points,
            world_points_conf=conf,
            detections=detections,
            reverse_mapping=reverse_mapping,
            conf_threshold=0.1,
            vggt_transforms=vggt_transforms,
        )

        logger.info(f"高效提取完成: {len(extracted_points)} 个3D点")
        logger.info(f"   唯一global_id数量: {len(np.unique(extracted_gids))}")

        # 匹配到target_points（GLB点云）
        if len(extracted_points) == 0:
            logger.warning("未提取到任何3D点，使用占位数据")
            return np.zeros(len(target_points), dtype=np.int32), np.zeros(len(target_points))

        # 使用KDTree将提取的点映射到GLB点云（仅需一次）
        distances, indices = nearest_neighbor_mapping(extracted_points, target_points, k=1)

        # 分配global_id和置信度
        final_gids = extracted_gids[indices]
        final_confs = extracted_confs[indices]

        logger.info(f"完成点云映射: {len(np.unique(final_gids))} 个唯一ID")

        return final_gids, final_confs

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

        # 3. 分配全局ID (60%)
        if progress and task_id is not None:
            progress.update(task_id, description="[cyan]分配全局ID...")

        global_ids, confidences = self.assign_global_ids_to_points(points)

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

        # 加载数据
        self.load_cache()
        self.mapper = GlobalIDMapper(str(self.global_mapping_path))

        # Viser服务器（延迟初始化）
        self.server = None

        # KD-Tree用于点拾取（延迟构建）
        self.kdtree = None
        self.points_centered = None

        # 拾取状态
        self.pick_mode_enabled = False
        self.last_picked_gid = None

        logger.info(f"ViserInteractive3DViewer initialized on port {port}")

    def load_cache(self) -> None:
        """加载缓存数据"""
        logger.info(f"Loading point cloud cache: {self.pcd_cache_path}")
        data = np.load(self.pcd_cache_path)
        self.points = data['points']
        self.colors = data['colors']
        self.global_ids = data['global_ids']
        self.confidences = data['confidences']
        logger.info(f"Loaded {len(self.points)} points")

        # 计算点云中心并缓存中心化后的坐标（避免在启动服务器时重复计算）
        self.scene_center = np.mean(self.points, axis=0)
        self.points_centered = self.points - self.scene_center
        logger.info(f"Computed scene center: {self.scene_center}")

        logger.info(f"Loading index cache: {self.index_cache_path}")
        with self.index_cache_path.open('r', encoding='utf-8') as f:
            self.index = json.load(f)
        logger.info(f"Loaded {len(self.index)} global IDs in index")

    def get_color_for_global_id(self, gid: int) -> Tuple[int, int, int]:
        """
        为全局ID生成独特颜色

        Args:
            gid: 全局ID

        Returns:
            (R, G, B) 颜色元组 (0-255)
        """
        # 使用gid的哈希值生成色调（黄金角分布）
        hue = (gid * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        return tuple(int(c * 255) for c in rgb)

    def build_kdtree_for_picking(self, points: np.ndarray) -> None:
        """构建KD-Tree用于快速点拾取（使用统一工具函数）"""
        self.kdtree = build_kdtree(points)

    def pick_point_by_sphere(
        self,
        click_position: np.ndarray,
        radius: float = 0.05
    ) -> Optional[int]:
        """
        通过球形区域拾取点（简化的点拾取）

        Args:
            click_position: 点击位置 (3D坐标)
            radius: 拾取半径

        Returns:
            拾取到的点的索引，如果没有则返回None
        """
        if self.kdtree is None:
            logger.warning("KD-Tree not built yet")
            return None

        # 查询半径范围内的所有点
        indices = self.kdtree.query_ball_point(click_position, r=radius)

        if len(indices) == 0:
            logger.debug(f"No points found within radius {radius}")
            return None

        # 返回最近的点
        distances, nearest_idx = self.kdtree.query(click_position, k=1)
        logger.info(f"Picked point {nearest_idx}, distance={distances:.4f}")
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

    def format_global_id_info(self, gid_str: str) -> str:
        """
        格式化全局ID信息为文本

        Args:
            gid_str: 全局ID字符串

        Returns:
            格式化的信息文本
        """
        if gid_str not in self.index:
            return f"Global ID {gid_str}: No info available"

        info = self.index[gid_str]
        info_lines = [
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"Global ID: {gid_str}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
            f"Images: {', '.join(map(str, info['images']))}",
            f"Objects: {', '.join(map(str, info['objects'][:10]))}{'...' if len(info['objects']) > 10 else ''}",
            f"Active: {info['active_count']}",
            f"Removed: {info['removed_count']}",
            f"━━━━━━━━━━━━━━━━━━━━━━━",
        ]
        return "\n".join(info_lines)


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

        # 构建KD-Tree用于点拾取
        self.build_kdtree_for_picking(points_centered)

        # 根据全局ID重新着色
        colors_by_gid = np.zeros((len(self.points), 3), dtype=np.uint8)
        unique_gids = np.unique(self.global_ids)
        for gid in unique_gids:
            mask = self.global_ids == gid
            colors_by_gid[mask] = self.get_color_for_global_id(int(gid))

        # ========== GUI Controls ==========
        gui_conf_threshold = self.server.gui.add_slider(
            "Confidence Percentile %", min=0, max=100, step=1, initial_value=25
        )

        gui_selected_id = self.server.gui.add_dropdown(
            "Show Global ID",
            options=["All"] + [str(gid) for gid in sorted(unique_gids)],
            initial_value="All"
        )

        # 拾取模式控件
        gui_pick_mode = self.server.gui.add_checkbox(
            "Pick Mode (Enter XYZ)",
            initial_value=False
        )

        gui_pick_radius = self.server.gui.add_slider(
            "Pick Radius",
            min=0.01,
            max=0.5,
            step=0.01,
            initial_value=0.05
        )

        # 显示当前场景中心（用于说明坐标已中心化）
        gui_scene_center = self.server.gui.add_text(
            "Scene Center (subtracted)",
            initial_value=f"{self.scene_center[0]:.3f}, {self.scene_center[1]:.3f}, {self.scene_center[2]:.3f}",
        )

        gui_info_text = self.server.gui.add_text(
            "Selected ID Info",
            initial_value="Enable Pick Mode and enter XYZ in centered coords",
        )

        # 初始点云
        init_threshold = np.percentile(self.confidences, 25)
        init_mask = self.confidences >= init_threshold
        point_cloud = self.server.scene.add_point_cloud(
            name="sku_pcd",
            points=points_centered[init_mask],
            colors=colors_by_gid[init_mask],
            point_size=0.002,
            point_shape="circle",
        )

        # 拾取球体指示器（初始隐藏）
        pick_sphere = self.server.scene.add_icosphere(
            name="pick_sphere",
            radius=gui_pick_radius.value,
            color=(255, 255, 0),  # 黄色
            opacity=0.3,
            position=(0.0, 0.0, 0.0),
            visible=False,
        )

        # ========== 点拾取逻辑 ==========
        # 由于Viser点云暂不支持原生on_click，我们提供手动输入3D坐标的拾取方式
        # 添加一个按钮组来输入3D坐标进行拾取（坐标已减去scene center）
        with self.server.gui.add_folder("Manual Pick (XYZ)"):
            gui_pick_x = self.server.gui.add_number("X", initial_value=0.0, step=0.01)
            gui_pick_y = self.server.gui.add_number("Y", initial_value=0.0, step=0.01)
            gui_pick_z = self.server.gui.add_number("Z", initial_value=0.0, step=0.01)
            gui_pick_button = self.server.gui.add_button("Pick Point")

        @gui_pick_button.on_click
        def _(_) -> None:
            """手动拾取按钮回调"""
            if not gui_pick_mode.value:
                gui_info_text.value = "Please enable Pick Mode first"
                return

            click_pos = np.array([gui_pick_x.value, gui_pick_y.value, gui_pick_z.value])

            # 显示拾取球体
            pick_sphere.position = tuple(click_pos)
            pick_sphere.radius = gui_pick_radius.value
            pick_sphere.visible = True

            # 执行点拾取
            point_idx = self.pick_point_by_sphere(click_pos, radius=gui_pick_radius.value)

            if point_idx is not None:
                gid_str = self.get_global_id_from_point_index(point_idx)
                if gid_str:
                    self.last_picked_gid = gid_str
                    info_text = self.format_global_id_info(gid_str)
                    gui_info_text.value = info_text

                    # 自动选择该全局ID
                    gui_selected_id.value = gid_str
                    update_point_cloud()

                    logger.info(f"Picked Global ID: {gid_str}")
                else:
                    gui_info_text.value = "Invalid point index"
            else:
                gui_info_text.value = f"No points found within radius {gui_pick_radius.value:.3f}"

        @gui_pick_mode.on_update
        def _(_) -> None:
            """拾取模式切换"""
            if gui_pick_mode.value:
                gui_info_text.value = "Pick Mode ON - Enter centered XYZ and click 'Pick Point'"
                pick_sphere.visible = True
            else:
                gui_info_text.value = "Pick Mode OFF"
                pick_sphere.visible = False

        @gui_pick_radius.on_update
        def _(_) -> None:
            """拾取半径更新"""
            pick_sphere.radius = gui_pick_radius.value


        def update_point_cloud() -> None:
            """更新点云显示"""
            threshold_val = np.percentile(self.confidences, gui_conf_threshold.value)
            conf_mask = self.confidences >= threshold_val

            if gui_selected_id.value == "All":
                id_mask = np.ones(len(self.global_ids), dtype=bool)
            else:
                selected_gid = int(gui_selected_id.value)
                id_mask = self.global_ids == selected_gid

            combined_mask = conf_mask & id_mask
            point_cloud.points = points_centered[combined_mask]
            point_cloud.colors = colors_by_gid[combined_mask]

        @gui_conf_threshold.on_update
        def _(_) -> None:
            update_point_cloud()

        @gui_selected_id.on_update
        def _(_) -> None:
            update_point_cloud()
            # 更新信息文本
            if gui_selected_id.value != "All":
                gui_info_text.value = self.format_global_id_info(gui_selected_id.value)
            else:
                gui_info_text.value = "Showing all global IDs"

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
        epilog="示例: uv run interactive_3d_viewer.py --global-mapping X.json --reconstruction Y.glb --image-dir Z"
    )

    # 核心参数（必需）
    parser.add_argument('--global-mapping', type=str, required=True,
                        help='global_mapping.json 路径')
    parser.add_argument('--reconstruction', type=str, required=True,
                        help='GLB/PLY 3D重建文件路径')
    parser.add_argument('--image-dir', type=str, required=True,
                        help='原始图像目录')

    # 可选参数
    parser.add_argument('--port', type=int, default=8080,
                        help='Viser服务端口 (默认: 8080)')
    parser.add_argument('--downsample', type=float, default=0.1,
                        help='点云下采样比例 0-1 (默认: 0.1)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='缓存目录（默认: 临时目录，退出后自动清理）')
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
            config={'downsample_ratio': args.downsample}
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
            downsample_ratio=args.downsample,
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
