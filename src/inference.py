import argparse
import logging
import sys
from pathlib import Path
import time
from typing import Sequence

# 使用主程序配置的日志；若独立运行且无处理器，则退回到控制台输出
logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    # 添加父目录到路径以便导入utils模块
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import (
        SKUMatchingConfig,
        SKUMatchingSystem
    )
    from utils.profiling import set_enabled, StageTimer
except ImportError as e:
    logger.error(f"模块导入错误: {e}")
    logger.error("请确保VGGT模块已正确安装和配置")
    sys.exit(1)


def _compute_output_dir(base: str, algorithm_type: str, ref_idx: int, backend: str | None = None) -> str:
    """根据算法类型和后端计算输出目录。

    - point_tracking: base/output_pt/<ref_idx>
    - 3d_mapping:
        - 若指定 backend，则 base/output_3dmapping_<backend>/<ref_idx>
        - 否则保持旧行为: base/output_3dmapping/<ref_idx>
    """
    if algorithm_type == "point_tracking":
        return f"{base}/output_pt/{ref_idx}"

    # 3D 映射匹配输出目录
    if backend:
        return f"{base}/output_3dmapping_{backend}/{ref_idx}"
    return f"{base}/output_3dmapping/{ref_idx}"


def _count_images_and_detections(image_folder: str, detection_dir: str) -> tuple[int, int]:
    """Count total images and numeric detection JSON files with valid structure.

    使用 utils.data_utils.load_detections 作为唯一标准源，避免重复的文件扫描逻辑。

    Returns (num_images, num_detections)."""
    from pathlib import Path
    from utils.data_utils import load_detections

    # 计数图片文件
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_dir = Path(image_folder)
    n_img = sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)

    # 使用标准load_detections计数有效检测文件
    try:
        detections_with_index = load_detections(detection_dir, return_index_map=True)
        n_det = len(detections_with_index)
    except (FileNotFoundError, ValueError) as e:
        logger.debug(f"Failed to load detections: {e}")
        n_det = 0

    return n_img, n_det


def create_config_from_args(args, algorithm_type: str = "point_tracking") -> SKUMatchingConfig:
    """根据命令行参数创建 SKUMatchingConfig 配置实例。
    
    此函数将命令行参数与预设配置模板相结合，生成适合
    指定算法类型的完整配置对象。
    
    Args:
        args: argparse.Namespace 命令行参数对象，包含用户输入的所有配置参数
        algorithm_type: 算法类型，'point_tracking' 或 '3d_mapping'，
            决定使用哪个预设配置模板
    
    Returns:
        SKUMatchingConfig: 配置好的 SKUMatchingConfig 实例，包含了
            用户参数和默认配置的组合
    
    Note:
        点追踪算法使用点追踪匹配，3D算法使用3D-2D投影匹配。
    """
    # 从parser获取基础output_dir，然后按算法类型和索引分组
    output_dir = _compute_output_dir(args.output_dir, algorithm_type, args.reference_idx, getattr(args, "backend", None))

    overrides = {
        "backend": args.backend,
        "device": args.device,
        "max_points_per_bbox": args.max_points_per_bbox,
        "confidence_threshold": args.confidence_threshold,
        "min_confident_points": args.min_confident_points,
        "min_hit_ratio": args.min_hit_ratio,
        "seed": args.seed,
        "save_json": args.save_json,
        "output_dir": output_dir,
        "sam3_mask_cache_root": str(
            Path(args.sam3_mask_cache_root)
            if args.sam3_mask_cache_root
            else Path(args.output_dir) / "sam3_mask_cache" / "v2"
        ),
    }
    # 可选 3D 阈值覆盖（网格扫描用，None=用 config 默认）
    for _k in ("plane_normal_alignment_threshold", "max_3d_distance", "max_depth", "depth_confidence_threshold", "min_3d_sample_points", "pairing_3d"):
        _v = getattr(args, _k, None)
        if _v is not None:
            overrides[_k] = _v

    if algorithm_type == "point_tracking":
        return SKUMatchingConfig.for_point_tracking(**overrides)
    backend = overrides.pop("backend", args.backend)
    return SKUMatchingConfig.for_3d_mapping(backend=backend, **overrides)


def _create_config_from_yaml(args, algorithm_type: str) -> SKUMatchingConfig:
    from utils import build_matching_config_from_yaml

    cfg = build_matching_config_from_yaml(args.config, algorithm=algorithm_type, backend=getattr(args, "backend", None))
    # Always route outputs to per-ref subdir like CLI path does
    cfg.output_dir = _compute_output_dir(args.output_dir, algorithm_type, args.reference_idx, getattr(args, "backend", None))
    cfg.sam3_mask_cache_root = str(
        Path(args.sam3_mask_cache_root)
        if args.sam3_mask_cache_root
        else Path(args.output_dir) / "sam3_mask_cache" / "v2"
    )
    # Override a few runtime knobs from CLI
    cfg.device = args.device
    cfg.save_json = bool(args.save_json)
    cfg.seed = args.seed
    # 可选 3D 阈值覆盖（网格扫描用，None=保留 YAML/默认值）
    # 注：main.py concise 路径总是传 --config，本函数是实际生效路径
    for _k in ("plane_normal_alignment_threshold", "max_3d_distance", "max_depth", "depth_confidence_threshold", "min_3d_sample_points", "pairing_3d"):
        _v = getattr(args, _k, None)
        if _v is not None:
            setattr(cfg, _k, _v)
    return cfg


def run_point_tracking_algorithm(args) -> dict:
    """执行传统点追踪SKU匹配算法。
    
    使用VGGT模型的点追踪功能，基于2D特征点的可见性和连续性
    进行SKU物体匹配。适合视角变化不大的场景。
    
    Args:
        args: 命令行参数对象，包含:
            - image_folder: 图像文件夹路径
            - detection_dir: 检测结果目录路径
            - reference_idx: 参考图像的索引
            - max_images: 最大处理图像数量
    
    Returns:
        dict: 匹配结果字典，格式为 {target_image_idx: [match_objects]}
    
    Raises:
        Exception: 当模型初始化、数据加载或匹配计算失败时
    
    Note:
        传统算法速度快、内存消耗低，但在大视角变化时可能不够稳定。
    """
    logger.info("=== 运行点追踪匹配算法 ===")
    num_images, num_dets = _count_images_and_detections(args.image_folder, args.detection_dir)
    logger.info(
        f"stage=match algo=point_tracking ref={args.reference_idx} images={num_images} detections={num_dets}"
    )

    config = create_config_from_args(args, "point_tracking")
    system = SKUMatchingSystem(config)

    t0 = time.time()
    correspondences = system.process_images(
        image_folder=args.image_folder,
        detection_dir=args.detection_dir,
        reference_image_idx=args.reference_idx,
        max_images=args.max_images
    )
    duration = time.time() - t0
    total_matches = sum(len(matches) for matches in correspondences.values())
    logger.info(
        f"matched_total={total_matches} saved_json={bool(config.save_json)} output_dir={config.output_dir} duration={duration:.2f}s"
    )
    
    system.cleanup()
    return correspondences


def run_3d_mapping_algorithm(args) -> dict:
    """执行3D-2D投影SKU匹配算法。
    
    基于VGGT模型的深度估计和相机姿态信息，将参考3D点投影到
    目标图像中进行匹配。提供更高的匹配精度和稳定性。
    
    Args:
        args: 命令行参数对象，包含:
            - image_folder: 图像文件夹路径
            - detection_dir: 检测结果目录路径
            - reference_idx: 参考图像的索引
            - max_images: 最大处理图像数量
    
    Returns:
        dict: 匹配结果字典，包含额外3D几何验证信息
    
    Raises:
        Exception: 当模型初始化、深度估计或投影计算失败时
    
    Note:
        3D算法计算复杂度更高，但在复杂场景和大视角变化下
        表现更优。需要更多的GPU内存和计算时间。
    """
    logger.info("=== 运行3D-2D投影匹配算法 ===")
    num_images, num_dets = _count_images_and_detections(args.image_folder, args.detection_dir)
    logger.info(
        f"stage=match algo=3d ref={args.reference_idx} images={num_images} detections={num_dets}"
    )

    config = create_config_from_args(args, "3d_mapping")
    system = SKUMatchingSystem(config)

    t0 = time.time()
    correspondences = system.process_images(
        image_folder=args.image_folder,
        detection_dir=args.detection_dir,
        reference_image_idx=args.reference_idx,
        max_images=args.max_images
    )
    duration = time.time() - t0
    total_matches = sum(len(matches) for matches in correspondences.values())
    logger.info(
        f"matched_total={total_matches} saved_json={bool(config.save_json)} output_dir={config.output_dir} duration={duration:.2f}s"
    )
    
    system.cleanup()
    return correspondences


def run_point_tracking(args) -> dict:
    if args.config:
        config = _create_config_from_yaml(args, "point_tracking")
        system = SKUMatchingSystem(config)
        num_images, num_dets = _count_images_and_detections(args.image_folder, args.detection_dir)
        logger.info(
            f"stage=match algo=point_tracking ref={args.reference_idx} images={num_images} detections={num_dets}"
        )
        t0 = time.time()
        correspondences = system.process_images(
            image_folder=args.image_folder,
            detection_dir=args.detection_dir,
            reference_image_idx=args.reference_idx,
            max_images=args.max_images
        )
        duration = time.time() - t0
        total_matches = sum(len(matches) for matches in correspondences.values())
        logger.info(
            f"matched_total={total_matches} saved_json={bool(config.save_json)} output_dir={config.output_dir} duration={duration:.2f}s"
        )
        system.cleanup()
        return correspondences
    else:
        return run_point_tracking_algorithm(args)


def run_3d_mapping(args) -> dict:
    if args.config:
        config = _create_config_from_yaml(args, "3d_mapping")
        system = SKUMatchingSystem(config)
        num_images, num_dets = _count_images_and_detections(args.image_folder, args.detection_dir)
        logger.info(
            f"stage=match algo=3d ref={args.reference_idx} images={num_images} detections={num_dets}"
        )
        t0 = time.time()
        correspondences = system.process_images(
            image_folder=args.image_folder,
            detection_dir=args.detection_dir,
            reference_image_idx=args.reference_idx,
            max_images=args.max_images
        )
        duration = time.time() - t0
        total_matches = sum(len(matches) for matches in correspondences.values())
        logger.info(
            f"matched_total={total_matches} saved_json={bool(config.save_json)} output_dir={config.output_dir} duration={duration:.2f}s"
        )
        StageTimer.record("per_ref_total", duration)
        system.cleanup()
        return correspondences
    else:
        return run_3d_mapping_algorithm(args)


def main(argv: Sequence[str] | None = None) -> None:
    """主函数，处理命令行参数并执行相应的SKU匹配算法。
    
    解析命令行参数，验证输入路径，并根据用户选择执行相应的
    匹配算法(点追踪、3D-2D投影、两者对比或演示模式)。
    
    支持的命令行参数:
        --algorithm: 算法选择 (point_tracking/3d/both)
        --image_folder: 图像文件夹路径
        --detection_dir: 检测结果目录路径
        --max_points_per_bbox: 每个检测框最大采样点数
        --visibility_threshold: 可见性阈值
        --device: 计算设备 (cuda/cpu)
        --save_json: 是否保存JSON结果
    
    Returns:
        None: 直接在控制台输出结果或退出程序
    
    Raises:
        FileNotFoundError: 当指定的图像文件夹或检测结果目录不存在时
        SystemExit: 当发生不可恢复的错误时退出程序
    
    Example:
        基本使用:
        >>> python inference.py --algorithm both
        
        使用3D算法处理指定数据集:
        >>> python inference.py --algorithm 3d --image_folder /path/to/images
        
        
    """
    parser = argparse.ArgumentParser(description="SKU匹配系统 - 物体跨图像匹配")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径（可选）")
    # 基本参数
    parser.add_argument("--image_folder", type=str, default="imdata/floor_display2/images", help="图像文件夹路径")
    parser.add_argument("--detection_dir", type=str, default="imdata/floor_display2/detections_results", help="检测结果目录路径")
    parser.add_argument("--output_dir", type=str, default="Output/floor_display2", help="输出根目录路径")
    parser.add_argument("--sam3_mask_cache_root", type=str, default=None,
                       help="canonical SAM3 v2 cache 根目录（默认由 output_dir 推导）")
    parser.add_argument("--reference_idx", type=int, default=0, help="参考图像索引")
    parser.add_argument("--max_images", type=int, default=50, help="最大处理图像数量")
    # 算法选择
    parser.add_argument("--algorithm", type=str, choices=["point_tracking", "3d", "both"], default="3d",
                       help="选择匹配算法: point_tracking(点追踪), 3d(3D投影), both(两种都运行)")
    parser.add_argument("--backend", type=str, choices=["vggt", "pi3", "da3"], default="da3",
                       help="3D重建模型后端 (vggt/pi3/da3)，用于3D算法时选择数据源")
    # 系统参数
    parser.add_argument("--device", type=str, default="cuda", help="计算设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_json", action="store_true", help="保存结果为JSON文件")
    # 匹配参数
    parser.add_argument("--max_points_per_bbox", type=int, default=30, help="每个检测框最大采样点数")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="点追踪置信度阈值")
    parser.add_argument("--min_confident_points", type=int, default=10, help="每个bbox的最小置信点数")
    parser.add_argument("--min_hit_ratio", type=float, default=0.5, help="最小命中率阈值")
    # 可选 3D 阈值覆盖（网格扫描用，default=None 表示用 config 默认）
    parser.add_argument("--plane_normal_alignment_threshold", type=float, default=None, help="平面法向对齐阈值(覆盖config)")
    parser.add_argument("--max_3d_distance", type=float, default=None, help="3D空间距离阈值(覆盖config)")
    parser.add_argument("--max_depth", type=float, default=None, help="最大深度(覆盖config)")
    parser.add_argument("--depth_confidence_threshold", type=float, default=None, help="深度置信度阈值(覆盖config)")
    parser.add_argument("--min_3d_sample_points", type=int, default=None, help="3D采样最少有效点数(覆盖config)")
    parser.add_argument("--pairing_3d", type=str, default=None, choices=["all","next"], help="3D配对策略(覆盖config)")
    parser.add_argument("--enable_profiling", action="store_true", default=False,
                       help="启用 per-stage 计时 instrumentation（默认关闭，零开销 no-op）")
    args = parser.parse_args(argv)
    
    try:
        if not Path(args.image_folder).exists():
            raise FileNotFoundError(f"图像文件夹不存在: {args.image_folder}")
        if not Path(args.detection_dir).exists():
            raise FileNotFoundError(f"检测结果目录不存在: {args.detection_dir}")
        
        logger.info("=== SKU匹配系统 ===")
        logger.info(f"图像文件夹: {args.image_folder}")
        logger.info(f"检测结果目录: {args.detection_dir}")
        logger.info(f"参考图像索引: {args.reference_idx}")
        logger.info(f"算法选择: {args.algorithm}")
        logger.info("==================")

        set_enabled(args.enable_profiling)

        correspondences_point_tracking = None
        correspondences_3d = None
        
        # 根据选择运行算法
        if args.algorithm in ["point_tracking", "both"]:
            correspondences_point_tracking = run_point_tracking(args)
            logger.info("")
        
        if args.algorithm in ["3d", "both"]:
            correspondences_3d = run_3d_mapping(args)
            logger.info("")
        
        # 总结比较结果
        if args.algorithm == "both" and correspondences_point_tracking and correspondences_3d:
            point_tracking_total = sum(len(matches) for matches in correspondences_point_tracking.values())
            projection_total = sum(len(matches) for matches in correspondences_3d.values())
            
            logger.info("=== 算法比较结果 ===")
            logger.info(f"点追踪算法: {point_tracking_total} 个匹配")
            logger.info(f"3D-2D投影算法: {projection_total} 个匹配")
            logger.info(f"差异: {abs(point_tracking_total - projection_total)} 个匹配")        
        logger.info("=== 处理完成 ===")
            
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as e:
        logger.error(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
