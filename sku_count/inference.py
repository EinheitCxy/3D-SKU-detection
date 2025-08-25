import argparse
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 恢复为INFO级别
    format='%(message)s',
    handlers=[
        logging.FileHandler('sku_matching.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 导入模块化组件
try:
    from module import (
        SKUMatchingConfig, 
        DEFAULT_POINT_TRACKING_CONFIG, 
        DEFAULT_3D_PROJECTION_CONFIG,
        SKUMatchingSystem
    )
except ImportError as e:
    print(f"模块导入错误: {e}")
    print("请确保VGGT模块已正确安装和配置")
    sys.exit(1)


def create_config_from_args(args, algorithm_type: str = "point_tracking") -> SKUMatchingConfig:
    """根据命令行参数创建 SKUMatchingConfig 配置实例。
    
    此函数将命令行参数与预设配置模板相结合，生成适合
    指定算法类型的完整配置对象。
    
    Args:
        args: argparse.Namespace 命令行参数对象，包含用户输入的所有配置参数
        algorithm_type: 算法类型，'point_tracking' 或 '3d_projection'，
            决定使用哪个预设配置模板
    
    Returns:
        SKUMatchingConfig: 配置好的 SKUMatchingConfig 实例，包含了
            用户参数和默认配置的组合
    
    Note:
        点追踪算法使用点追踪匹配，3D算法使用3D-2D投影匹配。
    """
    base_config = DEFAULT_POINT_TRACKING_CONFIG if algorithm_type == "point_tracking" else DEFAULT_3D_PROJECTION_CONFIG
    
    # 修改output_dir名称，加上reference_idx+1
    reference_num = args.reference_idx + 1
    if algorithm_type == "point_tracking":
        output_dir = f"output_point_tracking_ref{reference_num}"
    else:
        output_dir = f"output_3d_projection_ref{reference_num}"
    
    config_dict = {
        "device": args.device,
        "max_points_per_bbox": args.max_points_per_bbox,
        "visibility_threshold": args.visibility_threshold,
        "min_visible_points": args.min_visible_points,
        "correspondence_threshold": args.correspondence_threshold,
        "seed": args.seed,
        "save_json": args.save_json,
        "output_dir": output_dir,
        **{k: v for k, v in base_config.items() if k != "output_dir"}  # 排除base_config中的output_dir
    }
    
    return SKUMatchingConfig(**config_dict)


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
    print("=== 运行点追踪匹配算法 ===")
    
    config = create_config_from_args(args, "point_tracking")
    system = SKUMatchingSystem(config)
    
    correspondences = system.process_images(
        image_folder=args.image_folder,
        detection_dir=args.detection_dir,
        reference_image_idx=args.reference_idx,
        max_images=args.max_images
    )
    
    total_matches = sum(len(matches) for matches in correspondences.values())
    print(f"点追踪算法找到 {total_matches} 个匹配")
    
    system.cleanup()
    return correspondences


def run_3d_projection_algorithm(args) -> dict:
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
    print("=== 运行3D-2D投影匹配算法 ===")
    
    config = create_config_from_args(args, "3d_projection")
    system = SKUMatchingSystem(config)
    
    correspondences = system.process_images(
        image_folder=args.image_folder,
        detection_dir=args.detection_dir,
        reference_image_idx=args.reference_idx,
        max_images=args.max_images
    )
    
    total_matches = sum(len(matches) for matches in correspondences.values())
    print(f"3D-2D投影算法找到 {total_matches} 个匹配")
    
    system.cleanup()
    return correspondences


def demo(image_folder: str = "../imdata/total", 
         detection_dir: str = "../sku_detection.json", 
         reference_idx: int = 0, 
         max_images: int = 13) -> None:
    """演示SKU匹配系统的完整功能。
    
    顺序执行传统点追踪和3D-2D投影两种算法，展示它们的
    特点、性能和适用场景。用于系统测试和效果演示。
    
    Args:
        image_folder: 图像文件夹路径，包含要处理的所有图像
        detection_dir: 检测结果文件路径，可以是单个JSON或目录
        reference_idx: 参考图像的索引，用作其他图像的匹配基准
        max_images: 最大处理图像数量，避免内存溢出
    
    Returns:
        None: 直接在控制台输出演示结果
    
    Raises:
        Exception: 当模型初始化或数据处理失败时
    
    Example:
        基本使用:
        >>> demo()
        
        自定义参数:
        >>> demo("/path/to/images", "/path/to/detections", 0, 10)
    """
    try:
        print("=== SKU匹配系统演示 ===\n")
        
        # 计算reference图像编号（从1开始）
        reference_num = reference_idx + 1
        
        # 示例1: 点追踪算法
        print("1. 使用点追踪匹配算法:")
        config_point_tracking = SKUMatchingConfig(
            max_points_per_bbox=100,
            visibility_threshold=0.7,
            min_visible_points=10,
            output_dir=f"output_results_point_tracking_ref{reference_num}",
            enable_3d_projection_matching=False
        )
        
        system_point_tracking = SKUMatchingSystem(config_point_tracking)
        correspondences_point_tracking = system_point_tracking.process_images(
            image_folder=image_folder,
            detection_dir=detection_dir,
            reference_image_idx=reference_idx,
            max_images=max_images
        )
        system_point_tracking.cleanup()
        
        point_tracking_total = sum(len(matches) for matches in correspondences_point_tracking.values())
        print(f"点追踪算法找到 {point_tracking_total} 个匹配\n")
        
        # 示例2: 3D-2D投影算法
        print("2. 使用新的3D-2D投影匹配算法:")
        config_3d = SKUMatchingConfig(
            output_dir=f"output_results_3d_projection_ref{reference_num}",
            enable_3d_projection_matching=True,
            depth_confidence_threshold=0.15,
            point_3d_confidence_threshold=0.15,
            projection_match_threshold=0.7,
            max_3d_distance=1.0,
            max_depth_difference=2.0,
            min_depth_consistency=0.3
        )
        
        system_3d = SKUMatchingSystem(config_3d)
        correspondences_3d = system_3d.process_images(
            image_folder=image_folder,
            detection_dir=detection_dir,
            reference_image_idx=reference_idx,
            max_images=max_images
        )
        system_3d.cleanup()
        
        projection_total = sum(len(matches) for matches in correspondences_3d.values())
        print(f"3D-2D投影算法找到 {projection_total} 个匹配\n")
        
        print("=== 演示完成 ===")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


def main() -> None:
    """主函数，处理命令行参数并执行相应的SKU匹配算法。
    
    解析命令行参数，验证输入路径，并根据用户选择执行相应的
    匹配算法(点追踪、3D-2D投影、两者对比或演示模式)。
    
    支持的命令行参数:
        --algorithm: 算法选择 (point_tracking/3d/both/demo)
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
        
        演示模式:
        >>> python inference.py --algorithm demo
    """
    parser = argparse.ArgumentParser(description="SKU匹配系统 - 物体跨图像匹配")
    # 基本参数
    parser.add_argument("--image_folder", type=str, default="../imdata/total", help="图像文件夹路径")
    parser.add_argument("--detection_dir", type=str, default="../imdata/detections_results", help="检测结果目录路径")
    parser.add_argument("--reference_idx", type=int, default=0, help="参考图像索引")
    parser.add_argument("--max_images", type=int, default=20, help="最大处理图像数量")
    # 算法选择
    parser.add_argument("--algorithm", type=str, choices=["point_tracking", "3d", "both", "demo"], default="both", 
                       help="选择匹配算法: point_tracking(点追踪), 3d(3D投影), both(两种都运行), demo(演示模式)")
    # 系统参数
    parser.add_argument("--device", type=str, default="cuda", help="计算设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_json", action="store_true", help="保存结果为JSON文件")
    # 匹配参数
    parser.add_argument("--max_points_per_bbox", type=int, default=100, help="每个检测框最大采样点数")
    parser.add_argument("--visibility_threshold", type=float, default=0.8, help="可见性阈值")
    parser.add_argument("--min_visible_points", type=int, default=8, help="最小可见点数")
    parser.add_argument("--correspondence_threshold", type=float, default=0.5, help="对应关系阈值")
    args = parser.parse_args()
    
    try:
        # 验证输入路径
        if not Path(args.image_folder).exists():
            raise FileNotFoundError(f"图像文件夹不存在: {args.image_folder}")
        if not Path(args.detection_dir).exists():
            raise FileNotFoundError(f"检测结果目录不存在: {args.detection_dir}")
        
        print(f"=== SKU匹配系统 ===")
        print(f"图像文件夹: {args.image_folder}")
        print(f"检测结果目录: {args.detection_dir}")
        print(f"参考图像索引: {args.reference_idx}")
        print(f"最大图像数量: {args.max_images}")
        print(f"算法选择: {args.algorithm}")
        print()
        
        correspondences_point_tracking = None
        correspondences_3d = None
        
        # 根据选择运行算法
        if args.algorithm in ["point_tracking", "both"]:
            correspondences_point_tracking = run_point_tracking_algorithm(args)
            print()
        
        if args.algorithm in ["3d", "both"]:
            correspondences_3d = run_3d_projection_algorithm(args)
            print()
        
        if args.algorithm == "demo":
            demo(
                image_folder=args.image_folder,
                detection_dir=args.detection_dir,
                reference_idx=args.reference_idx,
                max_images=args.max_images
            )
        
        # 总结比较结果
        if args.algorithm == "both" and correspondences_point_tracking and correspondences_3d:
            point_tracking_total = sum(len(matches) for matches in correspondences_point_tracking.values())
            projection_total = sum(len(matches) for matches in correspondences_3d.values())
            
            print("=== 算法比较结果 ===")
            print(f"点追踪算法: {point_tracking_total} 个匹配")
            print(f"3D-2D投影算法: {projection_total} 个匹配")
            print(f"差异: {abs(point_tracking_total - projection_total)} 个匹配")        
        print("=== 处理完成 ===")
            
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()