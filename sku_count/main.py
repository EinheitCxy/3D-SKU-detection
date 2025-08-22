import argparse
import logging
import sys
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sku_matching.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 导入模块化组件
from utils import (
    SKUMatchingConfig, 
    DEFAULT_TRADITIONAL_CONFIG, 
    DEFAULT_3D_PROJECTION_CONFIG,
    SKUMatchingSystem
)


def create_config_from_args(args, algorithm_type: str = "traditional") -> SKUMatchingConfig:
    """根据命令行参数创建配置"""
    base_config = DEFAULT_TRADITIONAL_CONFIG if algorithm_type == "traditional" else DEFAULT_3D_PROJECTION_CONFIG
    
    config_dict = {
        "device": args.device,
        "max_points_per_bbox": args.max_points_per_bbox,
        "visibility_threshold": args.visibility_threshold,
        "min_visible_points": args.min_visible_points,
        "correspondence_threshold": args.correspondence_threshold,
        "seed": args.seed,
        "save_json": args.save_json,
        **base_config
    }
    
    return SKUMatchingConfig(**config_dict)


def run_traditional_algorithm(args) -> None:
    """运行传统点追踪匹配算法"""
    print("=== 运行传统点追踪匹配算法 ===")
    
    config = create_config_from_args(args, "traditional")
    system = SKUMatchingSystem(config)
    
    correspondences = system.process_images(
        image_folder=args.image_folder,
        detection_dir=args.detection_dir,
        reference_image_idx=args.reference_idx,
        max_images=args.max_images
    )
    
    total_matches = sum(len(matches) for matches in correspondences.values())
    print(f"传统算法找到 {total_matches} 个匹配")
    
    system.cleanup()
    return correspondences


def run_3d_projection_algorithm(args) -> None:
    """运行3D-2D投影匹配算法"""
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SKU匹配系统 - 物体跨图像匹配")
    
    # 基本参数
    parser.add_argument("--image_folder", type=str, default="../imdata",
                       help="图像文件夹路径")
    parser.add_argument("--detection_dir", type=str, default="../imdata/detections_results", 
                       help="检测结果目录路径")
    parser.add_argument("--reference_idx", type=int, default=0,
                       help="参考图像索引")
    parser.add_argument("--max_images", type=int, default=20,
                       help="最大处理图像数量")
    
    # 算法选择
    parser.add_argument("--algorithm", type=str, choices=["traditional", "3d", "both"], 
                       default="both", help="选择匹配算法: traditional(点追踪), 3d(3D投影), both(两种都运行)")
    
    # 系统参数
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--save_json", action="store_true",
                       help="保存结果为JSON文件")
    
    # 匹配参数
    parser.add_argument("--max_points_per_bbox", type=int, default=50,
                       help="每个检测框最大采样点数")
    parser.add_argument("--visibility_threshold", type=float, default=0.8,
                       help="可见性阈值")
    parser.add_argument("--min_visible_points", type=int, default=8,
                       help="最小可见点数")
    parser.add_argument("--correspondence_threshold", type=float, default=0.5,
                       help="对应关系阈值")
    
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
        
        correspondences_traditional = None
        correspondences_3d = None
        
        # 根据选择运行算法
        if args.algorithm in ["traditional", "both"]:
            correspondences_traditional = run_traditional_algorithm(args)
            print()
        
        if args.algorithm in ["3d", "both"]:
            correspondences_3d = run_3d_projection_algorithm(args)
            print()
        
        # 总结比较结果
        if args.algorithm == "both" and correspondences_traditional and correspondences_3d:
            traditional_total = sum(len(matches) for matches in correspondences_traditional.values())
            projection_total = sum(len(matches) for matches in correspondences_3d.values())
            
            print("=== 算法比较结果 ===")
            print(f"传统点追踪算法: {traditional_total} 个匹配")
            print(f"3D-2D投影算法: {projection_total} 个匹配")
            print(f"差异: {abs(traditional_total - projection_total)} 个匹配")
        
        print("=== 处理完成 ===")
        print("可以查看以下目录中的可视化结果:")
        if args.algorithm in ["traditional", "both"]:
            print("- output_results_traditional/ (传统点追踪算法结果)")
        if args.algorithm in ["3d", "both"]:
            print("- output_results_3d_projection/ (3D-2D投影算法结果)")
            
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        sys.exit(1)


def demo():
    """演示函数 - 展示两种算法的使用"""
    try:
        print("=== SKU匹配系统演示 ===\n")
        
        # 示例1: 传统点追踪算法
        print("1. 使用传统的点追踪匹配算法:")
        config_traditional = SKUMatchingConfig(
            max_points_per_bbox=100,
            visibility_threshold=0.7,
            min_visible_points=10,
            output_dir="output_results_traditional",
            enable_3d_projection_matching=False
        )
        
        system_traditional = SKUMatchingSystem(config_traditional)
        correspondences_traditional = system_traditional.process_images(
            image_folder="../imdata",
            detection_dir="../sku_detection.json",
            reference_image_idx=0,
            max_images=13
        )
        system_traditional.cleanup()
        
        traditional_total = sum(len(matches) for matches in correspondences_traditional.values())
        print(f"传统算法找到 {traditional_total} 个匹配\n")
        
        # 示例2: 3D-2D投影算法
        print("2. 使用新的3D-2D投影匹配算法:")
        config_3d = SKUMatchingConfig(
            output_dir="output_results_3d_projection",
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
            image_folder="../imdata",
            detection_dir="../sku_detection.json",
            reference_image_idx=0,
            max_images=13
        )
        system_3d.cleanup()
        
        projection_total = sum(len(matches) for matches in correspondences_3d.values())
        print(f"3D-2D投影算法找到 {projection_total} 个匹配\n")
        
        print("=== 演示完成 ===")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()