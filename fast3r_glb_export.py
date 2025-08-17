#!/usr/bin/env python3
"""
Fast3R GLB点云导出工具
使用Fast3R进行3D重建并导出为GLB格式
"""

import os
import sys
import torch
import numpy as np
import trimesh
from pathlib import Path
import argparse
from datetime import datetime

# 添加fast3r路径
current_dir = Path(__file__).parent
fast3r_dir = current_dir / "fast3r"
sys.path.insert(0, str(fast3r_dir))

from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.utils.image import load_images
from fast3r.dust3r.viz import pts3d_to_trimesh, cat_meshes
from fast3r.utils.checkpoint_utils import load_model


def extract_pointcloud_from_fast3r(result, min_conf_thr_percentile=30):
    """
    从Fast3R推理结果中提取点云数据
    
    Args:
        result: Fast3R推理结果字典
        min_conf_thr_percentile: 置信度阈值百分位数
        
    Returns:
        vertices: 3D点坐标 (N, 3)
        colors: 点云颜色 (N, 3)
    """
    all_points = []
    all_colors = []
    
    # Fast3R的结果结构
    views = result['views']
    preds = result['preds']
    
    for view, pred in zip(views, preds):
        # 获取3D点坐标和置信度
        pts3d = pred['pts3d']  # Shape: [H, W, 3]
        confidence = pred['conf']  # Shape: [H, W]
        
        # 获取图像数据
        img = view['img']  # Shape: [3, H, W] 或 [H, W, 3]
        
        # 确保数据在CPU上
        if isinstance(pts3d, torch.Tensor):
            pts3d = pts3d.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # 处理图像格式: [3, H, W] -> [H, W, 3]
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        # 规范化图像到[0, 255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        elif img.max() <= 2.0:  # [-1, 1] 范围
            img = ((img + 1) * 127.5).astype(np.uint8).clip(0, 255)
        
        # 应用置信度阈值
        if min_conf_thr_percentile > 0:
            conf_thr = np.percentile(confidence, min_conf_thr_percentile)
            mask = confidence > conf_thr
        else:
            mask = np.ones_like(confidence, dtype=bool)
        
        # 提取有效点和颜色
        valid_points = pts3d[mask]
        valid_colors = img[mask]
        
        if len(valid_points) > 0:
            all_points.append(valid_points)
            all_colors.append(valid_colors)
    
    if not all_points:
        raise ValueError("没有提取到有效的点云数据")
    
    # 合并所有点云数据
    vertices = np.vstack(all_points)
    colors = np.vstack(all_colors)
    
    return vertices, colors


def create_mesh_from_fast3r(result, min_conf_thr_percentile=30):
    """
    从Fast3R输出创建三角网格
    
    Args:
        result: Fast3R推理结果
        min_conf_thr_percentile: 置信度阈值百分位数
        
    Returns:
        trimesh.Trimesh: 三角网格对象
    """
    views = result['views']
    preds = result['preds']
    
    meshes = []
    
    for view, pred in zip(views, preds):
        # 获取数据并转换到CPU
        pts3d = pred['pts3d'].cpu().numpy()
        confidence = pred['conf'].cpu().numpy()
        img = view['img']
        
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # 处理图像格式
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        # 规范化图像
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        elif img.max() <= 2.0:
            img = ((img + 1) * 127.5).astype(np.uint8).clip(0, 255)
        
        # 应用置信度阈值
        if min_conf_thr_percentile > 0:
            conf_thr = np.percentile(confidence, min_conf_thr_percentile)
            mask = confidence > conf_thr
        else:
            mask = np.ones_like(confidence, dtype=bool)
        
        # 生成mesh
        mesh_dict = pts3d_to_trimesh(img, pts3d, valid=mask)
        meshes.append(mesh_dict)
    
    # 合并所有mesh
    combined_mesh_dict = cat_meshes(meshes)
    combined_mesh = trimesh.Trimesh(**combined_mesh_dict)
    
    return combined_mesh


def export_to_glb(data, output_path, is_mesh=False):
    """
    导出数据为GLB格式
    
    Args:
        data: trimesh对象或(vertices, colors)元组
        output_path: 输出文件路径
        is_mesh: 是否为网格数据
    """
    try:
        if is_mesh:
            # 直接导出网格
            data.export(output_path)
        else:
            # 导出点云
            vertices, colors = data
            point_cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
            point_cloud.export(output_path)
        
        print(f"✅ 已成功导出到: {output_path}")
    except Exception as e:
        print(f"❌ 导出失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Fast3R GLB导出工具')
    parser.add_argument('--images', type=str, required=True, 
                       help='输入图像目录路径')
    parser.add_argument('--model_path', type=str, 
                       default='jedyang97/Fast3R_ViT_Large_512',
                       help='模型路径或HuggingFace模型名称')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--min_conf_thr_percentile', type=int, default=30,
                       help='置信度阈值百分位数')
    parser.add_argument('--export_mesh', action='store_true',
                       help='导出三角网格而非点云')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备')
    parser.add_argument('--image_size', type=int, default=512,
                       help='图像分辨率')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Fast3R GLB导出工具 ===")
    print(f"输入图像目录: {args.images}")
    print(f"输出目录: {args.output_dir}")
    print(f"使用设备: {args.device}")
    
    # 检查设备可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # 加载模型
    print("\n🔄 正在加载Fast3R模型...")
    try:
        model, lit_module = load_model(args.model_path, device)
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 加载图像
    print(f"\n🔄 正在加载图像从: {args.images}")
    try:
        images = load_images(args.images, size=args.image_size, verbose=True)
        if len(images) == 0:
            print("❌ 错误: 未找到有效图像")
            return
        print(f"✅ 加载了 {len(images)} 张图像")
    except Exception as e:
        print(f"❌ 图像加载失败: {e}")
        return
    
    # 进行3D重建
    print("\n🔄 正在进行3D重建...")
    try:
        with torch.no_grad():
            # 根据Fast3R的实际调用方式
            result = inference(images, model, lit_module, device)
        print("✅ 3D重建完成")
    except Exception as e:
        print(f"❌ 3D重建失败: {e}")
        return
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if args.export_mesh:
            # 导出三角网格
            print("\n🔄 正在生成三角网格...")
            mesh = create_mesh_from_fast3r(result, args.min_conf_thr_percentile)
            
            # 导出GLB格式
            glb_path = os.path.join(args.output_dir, f"fast3r_mesh_{timestamp}.glb")
            export_to_glb(mesh, glb_path, is_mesh=True)
            
        else:
            # 导出点云
            print("\n🔄 正在提取点云数据...")
            vertices, colors = extract_pointcloud_from_fast3r(result, args.min_conf_thr_percentile)
            print(f"✅ 提取了 {len(vertices)} 个3D点")
            
            # 导出GLB格式
            glb_path = os.path.join(args.output_dir, f"fast3r_pointcloud_{timestamp}.glb")
            export_to_glb((vertices, colors), glb_path, is_mesh=False)
        
        print(f"\n🎉 === 导出完成 ===")
        print(f"输出文件保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()