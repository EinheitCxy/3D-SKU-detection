#!/usr/bin/env python3
"""
3D点云可视化工具
用于显示PLY格式的3D点云文件
"""

import open3d as o3d
import numpy as np
import argparse
import os
import sys


def load_point_cloud(ply_file):
    """加载PLY格式的点云文件"""
    if not os.path.exists(ply_file):
        raise FileNotFoundError(f"PLY文件不存在: {ply_file}")
    
    print(f"正在加载点云文件: {ply_file}")
    point_cloud = o3d.io.read_point_cloud(ply_file)
    
    if len(point_cloud.points) == 0:
        raise ValueError("点云文件为空或格式不支持")
    
    print(f"成功加载点云，包含 {len(point_cloud.points)} 个点")
    return point_cloud


def visualize_point_cloud(point_cloud, window_name="3D点云可视化"):
    """使用Open3D可视化点云"""
    print("正在启动3D可视化窗口...")
    
    # 设置点云颜色（如果没有颜色信息）
    if not point_cloud.has_colors():
        # 根据高度生成渐变色
        points = np.asarray(point_cloud.points)
        if len(points) > 0:
            z_coords = points[:, 2]
            z_min, z_max = z_coords.min(), z_coords.max()
            if z_max > z_min:
                # 归一化到0-1范围
                normalized_z = (z_coords - z_min) / (z_max - z_min)
                # 创建颜色映射（蓝色到红色）
                colors = np.zeros((len(points), 3))
                colors[:, 0] = normalized_z  # 红色通道
                colors[:, 2] = 1 - normalized_z  # 蓝色通道
                point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # 显示点云
    o3d.visualization.draw_geometries(
        [point_cloud],
        window_name=window_name,
        width=1200,
        height=800,
        left=50,
        top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )


def print_point_cloud_info(point_cloud):
    """打印点云信息"""
    points = np.asarray(point_cloud.points)
    
    print("\n" + "="*50)
    print("点云信息")
    print("="*50)
    print(f"点数: {len(points)}")
    
    if len(points) > 0:
        print(f"X范围: {points[:, 0].min():.3f} 到 {points[:, 0].max():.3f}")
        print(f"Y范围: {points[:, 1].min():.3f} 到 {points[:, 1].max():.3f}")
        print(f"Z范围: {points[:, 2].min():.3f} 到 {points[:, 2].max():.3f}")
        print(f"中心点: ({points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f})")
    
    if point_cloud.has_colors():
        print("包含颜色信息: 是")
    else:
        print("包含颜色信息: 否")
    
    if point_cloud.has_normals():
        print("包含法线信息: 是")
    else:
        print("包含法线信息: 否")


def main():
    parser = argparse.ArgumentParser(description="3D点云可视化工具")
    parser.add_argument("ply_file", help="PLY格式的点云文件路径")
    parser.add_argument("--window_name", default="3D点云可视化", help="窗口名称")
    
    args = parser.parse_args()
    
    try:
        # 加载点云
        point_cloud = load_point_cloud(args.ply_file)
        
        # 打印点云信息
        print_point_cloud_info(point_cloud)
        
        # 可视化点云
        visualize_point_cloud(point_cloud, args.window_name)
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()