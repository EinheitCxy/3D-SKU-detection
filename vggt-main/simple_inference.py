#!/usr/bin/env python3
"""
VGGT 推理脚本（精简版）
功能：输入一组图片，直接生成最终的 3D 点云文件 (.ply)。

用法：
1. 确保已安装所需库: pip install torch vggt open3d
2. 修改 main 函数中的 image_paths 指向你的图片文件夹。
3. 运行脚本: python simple_inference.py
"""

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import numpy as np

def save_point_cloud_to_ply(points_3d, output_file):
    """将 (S, H, W, 3) 格式的 3D 点云保存为 .ply 文件。"""
    try:
        import open3d as o3d
        
        # 将点云数据重塑为 (N, 3) 的标准格式
        points_reshaped = points_3d.reshape((-1, 3))
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_reshaped)
        
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"💾 最终点云已保存到: {output_file}")

    except ImportError:
        print("\n⚠️ 警告: open3d 库未安装 (请运行: pip install open3d)。")
        print("无法将点云保存为 .ply 文件。")
    except Exception as e:
        print(f"\n❌ 错误: 保存 .ply 文件时出错: {e}")


def generate_point_cloud(image_paths, output_dir="./output"):
    """
    运行 VGGT 推理并直接保存最终的 3D 点云。
    
    Args:
        image_paths: 图片路径列表或包含图片的目录。
        output_dir: 输出目录。
    """
    # --- 1. 环境与设备配置 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"🔧 使用设备: {device}, 数据类型: {dtype}")
    
    # --- 2. 加载模型 ---
    print("🔄 正在加载 VGGT 模型...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("✅ 模型加载完成")
    
    # --- 3. 加载和预处理图片 ---
    print(f"🔄 正在从 '{image_paths}' 加载图片...")
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_images = [f for ext in extensions for f in glob.glob(os.path.join(image_paths, ext))]
        image_paths = sorted(all_images)
    
    if not image_paths:
        print(f"❌ 错误: 在路径 '{image_paths}' 中未找到任何图片。")
        return

    print(f"📸 找到 {len(image_paths)} 张图片")
    images = load_and_preprocess_images(image_paths).to(device)
    
    # --- 4. 运行模型推理 ---
    print("🚀 正在运行 VGGT 推理...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    print("✅ 推理完成")
    
    # --- 5. 提取点云并保存 ---
    print("🔄 正在提取并保存最终的 3D 点云...")
    
    # 从预测结果中直接获取最终的世界坐标点云
    world_points = predictions["world_points"].cpu().numpy()
    
    os.makedirs(output_dir, exist_ok=True)
    output_ply_file = os.path.join(output_dir, "final_point_cloud.ply")
    save_point_cloud_to_ply(world_points, output_ply_file)


def main():
    """脚本主入口"""
    # --- 请修改这里 ---
    # 设置图片文件夹路径（可以使用相对或绝对路径）
    # 示例: 使用项目自带的 "examples/cup" 文件夹
    image_paths = "examples/cup"
    
    # 检查路径是否存在
    if not os.path.exists(image_paths):
        print(f"❌ 错误: 示例图片路径 '{image_paths}' 不存在。")
        print("请确保您正处于 vggt-main 目录下，或提供正确的图片路径。")
        return

    # 运行点云生成流程
    generate_point_cloud(image_paths)
    
    print("\n🎉 处理流程结束。")

if __name__ == "__main__":
    main()
