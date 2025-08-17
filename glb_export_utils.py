"""
GLB/GLTF导出功能扩展
可以添加到现有的SKU分析脚本中
"""

import trimesh
import numpy as np
from datetime import datetime
import os

def export_pointcloud_to_glb(points_3d, colors, output_path):
    """
    将3D点云导出为GLB格式
    
    Args:
        points_3d: numpy array (N, 3) - 3D点坐标
        colors: numpy array (N, 3) - RGB颜色值 [0-255]
        output_path: str - 输出文件路径
    """
    try:
        # 创建trimesh点云对象
        point_cloud = trimesh.PointCloud(vertices=points_3d, colors=colors)
        
        # 导出为GLB格式
        point_cloud.export(output_path)
        print(f"✅ 点云已导出为GLB格式: {output_path}")
        
        return True
    except Exception as e:
        print(f"❌ GLB导出失败: {e}")
        return False

def export_sku_clusters_to_glb(cluster_results, output_dir="./output"):
    """
    将SKU聚类结果导出为GLB格式的3D可视化
    
    Args:
        cluster_results: dict - 包含聚类结果的字典
        output_dir: str - 输出目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备点云数据
    all_points = []
    all_colors = []
    
    # 为每个聚类分配不同颜色
    cluster_colors = [
        [255, 0, 0],    # 红色
        [0, 255, 0],    # 绿色
        [0, 0, 255],    # 蓝色
        [255, 255, 0],  # 黄色
        [255, 0, 255],  # 品红
        [0, 255, 255],  # 青色
        [255, 128, 0],  # 橙色
        [128, 0, 255],  # 紫色
    ]
    
    for cluster_id, cluster_points in cluster_results.items():
        if cluster_id == -1:  # 噪声点
            color = [128, 128, 128]  # 灰色
        else:
            color = cluster_colors[cluster_id % len(cluster_colors)]
        
        # 为当前聚类的所有点分配相同颜色
        cluster_colors_array = np.array([color] * len(cluster_points))
        
        all_points.append(cluster_points)
        all_colors.append(cluster_colors_array)
    
    # 合并所有点云数据
    if all_points:
        vertices = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        # 导出GLB文件
        glb_path = os.path.join(output_dir, f"sku_clusters_3d_{timestamp}.glb")
        os.makedirs(output_dir, exist_ok=True)
        
        success = export_pointcloud_to_glb(vertices, colors, glb_path)
        
        if success:
            # 同时导出GLTF格式
            gltf_path = os.path.join(output_dir, f"sku_clusters_3d_{timestamp}.gltf")
            point_cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
            point_cloud.export(gltf_path)
            print(f"✅ GLTF格式也已保存: {gltf_path}")
            
            return glb_path, gltf_path
    
    return None, None

# 使用示例：
# 在SKU聚类分析完成后调用
# glb_path, gltf_path = export_sku_clusters_to_glb(cluster_results, "./sku_count")