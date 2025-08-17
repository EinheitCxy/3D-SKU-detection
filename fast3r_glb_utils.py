"""
Fast3R GLB导出扩展模块
可以集成到现有的SKU分析脚本中
"""

import trimesh
import numpy as np
import torch
from datetime import datetime
import os


def fast3r_result_to_glb(result, output_path, min_conf_thr_percentile=30, export_mesh=False):
    """
    将Fast3R结果直接导出为GLB格式
    
    Args:
        result: Fast3R推理结果字典，包含'views'和'preds'
        output_path: 输出GLB文件路径
        min_conf_thr_percentile: 置信度阈值百分位数
        export_mesh: 是否导出网格(True)或点云(False)
    
    Returns:
        bool: 导出是否成功
    """
    try:
        if export_mesh:
            # 导出三角网格
            mesh = _create_mesh_from_result(result, min_conf_thr_percentile)
            mesh.export(output_path)
        else:
            # 导出点云
            vertices, colors = _extract_pointcloud_from_result(result, min_conf_thr_percentile)
            point_cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
            point_cloud.export(output_path)
        
        print(f"✅ GLB文件已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ GLB导出失败: {e}")
        return False


def _extract_pointcloud_from_result(result, min_conf_thr_percentile=30):
    """提取点云数据"""
    all_points = []
    all_colors = []
    
    views = result['views']
    preds = result['preds']
    
    for view, pred in zip(views, preds):
        # 获取数据
        pts3d = pred['pts3d']
        confidence = pred['conf']
        img = view['img']
        
        # 转换到numpy
        if isinstance(pts3d, torch.Tensor):
            pts3d = pts3d.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # 处理图像格式: [3, H, W] -> [H, W, 3]
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        
        # 规范化图像到[0, 255]
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
        
        # 提取有效点和颜色
        valid_points = pts3d[mask]
        valid_colors = img[mask]
        
        if len(valid_points) > 0:
            all_points.append(valid_points)
            all_colors.append(valid_colors)
    
    if not all_points:
        raise ValueError("没有提取到有效的点云数据")
    
    vertices = np.vstack(all_points)
    colors = np.vstack(all_colors)
    
    return vertices, colors


def _create_mesh_from_result(result, min_conf_thr_percentile=30):
    """创建三角网格"""
    from fast3r.dust3r.viz import pts3d_to_trimesh, cat_meshes
    
    views = result['views']
    preds = result['preds']
    meshes = []
    
    for view, pred in zip(views, preds):
        # 获取数据
        pts3d = pred['pts3d'].cpu().numpy()
        confidence = pred['conf'].cpu().numpy()
        img = view['img']
        
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # 处理图像格式
        if len(img.shape) == 3 and img.shape[0] == 3:
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


def export_sku_clusters_to_glb_simple(cluster_points_3d, output_dir="./output"):
    """
    将SKU聚类结果导出为简单的GLB点云格式
    
    Args:
        cluster_points_3d: dict, 键为cluster_id，值为3D点坐标数组
        output_dir: 输出目录
    
    Returns:
        str or None: GLB文件路径，失败时返回None
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
    
    for cluster_id, points in cluster_points_3d.items():
        if len(points) == 0:
            continue
            
        # 选择颜色
        if cluster_id == -1:  # 噪声点
            color = [128, 128, 128]  # 灰色
        else:
            color = cluster_colors[cluster_id % len(cluster_colors)]
        
        # 为当前聚类的所有点分配相同颜色
        points = np.array(points)
        colors = np.array([color] * len(points))
        
        all_points.append(points)
        all_colors.append(colors)
    
    if not all_points:
        print("❌ 没有有效的聚类点云数据")
        return None
    
    # 合并所有点云数据
    vertices = np.vstack(all_points)
    colors = np.vstack(all_colors)
    
    # 导出GLB文件
    os.makedirs(output_dir, exist_ok=True)
    glb_path = os.path.join(output_dir, f"sku_clusters_3d_{timestamp}.glb")
    
    try:
        point_cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
        point_cloud.export(glb_path)  
        print(f"✅ SKU聚类GLB文件已保存: {glb_path}")
        return glb_path
    except Exception as e:
        print(f"❌ SKU聚类GLB导出失败: {e}")
        return None


# 使用示例函数
def demo_usage():
    """
    使用示例
    """
    print("=== Fast3R GLB导出模块使用示例 ===")
    print()
    print("1. 从Fast3R结果导出点云GLB:")
    print("   fast3r_result_to_glb(result, 'output.glb', min_conf_thr_percentile=30)")
    print()
    print("2. 从Fast3R结果导出网格GLB:")
    print("   fast3r_result_to_glb(result, 'mesh.glb', export_mesh=True)")
    print()
    print("3. 从SKU聚类结果导出GLB:")
    print("   cluster_points = {0: [[x1,y1,z1], [x2,y2,z2]], 1: [[x3,y3,z3]]}")
    print("   export_sku_clusters_to_glb_simple(cluster_points, './output')")


if __name__ == "__main__":
    demo_usage()