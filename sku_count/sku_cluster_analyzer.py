#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKU 3D聚类分析工具
使用VGGT with Bundle Adjustment进行3D重建，然后对不同图片中的SKU进行3D聚类分析
"""

import json
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Tuple, Optional
import argparse
from sklearn.cluster import DBSCAN
from datetime import datetime

# 添加VGGT路径
sys.path.append('../vggt-main')

def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

try:
    from advanced_3d_reconstruction import Advanced3DReconstructor
    ADVANCED_3D_AVAILABLE = True
except ImportError:
    print("警告: 无法导入高级3D重建模块")
    ADVANCED_3D_AVAILABLE = False

try:
    # VGGT已集成到Advanced3DReconstructor中
    VGGT_AVAILABLE = True
    print("✅ VGGT模块已通过Advanced3DReconstructor集成")
except ImportError as e:
    print(f"警告: 无法集成VGGT模块: {e}")
    VGGT_AVAILABLE = False

# 导入GLB/GLTF导出工具
try:
    from gltf_export_utils import export_point_cloud_to_gltf
    GLTF_AVAILABLE = True
    print("✅ GLB/GLTF导出功能已启用")
except ImportError as e:
    print(f"警告: 无法导入GLB/GLTF导出工具: {e}")
    GLTF_AVAILABLE = False

# 导入设备选择工具
try:
    from device_utils import get_optimal_device
    DEVICE_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入设备选择工具: {e}")
    DEVICE_UTILS_AVAILABLE = False

class SKUClusterAnalyzer:
    """SKU聚类分析器"""
    
    def __init__(self, image_dir: str, detection_file: str):
        """
        初始化分析器
        
        Args:
            image_dir: 图片目录路径
            detection_file: 检测结果JSON文件路径
        """
        self.image_dir = Path(image_dir)
        self.detection_file = Path(detection_file)
        self.images = []
        self.detections = []
        
        # SKU分析相关
        self.sku_centers_2d = []     # SKU检测框中心点 [{'center': (x, y), 'image_idx': int, 'confidence': float}, ...]
        self.sku_centers_3d = []     # SKU中心的3D位置 [{'2d': (x, y), '3d': (x, y, z), 'image_idx': int, 'confidence': float}, ...]
        self.sku_clusters = []       # SKU聚类结果
        self.cluster_analysis = {}   # 聚类分析结果
        
        # 3D重建相关
        self.reconstructor = None
        self.camera_poses = []
        self.camera_intrinsics = None
        self.point_cloud = None
        
        # VGGT相关（已集成到Advanced3DReconstructor中）
        self.use_ba = True  # 默认启用Bundle Adjustment
        self.vggt_output = None
        
    def load_data(self) -> None:
        """加载图片数据和检测结果"""
        print("正在加载图片数据和检测结果...")
        
        # 加载检测结果
        if self.detection_file.exists():
            with open(self.detection_file, 'r', encoding='utf-8') as f:
                self.detections = json.load(f)
            print(f"加载了 {len(self.detections)} 个检测结果")
        else:
            print(f"警告: 检测文件 {self.detection_file} 不存在")
            self.detections = []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.image_dir.glob(f'*{ext}'))
            
        self.images = sorted(image_files)
        print(f"找到 {len(self.images)} 张图片")
        
        if len(self.images) == 0:
            raise ValueError(f"在目录 {self.image_dir} 中未找到图片文件")
        
        # 提取SKU中心点
        self._extract_sku_centers()
    
    def initialize_3d_reconstruction(self) -> None:
        """初始化3D重建"""
        print("正在初始化3D重建...")
        
        if ADVANCED_3D_AVAILABLE:
            # 使用高级重建器（已集成VGGT with BA）
            self.reconstructor = Advanced3DReconstructor(
                str(self.image_dir), 
                str(self.detection_file)
            )
            self.reconstructor.load_data()
            self.reconstructor.reconstruct_with_vggt()
            
            # 获取相机参数和点云
            self.camera_poses = self.reconstructor.camera_poses
            self.camera_intrinsics = self.reconstructor.camera_intrinsics
            self.point_cloud = self.reconstructor.point_cloud
            
            # 初始化VGGT输出（如果使用高级重建器）
            if hasattr(self.reconstructor, 'vggt_output'):
                self.vggt_output = self.reconstructor.vggt_output
        else:
            # 使用简化的相机参数估计
            self._estimate_simple_camera_params()
            
        print(f"相机参数已初始化，包含 {len(self.camera_poses)} 个相机姿态")
    
        
    def _estimate_simple_camera_params(self) -> None:
        """估计简化的相机参数"""
        num_images = len(self.images)
        
        # 模拟相机内参
        self.camera_intrinsics = {
            'width': 1920,
            'height': 1080,
            'fx': 1000,
            'fy': 1000,
            'cx': 960,
            'cy': 540
        }
        
        # 模拟相机外参（围绕场景旋转）
        for i in range(num_images):
            angle = i * 2 * np.pi / num_images
            radius = 5.0
            
            # 相机位置
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 2.0
            
            # 相机朝向场景中心
            look_at = np.array([0, 0, 4])
            camera_pos = np.array([x, y, z])
            
            # 计算旋转矩阵
            forward = look_at - camera_pos
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, np.array([0, 0, 1]))
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            
            rotation_matrix = np.column_stack([right, up, -forward])
            
            pose = {
                'position': camera_pos,
                'rotation': rotation_matrix,
                'image_id': i
            }
            
            self.camera_poses.append(pose)
        
        # 创建简单点云
        self._create_simple_point_cloud()
    
    def _create_simple_point_cloud(self) -> None:
        """创建简单的点云"""
        points = []
        
        # 创建货架形状的点云
        for x in np.linspace(-4, 4, 50):
            for y in np.linspace(-2, 2, 20):
                for z in np.linspace(0, 8, 40):
                    if (abs(x) > 3.5 or abs(y) > 1.5 or 
                        z % 2.5 < 0.2):  # 货架结构
                        points.append([x, y, z])
        
        self.point_cloud = np.array(points)
    
    def backproject_point_to_3d(self, x: float, y: float, image_idx: int, 
                               depth_method: str = 'fixed') -> Optional[Tuple[float, float, float]]:
        """
        将2D点反投影到3D空间
        
        Args:
            x, y: 2D图像坐标
            image_idx: 图片索引
            depth_method: 深度估计方法 ('fixed', 'point_cloud', 'depth_map')
            
        Returns:
            3D点坐标 (x, y, z) 或 None
        """
        if image_idx >= len(self.camera_poses):
            print(f"错误: 图片索引 {image_idx} 超出范围")
            return None
        
        camera_pose = self.camera_poses[image_idx]
        
        # 获取相机内参
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        # 计算射线方向
        u = (x - cx) / fx
        v = (y - cy) / fy
        ray_direction = np.array([u, v, 1.0])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # 转换到世界坐标系
        rotation = camera_pose['rotation']
        ray_world = rotation @ ray_direction
        camera_pos = camera_pose['position']
        
        # 根据不同方法估计深度
        if depth_method == 'fixed':
            # 固定深度
            target_depth = 5.0
            point_3d = camera_pos + target_depth * ray_world
            
        elif depth_method == 'point_cloud' and self.point_cloud is not None:
            # 使用点云估计深度
            point_3d = self._estimate_depth_from_point_cloud(
                camera_pos, ray_world
            )
            
        elif depth_method == 'depth_map':
            # 使用深度图估计深度（需要实现深度估计）
            point_3d = self._estimate_depth_from_depth_map(
                x, y, image_idx, camera_pos, ray_world
            )
            
        else:
            # 默认使用平面投影
            target_z = 4.0
            if ray_world[2] != 0:
                t = (target_z - camera_pos[2]) / ray_world[2]
                point_3d = camera_pos + t * ray_world
            else:
                return None
        
        return tuple(point_3d.tolist())
    
    def _estimate_depth_from_point_cloud(self, camera_pos: np.ndarray, 
                                       ray_world: np.ndarray) -> np.ndarray:
        """从点云估计深度"""
        if self.point_cloud is None:
            return camera_pos + 5.0 * ray_world
        
        # 计算射线与点云的最近交点
        # 简化方法：找到射线附近的点云点
        min_distance = float('inf')
        best_point = None
        
        for point in self.point_cloud:
            # 计算点到射线的距离
            to_point = point - camera_pos
            projection_length = np.dot(to_point, ray_world)
            
            if projection_length > 0:  # 点在射线前方
                projection_point = camera_pos + projection_length * ray_world
                distance = np.linalg.norm(point - projection_point)
                
                if distance < min_distance:
                    min_distance = distance
                    best_point = point
        
        if best_point is not None and min_distance < 0.5:
            return best_point
        else:
            # 如果没有找到合适的点，使用默认深度
            return camera_pos + 5.0 * ray_world
    
    def _estimate_depth_from_depth_map(self, x: float, y: float, image_idx: int,
                                     camera_pos: np.ndarray, 
                                     ray_world: np.ndarray) -> np.ndarray:
        """从深度图估计深度"""
        if self.vggt_output is not None and image_idx < len(self.vggt_output['preds']):
            # 使用VGGT的深度预测
            pred = self.vggt_output['preds'][image_idx]
            
            # 获取3D点云 (1, H, W, 3)
            pts3d = pred['pts3d_in_other_view'].cpu().numpy()
            if len(pts3d.shape) == 4:
                pts3d = pts3d[0]  # 移除batch维度 (H, W, 3)
            
            # 将图像坐标转换为点云索引
            height, width = pts3d.shape[:2]
            
            # 获取原始图像尺寸
            original_height = self.camera_intrinsics['height']
            original_width = self.camera_intrinsics['width']
            
            # 缩放坐标到点云尺寸
            scaled_x = int(x * width / original_width)
            scaled_y = int(y * height / original_height)
            
            # 确保坐标在范围内
            scaled_x = max(0, min(scaled_x, width - 1))
            scaled_y = max(0, min(scaled_y, height - 1))
            
            # 获取对应的3D点
            point_3d = pts3d[scaled_y, scaled_x]
            
            # 检查深度有效性
            if point_3d[2] > 0 and point_3d[2] < 100:  # 深度在合理范围内
                return point_3d
            else:
                # 如果深度无效，使用邻近点的平均值
                window_size = 5
                y_start = max(0, scaled_y - window_size)
                y_end = min(height, scaled_y + window_size + 1)
                x_start = max(0, scaled_x - window_size)
                x_end = min(width, scaled_x + window_size + 1)
                
                region = pts3d[y_start:y_end, x_start:x_end]
                valid_depths = region[:, :, 2]
                valid_mask = (valid_depths > 0) & (valid_depths < 100)
                
                if np.any(valid_mask):
                    avg_depth = np.mean(valid_depths[valid_mask])
                    direction = ray_world / np.linalg.norm(ray_world)
                    return camera_pos + avg_depth * direction
        
        # 使用固定深度
        depth = 5.0
        return camera_pos + depth * ray_world
    
    def _extract_sku_centers(self) -> None:
        """从检测结果中提取SKU中心点"""
        print("正在提取SKU中心点...")
        
        self.sku_centers_2d = []
        
        for image_idx, detection_data in enumerate(self.detections):
            if image_idx >= len(self.images):
                break
                
            if 'objects' not in detection_data:
                continue
                
            for obj in detection_data['objects']:
                if 'position' in obj and 'confidences' in obj:
                    x1, y1, x2, y2 = obj['position']
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    confidence = obj['confidences'].get('det', 0.0)
                    
                    sku_center = {
                        'center': (center_x, center_y),
                        'image_idx': image_idx,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2)
                    }
                    self.sku_centers_2d.append(sku_center)
        
        print(f"提取了 {len(self.sku_centers_2d)} 个SKU中心点")
    
    def map_sku_centers_to_3d(self) -> None:
        """将SKU中心点映射到3D空间"""
        print("正在将SKU中心点映射到3D空间...")
        
        if not self.sku_centers_2d:
            print("没有SKU中心点需要映射")
            return
        
        if not self.camera_poses:
            print("错误: 请先初始化3D重建")
            return
        
        self.sku_centers_3d = []
        
        for sku_center in self.sku_centers_2d:
            center_x, center_y = sku_center['center']
            image_idx = sku_center['image_idx']
            
            if image_idx < len(self.camera_poses):
                # 使用depth_map方法获得最准确的3D位置
                point_3d = self.backproject_point_to_3d(
                    center_x, center_y, image_idx, 'depth_map'
                )
                
                if point_3d is not None:
                    sku_3d = {
                        '2d': (center_x, center_y),
                        '3d': point_3d,
                        'image_idx': image_idx,
                        'confidence': sku_center['confidence'],
                        'bbox': sku_center['bbox']
                    }
                    self.sku_centers_3d.append(sku_3d)
        
        print(f"成功映射 {len(self.sku_centers_3d)} 个SKU中心点到3D空间")
    
    def analyze_sku_clusters(self, eps: float = 0.5, min_samples: int = 2) -> None:
        """
        使用3D距离对SKU进行聚类分析
        重要：同一张图片上的物体不会被聚类，只有不同图片中3D空间位置相近的物体才会聚类
        
        Args:
            eps: DBSCAN聚类的距离阈值
            min_samples: DBSCAN聚类的最小样本数
        """
        print("正在进行SKU聚类分析...")
        print("注意：同一张图片上的物体不会被聚类")
        
        if not self.sku_centers_3d:
            print("错误: 请先映射SKU中心点到3D空间")
            return
        
        # 准备3D坐标数据和图片索引
        points_3d = []
        image_indices = []
        
        for i, sku in enumerate(self.sku_centers_3d):
            points_3d.append(sku['3d'])
            image_indices.append(sku['image_idx'])
        
        points_3d = np.array(points_3d)
        image_indices = np.array(image_indices)
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(points_3d)
        
        # **关键步骤：后处理聚类结果，确保同一图片的物体不会被分到同一聚类**
        # 创建一个映射来检查每个聚类中的图片
        cluster_to_images = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # 跳过噪声点
                continue
            if label not in cluster_to_images:
                cluster_to_images[label] = {}
            image_idx = image_indices[i]
            if image_idx not in cluster_to_images[label]:
                cluster_to_images[label][image_idx] = []
            cluster_to_images[label][image_idx].append(i)
        
        # 重新分配聚类标签，确保每个聚类中的物体来自不同图片
        new_cluster_labels = cluster_labels.copy()
        next_available_cluster_id = max(cluster_labels) + 1 if len(cluster_labels) > 0 else 0
        
        for cluster_id, images_dict in cluster_to_images.items():
            # 检查是否有图片包含多个物体
            images_with_multiple_objects = []
            for img_idx, point_indices in images_dict.items():
                if len(point_indices) > 1:
                    images_with_multiple_objects.append((img_idx, point_indices))
            
            if images_with_multiple_objects:
                # 对于每个有多个物体的图片，只保留第一个物体在原聚类中
                # 其余物体标记为噪声点
                for img_idx, point_indices in images_with_multiple_objects:
                    # 保留第一个物体在原聚类中
                    for i, point_idx in enumerate(point_indices[1:], 1):
                        new_cluster_labels[point_idx] = -1  # 标记为噪声点
        
        # 使用修正后的聚类标签
        cluster_labels = new_cluster_labels
        
        # 整理聚类结果
        self.sku_clusters = []
        unique_labels = set(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # 噪声点
                continue
                
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_skus = [self.sku_centers_3d[i] for i in cluster_indices]
            
            # 验证：确保这个聚类中的物体来自不同的图片
            images_in_cluster = set(sku['image_idx'] for sku in cluster_skus)
            if len(images_in_cluster) != len(cluster_skus):
                print(f"警告: 聚类 {cluster_id} 仍然包含来自同一图片的多个物体，这不应该发生")
                continue
            
            # 计算聚类中心
            cluster_points = points_3d[cluster_indices]
            cluster_center = np.mean(cluster_points, axis=0)
            
            # 计算聚类内部距离统计
            distances = []
            for i in range(len(cluster_points)):
                for j in range(i + 1, len(cluster_points)):
                    dist = np.linalg.norm(cluster_points[i] - cluster_points[j])
                    distances.append(dist)
            
            cluster_info = {
                'cluster_id': cluster_id,
                'skus': cluster_skus,
                'center_3d': cluster_center.tolist(),
                'size': len(cluster_skus),
                'avg_confidence': np.mean([sku['confidence'] for sku in cluster_skus]),
                'images_involved': list(set(sku['image_idx'] for sku in cluster_skus)),
                'internal_distances': distances,
                'avg_internal_distance': np.mean(distances) if distances else 0.0,
                'max_internal_distance': np.max(distances) if distances else 0.0
            }
            
            self.sku_clusters.append(cluster_info)
        
        # 处理噪声点
        noise_indices = np.where(cluster_labels == -1)[0]
        noise_skus = [self.sku_centers_3d[i] for i in noise_indices]
        
        # 生成分析报告
        self.cluster_analysis = {
            'total_skus': len(self.sku_centers_3d),
            'total_clusters': len(self.sku_clusters),
            'noise_points': len(noise_skus),
            'clustering_params': {'eps': eps, 'min_samples': min_samples},
            'timestamp': datetime.now().isoformat(),
            'clusters': self.sku_clusters,
            'noise_skus': noise_skus,
            'cross_image_only': True  # 标记这是跨图片聚类
        }
        
        print(f"聚类分析完成:")
        print(f"  - 总SKU数: {self.cluster_analysis['total_skus']}")
        print(f"  - 发现聚类: {self.cluster_analysis['total_clusters']} 个")
        print(f"  - 噪声点: {self.cluster_analysis['noise_points']} 个")
        print("✅ 聚类规则：仅对不同图片中的物体进行聚类")
        
        for cluster in self.sku_clusters:
            print(f"  - 聚类 {cluster['cluster_id']}: {cluster['size']} 个SKU, "
                  f"来自 {len(cluster['images_involved'])} 张不同图片, "
                  f"平均距离 {cluster['avg_internal_distance']:.2f}m")
    
    def generate_sku_analysis_report(self) -> str:
        """生成SKU分析报告"""
        if not self.cluster_analysis:
            return "尚未进行聚类分析"
        
        report = []
        report.append("=" * 60)
        report.append("3D SKU聚类分析报告（跨图片聚类）")
        report.append("=" * 60)
        report.append(f"分析时间: {self.cluster_analysis['timestamp']}")
        report.append(f"图片数量: {len(self.images)}")
        report.append(f"检测结果: {len(self.detections)} 个图片的检测数据")
        report.append("")
        
        report.append("聚类规则:")
        report.append("  ✅ 只对不同图片中的物体进行聚类")
        report.append("  ❌ 同一图片内的物体不会被聚类")
        report.append("  📍 基于3D空间距离进行聚类")
        report.append("")
        
        report.append("聚类参数:")
        params = self.cluster_analysis['clustering_params']
        report.append(f"  - 距离阈值 (eps): {params['eps']}m")
        report.append(f"  - 最小样本数 (min_samples): {params['min_samples']}")
        report.append("")
        
        report.append("总体统计:")
        report.append(f"  - 总SKU检测数: {self.cluster_analysis['total_skus']}")
        report.append(f"  - 识别的SKU类型: {self.cluster_analysis['total_clusters']} 个")
        report.append(f"  - 独立SKU (噪声点): {self.cluster_analysis['noise_points']} 个")
        report.append("")
        
        if self.sku_clusters:
            report.append("详细聚类信息:")
            for cluster in self.sku_clusters:
                report.append(f"  聚类 {cluster['cluster_id']}:")
                report.append(f"    - SKU数量: {cluster['size']}")
                report.append(f"    - 涉及图片: {len(cluster['images_involved'])} 张 {cluster['images_involved']}")
                report.append(f"    - 平均置信度: {cluster['avg_confidence']:.3f}")
                report.append(f"    - 3D中心位置: ({cluster['center_3d'][0]:.2f}, {cluster['center_3d'][1]:.2f}, {cluster['center_3d'][2]:.2f})")
                report.append(f"    - 平均内部距离: {cluster['avg_internal_distance']:.3f}m")
                report.append(f"    - 最大内部距离: {cluster['max_internal_distance']:.3f}m")
                report.append("")
        
        return "\n".join(report)
    
    def visualize_clusters_3d(self, save_path: Optional[str] = None) -> None:
        """显示3D聚类可视化结果"""
        print("正在生成3D聚类可视化...")
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制背景点云
        if self.point_cloud is not None:
            ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], self.point_cloud[:, 2], 
                      c='lightblue', s=1, alpha=0.3, label='背景点云')
        
        # 绘制SKU聚类结果
        if self.sku_clusters:
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.sku_clusters)))
            
            for i, cluster in enumerate(self.sku_clusters):
                cluster_points = np.array([sku['3d'] for sku in cluster['skus']])
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                          c=[colors[i]], s=80, alpha=0.7, 
                          label=f'SKU类型 {cluster["cluster_id"]} ({cluster["size"]}个)')
                
                # 绘制聚类中心
                center = cluster['center_3d']
                ax.scatter([center[0]], [center[1]], [center[2]], 
                          c=[colors[i]], s=200, alpha=1.0, marker='x', linewidths=3)
                
                # 添加聚类标签
                ax.text(center[0], center[1], center[2], 
                       f'C{cluster["cluster_id"]}', fontsize=12, fontweight='bold')
        
        # 绘制噪声点（独立SKU）
        if hasattr(self, 'cluster_analysis') and self.cluster_analysis and 'noise_skus' in self.cluster_analysis:
            noise_skus = self.cluster_analysis['noise_skus']
            if noise_skus:
                noise_points = np.array([sku['3d'] for sku in noise_skus])
                ax.scatter(noise_points[:, 0], noise_points[:, 1], noise_points[:, 2], 
                          c='gray', s=60, alpha=0.6, marker='o', label='独立SKU')
        
        # 绘制相机位置
        if self.camera_poses:
            camera_positions = np.array([pose['position'] for pose in self.camera_poses])
            ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                      c='green', s=50, alpha=0.7, label='相机位置', marker='^')
        
        ax.set_xlabel('X轴 (米)')
        ax.set_ylabel('Y轴 (米)')
        ax.set_zlabel('Z轴 (米)')
        
        # 设置标题
        if self.sku_clusters:
            title_parts = [
                '3D SKU聚类分析结果（跨图片聚类）',
                f'{len(self.sku_clusters)} 个SKU类型',
                f'{sum(cluster["size"] for cluster in self.sku_clusters)} 个检测'
            ]
            ax.set_title(' - '.join(title_parts))
        else:
            ax.set_title('3D SKU分析结果')
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D可视化已保存到: {save_path}")
        
        plt.show()
    
    def export_results(self, output_dir: str = "output") -> None:
        """导出分析结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 保存聚类分析结果 (JSON格式)
            if self.cluster_analysis:
                analysis_file = output_path / f"sku_cluster_analysis_{timestamp}.json"
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    # 使用convert_numpy_types处理所有numpy类型
                    analysis_serializable = convert_numpy_types(self.cluster_analysis)
                    json.dump(analysis_serializable, f, ensure_ascii=False, indent=2)
                print(f"聚类分析结果已保存到: {analysis_file}")
                
                # 保存文本报告
                report_file = output_path / f"sku_analysis_report_{timestamp}.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(self.generate_sku_analysis_report())
                print(f"分析报告已保存到: {report_file}")
            
            # 保存3D点云数据 (用于可视化)
            if self.sku_centers_3d:
                points_file = output_path / f"sku_centers_3d_{timestamp}.json"
                sku_centers_serializable = convert_numpy_types(self.sku_centers_3d)
                with open(points_file, 'w', encoding='utf-8') as f:
                    json.dump(sku_centers_serializable, f, ensure_ascii=False, indent=2)
                print(f"3D坐标已保存到: {points_file}")
            
            # 保存GLB格式的3D可视化
            if GLTF_AVAILABLE and self.sku_clusters:
                glb_file = output_path / f"sku_clusters_3d_{timestamp}.glb"
                self.export_clusters_to_glb(glb_file)
                
        except Exception as e:
            print(f"保存结果时出错: {e}")
    
    def export_clusters_to_glb(self, output_path: Path) -> None:
        """导出聚类结果到GLB格式"""
        try:
            print("正在导出GLB格式的聚类结果...")
            
            # 准备所有点云数据
            all_points = []
            all_colors = []
            
            # 添加背景点云（如果有）
            if self.point_cloud is not None:
                all_points.append(self.point_cloud)
                background_colors = np.array([[0.7, 0.7, 1.0]] * len(self.point_cloud))
                all_colors.append(background_colors)
            
            # 添加SKU聚类结果
            if self.sku_clusters:
                colors = plt.cm.Set1(np.linspace(0, 1, len(self.sku_clusters)))
                
                for i, cluster in enumerate(self.sku_clusters):
                    cluster_points = np.array([sku['3d'] for sku in cluster['skus']])
                    cluster_colors = np.array([colors[i][:3]] * len(cluster_points))
                    all_points.append(cluster_points)
                    all_colors.append(cluster_colors)
            
            # 添加独立SKU点
            if hasattr(self, 'cluster_analysis') and self.cluster_analysis and 'noise_skus' in self.cluster_analysis:
                noise_skus = self.cluster_analysis['noise_skus']
                if noise_skus:
                    noise_points = np.array([sku['3d'] for sku in noise_skus])
                    noise_colors = np.array([[0.5, 0.5, 0.5]] * len(noise_points))  # 灰色
                    all_points.append(noise_points)
                    all_colors.append(noise_colors)
            
            # 合并所有数据
            if all_points:
                combined_points = np.vstack(all_points)
                combined_colors = np.vstack(all_colors)
                
                # 导出GLB
                success = export_point_cloud_to_gltf(
                    combined_points,
                    combined_colors,
                    output_path,
                    point_size=0.02,
                    convert_to_mesh=True
                )
                
                if success:
                    print(f"✅ GLB文件已导出: {output_path}")
                else:
                    print(f"❌ GLB导出失败: {output_path}")
            else:
                print("❌ 没有有效的3D点数据可导出")
                
        except Exception as e:
            print(f"❌ 导出GLB时发生错误: {e}")
    
    def run_analysis(self, output_dir: str = "output", 
                    eps: float = 0.5, min_samples: int = 2,
                    visualize: bool = True) -> None:
        """运行完整的SKU聚类分析流程"""
        print("开始SKU 3D聚类分析...")
        print(f"聚类参数: eps={eps}, min_samples={min_samples}")
        print("重要：只对不同图片中的物体进行聚类")
        print("=" * 60)
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 初始化3D重建
            self.initialize_3d_reconstruction()
            
            # 3. 映射SKU中心点到3D
            self.map_sku_centers_to_3d()
            
            if not self.sku_centers_3d:
                print("❌ 没有找到可映射的SKU中心点")
                return
            
            # 4. 进行聚类分析
            self.analyze_sku_clusters(eps=eps, min_samples=min_samples)
            
            # 5. 生成并显示报告
            report = self.generate_sku_analysis_report()
            print("\n" + report)
            
            # 6. 导出结果
            self.export_results(output_dir)
            
            # 7. 3D可视化
            if visualize and self.cluster_analysis:
                output_path = Path(output_dir)
                viz_file = output_path / f"clusters_3d_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.visualize_clusters_3d(str(viz_file))
            
            print(f"\n✅ SKU聚类分析完成！")
            if self.cluster_analysis:
                print(f"发现 {self.cluster_analysis['total_clusters']} 个SKU类型")
                print(f"独立SKU: {self.cluster_analysis['noise_points']} 个")
                print(f"结果已保存到: {Path(output_dir).absolute()}")
            
        except Exception as e:
            print(f"❌ SKU分析失败: {str(e)}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="SKU 3D聚类分析工具",
        epilog="""
使用说明:
1. 自动提取检测框中心点，映射到3D空间
2. 使用DBSCAN对不同图片中的SKU进行3D聚类
3. 生成详细的分析报告和3D可视化
4. 自动保存所有结果到指定目录

重要特性:
- 使用VGGT with Bundle Adjustment进行高质量3D重建和深度估计
- 只对不同图片中的物体进行聚类（同一图片内的物体不会被聚类）
- 支持GLB格式导出，可在Blender等3D软件中查看
- 自动生成详细的分析报告
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image_dir", default="../imdata/sample", help="图片目录路径")
    parser.add_argument("--detection_file", default="../sku_detection.json", help="检测结果文件路径")
    parser.add_argument("--output_dir", default="output", help="输出目录路径")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN聚类距离阈值 (默认: 0.5)")
    parser.add_argument("--min_samples", type=int, default=2, help="DBSCAN最小样本数 (默认: 2)")
    parser.add_argument("--no_viz", action="store_true", help="跳过3D可视化")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SKU 3D聚类分析工具")
    print("=" * 60)
    print("功能:")
    print("  • 自动SKU中心提取和3D映射")
    print("  • 基于3D距离的跨图片SKU聚类分析")
    print("  • 同一图片内物体不会被聚类")
    print("  • 生成详细分析报告")
    print("  • 3D可视化结果")
    print("  • GLB格式导出")
    print("")
    
    # 创建分析器
    analyzer = SKUClusterAnalyzer(args.image_dir, args.detection_file)
    
    # 运行分析
    analyzer.run_analysis(
        output_dir=args.output_dir,
        eps=args.eps,
        min_samples=args.min_samples,
        visualize=not args.no_viz
    )


if __name__ == "__main__":
    main()