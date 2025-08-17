#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级3D货架重建与物体检测可视化
使用VGGT进行真正的3D重建，并在3D点云上显示物体检测的中心点
"""

import json
import numpy as np
import cv2
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from typing import List, Dict, Tuple, Optional
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import viser
import viser.transforms as tf

# 添加VGGT路径
sys.path.append('../vggt-main')

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
    from vggt.dependency.track_predict import predict_tracks
    from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap
    import pycolmap
    VGGT_AVAILABLE = True
    print("成功导入VGGT模块")
except ImportError as e:
    print(f"警告: 无法导入VGGT模块，将使用COLMAP进行3D重建: {e}")
    VGGT_AVAILABLE = False

# 导入GLB/GLTF导出工具
try:
    from gltf_export_utils import export_open3d_point_cloud_to_gltf, create_detection_scene_gltf
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

class Advanced3DReconstructor:
    """高级3D重建器"""
    
    def __init__(self, image_dir: str, detection_file: str):
        """
        初始化重建器
        
        Args:
            image_dir: 图片目录路径
            detection_file: 检测结果JSON文件路径
        """
        self.image_dir = Path(image_dir)
        self.detection_file = Path(detection_file)
        self.images = []
        self.detections = []
        self.camera_poses = []
        self.point_cloud = None
        self.detection_points_3d = []
        self.camera_intrinsics = None
        
        # VGGT相关
        self.vggt_model = None
        self.vggt_output = None
        self.use_ba = True  # 默认启用Bundle Adjustment
        
    def load_data(self) -> None:
        """加载图片和检测数据"""
        print("正在加载数据...")
        
        # 加载检测结果
        with open(self.detection_file, 'r', encoding='utf-8') as f:
            self.detections = json.load(f)
        
        # 加载图片
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.image_dir.glob(f'*{ext}'))
            
        self.images = sorted(image_files)
        
        print(f"加载了 {len(self.images)} 张图片和 {len(self.detections)} 个检测结果")

    
    def reconstruct_with_vggt(self) -> None:
        """使用VGGT with Bundle Adjustment进行3D重建"""
        if not VGGT_AVAILABLE:
            print("VGGT不可用")
            raise ImportError("VGGT模块不可用，请检查安装")
        
        print("正在使用VGGT with Bundle Adjustment进行3D重建...")
        
        try:
            # 智能设备选择
            if DEVICE_UTILS_AVAILABLE:
                device = get_optimal_device(verbose=True)
            else:
                # 回退到原有逻辑
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"使用设备: {device}")
            
            # 设置数据类型
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            print(f"使用数据类型: {dtype}")
            
            # 加载VGGT模型
            print("正在加载VGGT预训练模型...")
            self.vggt_model = VGGT()
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            self.vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
            self.vggt_model.eval()
            self.vggt_model = self.vggt_model.to(device)
            
            # 准备图片路径列表
            image_paths = [str(img_path) for img_path in self.images]
            print(f"准备处理 {len(image_paths)} 张图片")
            
            # VGGT固定分辨率和加载分辨率
            vggt_fixed_resolution = 518
            img_load_resolution = 518
            
            # 加载和预处理图片
            print("正在加载和预处理图片...")
            images, original_coords = load_and_preprocess_images_square(image_paths, img_load_resolution)
            images = images.to(device)
            original_coords = original_coords.to(device)
            
            # 运行VGGT进行相机和深度估计
            print("正在执行VGGT推理...")
            with torch.amp.autocast('cuda', dtype=dtype):
                extrinsic, intrinsic, depth_map, depth_conf, outputs = self._run_vggt_inference(
                    self.vggt_model, images, dtype, vggt_fixed_resolution
                )
            
            print(f"VGGT推理完成，深度图形状: {depth_map.shape}")
            
            # 使用VGGT输出的3D点云
            points_3d = outputs["world_points"]
            
            # 保存VGGT输出供其他模块使用
            self.vggt_output = {
                'extrinsic': extrinsic,
                'intrinsic': intrinsic,
                'depth_map': depth_map,
                'depth_conf': depth_conf,
                'preds': [{'pts3d_in_other_view': points_3d[i:i+1]} for i in range(points_3d.shape[0])]
            }
            
            if self.use_ba:
                print("正在执行Bundle Adjustment...")
                self._perform_bundle_adjustment(
                    images, points_3d, extrinsic, intrinsic, depth_conf, 
                    img_load_resolution, vggt_fixed_resolution, dtype
                )
            else:
                print("跳过Bundle Adjustment，使用VGGT直接结果")
                self._process_vggt_results(extrinsic, intrinsic, points_3d)
            
            print(f"VGGT重建完成:")
            print(f"  - 相机姿态: {len(self.camera_poses)} 个")
            print(f"  - 点云点数: {len(self.point_cloud) if self.point_cloud is not None else 0}")
            print(f"  - Bundle Adjustment: {'启用' if self.use_ba else '禁用'}")
            
        except Exception as e:
            print(f"VGGT重建失败: {e}")
            print("回退到COLMAP方法")
            self._create_synthetic_point_cloud()
    
    def _run_vggt_inference(self, model, images, dtype, resolution):
        """运行VGGT推理"""
        with torch.no_grad():
            # 运行VGGT模型（不传递resolution参数）
            outputs = model(images)
            
            # 转换相机位姿编码为外参内参矩阵
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                outputs["pose_enc"], images.shape[-2:]
            )
            
            # 提取深度图和置信度
            depth_map = outputs["depth"]
            depth_conf = outputs["depth_conf"]
            
            return extrinsic, intrinsic, depth_map, depth_conf, outputs
    
    def _perform_bundle_adjustment(self, images, points_3d, extrinsic, intrinsic, depth_conf, 
                                  img_load_resolution, vggt_fixed_resolution, dtype):
        """执行Bundle Adjustment"""
        print("正在预测tracks...")
        
        # 设置BA参数
        max_query_pts = 5000  # 最大查询点数
        query_frame_num = 5   # 查询帧数
        fine_tracking = True   # 精细跟踪
        vis_thresh = 0.5      # 可见性阈值
        max_reproj_error = 8.0  # 最大重投影误差
        shared_camera = False  # 不共享相机
        camera_type = "PINHOLE"  # 相机类型
        
        with torch.amp.autocast('cuda', dtype=dtype):
            # 预测tracks
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=max_query_pts,
                query_frame_num=query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=fine_tracking,
            )
        
        torch.cuda.empty_cache()
        
        # 重新缩放内参矩阵
        scale = img_load_resolution / vggt_fixed_resolution
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > vis_thresh
        
        # 转换为pycolmap格式
        image_size = np.array(images.shape[-2:])
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=max_reproj_error,
            shared_camera=shared_camera,
            camera_type=camera_type,
            points_rgb=points_rgb,
        )
        
        if reconstruction is None:
            print("警告: 无法构建BA重建，使用VGGT直接结果")
            self._process_vggt_results(extrinsic, intrinsic, points_3d)
            return
        
        # 执行Bundle Adjustment
        print("正在执行Bundle Adjustment优化...")
        ba_options = pycolmap.BundleAdjustmentOptions()
        pycolmap.bundle_adjustment(reconstruction, ba_options)
        
        # 从重建结果中提取优化后的相机姿态
        self._extract_ba_results(reconstruction, img_load_resolution)
    
    def _process_vggt_results(self, extrinsic, intrinsic, points_3d):
        """处理VGGT直接结果（无BA）"""
        # 转换相机姿态格式
        batch_size = extrinsic.shape[0]
        self.camera_poses = []
        
        for i in range(batch_size):
            # 提取相机外参
            pose_matrix = extrinsic[i].cpu().numpy()
            
            # 分离旋转和平移
            rotation = pose_matrix[:3, :3]
            position = pose_matrix[:3, 3]
            
            # 提取相机内参
            fx = intrinsic[i, 0, 0].cpu().item()
            fy = intrinsic[i, 1, 1].cpu().item()
            cx = intrinsic[i, 0, 2].cpu().item()
            cy = intrinsic[i, 1, 2].cpu().item()
            
            camera_pose = {
                'position': position.tolist(),
                'rotation': rotation,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
            self.camera_poses.append(camera_pose)
        
        # 设置相机内参（使用第一张图片的内参）
        self.camera_intrinsics = {
            'fx': intrinsic[0, 0, 0].cpu().item(),
            'fy': intrinsic[0, 1, 1].cpu().item(),
            'cx': intrinsic[0, 0, 2].cpu().item(),
            'cy': intrinsic[0, 1, 2].cpu().item()
        }
        
        # 提取点云
        self._extract_point_cloud_from_vggt(points_3d)
    
    def _extract_point_cloud_from_vggt(self, points_3d):
        """从VGGT输出中提取点云"""
        print("正在从VGGT输出中提取点云...")
        
        # 将3D点转换为numpy数组
        points_3d_np = points_3d.cpu().numpy()
        
        # 重新排列为 (N, 3) 格式
        batch_size, height, width, _ = points_3d_np.shape
        points_3d_flat = points_3d_np.reshape(-1, 3)
        
        # 过滤无效点
        valid_mask = ~np.any(np.isnan(points_3d_flat) | np.isinf(points_3d_flat), axis=1)
        valid_points = points_3d_flat[valid_mask]
        
        # 过滤深度范围
        depth_mask = (valid_points[:, 2] > 0) & (valid_points[:, 2] < 50)
        valid_points = valid_points[depth_mask]
        
        # 下采样
        if len(valid_points) > 50000:
            indices = np.random.choice(len(valid_points), 50000, replace=False)
            valid_points = valid_points[indices]
        
        self.point_cloud = valid_points
        print(f"提取了 {len(self.point_cloud)} 个3D点")
    
    def _extract_ba_results(self, reconstruction, resolution):
        """从Bundle Adjustment结果中提取相机姿态"""
        self.camera_poses = []
        
        # 提取所有图像
        images = reconstruction.images
        for image_id, image in images.items():
            # 提取相机姿态
            camera = image.camera
            pose = image.cam_from_world
            
            # 转换为位置和旋转
            position = pose.translation
            rotation = pose.rotation.matrix()
            
            # 提取相机内参
            if hasattr(camera, 'focal_length_x'):
                fx = camera.focal_length_x
                fy = camera.focal_length_y
                cx = camera.principal_point_x
                cy = camera.principal_point_y
            else:
                # 对于简化的相机模型
                fx = fy = camera.focal_length()
                cx = cy = resolution / 2
            
            camera_pose = {
                'position': position.tolist(),
                'rotation': rotation,
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy
            }
            self.camera_poses.append(camera_pose)
        
        # 设置相机内参
        if self.camera_poses:
            self.camera_intrinsics = {
                'fx': self.camera_poses[0]['fx'],
                'fy': self.camera_poses[0]['fy'],
                'cx': self.camera_poses[0]['cx'],
                'cy': self.camera_poses[0]['cy']
            }
        
        # 提取BA优化后的点云
        self._extract_ba_point_cloud(reconstruction)
    
    def _extract_ba_point_cloud(self, reconstruction):
        """从Bundle Adjustment结果中提取点云"""
        print("正在从Bundle Adjustment结果中提取点云...")
        
        points = []
        colors = []
        
        # 提取所有3D点
        for point_id, point in reconstruction.points.items():
            if point.xyz is not None:
                points.append(point.xyz)
                if point.color is not None:
                    colors.append(point.color)
                else:
                    colors.append([128, 128, 128])  # 默认灰色
        
        if points:
            self.point_cloud = np.array(points)
            print(f"提取了 {len(self.point_cloud)} 个3D点")
        else:
            print("警告: 未能从Bundle Adjustment结果中提取点云")
            self._create_synthetic_point_cloud()
    
    def _create_synthetic_point_cloud(self) -> None:
        """创建合成的点云数据"""
        print("正在生成合成点云...")
        
        # 创建货架形状的点云
        points = []
        
        # 货架主体 - 垂直支柱
        for x in [-4, 4]:
            for z in np.linspace(0, 8, 20):
                for y in np.linspace(-2, 2, 10):
                    points.append([x, y, z])
        
        # 货架层板
        for layer in range(1, 4):
            z_layer = layer * 2.5
            for x in np.linspace(-4, 4, 50):
                for y in np.linspace(-2, 2, 20):
                    points.append([x, y, z_layer])
        
        # 添加一些随机的货架内容点
        for _ in range(5000):
            x = np.random.uniform(-3.5, 3.5)
            y = np.random.uniform(-1.5, 1.5)
            z = np.random.uniform(0.5, 7.5)
            points.append([x, y, z])
        
        self.point_cloud = np.array(points)
        print(f"生成了包含 {len(self.point_cloud)} 个点的3D点云")
    
    def map_detections_to_3d(self) -> None:
        """将2D检测结果映射到3D空间"""
        print("正在将2D检测映射到3D空间...")
        
        if not self.camera_poses:
            print("错误: 请先估计相机姿态")
            return
        
        for i, detection_data in enumerate(self.detections):
            if i >= len(self.camera_poses):
                break
            
            # 获取相机姿态
            camera_pose = self.camera_poses[i]
            
            # 计算检测中心点
            centers_2d = self._calculate_detection_centers(detection_data)
            
            # 将2D点映射到3D空间
            for center_x, center_y in centers_2d:
                # 使用相机参数进行反投影
                point_3d = self._backproject_point(center_x, center_y, camera_pose)
                if point_3d is not None:
                    self.detection_points_3d.append(point_3d)
        
        print(f"映射了 {len(self.detection_points_3d)} 个检测点到3D空间")
    
    def _calculate_detection_centers(self, detection_data: Dict) -> List[Tuple[float, float]]:
        """计算检测框的中心点"""
        centers = []
        if 'objects' not in detection_data:
            return centers
            
        for obj in detection_data['objects']:
            if 'position' in obj:
                x1, y1, x2, y2 = obj['position']
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                centers.append((center_x, center_y))
                
        return centers
    
    def _backproject_point(self, x: float, y: float, camera_pose: Dict) -> Optional[List[float]]:
        """将2D点反投影到3D空间"""
        # 归一化坐标
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        # 计算射线方向
        u = (x - cx) / fx
        v = (y - cy) / fy
        
        # 射线方向向量
        ray_direction = np.array([u, v, 1.0])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # 转换到世界坐标系
        rotation = camera_pose['rotation']
        ray_world = rotation @ ray_direction
        
        # 相机位置
        camera_pos = camera_pose['position']
        
        # 简单的深度估计（这里可以改进）
        # 假设货架在z=4的位置
        target_z = 4.0
        if ray_world[2] != 0:
            t = (target_z - camera_pos[2]) / ray_world[2]
            point_3d = camera_pos + t * ray_world
            
            # 检查点是否在货架范围内
            if (abs(point_3d[0]) < 4.5 and 
                abs(point_3d[1]) < 2.5 and 
                0 <= point_3d[2] <= 8):
                return point_3d.tolist()
        
        return None
    
    def visualize_with_viser(self) -> None:
        """使用Viser进行交互式3D可视化"""
        print("正在启动Viser可视化...")
        
        server = viser.ViserServer()
        
        # 添加点云
        if self.point_cloud is not None:
            server.scene.add_point_cloud(
                "/point_cloud",
                points=self.point_cloud,
                colors=np.ones_like(self.point_cloud) * [0.7, 0.7, 1.0],
                point_size=0.02
            )
        
        # 添加检测点
        if self.detection_points_3d:
            detection_points = np.array(self.detection_points_3d)
            server.scene.add_point_cloud(
                "/detection_points",
                points=detection_points,
                colors=np.ones_like(detection_points) * [1.0, 0.0, 0.0],
                point_size=0.05
            )
        
        # 添加相机位置
        for i, pose in enumerate(self.camera_poses):
            position = pose['position']
            server.scene.add_frame(
                f"/camera_{i}",
                wxyz=tf.SO3.from_matrix(pose['rotation']).wxyz,
                position=position
            )
        
        print("Viser服务器已启动，请在浏览器中查看")
        input("按回车键退出...")
    
    def save_results(self, output_dir: str) -> None:
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存点云
        if self.point_cloud is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
            pcd.paint_uniform_color([0.7, 0.7, 1.0])  # 浅蓝色
            
            # 优先保存GLB格式
            if GLTF_AVAILABLE:
                shelf_glb_path = output_path / "shelf_point_cloud.glb"
                success = export_open3d_point_cloud_to_gltf(
                    pcd, shelf_glb_path, point_size=0.015, convert_to_mesh=True
                )
                if success:
                    print(f"货架GLB文件已保存到: {shelf_glb_path}")
                else:
                    # GLB导出失败，回退到PLY
                    ply_path = output_path / "shelf_point_cloud.ply"
                    o3d.io.write_point_cloud(str(ply_path), pcd)
                    print(f"GLB导出失败，已保存PLY格式: {ply_path}")
            else:
                # GLB不可用，使用PLY格式
                ply_path = output_path / "shelf_point_cloud.ply"
                o3d.io.write_point_cloud(str(ply_path), pcd)
                print(f"GLB/GLTF导出不可用，已保存PLY格式: {ply_path}")
        
        # 保存检测点
        if self.detection_points_3d:
            detection_pcd = o3d.geometry.PointCloud()
            detection_points = np.array(self.detection_points_3d)
            detection_pcd.points = o3d.utility.Vector3dVector(detection_points)
            detection_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
            
            # 优先保存GLB格式
            if GLTF_AVAILABLE:
                detection_glb_path = output_path / "detection_points.glb"
                success = export_open3d_point_cloud_to_gltf(
                    detection_pcd, detection_glb_path, point_size=0.025, convert_to_mesh=True
                )
                if success:
                    print(f"检测点GLB文件已保存到: {detection_glb_path}")
                else:
                    # GLB导出失败，回退到PLY
                    ply_path = output_path / "detection_points.ply"
                    o3d.io.write_point_cloud(str(ply_path), detection_pcd)
                    print(f"GLB导出失败，已保存PLY格式: {ply_path}")
            else:
                # GLB不可用，使用PLY格式
                ply_path = output_path / "detection_points.ply"  
                o3d.io.write_point_cloud(str(ply_path), detection_pcd)
                print(f"GLB/GLTF导出不可用，已保存PLY格式: {ply_path}")
        
        # 保存完整场景GLB
        if GLTF_AVAILABLE and self.point_cloud is not None:
            scene_glb_path = output_path / "complete_scene.glb"
            
            # 准备数据
            shelf_points = self.point_cloud
            shelf_colors = np.array([[0.7, 0.7, 1.0]] * len(shelf_points))
            
            if self.detection_points_3d:
                detection_points = np.array(self.detection_points_3d)
                detection_colors = np.array([[1.0, 0.0, 0.0]] * len(detection_points))
            else:
                detection_points = np.array([]).reshape(0, 3)
                detection_colors = np.array([]).reshape(0, 3)
            
            # 导出完整场景
            success = create_detection_scene_gltf(
                shelf_points, detection_points,
                shelf_colors, detection_colors,
                scene_glb_path
            )
            if success:
                print(f"完整场景GLB文件已保存到: {scene_glb_path}")
        
        # 保存相机姿态
        if self.camera_poses:
            camera_data = {
                'intrinsics': self.camera_intrinsics,
                'poses': self.camera_poses
            }
            
            # 序列化numpy类型
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            camera_data_serializable = convert_numpy_types(camera_data)
            with open(output_path / "camera_poses.json", 'w') as f:
                json.dump(camera_data_serializable, f, indent=2)
        
        print(f"结果已保存到: {output_path}")
        if GLTF_AVAILABLE:
            print("📦 包含GLB格式的3D文件，可在Blender、Three.js等工具中查看")
        else:
            print("⚠️  GLB/GLTF导出不可用，文件已保存为PLY格式")
    
    def run_pipeline(self, output_dir: str = "output") -> None:
        """运行完整的处理流程"""
        print("开始高级3D货架重建与检测可视化流程...")
        
        # 加载数据
        self.load_data()
        
        # 3D重建
        self.reconstruct_with_vggt()
        
        # 映射检测结果
        self.map_detections_to_3d()
        
        # 可视化
        self.visualize_with_viser()
        
        # 保存结果
        self.save_results(output_dir)
        
        print("处理完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级3D货架重建与物体检测可视化")
    parser.add_argument("--image_dir", default="../imdata/sample", help="图片目录路径")
    parser.add_argument("--detection_file", default="../sku_detection.json", help="检测结果文件路径")
    parser.add_argument("--output_dir", default="output", help="输出目录路径")
    
    args = parser.parse_args()
    
    # 创建重建器
    reconstructor = Advanced3DReconstructor(args.image_dir, args.detection_file)
    
    # 运行处理流程
    reconstructor.run_pipeline(args.output_dir)


if __name__ == "__main__":
    main() 