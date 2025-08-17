#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GLB/GLTF导出工具模块
提供点云和网格导出为GLB/GLTF格式的功能
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Union, List, Tuple
import json
import base64
import struct
from pygltflib import GLTF2, Buffer, BufferView, Accessor, Mesh, Primitive, Node, Scene, Material
from pygltflib.constants import *
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointCloudToGLTFExporter:
    """点云到GLTF/GLB导出器"""
    
    def __init__(self):
        self.gltf = None
        self.buffers = []
        self.buffer_views = []
        self.accessors = []
        self.materials = []
        self.meshes = []
        self.nodes = []
        
    def export_point_cloud(self, 
                          points: np.ndarray,
                          colors: Optional[np.ndarray] = None,
                          output_path: Union[str, Path] = "output.glb",
                          point_size: float = 0.01,
                          convert_to_mesh: bool = True) -> bool:
        """
        导出点云为GLB/GLTF格式
        
        Args:
            points: 点云坐标数组 (N, 3)
            colors: 点云颜色数组 (N, 3) 或 (N, 4), 值范围[0,1]
            output_path: 输出文件路径
            point_size: 点的大小（仅在转换为球体时使用）
            convert_to_mesh: 是否将点云转换为网格
            
        Returns:
            是否导出成功
        """
        try:
            output_path = Path(output_path)
            
            if convert_to_mesh:
                # 方法1: 将点云转换为小球体网格
                success = self._export_as_sphere_mesh(points, colors, output_path, point_size)
            else:
                # 方法2: 使用GLB扩展支持点云（某些查看器可能不支持）
                success = self._export_as_point_primitive(points, colors, output_path)
            
            if success:
                logger.info(f"点云已成功导出为: {output_path}")
                return True
            else:
                logger.error(f"导出失败: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"导出点云时发生错误: {e}")
            return False
    
    def _export_as_sphere_mesh(self, 
                              points: np.ndarray, 
                              colors: Optional[np.ndarray],
                              output_path: Path,
                              point_size: float) -> bool:
        """将点云转换为小球体网格导出"""
        try:
            # 使用trimesh创建点云场景
            if colors is not None:
                # 确保颜色在正确范围内
                if colors.max() <= 1.0:
                    colors_255 = (colors * 255).astype(np.uint8)
                else:
                    colors_255 = colors.astype(np.uint8)
                
                # 如果是RGB，添加alpha通道
                if colors_255.shape[1] == 3:
                    alpha = np.ones((colors_255.shape[0], 1), dtype=np.uint8) * 255
                    colors_255 = np.hstack([colors_255, alpha])
            else:
                # 默认颜色：白色
                colors_255 = np.ones((len(points), 4), dtype=np.uint8) * 255
            
            # 创建点云对象
            point_cloud = trimesh.points.PointCloud(points, colors=colors_255)
            
            # 将点云转换为球体网格
            spheres = []
            for i, (point, color) in enumerate(zip(points, colors_255)):
                # 创建小球体
                sphere = trimesh.creation.icosphere(subdivisions=1, radius=point_size)
                sphere.vertices += point
                
                # 设置颜色
                sphere.visual.vertex_colors = np.tile(color, (len(sphere.vertices), 1))
                spheres.append(sphere)
                
                # 避免创建过多球体（性能考虑）
                if i >= 5000:  # 限制最多5000个球体
                    logger.warning(f"点云过大，仅导出前5000个点")
                    break
            
            # 合并所有球体
            if spheres:
                combined_mesh = trimesh.util.concatenate(spheres)
                
                # 导出为GLB
                if output_path.suffix.lower() == '.glb':
                    combined_mesh.export(str(output_path))
                else:
                    # 导出为GLTF
                    gltf_path = output_path.with_suffix('.gltf')
                    combined_mesh.export(str(gltf_path))
                
                return True
            else:
                logger.error("没有创建任何球体")
                return False
                
        except Exception as e:
            logger.error(f"球体网格导出失败: {e}")
            return False
    
    def _export_as_point_primitive(self, 
                                  points: np.ndarray,
                                  colors: Optional[np.ndarray], 
                                  output_path: Path) -> bool:
        """将点云作为点图元导出"""
        try:
            # 使用pygltflib手动创建GLTF
            gltf = GLTF2()
            
            # 准备顶点数据
            vertices = points.astype(np.float32).flatten()
            
            # 准备颜色数据
            if colors is not None:
                if colors.max() > 1.0:
                    colors = colors / 255.0
                if colors.shape[1] == 3:
                    # 添加alpha通道
                    alpha = np.ones((colors.shape[0], 1), dtype=np.float32)
                    colors = np.hstack([colors, alpha])
                colors_data = colors.astype(np.float32).flatten()
            else:
                # 默认白色
                colors_data = np.ones(len(points) * 4, dtype=np.float32)
            
            # 创建缓冲区数据
            vertex_buffer = vertices.tobytes()
            color_buffer = colors_data.tobytes()
            
            # 合并所有数据到一个缓冲区
            total_buffer = vertex_buffer + color_buffer
            
            # 创建GLTF结构
            buffer = Buffer(byteLength=len(total_buffer))
            gltf.buffers.append(buffer)
            
            # 顶点缓冲区视图
            vertex_buffer_view = BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(vertex_buffer),
                target=ARRAY_BUFFER
            )
            gltf.bufferViews.append(vertex_buffer_view)
            
            # 颜色缓冲区视图
            color_buffer_view = BufferView(
                buffer=0,
                byteOffset=len(vertex_buffer),
                byteLength=len(color_buffer),
                target=ARRAY_BUFFER
            )
            gltf.bufferViews.append(color_buffer_view)
            
            # 顶点访问器
            vertex_accessor = Accessor(
                bufferView=0,
                byteOffset=0,
                componentType=FLOAT,
                count=len(points),
                type=VEC3,
                max=points.max(axis=0).tolist(),
                min=points.min(axis=0).tolist()
            )
            gltf.accessors.append(vertex_accessor)
            
            # 颜色访问器
            color_accessor = Accessor(
                bufferView=1,
                byteOffset=0,
                componentType=FLOAT,
                count=len(points),
                type=VEC4,
                max=[1.0, 1.0, 1.0, 1.0],
                min=[0.0, 0.0, 0.0, 0.0]
            )
            gltf.accessors.append(color_accessor)
            
            # 创建材质
            material = Material(
                pbrMetallicRoughness={
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0
                }
            )
            gltf.materials.append(material)
            
            # 创建图元（使用POINTS模式）
            primitive = Primitive(
                attributes={
                    "POSITION": 0,
                    "COLOR_0": 1
                },
                mode=POINTS,
                material=0
            )
            
            # 创建网格
            mesh = Mesh(primitives=[primitive])
            gltf.meshes.append(mesh)
            
            # 创建节点
            node = Node(mesh=0)
            gltf.nodes.append(node)
            
            # 创建场景
            scene = Scene(nodes=[0])
            gltf.scenes.append(scene)
            gltf.scene = 0
            
            # 保存文件
            if output_path.suffix.lower() == '.glb':
                # GLB格式（二进制）
                gltf.set_binary_blob(total_buffer)
                gltf.save_binary(str(output_path))
            else:
                # GLTF格式（JSON + bin）
                bin_path = output_path.with_suffix('.bin')
                with open(bin_path, 'wb') as f:
                    f.write(total_buffer)
                
                buffer.uri = bin_path.name
                gltf.save(str(output_path.with_suffix('.gltf')))
            
            return True
            
        except Exception as e:
            logger.error(f"点图元导出失败: {e}")
            return False

def export_point_cloud_to_gltf(points: np.ndarray,
                              colors: Optional[np.ndarray] = None,
                              output_path: Union[str, Path] = "point_cloud.glb",
                              point_size: float = 0.01,
                              convert_to_mesh: bool = True) -> bool:
    """
    便捷函数：导出点云为GLB/GLTF格式
    
    Args:
        points: 点云坐标数组 (N, 3)
        colors: 点云颜色数组 (N, 3) 或 (N, 4), 值范围[0,1]
        output_path: 输出文件路径
        point_size: 点的大小（转换为网格时使用）
        convert_to_mesh: 是否转换为网格（推荐为True，兼容性更好）
        
    Returns:
        是否导出成功
    """
    exporter = PointCloudToGLTFExporter()
    return exporter.export_point_cloud(points, colors, output_path, point_size, convert_to_mesh)

def export_open3d_point_cloud_to_gltf(pcd,
                                     output_path: Union[str, Path] = "point_cloud.glb",
                                     point_size: float = 0.01,
                                     convert_to_mesh: bool = True) -> bool:
    """
    从Open3D点云对象导出为GLB/GLTF格式
    
    Args:
        pcd: Open3D点云对象
        output_path: 输出文件路径  
        point_size: 点的大小
        convert_to_mesh: 是否转换为网格
        
    Returns:
        是否导出成功
    """
    try:
        import open3d as o3d
        
        # 提取点坐标
        points = np.asarray(pcd.points)
        
        # 提取颜色（如果有）
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        
        return export_point_cloud_to_gltf(points, colors, output_path, point_size, convert_to_mesh)
        
    except Exception as e:
        logger.error(f"Open3D点云导出失败: {e}")
        return False

def create_detection_scene_gltf(shelf_points: np.ndarray,
                               detection_points: np.ndarray,
                               shelf_colors: Optional[np.ndarray] = None,
                               detection_colors: Optional[np.ndarray] = None,
                               output_path: Union[str, Path] = "detection_scene.glb") -> bool:
    """
    创建包含货架点云和检测点的完整场景
    
    Args:
        shelf_points: 货架点云坐标
        detection_points: 检测点坐标
        shelf_colors: 货架点云颜色
        detection_colors: 检测点颜色
        output_path: 输出路径
        
    Returns:
        是否导出成功
    """
    try:
        # 合并点云
        all_points = np.vstack([shelf_points, detection_points])
        
        # 准备颜色
        if shelf_colors is None:
            shelf_colors = np.array([[0.7, 0.7, 1.0]] * len(shelf_points))  # 浅蓝色
        if detection_colors is None:
            detection_colors = np.array([[1.0, 0.0, 0.0]] * len(detection_points))  # 红色
            
        all_colors = np.vstack([shelf_colors, detection_colors])
        
        # 导出
        return export_point_cloud_to_gltf(
            all_points, 
            all_colors, 
            output_path, 
            point_size=0.02,  # 稍大的点用于更好的可视化
            convert_to_mesh=True
        )
        
    except Exception as e:
        logger.error(f"检测场景导出失败: {e}")
        return False

if __name__ == "__main__":
    # 测试代码
    logger.info("测试GLB/GLTF导出功能...")
    
    # 创建测试点云
    n_points = 100
    test_points = np.random.rand(n_points, 3) * 10
    test_colors = np.random.rand(n_points, 3)
    
    # 测试导出
    success = export_point_cloud_to_gltf(
        test_points, 
        test_colors, 
        "test_point_cloud.glb",
        convert_to_mesh=True
    )
    
    if success:
        logger.info("✅ 测试成功！")
    else:
        logger.error("❌ 测试失败！")