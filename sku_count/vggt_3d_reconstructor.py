#!/usr/bin/env python3
"""
基于VGGT模型的3D重构GLB导出器

功能：
1. 输入一系列图片
2. 通过VGGT模型进行3D重构
3. 保存为GLB文件

使用方法:
python vggt_3d_reconstructor.py --input_dir /path/to/images --output_file output.glb
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path
from datetime import datetime
import gc
import time
from contextlib import nullcontext

# 添加VGGT模块路径
sys.path.append("../vggt-main")
sys.path.append("../vggt-main/vggt")

try:
    from visual_util import predictions_to_glb
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError:
    # 尝试相对导入
    sys.path.insert(0, '../vggt-main')
    from visual_util import predictions_to_glb
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

# 使用统一的设备与精度选择逻辑
try:
    from module.config import get_optimal_device_config
except Exception:
    # 兼容在不同工作目录下运行的情形
    sys.path.insert(0, '.')
    from module.config import get_optimal_device_config

class VGGT3DReconstructor:
    """VGGT 3D重构器"""
    
    def __init__(self, device=None, model_path=None):
        """
        初始化3D重构器
        
        Args:
            device: 计算设备 ("cuda" 或 "cpu")
            model_path: 预训练模型路径 (可选，默认从HuggingFace下载)
        """
        # 统一设备与dtype选择
        optimal_device, optimal_dtype = get_optimal_device_config(verbose=True)

        # 用户覆盖（若提供）
        if device is not None:
            if device == "cuda" and not torch.cuda.is_available():
                print("警告: CUDA不可用，回退到CPU")
                self.device = "cpu"
                self.dtype = torch.float32
            elif device == "cpu":
                self.device = "cpu"
                self.dtype = torch.float32
            else:
                self.device = device
                self.dtype = optimal_dtype if str(optimal_device).startswith("cuda") else torch.float32
        else:
            self.device = optimal_device
            self.dtype = optimal_dtype if str(optimal_device).startswith("cuda") else torch.float32

        self.model = None
        self.model_path = model_path
        
        print(f"使用设备: {self.device}")
    
    def load_model(self):
        """加载VGGT模型"""
        print("正在加载VGGT模型...")
        
        try:
            self.model = VGGT()
            
            if self.model_path and os.path.exists(self.model_path):
                # 从本地加载
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"从本地加载模型: {self.model_path}")
            else:
                # 从HuggingFace下载
                _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
                state_dict = torch.hub.load_state_dict_from_url(_URL, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("从HuggingFace下载并加载模型")
            
            self.model.eval()
            self.model = self.model.to(self.device)
            print("模型加载完成")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def load_images(self, input_dir):
        """
        加载并预处理图片
        
        Args:
            input_dir: 图片目录路径
            
        Returns:
            images: 预处理后的图片张量
            image_paths: 图片文件路径列表
        """
        print(f"从目录加载图片: {input_dir}")
        
        # 支持的图片格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        
        image_paths = sorted(image_paths)
        print(f"找到 {len(image_paths)} 张图片")
        
        if len(image_paths) == 0:
            raise ValueError(f"在目录 {input_dir} 中未找到图片")
        
        if len(image_paths) < 2:
            print("警告: 图片数量少于2张，3D重构效果可能不佳")
        
        # 预处理图片
        try:
            images = load_and_preprocess_images(image_paths).to(self.device)
            print(f"图片预处理完成，张量形状: {images.shape}")
            return images, image_paths
        except Exception as e:
            print(f"图片预处理失败: {e}")
            raise
    
    def run_inference(self, images):
        """
        运行VGGT模型推理
        
        Args:
            images: 预处理后的图片张量
            
        Returns:
            predictions: 模型预测结果
        """
        print("开始3D重构推理...")
        start_time = time.time()
        
        try:
            use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
            amp_ctx = torch.cuda.amp.autocast(dtype=self.dtype) if use_amp else nullcontext()
            with torch.no_grad():
                with amp_ctx:
                    predictions = self.model(images)
            
            print("转换姿态编码为外参和内参矩阵...")
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                predictions["pose_enc"], 
                images.shape[-2:]
            )
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            
            # 转换为numpy数组
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            
            predictions['pose_enc_list'] = None
            
            # 从深度图生成世界坐标点
            print("从深度图计算3D点云...")
            depth_map = predictions["depth"]
            world_points = unproject_depth_map_to_point_map(
                depth_map, 
                predictions["extrinsic"], 
                predictions["intrinsic"]
            )
            predictions["world_points_from_depth"] = world_points
            
            end_time = time.time()
            print(f"推理完成，耗时: {end_time - start_time:.2f}秒")
            
            return predictions
            
        except Exception as e:
            print(f"推理过程出错: {e}")
            raise
    
    def export_glb(self, predictions, output_path, 
                   conf_thres=50.0, 
                   show_cam=True,
                   mask_black_bg=False,
                   mask_white_bg=False,
                   mask_sky=False,
                   prediction_mode="Depthmap and Camera Branch"):
        """
        将预测结果导出为GLB文件
        
        Args:
            predictions: 模型预测结果
            output_path: 输出GLB文件路径
            conf_thres: 置信度阈值 (0-100)
            show_cam: 是否显示相机
            mask_black_bg: 是否遮罩黑色背景
            mask_white_bg: 是否遮罩白色背景
            mask_sky: 是否遮罩天空
            prediction_mode: 预测模式
        """
        print(f"导出GLB文件: {output_path}")
        
        try:
            # 创建输出目录
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 转换预测结果为3D场景
            glb_scene = predictions_to_glb(
                predictions,
                conf_thres=conf_thres,
                filter_by_frames="All",
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                show_cam=show_cam,
                mask_sky=mask_sky,
                target_dir=None,
                prediction_mode=prediction_mode,
            )
            
            # 导出GLB文件
            glb_scene.export(file_obj=output_path)
            print(f"GLB文件成功导出到: {output_path}")
            
            # 文件信息
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            print(f"GLB导出失败: {e}")
            raise
    
    def reconstruct_from_directory(self, input_dir, output_path, **kwargs):
        """
        从目录中的图片进行3D重构并导出GLB
        
        Args:
            input_dir: 输入图片目录
            output_path: 输出GLB文件路径
            **kwargs: 导出参数
        """
        print("="*60)
        print("开始VGGT 3D重构流程")
        print("="*60)
        
        total_start = time.time()
        
        try:
            # 1. 加载模型
            if self.model is None:
                self.load_model()
            
            # 2. 加载图片
            images, image_paths = self.load_images(input_dir)
            print(f"处理图片: {[os.path.basename(p) for p in image_paths]}")
            
            # 3. 运行推理
            predictions = self.run_inference(images)
            
            # 4. 导出GLB
            self.export_glb(predictions, output_path, **kwargs)
            
            total_time = time.time() - total_start
            print(f"\n总流程耗时: {total_time:.2f}秒")
            print("3D重构完成!")
            
            return output_path
            
        except Exception as e:
            print(f"3D重构失败: {e}")
            raise
        finally:
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VGGT 3D重构GLB导出器")
    parser.add_argument("--input_dir", type=str, default="../imdata/floor_display2/images",
                       help="输入图片目录路径 (default: ../imdata/floor_display2/images)")
    parser.add_argument("--output_file", type=str, default="../imdata/floor_display2/reconstruction.glb",
                       help="输出GLB文件路径 (default: ../imdata/floor_display2/reconstruction.glb)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                       help="计算设备 (默认自动选择)")
    parser.add_argument("--model_path", type=str,
                       help="预训练模型路径 (可选)")
    parser.add_argument("--conf_thres", type=float, default=50.0,
                       help="置信度阈值 (0-100, 默认50)")
    parser.add_argument("--show_cam", action="store_true", default=True,
                       help="显示相机位置")
    parser.add_argument("--mask_black_bg", action="store_true",
                       help="遮罩黑色背景")
    parser.add_argument("--mask_white_bg", action="store_true",
                       help="遮罩白色背景")
    parser.add_argument("--mask_sky", action="store_true",
                       help="遮罩天空区域")
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 自动生成输出文件名
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_dirname = os.path.basename(args.input_dir.rstrip("/"))
        args.output_file = f"vggt_3d_{input_dirname}_{timestamp}.glb"
    
    # 确保输出文件有.glb扩展名
    if not args.output_file.endswith('.glb'):
        args.output_file += '.glb'
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出文件: {args.output_file}")
    
    # 创建重构器
    reconstructor = VGGT3DReconstructor(
        device=args.device,
        model_path=args.model_path
    )
    
    # 执行重构
    try:
        result_path = reconstructor.reconstruct_from_directory(
            input_dir=args.input_dir,
            output_path=args.output_file,
            conf_thres=args.conf_thres,
            show_cam=args.show_cam,
            mask_black_bg=args.mask_black_bg,
            mask_white_bg=args.mask_white_bg,
            mask_sky=args.mask_sky
        )
        print(f"\n✅ 成功! GLB文件已保存到: {result_path}")
        
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
