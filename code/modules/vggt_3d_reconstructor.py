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
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import gc
import time
from contextlib import nullcontext

# 添加父目录到路径以便导入utils模块
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 先导入 utils 以统一配置 VGGT 路径（由 utils/__init__.py 负责）
from utils.config import get_optimal_device_config
from utils.transforms import build_vggt_transforms

try:
    from visual_util import predictions_to_glb
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError:
    # 如果此处仍失败，说明 utils/__init__.py 未被正确导入或 vggt-main 不存在
    raise

# 使用统一的设备与精度选择逻辑：已在顶部导入

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
                logger.warning("警告: CUDA不可用，回退到CPU")
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
        
        logger.info(f"使用设备: {self.device}")

    def _extract_image_ids(self, image_names):
        """
        从图片文件名中提取ID（通常是数字）

        Args:
            image_names: 图片文件名列表

        Returns:
            image_ids: 提取的ID列表
        """
        import re

        image_ids = []
        for name in image_names:
            # 尝试提取数字ID（支持多种格式：1.jpg, image_01.png, IMG001.jpeg等）
            match = re.search(r'(\d+)', Path(name).stem)
            if match:
                img_id = int(match.group(1))
                image_ids.append(img_id)
            else:
                # 如果无法提取数字，使用文件名顺序作为ID
                logger.warning(f"无法从文件名提取ID: {name}，使用顺序编号")
                img_id = len(image_ids)
                image_ids.append(img_id)

        return image_ids
    
    def load_model(self):
        """加载VGGT模型"""
        logger.info("正在加载VGGT模型...")
        
        try:
            self.model = VGGT()
            
            if self.model_path and os.path.exists(self.model_path):
                # 从本地加载
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"从本地加载模型: {self.model_path}")
            else:
                self.model = VGGT.from_pretrained("facebook/VGGT-1B")
            
            self.model.eval()
            self.model = self.model.to(self.device)
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def load_images(self, input_dir):
        """
        加载并预处理图片

        Args:
            input_dir: 图片目录路径

        Returns:
            images: 预处理后的图片张量
            image_paths: 图片文件路径列表
            image_names: 图片文件名列表（不含路径）
            image_ids: 从文件名提取的图片ID列表
        """
        logger.info(f"从目录加载图片: {input_dir}")

        # 支持的图片格式
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []

        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

        image_paths = sorted(image_paths)
        logger.info(f"找到 {len(image_paths)} 张图片")

        if len(image_paths) == 0:
            raise ValueError(f"在目录 {input_dir} 中未找到图片")

        if len(image_paths) < 2:
            logger.warning("警告: 图片数量少于2张，3D重构效果可能不佳")

        # 提取图片名字和ID
        image_names = [os.path.basename(path) for path in image_paths]
        image_ids = self._extract_image_ids(image_names)

        logger.info(f"图片顺序和ID映射:")
        for i, (name, img_id) in enumerate(zip(image_names, image_ids)):
            logger.info(f"  Frame[{i}] → {name} (ID: {img_id})")

        # 预处理图片
        try:
            images = load_and_preprocess_images(image_paths).to(self.device)
            logger.info(f"图片预处理完成，张量形状: {images.shape}")
            return images, image_paths, image_names, image_ids
        except Exception as e:
            logger.error(f"图片预处理失败: {e}")
            raise
    
    def run_inference(self, images):
        """
        运行VGGT模型推理
        
        Args:
            images: 预处理后的图片张量
            
        Returns:
            predictions: 模型预测结果
        """
        logger.info("开始3D重构推理...")
        start_time = time.time()
        
        try:
            use_amp = torch.cuda.is_available() and str(self.device).startswith("cuda")
            amp_ctx = torch.cuda.amp.autocast(dtype=self.dtype) if use_amp else nullcontext()
            with torch.no_grad():
                with amp_ctx:
                    predictions = self.model(images)
            
            logger.info("转换姿态编码为外参和内参矩阵...")
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
            logger.info("从深度图计算3D点云...")
            depth_map = predictions["depth"]
            world_points = unproject_depth_map_to_point_map(
                depth_map, 
                predictions["extrinsic"], 
                predictions["intrinsic"]
            )
            predictions["world_points_from_depth"] = world_points
            
            end_time = time.time()
            logger.info(f"推理完成，耗时: {end_time - start_time:.2f}秒")
            
            return predictions
            
        except Exception as e:
            logger.error(f"推理过程出错: {e}")
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
            logger.info(f"GLB文件成功导出到: {output_path}")
            
            # 文件信息
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"文件大小: {file_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"GLB导出失败: {e}")
            raise
    
    def reconstruct_from_directory(self, input_dir, output_path, save_predictions=True, **kwargs):
        """
        从目录中的图片进行3D重构并导出GLB

        Args:
            input_dir: 输入图片目录
            output_path: 输出GLB文件路径
            save_predictions: 是否保存VGGT predictions中间数据（用于后续可视化）
            **kwargs: 导出参数
        """
        logger.info("="*60)
        logger.info("开始VGGT 3D重构流程")
        logger.info("="*60)

        total_start = time.time()

        try:
            # 1. 加载模型
            if self.model is None:
                self.load_model()

            # 2. 加载图片
            images, image_paths, image_names, image_ids = self.load_images(input_dir)
            # 2.1 保存与VGGT预处理完全一致的裁剪/填充变换参数（transforms.json）
            try:
                self._save_transforms_cache(image_paths, image_ids, output_path, target_size=518)
            except Exception as e:
                logger.warning(f"保存 transforms.json 失败（不影响后续流程）: {e}")
            logger.info(f"处理图片: {[os.path.basename(p) for p in image_paths]}")

            # 3. 运行推理
            predictions = self.run_inference(images)

            # 4. 导出GLB
            self.export_glb(predictions, output_path, **kwargs)

            # 5. 保存VGGT predictions中间数据（用于后续可视化）
            if save_predictions:
                self._save_predictions_cache(predictions, output_path, image_ids)

            total_time = time.time() - total_start
            logger.info(f"\n总流程耗时: {total_time:.2f}秒")
            logger.info("3D重构完成!")

            return output_path

        except Exception as e:
            logger.error(f"3D重构失败: {e}")
            raise
        finally:
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()

    def _save_transforms_cache(self, image_paths, image_ids, output_path, *, target_size: int = 518) -> None:
        """保存VGGT裁剪/填充变换到 vggt_cache/transforms.json（精简字段）。

        每帧仅保存映射必需参数：scales、crop_start_y、batch_padding，加上标识 frame_idx/image_id/source_path；
        顶层保存 padded_size 与 target_size 便于校验与可视化。
        """
        from pathlib import Path as _Path
        import json

        # 目标目录：与 predictions.npz 相同的 vggt_cache 目录
        out_dir = _Path(output_path).parent / "vggt_cache"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "transforms.json"

        # 按VGGT crop 逻辑构建批次变换，并自动应用批内居中填充
        transforms = build_vggt_transforms(image_paths, target_size=target_size)

        frames = []
        # padded_size 对整个批次相同，直接取第一帧
        first_info = transforms[0].get_transform_info()
        padded_w, padded_h = int(first_info["padded_size"][0]), int(first_info["padded_size"][1])

        for idx, (img_path, img_id, t) in enumerate(zip(image_paths, image_ids, transforms)):
            info = t.get_transform_info()
            frames.append({
                "frame_idx": int(idx),
                "image_id": int(img_id),
                "source_path": str(img_path),
                "scales": [float(info["scales"][0]), float(info["scales"][1])],
                "crop_start_y": int(info["crop_start_y"]),
                "batch_padding": [int(info["batch_padding"][0]), int(info["batch_padding"][1])],
            })

        payload = {
            "target_size": int(target_size),
            "padded_size": [int(padded_w), int(padded_h)],
            "frames": frames,
        }

        with out_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"保存VGGT transforms参数: {out_file}")

    def _save_predictions_cache(self, predictions, output_path, image_ids):
        """
        保存VGGT predictions中间数据

        保存内容：
        - world_points: 世界坐标点云 (S,H,W,3) - S为图片数量
        - depth: 深度图 (S,H,W)
        - conf: 置信度图 (S,H,W)
        - image_ids: 图片ID列表 (S,) - 从文件名提取的数字ID

        Args:
            predictions: VGGT模型预测结果
            output_path: GLB输出文件路径
            image_ids: 图片ID列表
        """
        try:
            output_dir = Path(output_path).parent
            vggt_cache_dir = output_dir / "vggt_cache"
            vggt_cache_dir.mkdir(exist_ok=True)

            cache_path = vggt_cache_dir / "predictions.npz"


            # 验证数据一致性
            world_points = predictions.get('world_points_from_depth')
            depth = predictions.get('depth')
            conf = predictions.get('conf')

            if world_points is not None:
                if world_points.ndim == 4:  # (S,H,W,3)
                    frame_count = world_points.shape[0]
                elif world_points.ndim == 3:  # (H,W,3) - 单帧
                    frame_count = 1
                    # 扩展为批次维度
                    world_points = world_points[np.newaxis, ...]
                    if depth is not None and depth.ndim == 2:
                        depth = depth[np.newaxis, ...]
                    if conf is not None and conf.ndim == 2:
                        conf = conf[np.newaxis, ...]
                else:
                    logger.warning(f"world_points维度异常: {world_points.shape}")
                    frame_count = len(image_ids)

                # 验证数据一致性
                if len(image_ids) != frame_count:
                    logger.warning(f"image_ids数量({len(image_ids)})与world_points帧数({frame_count})不一致")


            # 保存数据
            save_data = {
                'world_points': world_points,
                'depth': depth,
                'conf': conf,
                'image_ids': np.array(image_ids, dtype=np.int32),
                'frame_count': frame_count if world_points is not None else len(image_ids),
            }

            # 过滤掉None值
            save_data = {k: v for k, v in save_data.items() if v is not None}

            np.savez_compressed(cache_path, **save_data)

            file_size = cache_path.stat().st_size / (1024 * 1024)
            logger.info(f"保存成功: {file_size:.2f} MB")
            logger.info(f"   包含数据: {list(save_data.keys())}")


        except Exception as e:
            logger.warning(f"保存predictions缓存失败: {e}")
            logger.warning("   可视化功能可能受限，但不影响GLB导出")

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
        logger.error(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    # 自动生成输出文件名
    if not args.output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_dirname = os.path.basename(args.input_dir.rstrip("/"))
        args.output_file = f"vggt_3d_{input_dirname}_{timestamp}.glb"
    
    # 确保输出文件有.glb扩展名
    if not args.output_file.endswith('.glb'):
        args.output_file += '.glb'
    
    logger.info(f"输入目录: {args.input_dir}")
    logger.info(f"输出文件: {args.output_file}")
    
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
        logger.info(f"\n成功! GLB文件已保存到: {result_path}")
        
    except Exception as e:
        logger.error(f"\n失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
