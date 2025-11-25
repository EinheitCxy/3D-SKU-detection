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
from typing import Optional, List, Any

# 添加父目录到路径以便导入utils模块
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ====== VGGT路径注入策略 ======
# 1. 先导入 utils 模块，触发 utils/__init__.py 的路径注入逻辑
# 2. utils/__init__.py 会自动将 vggt-main 添加到 sys.path
# 3. 然后再导入 vggt 相关模块，确保路径已正确配置
# =============================

from utils.transforms import build_transforms
from utils import get_vggt_root  # 确保VGGT路径已注入

# 验证VGGT路径可用性
VGGT_ROOT = get_vggt_root()
if not VGGT_ROOT.exists():
    logger.error(f"VGGT路径不存在: {VGGT_ROOT}")
    logger.error("请确保vggt-main目录存在于项目根目录")
    sys.exit(1)

# 现在安全地导入VGGT模块（路径已由utils/__init__.py注入）
try:
    from visual_util import predictions_to_glb
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError as e:
    logger.error(f"VGGT模块导入失败: {e}")
    logger.error(f"VGGT路径: {VGGT_ROOT}")
    logger.error("请确保VGGT模块已正确安装: pip install -r vggt-main/requirements.txt")
    sys.exit(1)

# 使用统一的设备与精度选择逻辑：已在顶部导入

from .reconstructor_base import ReconstructorBase


class VGGT3DReconstructor(ReconstructorBase):
    """VGGT 3D重构器"""

    def __init__(self, device=None, model_path=None):
        super().__init__(device=device, model_path=model_path, backend_name="vggt")
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
            
        except (ImportError, RuntimeError, FileNotFoundError) as e:
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
        image_ids = self.extract_image_ids(image_names)

        logger.info(f"图片顺序和ID映射:")
        for i, (name, img_id) in enumerate(zip(image_names, image_ids)):
            logger.info(f"  Frame[{i}] → {name} (ID: {img_id})")

        # 预处理图片
        try:
            images = load_and_preprocess_images(image_paths).to(self.device)
            logger.info(f"图片预处理完成，张量形状: {images.shape}")
            # 基类流程只需要图像张量
            return images
        except (RuntimeError, FileNotFoundError) as e:
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

        except (RuntimeError, ValueError, KeyError) as e:
            logger.error(f"推理过程出错: {e}")
            raise
    
    def export_glb(self, predictions, output_path: Path, *,
                   conf_thres: float = 50.0,
                   show_cam: bool = True,
                   mask_black_bg: bool = False,
                   mask_white_bg: bool = False,
                   mask_sky: bool = False,
                   prediction_mode: str = "Depthmap and Camera Branch") -> None:
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
            glb_scene.export(file_obj=str(output_path))
            logger.info(f"GLB文件成功导出到: {output_path}")
            
            # 文件信息
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"文件大小: {file_size:.2f} MB")

        except (FileNotFoundError, PermissionError, OSError, RuntimeError) as e:
            logger.error(f"GLB导出失败: {e}")
            raise
    
    def reconstruct_from_directory(self, *, input_dir: str, output_path: str,
                                  conf_thres: float = 50.0, show_cam: bool = True,
                                  save_predictions: bool = True, **kwargs):
        """使用基类模板流程执行重建。"""
        return super().reconstruct_from_directory(
            input_dir=input_dir,
            output_path=output_path,
            conf_thres=conf_thres,
            show_cam=show_cam,
            save_predictions=save_predictions,
            **kwargs,
        )

    def _save_transforms_cache_direct(self, image_paths, image_ids, cache_dir: Path, *, target_size: int = 518) -> None:
        """保存VGGT裁剪/填充变换到 transforms.json（精简字段）。

        Args:
            image_paths: 图像路径列表
            image_ids: 图像ID列表
            cache_dir: 缓存目录（直接传入，不再创建嵌套）
            target_size: VGGT目标尺寸

        每帧仅保存映射必需参数：scales、crop_start_y、batch_padding，加上标识 frame_idx/image_id/source_path；
        顶层保存 padded_size 与 target_size 便于校验与可视化。
        """
        import json

        out_file = cache_dir / "transforms.json"

        # 按VGGT crop 逻辑构建批次变换，并自动应用批内居中填充
        transforms = build_transforms(image_paths, model_type="vggt", target_size=target_size)

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

    def _save_predictions_cache_direct(
        self,
        predictions,
        cache_path: Path,
        image_ids: Optional[List[int]] = None,
        *,
        image_paths: Optional[List[str]] = None,
        mask_black_bg: bool = False,
        mask_white_bg: bool = False,
        mask_sky: bool = False,
    ) -> None:
        """保存与 viewer 兼容的 predictions.npz 缓存。

        Args:
            predictions: VGGT 预测结果字典
            cache_path: 缓存文件的完整路径（如 <out_dir>/vggt_cache/predictions.npz）
            image_ids: 可选的图像ID列表
            image_paths: 可选的图像路径列表
            mask_black_bg: 是否遮罩黑色背景
            mask_white_bg: 是否遮罩白色背景
            mask_sky: 是否遮罩天空

        约定：
        - 必需键：world_points (S,H,W,3), conf (S,H,W)
        - 可选键：extrinsic, intrinsic, image_ids
        """
        try:
            # 确保父目录存在
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            world_points = predictions.get("world_points")
            if world_points is None:
                logger.warning("VGGT 预测结果缺少 'world_points'，跳过 predictions 缓存保存")
                return

            import numpy as np

            # 置信度：优先使用 world_points_conf，其次 depth_conf，最后回退为全1
            conf = predictions.get("world_points_conf")
            if conf is None:
                conf = predictions.get("depth_conf")
            if conf is None:
                conf = np.ones(world_points.shape[:-1], dtype=np.float32)

            save_dict: dict[str, Any] = {
                "world_points": world_points.astype(np.float32, copy=False),
                "conf": conf.astype(np.float32, copy=False),
            }

            extr = predictions.get("extrinsic")
            intr = predictions.get("intrinsic")
            if extr is not None:
                save_dict["extrinsic"] = np.asarray(extr, dtype=np.float32)
            if intr is not None:
                save_dict["intrinsic"] = np.asarray(intr, dtype=np.float32)

            if image_ids:
                save_dict["image_ids"] = np.asarray(image_ids, dtype=np.int32)

            np.savez_compressed(cache_path, **save_dict)
            logger.info(f"保存VGGT预测缓存: {cache_path}")
        except Exception as e:  # noqa: BLE001 - 缓存失败不影响主流程
            logger.warning(f"保存 VGGT predictions 缓存失败（不影响GLB导出）：{e}")

    def save_predictions_cache(
        self,
        predictions,
        images_tensor,
        out_dir: Path,
        *,
        image_names: Optional[List[str]] = None,
        input_dir: Optional[str] = None,
        target_size: int = 518,
        mask_black_bg: bool = False,
        mask_white_bg: bool = False,
        mask_sky: bool = False,
        **_: Any,
    ) -> None:
        """保存 transforms.json 与 predictions.npz（viewer 兼容）。"""
        # 组装 image_paths 与 image_ids
        image_paths: list[str] = []
        image_ids: list[int] = []
        if image_names and input_dir:
            image_paths = [str(Path(input_dir) / n) for n in image_names]
            image_ids = self.extract_image_ids(image_names)

        # out_dir 已经是 cache 目录（如 Output/floor_display7/vggt_cache），不需要再嵌套
        cache_dir = out_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1) 保存 transforms.json（严格对齐 VGGT 预处理）
        try:
            if image_paths and image_ids:
                self._save_transforms_cache_direct(image_paths, image_ids, cache_dir, target_size=target_size)
        except Exception as e:
            logger.warning(f"保存 transforms.json 失败（不影响后续流程）: {e}")

        # 2) 保存 predictions.npz（直接保存到 cache_dir，避免嵌套）
        try:
            cache_path = cache_dir / "predictions.npz"
            self._save_predictions_cache_direct(
                predictions,
                cache_path,
                image_ids,
                image_paths=image_paths,
                mask_black_bg=mask_black_bg,
                mask_white_bg=mask_white_bg,
                mask_sky=mask_sky,
            )
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

    except (RuntimeError, ValueError, FileNotFoundError, OSError) as e:
        logger.error(f"\n失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
