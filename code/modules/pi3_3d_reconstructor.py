#!/usr/bin/env python3
"""
基于 PI3 的3D重建与GLB导出器

目标：仅替换重建（reconstruct）环节，SKU匹配仍使用VGGT。

功能：
1. 从目录加载图像
2. 使用 Pi3 模型推理，得到稠密点云与相机位姿
3. 导出 GLB（支持可选相机可视化）
4. 兼容 viewer 的 predictions 缓存（保存为 pi3_cache/predictions.npz，包含 world_points 与 images）

使用：
  uv run python -m modules.pi3_3d_reconstructor --input_dir <images_dir> --output_file <out.glb>
"""

from __future__ import annotations

import os
import sys
import glob
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    import sys as _sys
    _h = logging.StreamHandler(_sys.stdout)
    _h.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# 路径注入：确保可以导入 Pi3（仓库根目录/Pi3）和 code/utils
THIS_DIR = Path(__file__).resolve().parent  # code/modules
CODE_ROOT = THIS_DIR.parent  # code
REPO_ROOT = CODE_ROOT.parent  # 3D_Recognization 或 3D_SKU_Detection
PI3_ROOT = REPO_ROOT / "Pi3"

if str(PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(PI3_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# utils 中的通用设备与dtype选择
from .reconstructor_base import ReconstructorBase


def _to_nhwc_uint8(imgs: torch.Tensor | np.ndarray) -> np.ndarray:
    """将图像张量/数组转为 NHWC uint8（范围[0,255]）"""
    if isinstance(imgs, torch.Tensor):
        arr = imgs.detach().cpu().numpy()
    else:
        arr = imgs
    # 支持 (N,3,H,W) 或 (N,H,W,3)
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    # 归一化到 uint8
    if arr.dtype != np.uint8:
        maxv = float(arr.max()) if arr.size > 0 else 1.0
        scale = 255.0 if maxv <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr


def _estimate_intrinsics_from_local_points(
    local_points: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    max_points_per_view: int = 50000,
) -> torch.Tensor:
    """基于相机坐标系下的局部3D点与像素坐标，拟合每帧的内参矩阵K。

    Args:
        local_points: (B, N, H, W, 3)，相机坐标系下的3D点 (X, Y, Z)。
        conf: (B, N, H, W, 1) 或 (B, N, H, W)，对应置信度，可为None。
        max_points_per_view: 每个视角用于拟合的最大点数（随机子采样）。

    Returns:
        intrinsic: (B, N, 3, 3) 相机内参矩阵。
    """
    if local_points.ndim != 5 or local_points.shape[-1] != 3:
        raise ValueError(f"local_points shape must be (B,N,H,W,3), got {local_points.shape}")

    bsz, num_views, height, width, _ = local_points.shape
    device = local_points.device
    dtype = torch.float32

    # 像素坐标网格 (u,v)，带 0.5 像素偏移
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    u_grid = xs + 0.5  # 水平
    v_grid = ys + 0.5  # 垂直

    intrinsics = torch.zeros((bsz, num_views, 3, 3), device=device, dtype=dtype)

    # 默认K（当样本不足或拟合失败时退化使用）
    f_default = float(max(height, width))
    cx_default = (width - 1) / 2.0
    cy_default = (height - 1) / 2.0
    default_K = torch.tensor(
        [[f_default, 0.0, cx_default], [0.0, f_default, cy_default], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )

    if conf is not None:
        if conf.ndim == 5 and conf.shape[-1] == 1:
            conf = conf[..., 0]
        if conf.shape[:4] != (bsz, num_views, height, width):
            raise ValueError(f"conf shape mismatch, expect (B,N,H,W[,1]), got {conf.shape}")

    for b in range(bsz):
        for n in range(num_views):
            pts = local_points[b, n]  # (H, W, 3)
            pts = pts.to(dtype)
            X = pts[..., 0]
            Y = pts[..., 1]
            Z = pts[..., 2]

            valid_mask = torch.isfinite(pts).all(dim=-1) & (Z > 1e-4)
            if conf is not None:
                valid_mask = valid_mask & (conf[b, n] > 0.05)

            if not valid_mask.any():
                intrinsics[b, n] = default_K
                continue

            a_u = (X / Z)[valid_mask]
            a_v = (Y / Z)[valid_mask]
            u = u_grid[valid_mask]
            v = v_grid[valid_mask]

            num_samples = a_u.numel()
            if num_samples < 10:
                intrinsics[b, n] = default_K
                continue

            if num_samples > max_points_per_view:
                idx = torch.randperm(num_samples, device=device)[:max_points_per_view]
                a_u = a_u[idx]
                a_v = a_v[idx]
                u = u[idx]
                v = v[idx]

            # 线性最小二乘拟合:
            #   u ≈ fx * (X/Z) + cx
            #   v ≈ fy * (Y/Z) + cy
            A_u = torch.stack([a_u, torch.ones_like(a_u)], dim=1)  # (M, 2)
            A_v = torch.stack([a_v, torch.ones_like(a_v)], dim=1)  # (M, 2)

            try:
                ATA_u = A_u.T @ A_u
                ATA_v = A_v.T @ A_v

                # 检查条件数，避免病态矩阵求解
                cond_u = torch.linalg.cond(ATA_u).item()
                cond_v = torch.linalg.cond(ATA_v).item()
                max_cond = max(cond_u, cond_v)

                if max_cond > 1e6:
                    # 条件数过大，矩阵接近奇异
                    intrinsics[b, n] = default_K
                    continue

                ATu = A_u.T @ u
                theta_u = torch.linalg.solve(ATA_u, ATu)  # (2,)

                ATv = A_v.T @ v
                theta_v = torch.linalg.solve(ATA_v, ATv)  # (2,)

                fx, cx = theta_u[0].item(), theta_u[1].item()
                fy, cy = theta_v[0].item(), theta_v[1].item()

                # 图像尺寸（用于范围检查）
                W = float(width)
                H = float(height)

                # === 必要检查（硬性条件）===
                # 1. 焦距必须为正
                if fx <= 0 or fy <= 0:
                    logger.warning(f"Pi3 内参拟合失败 (batch={b}, view={n}): 焦距非正 (fx={fx:.1f}, fy={fy:.1f}) → 使用默认内参")
                    intrinsics[b, n] = default_K
                    continue

                # 2. 计算拟合残差（主判据）
                u_pred = fx * a_u + cx
                v_pred = fy * a_v + cy
                residual_u = torch.abs(u - u_pred).mean().item()
                residual_v = torch.abs(v - v_pred).mean().item()
                max_residual = max(residual_u, residual_v)

                # 主判据：残差过大 → 拟合质量差 → 使用默认内参
                RESIDUAL_THRESHOLD = 50.0  # 平均误差阈值（像素）
                if max_residual > RESIDUAL_THRESHOLD:
                    logger.warning(
                        f"Pi3 内参拟合质量差 (batch={b}, view={n}):\n"
                        f"   残差: u={residual_u:.1f}px, v={residual_v:.1f}px (阈值={RESIDUAL_THRESHOLD}px)\n"
                        f"   拟合结果: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}\n"
                        f"   → 使用默认内参 (fx={f_default:.1f}, cx={W/2:.1f}, cy={H/2:.1f})"
                    )
                    intrinsics[b, n] = default_K
                    continue

                # === 启发式检查（仅警告，不影响采用）===
                warnings = []
                if not (0.1 * f_default <= fx <= 10 * f_default):
                    warnings.append(f"fx={fx:.1f} 超出合理范围 [{0.1*f_default:.1f}, {10*f_default:.1f}]")
                if not (0.1 * f_default <= fy <= 10 * f_default):
                    warnings.append(f"fy={fy:.1f} 超出合理范围 [{0.1*f_default:.1f}, {10*f_default:.1f}]")
                if not (-0.2 * W <= cx <= 1.2 * W):
                    warnings.append(f"cx={cx:.1f} 超出图像范围 [{-0.2*W:.1f}, {1.2*W:.1f}]")
                if not (-0.2 * H <= cy <= 1.2 * H):
                    warnings.append(f"cy={cy:.1f} 超出图像范围 [{-0.2*H:.1f}, {1.2*H:.1f}]")

                # 防止除零
                aspect_ratio = fx / fy if abs(fy) > 1e-6 else float("inf")
                if not (0.7 <= aspect_ratio <= 1.3):
                    warnings.append(f"焦距比例 fx/fy={aspect_ratio:.3f} 异常（预期接近1.0）")

                if warnings:
                    logger.warning(
                        f"Pi3 内参拟合异常 (batch={b}, view={n})，但残差可接受 ({max_residual:.1f}px):\n" +
                        "\n".join(f"   - {w}" for w in warnings) +
                        f"\n   → 仍采用拟合结果"
                    )

                # 采用拟合结果
                K = torch.zeros((3, 3), device=device, dtype=dtype)
                K[0, 0] = fx
                K[1, 1] = fy
                K[0, 2] = cx
                K[1, 2] = cy
                K[2, 2] = 1.0
                intrinsics[b, n] = K
            except RuntimeError:
                # 矩阵奇异等情况，退化为默认K
                intrinsics[b, n] = default_K

    return intrinsics


def _save_predictions_npz(
    pred: Dict[str, Any],
    image_tensor: torch.Tensor,
    cache_path: Path,
    *,
    image_ids: Optional[List[int]] = None,
) -> Path:
    """保存与 viewer 兼容的 predictions 缓存。

    Args:
        pred: Pi3 预测结果字典
        image_tensor: 图像张量
        cache_path: 缓存文件的完整路径（如 <out_dir>/pi3_cache/predictions.npz）
        image_ids: 可选的图像ID列表

    Returns:
        保存的缓存文件路径
    """
    # 确保父目录存在
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    points = pred.get("points")  # 期望 (N,H,W,3)
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if points is None:
        raise ValueError("Pi3 predictions missing 'points'")

    imgs_nhwc = _to_nhwc_uint8(image_tensor)

    # Pi3 输出通常是 (N,H,W,3)。若包含 batch 维且为单batch，去掉batch
    if points.ndim == 5 and points.shape[0] == 1:
        points = points[0]

    # 置信度（用于后续 gid 分配中的 bbox 采样）
    conf = pred.get("conf")
    if isinstance(conf, torch.Tensor):
        conf_np = conf.detach().cpu().numpy()
    else:
        conf_np = conf
    if conf_np is not None and conf_np.ndim >= 4 and conf_np.shape[0] == 1:
        # 去掉batch维
        conf_np = conf_np[0]
    if conf_np is not None and conf_np.ndim == 4 and conf_np.shape[-1] == 1:
        # (N,H,W,1) -> (N,H,W)
        conf_np = conf_np[..., 0]

    save_kwargs: Dict[str, Any] = {
        "world_points": points.astype(np.float32, copy=False),
        "images": imgs_nhwc,
    }
    if conf_np is not None:
        save_kwargs["conf"] = conf_np.astype(np.float32, copy=False)
    if image_ids is not None:
        save_kwargs["image_ids"] = np.asarray(image_ids, dtype=np.int32)
    # 附加相机位姿（若存在）
    cam = pred.get("camera_poses")
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
    if cam is not None:
        if cam.ndim >= 3 and cam.shape[0] == 1:
            cam = cam[0]
        save_kwargs["camera_poses"] = cam
    # 可选：保存从 Pi3 估算得到的相机外参和内参、深度等（若存在）
    extrinsic = pred.get("extrinsic")
    if isinstance(extrinsic, torch.Tensor):
        extrinsic_np = extrinsic.detach().cpu().numpy()
    else:
        extrinsic_np = extrinsic
    if extrinsic_np is not None:
        if extrinsic_np.ndim >= 3 and extrinsic_np.shape[0] == 1:
            extrinsic_np = extrinsic_np[0]
        save_kwargs["extrinsic"] = extrinsic_np.astype(np.float32, copy=False)

    intrinsic = pred.get("intrinsic")
    if isinstance(intrinsic, torch.Tensor):
        intrinsic_np = intrinsic.detach().cpu().numpy()
    else:
        intrinsic_np = intrinsic
    if intrinsic_np is not None:
        if intrinsic_np.ndim >= 3 and intrinsic_np.shape[0] == 1:
            intrinsic_np = intrinsic_np[0]
        save_kwargs["intrinsic"] = intrinsic_np.astype(np.float32, copy=False)

    depth = pred.get("depth")
    if isinstance(depth, torch.Tensor):
        depth_np = depth.detach().cpu().numpy()
    else:
        depth_np = depth
    if depth_np is not None:
        if depth_np.ndim >= 5 and depth_np.shape[0] == 1:
            depth_np = depth_np[0]
        save_kwargs["depth"] = depth_np.astype(np.float32, copy=False)

    depth_conf = pred.get("depth_conf")
    if isinstance(depth_conf, torch.Tensor):
        depth_conf_np = depth_conf.detach().cpu().numpy()
    else:
        depth_conf_np = depth_conf
    if depth_conf_np is not None:
        if depth_conf_np.ndim >= 4 and depth_conf_np.shape[0] == 1:
            depth_conf_np = depth_conf_np[0]
        save_kwargs["depth_conf"] = depth_conf_np.astype(np.float32, copy=False)

    world_points_conf = pred.get("world_points_conf")
    if isinstance(world_points_conf, torch.Tensor):
        world_points_conf_np = world_points_conf.detach().cpu().numpy()
    else:
        world_points_conf_np = world_points_conf
    if world_points_conf_np is not None:
        if world_points_conf_np.ndim >= 4 and world_points_conf_np.shape[0] == 1:
            world_points_conf_np = world_points_conf_np[0]
        save_kwargs["world_points_conf"] = world_points_conf_np.astype(np.float32, copy=False)

    local_points = pred.get("local_points")
    if isinstance(local_points, torch.Tensor):
        local_points_np = local_points.detach().cpu().numpy()
    else:
        local_points_np = local_points
    if local_points_np is not None:
        if local_points_np.ndim >= 5 and local_points_np.shape[0] == 1:
            local_points_np = local_points_np[0]
        save_kwargs["local_points"] = local_points_np.astype(np.float32, copy=False)

    # 标注来源模型，便于下游判断
    save_kwargs["source_model"] = np.array(["pi3"], dtype=object)

    # 方案2优化：预计算帧对齐索引，避免运行时重复计算
    if image_ids is not None:
        image_ids_array = np.asarray(image_ids, dtype=np.int32)
        # 保存排序后的索引映射（用于快速查找）
        sorted_indices = np.argsort(image_ids_array)
        save_kwargs["frame_alignment_sorted_indices"] = sorted_indices

        # 保存逆映射（从 image_id 到帧索引的映射字典）
        # 格式: {image_id: frame_index}
        id_to_frame_map = {int(img_id): int(idx) for idx, img_id in enumerate(image_ids_array)}
        # 将字典转换为两个数组保存（npz不直接支持字典）
        map_keys = np.array(list(id_to_frame_map.keys()), dtype=np.int32)
        map_values = np.array(list(id_to_frame_map.values()), dtype=np.int32)
        save_kwargs["frame_alignment_map_keys"] = map_keys
        save_kwargs["frame_alignment_map_values"] = map_values

        logger.info(f"已预计算帧对齐索引: {len(image_ids)} 帧 (image_ids: {image_ids[:5]}...)")

    np.savez_compressed(cache_path, **save_kwargs)
    logger.info(f"保存Pi3预测缓存: {cache_path}")
    return cache_path


@dataclass
class _ExportOptions:
    conf_thres: float = 50.0  # 百分比 0-100
    show_cam: bool = True


class PI33DReconstructor(ReconstructorBase):
    """Pi3 3D重建器（仅用于重建，可无缝替换现有 VGGT 重建流程）。"""

    def __init__(self, device: str | None = None, model_path: str | None = None) -> None:
        super().__init__(device=device, model_path=model_path, backend_name="pi3")

        # 延迟导入，以便路径注入生效
        try:
            from pi3.models.pi3 import Pi3  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                f"无法导入 Pi3 包，请检查子模块或路径。Pi3路径: {PI3_ROOT}. 错误: {e}"
            )

    # ---- 加载 ----
    def load_model(self) -> None:
        """加载 Pi3 模型（支持 from_pretrained 或本地 ckpt）。"""
        logger.info("加载 Pi3 模型...")
        from pi3.models.pi3 import Pi3
        if self.model_path:
            self.model = Pi3().to(self.device).eval()
            if self.model_path.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(self.model_path)
            else:
                weight = torch.load(self.model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(weight)
            logger.info(f"已从本地权重加载: {self.model_path}")
        else:
            # 需 huggingface-hub 支持
            self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device).eval()
            logger.info("已从 HuggingFace 加载预训练 Pi3 模型")

    def load_images(self, input_dir: str) -> torch.Tensor:
        """使用 Pi3 自带工具加载图像 (N,3,H,W)"""
        from pi3.utils.basic import load_images_as_tensor
        # 仅目录输入；视频输入不在本模块范围
        imgs = load_images_as_tensor(input_dir, interval=1)
        return imgs.to(self.device)

    # ---- 推理 ----
    def run_inference(self, images_nchw: torch.Tensor) -> Dict[str, Any]:
        """运行 Pi3 推理，返回包含 points/Conf/camera_poses/images 的字典。"""
        # Pi3 需要 batch 维
        x = images_nchw[None]
        # dtype: Amp 在 Ampere 及以上默认 bfloat16；否则 float16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        logger.info("运行 Pi3 推理...")
        t0 = time.time()
        with torch.no_grad():
            amp_ctx = torch.amp.autocast('cuda', dtype=dtype) if self.device.startswith("cuda") and torch.cuda.is_available() else torch.cuda.amp.autocast(enabled=False)
            with amp_ctx:
                pred: Dict[str, torch.Tensor] = self.model(x)
        # 后处理：置信度与图像备份
        from pi3.utils.geometry import depth_edge

        # 置信度后处理：sigmoid + 深度边缘抑制
        if "conf" in pred:
            pred["conf"] = torch.sigmoid(pred["conf"])
            try:
                edge = depth_edge(pred["local_points"][..., 2], rtol=0.03)
                pred["conf"][edge] = 0.0
            except Exception as e:  # pragma: no cover - 仅日志，不中断流程
                logger.warning(f"depth_edge 处理失败，跳过边缘抑制: {e}")

        # Extrinsic：由 C2W 的 camera_poses 反求 W2C
        if "camera_poses" in pred:
            try:
                pred["extrinsic"] = torch.linalg.inv(pred["camera_poses"])
            except RuntimeError as e:
                logger.warning(f"无法从 camera_poses 反求外参矩阵: {e}")

        # Depth / depth_conf / world_points_conf：
        # 直接使用相机坐标系下的 Z 分量作为深度，并复用点置信度
        if "local_points" in pred:
            # (B, N, H, W, 1)
            pred["depth"] = pred["local_points"][..., 2:3]
        if "conf" in pred:
            # (B, N, H, W)
            depth_conf = pred["conf"]
            if depth_conf.ndim == 5 and depth_conf.shape[-1] == 1:
                depth_conf = depth_conf[..., 0]
            pred["depth_conf"] = depth_conf
            pred["world_points_conf"] = depth_conf

        # Intrinsic：基于 local_points 估计每帧内参矩阵
        if "local_points" in pred:
            try:
                pred["intrinsic"] = _estimate_intrinsics_from_local_points(
                    pred["local_points"],
                    conf=pred.get("conf"),
                    max_points_per_view=50000,
                )
            except Exception as e:
                logger.warning(f"估计相机内参失败，将在后续流程中回退默认K: {e}")

        # 保留 local_points 用于 SKU 匹配（包含相机坐标系深度）
        # if 'local_points' in pred:
        #     del pred['local_points']

        # 存储原始图像 (BNHWC，0-1范围) 以兼容 Pi3 viewer
        pred["images"] = x.permute(0, 1, 3, 4, 2)  # BNCHW->BNHWC（0-1范围）

        # 验证必需的键存在
        elapsed = time.time() - t0
        logger.info(f"Pi3 推理完成，用时 {elapsed:.2f}s")
        logger.info(f"返回的键: {list(pred.keys())}")
        if 'camera_poses' not in pred:
            logger.warning("Pi3 模型输出缺少 'camera_poses'，相机将不会显示在 GLB 中")
        else:
            logger.info(f"camera_poses 形状: {pred['camera_poses'].shape}")
        return pred  # tensors（带batch维）

    # ---- 导出 ----
    def export_glb(self, pred: Dict[str, Any], output_path: Path, *, conf_thres: float = 50.0, show_cam: bool = True) -> None:
        """将 Pi3 预测结果导出为 GLB 文件。"""
        try:
            # 直接复用 Pi3/demo_gradio.py 中的 predictions_to_glb（不修改vendor，仅导入）
            from demo_gradio import predictions_to_glb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(f"无法从 Pi3/demo_gradio 导入 predictions_to_glb: {e}")

        # 将 torch.Tensor 转为 numpy 以用于 trimesh
        pred_np: Dict[str, Any] = {}
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                arr = v.detach().cpu().numpy()
                # 去 batch 维
                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr[0]
                pred_np[k] = arr
            else:
                pred_np[k] = v

        out_dir = output_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        scene = predictions_to_glb(
            pred_np,
            conf_thres=conf_thres,
            filter_by_frames="all",
            show_cam=show_cam,
        )
        scene.export(file_obj=str(output_path))
        logger.info(f"GLB导出成功: {output_path}")

    # ---- 主流程 ----
    # 由基类统一流程调用
    def save_predictions_cache(
        self,
        predictions: Dict[str, Any],
        images: torch.Tensor,
        out_dir: Path,
        *,
        image_names: Optional[List[str]] = None,
        input_dir: Optional[str] = None,
        **_: Any,
    ) -> None:
        """保存 Pi3 预测缓存（viewer 兼容）。"""
        image_ids: Optional[List[int]] = None
        image_paths: List[str] = []
        if image_names is not None:
            image_ids = self.extract_image_ids(image_names)
            if input_dir is not None:
                image_paths = [str(Path(input_dir) / n) for n in image_names]

        # out_dir 已经是 cache 目录（如 Output/floor_display7/pi3_cache），不需要再嵌套
        cache_dir = out_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 1) 保存 transforms.json（Pi3 仅包含缩放，无裁剪/填充）
        try:
            if image_paths:
                from utils.transforms import build_transforms
                transforms = build_transforms(image_paths, model_type="pi3", pixel_limit=255000)
                # 取统一目标尺寸
                first_info = transforms[0].get_transform_info()
                tw, th = int(first_info["target_size"][0]), int(first_info["target_size"][1])
                tf_path = cache_dir / "transforms.json"

                frames = []
                for idx, (name, t) in enumerate(zip(image_names or [], transforms)):
                    info = t.get_transform_info()
                    sx, sy = float(info["scales"][0]), float(info["scales"][1])
                    frames.append({
                        "frame_idx": int(idx),
                        "image_id": int(image_ids[idx]) if image_ids else int(idx),
                        "source_path": str(Path(input_dir or "") / name) if input_dir and name else "",
                        "scales": [sx, sy],
                        "crop_start_y": 0,
                        "batch_padding": [0, 0],
                    })

                import json
                payload = {
                    "target_size": [tw, th],
                    "padded_size": [tw, th],
                    "frames": frames,
                }
                with tf_path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存 Pi3 transforms.json 失败（不影响GLB/NPZ）：{e}")

        # 2) 保存 predictions.npz（直接保存到 cache_dir，避免嵌套）
        cache_path = cache_dir / "predictions.npz"
        _save_predictions_npz(predictions, images, cache_path, image_ids=image_ids)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Pi3 3D重建GLB导出器")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图片目录（包含图像文件）")
    parser.add_argument("--output_file", type=str, required=True, help="输出GLB文件路径")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="计算设备")
    parser.add_argument("--model_path", type=str, default=None, help="Pi3权重路径（可选）")
    parser.add_argument("--conf_thres", type=float, default=50.0, help="置信度阈值(0-100)")
    parser.add_argument("--no_show_cam", action="store_true", help="GLB中不显示相机")
    args = parser.parse_args()

    recon = PI33DReconstructor(device=args.device, model_path=args.model_path)
    try:
        path = recon.reconstruct_from_directory(
            input_dir=args.input_dir,
            output_path=args.output_file,
            conf_thres=args.conf_thres,
            show_cam=(not args.no_show_cam),
            save_predictions=True,
        )
        logger.info(f"成功: {path}")
        return 0
    except Exception as e:
        logger.error(f"失败: {e}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
