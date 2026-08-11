#!/usr/bin/env python3
"""DA3 推理脚本（在 Depth-Anything-3/.venv 下运行）。

由 modules/da3_3d_reconstructor.py 通过 subprocess 调用，不依赖 code/ 任何模块。
输入图片目录，用 Depth-Anything-3 多视图推理，输出 da3_cache/predictions.npz，
字段与 Pi3 缓存完全兼容（供 SKU matching 消费）。

用法（由父进程调用，勿直接运行）：
  Depth-Anything-3/.venv/bin/python modules/da3_runner.py \
      --input_dir <images> --output_npz <da3_cache/predictions.npz>
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

# ---- 路径注入：DA3 源码 ----
# 本脚本位于 code/modules/da3_runner.py，仓库根为 parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
DA3_SRC = REPO_ROOT / "Depth-Anything-3" / "src"
if str(DA3_SRC) not in sys.path:
    sys.path.insert(0, str(DA3_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("da3_runner")

DEFAULT_HF_REPO = "depth-anything/DA3NESTED-GIANT-LARGE"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PATCH_SIZE = 14


def _depth_to_world_points(depth: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """深度图 + 内外参 -> 世界坐标系点云 (N,H,W,3)。extrinsics 为 w2c (N,4,4)。"""
    N, H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=-1).astype(np.float32)  # (H,W,3)
    pixels_flat = pixels.reshape(-1, 3)  # (H*W,3)

    world_points = np.zeros((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        K = intrinsics[i]
        W2C = extrinsics[i]
        # DA3 extrinsics 可能是 (3,4) [R|t] 或 (4,4)；补齐成 4x4 方阵以求逆
        if W2C.shape == (3, 4):
            W2C = np.vstack([W2C, np.array([[0, 0, 0, 1]], dtype=np.float32)])
        C2W = np.linalg.inv(W2C)
        K_inv = np.linalg.inv(K)
        d = depth[i].reshape(-1)
        # 无效深度过滤（参考 DA3 export/glb.py: isfinite(d) & (d > 0)）
        valid = np.isfinite(d) & (d > 0)
        p_cam = (K_inv @ pixels_flat.T).T * d[:, None]  # (H*W,3)
        p_cam_h = np.concatenate([p_cam, np.ones((len(p_cam), 1), dtype=np.float32)], axis=-1)
        p_world = (C2W @ p_cam_h.T).T[:, :3]
        # 无效像素的 world_points 置 0
        p_world[~valid] = 0.0
        world_points[i] = p_world.reshape(H, W, 3)
    return world_points


def _extract_image_ids(paths: list[str]) -> list[int]:
    ids = []
    for p in paths:
        m = re.search(r"(\d+)", Path(p).stem)
        ids.append(int(m.group(1)) if m else len(ids))
    return ids


def _nearest_patch_multiple(value: int) -> int:
    down = (value // PATCH_SIZE) * PATCH_SIZE
    up = down + PATCH_SIZE
    return max(1, up if abs(up - value) <= abs(value - down) else down)


def _source_to_processed_affines(
    image_sizes: list[tuple[int, int]],
    output_height: int,
    output_width: int,
    process_res: int,
) -> np.ndarray:
    """Return original-pixel to final DA3-grid affine transforms for this runner."""
    transforms: list[np.ndarray] = []
    for original_width, original_height in image_sizes:
        scale = process_res / float(max(original_width, original_height))
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        processed_width = _nearest_patch_multiple(resized_width)
        processed_height = _nearest_patch_multiple(resized_height)
        crop_left = (processed_width - output_width) // 2
        crop_top = (processed_height - output_height) // 2
        if crop_left < 0 or crop_top < 0:
            raise ValueError(
                "DA3 output is larger than the pre-unification processed image"
            )
        transforms.append(
            np.array(
                [
                    [processed_width / original_width, 0.0, -crop_left],
                    [0.0, processed_height / original_height, -crop_top],
                ],
                dtype=np.float32,
            )
        )
    return np.stack(transforms)


def main() -> None:
    ap = argparse.ArgumentParser(description="DA3 推理 -> predictions.npz")
    ap.add_argument("--input_dir", required=True, help="图片目录")
    ap.add_argument("--output_npz", required=True, help="输出 npz 路径")
    ap.add_argument("--model_path", default=DEFAULT_HF_REPO, help=f"模型路径/HF repo（默认 {DEFAULT_HF_REPO}）")
    ap.add_argument("--device", default="cuda", help="推理设备（默认 cuda）")
    ap.add_argument("--process_res", type=int, default=504, help="推理分辨率（默认 504）")
    args = ap.parse_args()

    from PIL import Image
    from depth_anything_3.api import DepthAnything3

    input_dir = Path(args.input_dir)
    paths = sorted((str(p) for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()), key=lambda p: int(re.search(r"(\d+)", Path(p).stem).group(1)))
    if not paths:
        raise SystemExit(f"目录中未找到图片: {input_dir}")
    logger.info(f"[da3_runner] {len(paths)} imgs from {input_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(args.model_path).to(device)
    model.eval()
    logger.info(f"[da3_runner] model loaded on {device} ({time.time()-t0:.1f}s)")

    pil_images = [Image.open(p).convert("RGB") for p in paths]
    source_image_sizes = [(image.width, image.height) for image in pil_images]
    t1 = time.time()
    prediction = model.inference(pil_images, process_res=args.process_res, process_res_method="upper_bound_resize")
    logger.info(f"[da3_runner] inference done ({time.time()-t1:.1f}s)")

    depth = np.asarray(prediction.depth, dtype=np.float32)          # (N,H,W)
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)  # (N,3,4) [R|t] w2c（非方阵，求逆时在 _depth_to_world_points 内补齐为 4x4）
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)  # (N,3,3)
    conf = prediction.conf
    conf = np.asarray(conf, dtype=np.float32) if conf is not None else np.ones_like(depth)
    proc_imgs = prediction.processed_images
    N, H, W = depth.shape
    logger.info(f"[da3_runner] output N={N} H={H} W={W} depth_range=[{depth.min():.2f},{depth.max():.2f}]")

    world_points = _depth_to_world_points(depth, intrinsics, extrinsics)  # (N,H,W,3)
    images_np = np.asarray(proc_imgs, dtype=np.uint8) if proc_imgs is not None else np.zeros((N, H, W, 3), dtype=np.uint8)
    image_ids = _extract_image_ids(paths)
    image_ids_array = np.asarray(image_ids, dtype=np.int32)
    source_to_processed_affine = _source_to_processed_affines(
        source_image_sizes, H, W, args.process_res
    )

    # 帧对齐索引（对齐 pi3 schema：sorted_indices + id->frame 映射）
    sorted_indices = np.argsort(image_ids_array)
    id_to_frame_map = {int(img_id): int(idx) for idx, img_id in enumerate(image_ids_array)}
    map_keys = np.array(list(id_to_frame_map.keys()), dtype=np.int32)
    map_values = np.array(list(id_to_frame_map.values()), dtype=np.int32)

    out = Path(args.output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        depth=depth[..., None].astype(np.float32),            # (N,H,W,1) - matcher 要求最后一维
        depth_conf=conf.astype(np.float32),                    # (N,H,W)
        world_points=world_points.astype(np.float32),          # (N,H,W,3)
        world_points_conf=conf.astype(np.float32),              # (N,H,W)
        extrinsic=extrinsics.astype(np.float32),                # (N,3,4) [R|t] w2c（保存原始非方阵；matcher 与 _depth_to_world_points 内部按需补齐）
        intrinsic=intrinsics.astype(np.float32),               # (N,3,3)
        images=images_np.astype(np.uint8),                      # (N,H,W,3)
        image_ids=image_ids_array,                              # (N,)
        source_image_sizes=np.asarray(source_image_sizes, dtype=np.int32),
        source_to_processed_affine=source_to_processed_affine,
        source_model=np.array(["depth-anything/DA3NESTED-GIANT-LARGE"], dtype=object),
        frame_alignment_sorted_indices=sorted_indices,
        frame_alignment_map_keys=map_keys,
        frame_alignment_map_values=map_values,
    )
    logger.info(f"[da3_runner] saved {out} (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
