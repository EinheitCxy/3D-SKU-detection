# Copyright (c) Meta Platforms, Inc. and affiliates.
# 分段追踪视频推理脚本 - 使用文本prompt进行视频分割

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import sam3
from sam3.model_builder import build_sam3_video_model
from sam3.visualization_utils import draw_masks_to_frame


@dataclass
class VideoSegment:
    """视频分段信息"""
    segment_id: int
    start_frame: int
    end_frame: int
    overlap_frames: int  # 与下一段的重叠帧数


@dataclass
class SegmentResult:
    """分段处理结果"""
    segment_id: int
    start_frame: int
    end_frame: int
    masks: Dict[int, Dict[int, np.ndarray]]  # {frame_idx: {obj_id: mask}}
    success: bool
    error_msg: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分段追踪视频推理 - 支持长视频并行处理"
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="输入视频路径",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/sam3.pt",
        help="SAM3模型checkpoint路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备: cuda, mps, cpu",
    )
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=500,
        help="每段最大帧数（默认500）",
    )
    parser.add_argument(
        "--overlap-frames",
        type=int,
        default=30,
        help="分段重叠帧数（默认30，用于平滑过渡）",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        required=True,
        help="文本prompt（如 'bag', 'person', 'bottle'）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_masks",
        help="输出目录",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        default=False,
        help="保存mask图片到文件",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="保存带mask叠加的视频（默认开启）",
    )
    parser.add_argument(
        "--vis-stride",
        type=int,
        default=30,
        help="可视化帧间隔",
    )
    return parser.parse_args()


def get_video_info(video_path: str) -> Tuple[int, int, int, float]:
    """快速获取视频信息（不加载全部帧）

    Returns:
        (frame_count, width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    return frame_count, width, height, fps


def plan_segments(
    total_frames: int,
    segment_frames: int,
    overlap_frames: int,
) -> List[VideoSegment]:
    """规划视频分段

    Args:
        total_frames: 总帧数
        segment_frames: 每段最大帧数
        overlap_frames: 重叠帧数

    Returns:
        分段列表
    """
    segments = []
    segment_id = 0
    start = 0

    while start < total_frames:
        end = min(start + segment_frames, total_frames)
        overlap = overlap_frames if end < total_frames else 0

        segments.append(VideoSegment(
            segment_id=segment_id,
            start_frame=start,
            end_frame=end,
            overlap_frames=overlap,
        ))

        # 下一段从 end - overlap 开始（保证重叠）
        start = end - overlap if end < total_frames else total_frames
        segment_id += 1

    return segments


def resolve_device(device_arg: str) -> torch.device:
    """解析设备"""
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mask_to_numpy(mask_tensor: torch.Tensor) -> np.ndarray:
    """将mask tensor转换为numpy数组"""
    mask = (mask_tensor > 0.0).cpu().numpy()
    if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
        return mask[0, 0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        return mask[0]
    return mask


def process_segment(
    video_path: str,
    segment: VideoSegment,
    checkpoint_path: str,
    device: str,
    text_prompt: str,
) -> SegmentResult:
    """处理单个视频分段

    Args:
        text_prompt: 文本prompt（如 'bag'），用于语义分割
    """
    try:
        device_obj = resolve_device(device)

        autocast_ctx = contextlib.nullcontext()
        if device_obj.type == "cuda":
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        with autocast_ctx:
            # 构建模型
            sam3_model = build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                load_from_HF=checkpoint_path is None,
                device=device_obj,
            )

            # 初始化状态
            inference_state = sam3_model.init_state(resource_path=video_path)

            # 添加prompts
            ann_frame_idx = segment.start_frame

            # 使用文本prompt进行检测
            # 注意：SAM3的文本prompt会自动检测所有匹配的对象
            sam3_model.add_prompt(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                text_str=text_prompt,
            )

            # 追踪
            segment_masks = {}
            frames_to_track = segment.end_frame - segment.start_frame

            for frame_idx, out in sam3_model.propagate_in_video(
                inference_state,
                start_frame_idx=segment.start_frame,
                max_frame_num_to_track=frames_to_track,
                reverse=False,
            ):
                # out 包含 out_obj_ids, out_binary_masks 等
                if out is not None:
                    obj_ids = out.get("out_obj_ids", [])
                    video_res_masks = out.get("out_binary_masks", [])
                    segment_masks[frame_idx] = {
                        obj_id: mask_to_numpy(torch.tensor(video_res_masks[i]))
                        for i, obj_id in enumerate(obj_ids)
                    }

        return SegmentResult(
            segment_id=segment.segment_id,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            masks=segment_masks,
            success=True,
        )

    except Exception as e:
        return SegmentResult(
            segment_id=segment.segment_id,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
            masks={},
            success=False,
            error_msg=str(e),
        )


def merge_segment_results(
    results: List[SegmentResult],
) -> Dict[int, Dict[int, np.ndarray]]:
    """合并分段结果，处理重叠区域

    重叠区域使用加权平均进行平滑过渡
    """
    merged_masks = {}

    # 按segment_id排序
    sorted_results = sorted(results, key=lambda r: r.segment_id)

    for result in sorted_results:
        if not result.success:
            print(f"警告: 分段 {result.segment_id} 处理失败: {result.error_msg}")
            continue

        for frame_idx, obj_masks in result.masks.items():
            if frame_idx not in merged_masks:
                merged_masks[frame_idx] = {}

            for obj_id, mask in obj_masks.items():
                if obj_id not in merged_masks[frame_idx]:
                    merged_masks[frame_idx][obj_id] = mask.astype(np.float32)
                else:
                    # 重叠区域：简单取并集（也可以用加权平均）
                    existing = merged_masks[frame_idx][obj_id]
                    merged_masks[frame_idx][obj_id] = np.maximum(existing, mask.astype(np.float32))

    # 转换回bool
    for frame_idx in merged_masks:
        for obj_id in merged_masks[frame_idx]:
            merged_masks[frame_idx][obj_id] = merged_masks[frame_idx][obj_id] > 0.5

    return merged_masks


def save_masks_to_disk(
    masks: Dict[int, Dict[int, np.ndarray]],
    output_dir: str,
) -> None:
    """保存masks到磁盘"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for frame_idx, obj_masks in masks.items():
        frame_dir = output_path / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(exist_ok=True)

        for obj_id, mask in obj_masks.items():
            mask_path = frame_dir / f"obj_{obj_id}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

    print(f"Masks保存到: {output_dir}")


def save_video_with_masks(
    video_path: str,
    masks: Dict[int, Dict[int, np.ndarray]],
    output_path: str,
    fps: float,
) -> None:
    """保存带mask叠加的视频"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出目录
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 颜色映射（不同对象用不同颜色）
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
    ]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用SAM3的可视化工具绘制mask
        if frame_idx in masks:
            mask_list = list(masks[frame_idx].values())
            if mask_list:
                masks_array = np.stack(mask_list, axis=0)
                n_masks = len(mask_list)
                colors_list = colors[:n_masks] if n_masks <= len(colors) else colors * (n_masks // len(colors) + 1)
                colors_array = np.array(colors_list[:n_masks], dtype=np.uint8)
                frame = draw_masks_to_frame(frame, masks_array, colors_array)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"视频保存到: {output_path}")


def main() -> None:
    """主函数"""
    args = parse_args()

    # 1. 获取视频信息
    video_path = args.video_path
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频不存在: {video_path}")

    frame_count, width, height, fps = get_video_info(video_path)
    print(f"\n{'='*60}")
    print(f"视频信息:")
    print(f"  路径: {video_path}")
    print(f"  帧数: {frame_count}")
    print(f"  分辨率: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"{'='*60}\n")

    # 2. 规划分段
    segments = plan_segments(
        total_frames=frame_count,
        segment_frames=args.segment_frames,
        overlap_frames=args.overlap_frames,
    )

    print(f"分段规划:")
    print(f"  每段最大帧数: {args.segment_frames}")
    print(f"  重叠帧数: {args.overlap_frames}")
    print(f"  总分段数: {len(segments)}")
    for seg in segments:
        print(f"    段{seg.segment_id}: 帧 {seg.start_frame} -> {seg.end_frame} (共{seg.end_frame - seg.start_frame}帧)")
    print()

    # 3. 显示文本prompt
    print(f"文本prompt: {args.text_prompt}")

    # 4. 串行处理分段
    print(f"\n开始处理...")
    results = []

    for seg in segments:
        print(f"  处理段 {seg.segment_id}/{len(segments)-1}...")
        result = process_segment(
            video_path=video_path,
            segment=seg,
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            text_prompt=args.text_prompt,
        )
        results.append(result)

        if result.success:
            print(f"    完成: {len(result.masks)} 帧")
        else:
            print(f"    失败: {result.error_msg}")

    # 5. 合并结果
    print(f"\n合并分段结果...")
    merged_masks = merge_segment_results(results)
    print(f"  合并完成: {len(merged_masks)} 帧")

    # 6. 保存结果
    if args.save_masks:
        save_masks_to_disk(merged_masks, args.output_dir)

    if args.save_video:
        output_video_path = str(Path(args.output_dir) / "output_with_masks.mp4")
        save_video_with_masks(video_path, merged_masks, output_video_path, fps)

    # 7. 统计
    success_count = sum(1 for r in results if r.success)
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"  成功分段: {success_count}/{len(segments)}")
    print(f"  总帧数: {len(merged_masks)}")
    print(f"{'='*60}")

    return merged_masks


if __name__ == "__main__":
    main()
