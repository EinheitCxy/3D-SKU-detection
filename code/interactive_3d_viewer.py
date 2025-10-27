"""
Interactive 3D Viewer (wrapper)

此文件为薄封装，真正的实现迁移至 code/viewer/ 包。
保持原有 CLI 用法不变：

uv run python interactive_3d_viewer.py \
  --global-mapping Output/.../global_mapping.json \
  --reconstruction Output/.../reconstruction.glb \
  --image-dir ../imdata/.../images \
  --detection-dir ../imdata/.../detections_results [--points-source glb|predictions]
"""

import argparse
import logging
from pathlib import Path

from viewer.types import ViewerConfig, ViewerRuntimeConfig
from viewer.cache import build_cache
from viewer.runtime import ViserViewer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D SKU 交互式可视化系统（基于Viser）",
        epilog="""
        示例用法：
        uv run python interactive_3d_viewer.py \
          --global-mapping Output/floor_display2/dedup_detections/global_mapping.json \
          --reconstruction Output/floor_display2/reconstruction.glb \
          --image-dir ../imdata/floor_display2/images \
          --detection-dir ../imdata/floor_display2/detections_results
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 核心参数（必需）
    parser.add_argument("--global-mapping", type=str, required=True, help="global_mapping.json 路径")
    parser.add_argument("--reconstruction", type=str, required=True, help="GLB/PLY 3D重建文件路径")
    parser.add_argument("--image-dir", type=str, required=True, help="原始图像目录")

    # 可选参数
    parser.add_argument("--detection-dir", type=str, default=None, help="检测结果目录（默认：自动推断）")
    parser.add_argument("--port", type=int, default=8080, help="Viser服务端口 (默认: 8080)")
    parser.add_argument("--downsample", type=float, default=1.0, help="点云下采样比例 0-1 (默认: 1.0)")
    parser.add_argument("--cache-dir", type=str, default=None, help="缓存目录（默认: 临时目录，退出后自动清理）")
    parser.add_argument(
        "--points-source",
        type=str,
        default="glb",
        choices=["glb", "predictions"],
        help="点云来源：glb（默认）或 predictions（更高几何精度）。",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="强制重建缓存（忽略现有缓存）")

    args = parser.parse_args()

    import tempfile

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix="3d_viewer_cache_"))
    vcfg = ViewerConfig(
        global_mapping=Path(args.global_mapping),
        reconstruction=Path(args.reconstruction),
        image_dir=Path(args.image_dir),
        detection_dir=(Path(args.detection_dir) if args.detection_dir else None),
        cache_dir=cache_dir,
        downsample_ratio=float(args.downsample),
        points_source=str(args.points_source),
        force_rebuild=bool(args.force_rebuild),
    )
    artifacts = build_cache(vcfg)
    vrt = ViewerRuntimeConfig(port=int(args.port))
    viewer = ViserViewer(artifacts, vrt, image_dir=vcfg.image_dir, global_mapping=vcfg.global_mapping)
    viewer.start()


if __name__ == "__main__":
    main()

