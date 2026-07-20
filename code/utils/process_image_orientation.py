#!/usr/bin/env python3
"""
修复图像方向脚本
自动处理EXIF方向信息并重新保存图像，解决图像旋转90度的问题
"""

import argparse
import logging
import os
from pathlib import Path

from PIL import Image, ImageOps

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fix_image_orientation(image_path: str, output_path: str = None) -> bool:
    """修复单个图像的方向

    Args:
        image_path: 输入图像路径
        output_path: 输出图像路径，如果为None则覆盖原文件

    Returns:
        bool: 是否成功处理
    """
    try:
        # 打开图像
        with Image.open(image_path) as img:
            # 获取EXIF方向信息
            exif_dict = img.getexif()
            orientation = exif_dict.get(274, 1)  # 274是方向标签，默认为1（正常）

            logger.info(
                f"Processing {Path(image_path).name}, EXIF orientation: {orientation}"
            )

            # 应用EXIF方向并移除EXIF数据
            img_fixed = ImageOps.exif_transpose(img)

            # 确定输出路径
            if output_path is None:
                output_path = image_path

            # 保存图像（不保留EXIF数据）
            if img_fixed.mode in ("RGBA", "LA", "P"):
                # 如果有透明通道，转换为RGB
                rgb_img = Image.new("RGB", img_fixed.size, (255, 255, 255))
                if img_fixed.mode == "P":
                    img_fixed = img_fixed.convert("RGBA")
                rgb_img.paste(
                    img_fixed,
                    mask=(
                        img_fixed.split()[-1]
                        if img_fixed.mode in ("RGBA", "LA")
                        else None
                    ),
                )
                img_fixed = rgb_img

            img_fixed.save(output_path, "JPEG", quality=95, optimize=True)

            if orientation != 1:
                logger.info(f"Fixed orientation for {Path(image_path).name}")
            else:
                logger.info(f"No orientation fix needed for {Path(image_path).name}")

            return True

    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return False


def fix_directory_orientations(
    input_dir: str, output_dir: str = None, backup: bool = True
) -> None:
    """批量修复目录中所有图像的方向

    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则原地修改
        backup: 是否备份原文件
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    # 支持的图像格式
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} image files in {input_dir}")

    # 创建输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
    else:
        output_path = input_path
        if backup:
            backup_dir = input_path / "backup_original"
            backup_dir.mkdir(exist_ok=True)
            logger.info(f"Backup directory: {backup_dir}")

    success_count = 0

    for image_file in image_files:
        try:
            # 确定输出文件路径
            if output_dir:
                output_file = output_path / image_file.name
            else:
                output_file = image_file
                # 备份原文件
                if backup:
                    backup_file = backup_dir / image_file.name
                    if not backup_file.exists():
                        import shutil

                        shutil.copy2(image_file, backup_file)
                        logger.debug(f"Backed up {image_file.name}")

            # 修复图像方向
            if fix_image_orientation(str(image_file), str(output_file)):
                success_count += 1

        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Failed to process {image_file.name}: {e}")
            continue

    logger.info(f"\n=== 处理完成 ===")
    logger.info(f"成功处理: {success_count}/{len(image_files)} 张图片")

    if success_count > 0:
        logger.info("现在图像方向已修复，可以重新运行SKU匹配算法")


def main():
    parser = argparse.ArgumentParser(description="修复图像EXIF方向信息")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../imdata/floor_display2/images",
        help="输入图像目录路径 (default: imdata/floor_display2/images)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出图像目录路径，如果不指定则原地修改",
    )
    parser.add_argument(
        "--no_backup", action="store_true", help="不备份原文件（仅在原地修改时有效）"
    )

    args = parser.parse_args()

    # 修复图像方向
    fix_directory_orientations(
        input_dir=args.input_dir, output_dir=args.output_dir, backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
