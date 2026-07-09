#!/usr/bin/env python3
"""
中间节点 Pipeline:接收上游[images]+[skus JSON] → 3D匹配 → 去重 → 生成带global_id的detection JSON

仅支持工作流中间节点输入：
- --images <dir>
- --skus <dir|file>

输出：
- /output/detection_with_global_id.json  带global_id的detection JSON(保持原始结构)
- /output/dedup/...                      保留完整中间产物（去重/映射等）
"""

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import bson

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd: List[str], description: str, cwd: Optional[Path] = None) -> bool:
    """运行命令并返回是否成功"""
    logger.info(f"开始: {description}")
    logger.info(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"完成: {description}")
        if result.stdout:
            logger.debug(f"输出:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"失败: {description}")
        logger.error(f"错误码: {e.returncode}")
        if e.stdout:
            logger.error(f"标准输出:\n{e.stdout}")
        if e.stderr:
            logger.error(f"错误输出:\n{e.stderr}")
        return False


def process(inputs):
    """
    处理输入的images和skus，生成带global_id的detection JSON

    输入格式（兼容两种）:
    格式1: {
        "images": [bytes, bytes, ...],  # 图片二进制数据列表
        "skus": [dict, dict, ...]       # SKU检测结果列表（每个dict对应一张图片）
    }
    格式2: {
        "images": [bytes, bytes, ...],  # 图片二进制数据列表
        "skus": "json_string"           # JSON字符串，包含多张图片的bbox数据
    }

    输出格式:
    {
        "detection_with_global_id": [...],  # 带global_id的detection JSON
        "global_mapping": {...}             # 全局ID映射表
    }
    """
    try:
        # 解析输入
        images_bytes = inputs.get("images", [])
        skus_raw = inputs.get("skus", [])

        if not images_bytes or not skus_raw:
            raise ValueError("缺少必需参数: images 或 json")

        # skus可能是字符串或列表，支持多种格式
        if isinstance(skus_raw, str):
            # 格式2：skus是JSON字符串（支持单文件多图片数组格式）
            logger.info("检测到skus为JSON字符串格式，正在解析...")
            try:
                skus_data = json.loads(skus_raw)
                if not isinstance(skus_data, list):
                    raise ValueError("skus JSON字符串解析后应为列表")
                logger.info(f"成功解析JSON字符串，包含 {len(skus_data)} 张图片的检测结果")
            except json.JSONDecodeError as e:
                raise ValueError(f"skus JSON字符串解析失败: {e}")
        elif isinstance(skus_raw, list):
            skus_data = skus_raw
            logger.info(f"接收到列表格式，包含 {len(skus_data)} 个元素")
        else:
            raise ValueError(f"skus格式不支持，应为列表或JSON字符串，实际类型: {type(skus_raw)}")

        # 兼容处理：检测是否为嵌套的单文件格式
        # 格式1: [{"classes": {...}, "objects": [...]}, ...]  标准格式，每个元素是一张图片
        # 格式2: [[{"classes": {...}, "objects": [...]}, ...]]  嵌套格式，外层列表只有1个元素
        if len(skus_data) == 1 and isinstance(skus_data[0], dict):
            # 检查是否为 {"skus": [...]} 包装格式
            if "skus" in skus_data[0] and isinstance(skus_data[0]["skus"], list):
                logger.info("检测到 {'skus': [...]} 包装格式，展开...")
                skus_data = skus_data[0]["skus"]

        # 验证数据格式：每个元素应该是包含 classes 和 objects 的字典
        for idx, item in enumerate(skus_data):
            if not isinstance(item, dict):
                raise ValueError(f"skus[{idx}] 应为字典，实际类型: {type(item)}")
            if "objects" not in item:
                logger.warning(f"skus[{idx}] 缺少 'objects' 字段")

        logger.info(f"最终处理后包含 {len(skus_data)} 张图片的检测结果")

        if len(images_bytes) != len(skus_data):
            raise ValueError(f"图片数量({len(images_bytes)})与SKU数量({len(skus_data)})不一致")

        logger.info(f"接收到 {len(images_bytes)} 张图片和 {len(skus_data)} 个SKU检测结果")

        # 创建临时工作目录
        work_root = Path(tempfile.mkdtemp(prefix="sku_mapping_"))
        dataset_dir = work_root / "dataset"
        images_dir = dataset_dir / "images"
        det_dir = dataset_dir / "detections_results"
        images_dir.mkdir(parents=True, exist_ok=True)
        det_dir.mkdir(parents=True, exist_ok=True)

        # 保存图片到临时目录（按顺序编号0, 1, 2, ...，与VGGT索引一致）
        for idx, img_bytes in enumerate(images_bytes):
            # 解码图片以确定格式
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法解码第 {idx} 张图片")

            # 保存为JPG格式
            img_path = images_dir / f"{idx}.jpg"
            cv2.imwrite(str(img_path), img)
            logger.info(f"保存图片: {img_path}")

        # 保存SKU检测结果到临时目录
        for idx, sku_item in enumerate(skus_data):
            det_path = det_dir / f"{idx}.json"
            with det_path.open("w", encoding="utf-8") as f:
                json.dump(sku_item, f, ensure_ascii=False, indent=2)
            logger.info(f"保存检测结果: {det_path}")

        # Step 1: 运行3D匹配（对每张图片作为参考图）
        logger.info("[步骤 1/2] 运行VGGT 3D匹配...")
        code_modules_dir = Path('/app/code')

        # 对每张图片作为参考图进行匹配（使用0-based索引）
        num_images = len(images_bytes)
        for ref_idx in range(num_images):
            logger.info(f"匹配参考图 {ref_idx}/{num_images-1}...")
            matching_cmd = [
                sys.executable, '-m', 'modules.inference',
                '--algorithm', 'point_tracking',
                '--image_folder', str(images_dir),
                '--detection_dir', str(det_dir),
                '--output_dir', str(dataset_dir),
                '--reference_idx', str(ref_idx),
                '--device', 'cuda',
                '--save_json'
            ]

            if not run_command(matching_cmd, f"VGGT 3D匹配 (ref={ref_idx})", cwd=code_modules_dir):
                raise RuntimeError(f"3D匹配失败 (ref={ref_idx})")

        # 调试：检查匹配结果（检查第一个参考图）
        summary_file = dataset_dir / 'output_pt' / '0' / 'matching_summary.txt'
        if summary_file.exists():
            summary_content = summary_file.read_text(encoding='utf-8')
            match_count = summary_content.count('Matched ref')
            found_count = summary_content.count('Found')
            logger.info(f"matching_summary.txt: {match_count} 条Matched记录, {found_count} 条Found记录")
            if match_count == 0:
                logger.warning(f"matching_summary.txt内容:\n{summary_content}")
        else:
            logger.warning(f"matching_summary.txt不存在: {summary_file}")

        # 检查匹配结果
        output_pt_dir = dataset_dir / 'output_pt'
        if not output_pt_dir.exists():
            raise RuntimeError(f"匹配结果目录不存在: {output_pt_dir}")

        # Step 2: 执行去重并生成带global_id的JSON
        logger.info("[步骤 2/2] 执行去重并生成带global_id的JSON...")
        dedup_output = work_root / 'dedup'
        dedup_cmd = [
            sys.executable, '-m', 'modules.deduplicate_detections',
            '--dataset', str(dataset_dir),
            '--dedup_mode', 'any',
            '--min_hit_ratio', '0.0',
            '--output_dir', str(dedup_output),
        ]

        if not run_command(dedup_cmd, "去重并生成global_id JSON", cwd=code_modules_dir):
            raise RuntimeError("去重失败")

        # 读取生成的结果
        dataset_name = dataset_dir.name
        final_output_dir = dedup_output / dataset_name
        merged_gid_path = final_output_dir / 'all_images_with_global_id.json'
        global_mapping_path = final_output_dir / 'global_mapping.json'

        if not merged_gid_path.exists():
            raise RuntimeError(f"输出文件未生成: {merged_gid_path}")

        # 读取结果
        with merged_gid_path.open('r', encoding='utf-8') as f:
            detection_with_global_id = json.load(f)

        global_mapping = {}
        if global_mapping_path.exists():
            with global_mapping_path.open('r', encoding='utf-8') as f:
                global_mapping = json.load(f)

        logger.info(f"Pipeline执行成功，生成 {len(detection_with_global_id)} 个带global_id的检测结果")
        logger.info(f"全局ID数量: {len(global_mapping)} (原始检测框: 436)")

        # 统计去重效果
        total_boxes = sum(len(img_data.get('objects', [])) for img_data in detection_with_global_id)
        logger.info(f"去重后检测框总数: {total_boxes}")

        # 清理临时目录
        try:
            shutil.rmtree(work_root)
            logger.info(f"清理临时目录: {work_root}")
        except Exception as e:
            logger.warning(f"清理临时目录失败: {e}")

        # 返回结果
        return {
            "detection_with_global_id": detection_with_global_id,
            "global_mapping": global_mapping
        }

    except Exception as e:
        logger.error(f"处理失败: {e}")
        logger.error(traceback.format_exc())
        raise
