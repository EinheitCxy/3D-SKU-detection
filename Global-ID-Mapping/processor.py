import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
# ===========================================================================
# 路径配置：动态定位 Global-ID-Mapping/code 目录
# ============================================================================
PROCESSOR_DIR = Path(__file__).resolve().parent  # Global-ID-Mapping/
CODE_DIR = PROCESSOR_DIR / "code"  # Global-ID-Mapping/code/

if not CODE_DIR.exists():
    raise RuntimeError(f"Code directory not found: {CODE_DIR}")

# 将code目录放到sys.path最前，避免导入到 /app/main.py 造成循环依赖
code_dir_str = str(CODE_DIR)
if code_dir_str in sys.path:
    sys.path.remove(code_dir_str)
sys.path.insert(0, code_dir_str)

# ============================================================================
# 导入 Global-ID-Mapping/code 中的模块
# ============================================================================
from main import SKUDetectionMain

logger = logging.getLogger(__name__)
logger.info(f"✓ 成功导入 Global-ID-Mapping/code 模块")
logger.info(f"  - CODE_DIR: {CODE_DIR}")


# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process(inputs: dict) -> dict:
    """
    处理输入的images和skus，生成带global_id的detection JSON

    处理流程:
    1. Pi3 3D重建：多视图场景重建
    2. 3D投影匹配：跨图像对象匹配
    3. 智能去重：基于并查集的全局ID分配
       - 防止同图片物体合并到同一全局ID
       - 支持间接去重（传递性匹配自动标记）
       - 每个全局ID只有最早出现的物体标记为is_deduplicated=False

    输入格式:
    {
        "images": [bytes, bytes, ...],  # 图片二进制数据列表
        "skus": [str, str, ...]         # SKU检测结果列表（每个str是JSON字符串，对应一张图片）
    }

    输出格式:
    {
        "global_skus": [str, str, ...],  # 带全局ID的SKU列表，每个str是JSON字符串，包含:
                                         #   - global_id: 全局唯一ID
                                         #   - is_deduplicated: True表示该物体在更早图片中已出现
    }

    """
    images_bytes = inputs.get("images", [])
    skus_raw = inputs.get("skus", [])

    logger.info(f"接收到 {len(images_bytes)} 张图片和 {len(skus_raw)} 个 detection 文件")

    # 解析 skus：每个元素是 JSON 字符串
    skus_data = []
    for idx, item in enumerate(skus_raw):
        parsed_item = json.loads(item)
        if not isinstance(parsed_item, dict):
            raise ValueError(f"skus[{idx}] JSON解析后应为字典，实际类型: {type(parsed_item)}")
        skus_data.append(parsed_item)

    logger.info(f"成功解析 {len(skus_data)} 个 detection 文件")

    # 验证数据格式
    for idx, item in enumerate(skus_data):
        if "objects" not in item:
            logger.warning(f"skus[{idx}] 缺少 'objects' 字段")

    if len(images_bytes) != len(skus_data):
        raise ValueError(f"图片数量({len(images_bytes)})与SKU数量({len(skus_data)})不一致")

    logger.info(f"验证通过: {len(images_bytes)} 张图片和 {len(skus_data)} 个SKU检测结果")

    # 创建临时工作目录
    work_root = Path(tempfile.mkdtemp(prefix="sku_mapping_"))
    dataset_dir = work_root / "dataset"
    images_dir = dataset_dir / "images"
    det_dir = dataset_dir / "detections_results"
    images_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"保存图片: {images_dir}")
    for idx, img_bytes in enumerate(images_bytes):
        # 解码图片以确定格式
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"无法解码第 {idx} 张图片")
        
        # 保存为JPG格式
        img_path = images_dir / f"{idx}.jpg"
        cv2.imwrite(str(img_path), img)
        

    logger.info(f"保存检测结果: {det_dir}")
    for idx, sku_item in enumerate(skus_data):
        det_path = det_dir / f"{idx}.json"
        with det_path.open("w", encoding="utf-8") as f:
            json.dump(sku_item, f, ensure_ascii=False, indent=2)

    # Step 1: 运行Pi3 3D重建 + 3D投影匹配
    logger.info("[步骤 1/3] 运行Pi3 3D重建...")

    system = SKUDetectionMain()
    system.save_root = work_root  # 设置输出根目录
    system.match_backend = 'pi3'  # 使用Pi3后端

    # 1.1 运行Pi3 3D重建
    recon_result = system.run_reconstruction(
        dataset_path=str(dataset_dir),
        model_path='/app/Pi3/checkpoints/snapshots/ae722e7039287d0c8fde9f11f197f804f44b510c/model.safetensors',
        backend='pi3',
        device='cuda',
    )

    if not recon_result.get('success', False):
        error_msg = recon_result.get('error', 'Unknown error')
        raise RuntimeError(f"Pi3 3D重建失败: {error_msg}")

    logger.info("[步骤 2/3] 运行Pi3 3D投影匹配...")

    # 1.2 运行3D投影匹配（使用Pi3后端）
    match_result = system.run_sku_matching(
        dataset_path=str(dataset_dir),
        algorithm='3d',  # 使用3D投影算法
        batch_all_refs=True,  # 批量处理所有参考图
        device='cuda',
        save_json=True,
        backend='pi3',  # 使用Pi3后端
    )

    if not match_result.get('success', False):
        error_msg = match_result.get('error', 'Unknown error')
        raise RuntimeError(f"Pi3 3D投影匹配失败: {error_msg}")

    # 调试：检查匹配结果（Pi3使用output_3dmapping_pi3目录）
    output_dir = work_root / dataset_dir.name / 'output_3dmapping_pi3'
    summary_file = output_dir / '0' / 'matching_summary.txt'
    if summary_file.exists():
        summary_content = summary_file.read_text(encoding='utf-8')
        match_count = summary_content.count('Matched ref')
        found_count = summary_content.count('Found')
        logger.info(f"✓ matching_summary.txt: {match_count} 条Matched记录, {found_count} 条Found记录")
        if match_count == 0:
            logger.warning(f"⚠ matching_summary.txt内容:\n{summary_content[:500]}")
    else:
        logger.warning(f"⚠ matching_summary.txt不存在: {summary_file}")

    # 检查匹配结果目录
    if not output_dir.exists():
        raise RuntimeError(f"匹配结果目录不存在: {output_dir}")

    logger.info("[步骤 3/3] 执行去重并生成带global_id的JSON...")

    # Step 2: 执行去重
    dedup_result = system.run_dedup_sequence(
        dataset_path=str(dataset_dir)
    )

    if not dedup_result.get('success', False):
        error_msg = dedup_result.get('error', 'Unknown error')
        raise RuntimeError(f"去重失败: {error_msg}")

    # 读取生成的结果（去重输出在work_root/<dataset_name>/dedup_detections/）
    dataset_name = dataset_dir.name
    final_output_dir = work_root / dataset_name / 'dedup_detections'
    merged_gid_path = final_output_dir / 'global_skus.json'
    global_mapping_path = final_output_dir / 'global_mapping.json'

    if not merged_gid_path.exists():
        raise RuntimeError(f"输出文件未生成: {merged_gid_path}")

    # 读取 global_mapping
    global_mapping = {}
    if global_mapping_path.exists():
        with global_mapping_path.open('r', encoding='utf-8') as f:
            global_mapping = json.load(f)

    logger.info(f"全局ID数量: {len(global_mapping)}")

    # 读取 global_skus.json（字符串列表格式）
    with merged_gid_path.open('r', encoding='utf-8') as f:
        global_skus = json.load(f)

    logger.info(f"从 global_skus.json 读取 {len(global_skus)} 张图片的数据")

    # 获取所有数字命名的JSON文件并排序
    json_files = sorted([f for f in final_output_dir.glob("*.json") if f.stem.isdigit()],
                       key=lambda x: int(x.stem))

    logger.info(f"从 {final_output_dir} 读取 {len(json_files)} 个JSON文件")

    # 逐个读取JSON文件并使用json.dumps转换为字符串
    total_objects = 0
    for json_file in json_files:
        with json_file.open('r', encoding='utf-8') as f:
            img_data = json.load(f)

        # # 使用 json.dumps 转换为字符串
        # img_data_str = json.dumps(img_data, ensure_ascii=False)
        # deduped_skus_strings.append(img_data_str)

        objects = img_data.get('skus', [{}])[0].get('objects', []) if 'skus' in img_data else img_data.get('objects', [])
        total_objects += len(objects)
        logger.info(f"图片 {json_file.stem}: objects数量={len(objects)}")

    # logger.info(f"总图片={len(deduped_skus_strings)}, 总对象={total_objects}")


    # return images with bbox

    # dedup_viz = system.run_detection_visualization(
    #     dataset_path=str(dataset_dir),
    #     detection_dir=str(final_output_dir),
    #     output_suffix="dedup_imgs_w_bboxes",
    # )
    # if not dedup_viz.get("success", False):
    #     raise RuntimeError(f"去重可视化失败: {dedup_viz.get('error', 'unknown error')}")

    # # 直接从 dataset_dir/dedup_imgs_w_bboxes 读取可视化图片
    # dedup_viz_dir = dataset_dir / "dedup_imgs_w_bboxes"
    # if not dedup_viz_dir.exists():
    #     raise RuntimeError(f"去重可视化目录不存在: {dedup_viz_dir}")


    # dedup_viz_images: Union[List[str], List[bytes]] = []
    # if output_root:
    #     output_root_path = Path(output_root)
    #     try:
    #         output_root_path.mkdir(parents=True, exist_ok=True)
    #         # 复制去重检测结果目录
    #         dedup_output_dir = output_root_path / "dedup"
    #         shutil.copytree(final_output_dir, dedup_output_dir, dirs_exist_ok=True)
    #         # 复制去重可视化图片目录
    #         dedup_viz_output_dir = output_root_path / "dedup_imgs_w_bboxes"
    #         shutil.copytree(dedup_viz_dir, dedup_viz_output_dir, dirs_exist_ok=True)
    #         # 返回图片路径列表
    #         dedup_viz_images = [
    #             str(p)
    #             for p in sorted(dedup_viz_output_dir.iterdir())
    #             if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    #         ]
    #         logger.info(f"输出结果到: {output_root_path}")
    #     except Exception as e:
    #         raise RuntimeError(f"写入输出目录失败: {output_root}") from e
    # else:
    #     # 无输出目录：在清理临时目录前，将图片读取到内存
    #     logger.info(f"未提供输出目录，将图片数据加载到内存中返回")
    #     for p in sorted(dedup_viz_dir.iterdir()):
    #         if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
    #             try:
    #                 img_bytes = p.read_bytes()
    #                 dedup_viz_images.append(img_bytes)
    #                 logger.debug(f"加载图片到内存: {p.name} ({len(img_bytes)/1024:.1f} KB)")
    #             except Exception as e:
    #                 logger.warning(f"读取图片失败 {p.name}: {e}")
    #     total_mb = sum(len(img) for img in dedup_viz_images) / 1024 / 1024
    #     logger.info(f"已加载 {len(dedup_viz_images)} 张图片到内存 (总大小: {total_mb:.2f} MB)")

    try:
        shutil.rmtree(work_root)
        logger.info(f"清理临时目录: {work_root}")
    except Exception as e:
        logger.warning(f"清理临时目录失败: {e}")

    # logger.info(f"返回 {len(deduped_skus_strings)} 个去重后的json")

    return {
        "global_skus": global_skus,
    }
