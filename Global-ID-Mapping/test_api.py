#!/usr/bin/env python3
"""测试API：传入floor_display2的图片和检测框"""
import json
import bson
import requests
from pathlib import Path
import sys
import base64
from PIL import Image
import io
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 配置
API_URL = "http://localhost:8011/api"
IMAGE_DIR = Path("../imdata/floor_display2/images")
# 使用 api_result.json 中的 global_skus 作为输入
DETECTION_PATH = Path("../imdata/floor_display2/detections_results")  # 使用上次的输出结果
OUTPUT_DIR = None  # 设置为None，让API返回图片bytes数据而不是容器内路径
REQUEST_TIMEOUT = 600  # 请求超时时间（秒），3D重建可能需要较长时间

def load_data(include_output_dir=True):
    """加载图片和检测结果（兼容目录和单个JSON文件）

    Args:
        include_output_dir: 是否在请求中包含output_dir参数
    """
    images = []

    # 按文件名排序（支持大小写）
    image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.JPG")))

    if not image_files:
        raise FileNotFoundError(f"未找到图片文件: {IMAGE_DIR}")

    # 读取所有图片
    print(f"正在加载 {len(image_files)} 张图片...")
    for img_file in image_files:
        with open(img_file, "rb") as f:
            images.append(f.read())

    # 从指定目录逐个读取JSON文件
    # dedup_source_dir = Path("/Users/ahs/Downloads/RetailEye_Docs/3D_SKU_Detection/code/Output1129/floor_display3/")


    # 获取所有数字命名的JSON文件并排序
    json_files = sorted([f for f in DETECTION_PATH.glob("*.json") if f.stem.isdigit()],
                       key=lambda x: int(x.stem))



    m = []  # JSON字符串列表
    for json_file in json_files:
        with json_file.open('r', encoding='utf-8') as f:
            img_data = json.load(f)

        # 使用 json.dumps 转换为字符串
        img_data_str = json.dumps(img_data, ensure_ascii=False)
        m.append(img_data_str)

        # 统计信息
        objects = img_data.get('skus', [{}])[0].get('objects', []) if 'skus' in img_data else img_data.get('objects', [])
        print(f"  ✓ 图片 {json_file.stem}: objects数量={len(objects)}")

    print(f"✓ 提取到 {len(m)} 张图片的检测结果")

    # 构建请求数据
    data = {"images": images, "skus": m}

    # 可选：添加输出目录参数
    if include_output_dir and OUTPUT_DIR:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        data["output_dir"] = str(OUTPUT_DIR.resolve())
        print(f"输出目录: {data['output_dir']}")

    return data

def load_and_display_images(viz_images, max_display=3):
    """加载并显示图片（支持路径列表、bytes列表或字典列表）

    Args:
        viz_images: 图片数据，可以是：
            - 路径列表: ['path1.jpg', 'path2.jpg', ...]
            - bytes列表: [b'\\xff\\xd8...', b'\\xff\\xd8...', ...]
            - 字典列表: [{'filename': 'img.jpg', 'data': bytes}, ...]
        max_display: 最多显示的图片数量

    Returns:
        list: 包含图片信息的字典列表
    """
    images_data = []

    for i, img_item in enumerate(viz_images[:max_display]):
        try:
            # 判断输入格式
            if isinstance(img_item, bytes):
                # 格式1：直接的bytes数据（processor.py无output_dir时返回）
                img_bytes = img_item
                filename = f'image_{i}.jpg'
            elif isinstance(img_item, dict) and 'data' in img_item:
                # 格式2：字典（包含二进制数据）
                img_bytes = img_item['data']
                filename = img_item.get('filename', f'image_{i}.jpg')
            elif isinstance(img_item, str):
                # 格式3：文件路径（processor.py有output_dir时返回）
                with open(img_item, 'rb') as f:
                    img_bytes = f.read()
                filename = Path(img_item).name
            else:
                logger.warning(f"未知的图片格式: {type(img_item)}")
                continue

            # 转换为PIL Image以获取尺寸信息
            img = Image.open(io.BytesIO(img_bytes))
            width, height = img.size

            # Base64编码
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            images_data.append({
                'filename': filename,
                'width': width,
                'height': height,
                'size_kb': len(img_bytes) / 1024,
                'base64': img_base64
            })

            print(f"  ✓ 图片 {i+1}: {filename} ({width}x{height}, {len(img_bytes)/1024:.1f} KB)")

        except Exception as e:
            print(f"  ✗ 加载图片失败: {e}")

    return images_data

def main():
    """主测试函数"""
    try:
        print("=" * 60)
        print("Global-ID-Mapping API 测试")
        print("=" * 60)

        # 加载数据
        print(f"\n[1/3] 加载数据...")
        data = load_data()
        print(f"✓ 图片数量: {len(data['images'])}")

        # 检测结果数量统计
        if isinstance(data['skus'], str):
            skus_count = len(json.loads(data['skus']))
        else:
            skus_count = len(data['skus'])
        print(f"✓ 检测结果数量: {skus_count}")

        # 发送请求
        print(f"\n[2/3] 发送请求到 {API_URL}...")
        print(f"⏳ 处理中（可能需要数分钟，请耐心等待）...")

        bson_data = bson.dumps(data)
        print(f"✓ 请求数据大小: {len(bson_data) / 1024 / 1024:.2f} MB")

        response = requests.post(
            API_URL,
            data=bson_data,
            headers={"Content-Type": "application/bson"},
            timeout=REQUEST_TIMEOUT
        )

        # 处理响应
        print(f"\n[3/3] 处理响应...")
        if response.status_code == 200:
            result = bson.loads(response.content)

            print(f"\n{'=' * 60}")
            print("✓ 处理成功!")
            print(f"{'=' * 60}")

            # 统计结果
            global_skus = result.get('global_skus', [])

            print(f"\n结果统计:")
            print(f"  - 带全局ID的SKU数量: {len(global_skus)} 张图片")

            # 统计全局ID和去重信息
            total_objects = 0
            total_kept = 0
            total_removed = 0
            global_ids = set()

            for i, sku_str in enumerate(global_skus):
                try:
                    sku_data = json.loads(sku_str) if isinstance(sku_str, str) else sku_str
                    objects = sku_data.get('objects', [])
                    total_objects += len(objects)

                    for obj in objects:
                        gid = obj.get('global_id')
                        if gid is not None:
                            global_ids.add(gid)
                        if obj.get('is_deduplicated', False):
                            total_removed += 1
                        else:
                            total_kept += 1
                except Exception as e:
                    logger.warning(f"解析第{i}张图片SKU数据失败: {e}")

            print(f"  - 全局ID数量: {len(global_ids)} 个")
            print(f"  - 总物体数: {total_objects}")
            print(f"  - 保留物体数: {total_kept} (is_deduplicated=false)")
            print(f"  - 去重物体数: {total_removed} (is_deduplicated=true)")

            # 保存JSON结果
            output_file = Path("api_result.json")
            save_result = {
                "global_skus": global_skus,
            }
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(save_result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ JSON结果已保存到: {output_file}")

            return 0

        else:
            print(f"\n{'=' * 60}")
            print(f"✗ 请求失败!")
            print(f"{'=' * 60}")
            print(f"状态码: {response.status_code}")
            print(f"错误信息:\n{response.text}")
            return 1

    except requests.exceptions.Timeout:
        print(f"\n✗ 请求超时（超过 {REQUEST_TIMEOUT} 秒）")
        print(f"提示: 3D重建可能需要较长时间，可以增加 REQUEST_TIMEOUT 参数")
        return 1
    except requests.exceptions.ConnectionError:
        print(f"\n✗ 连接失败: 无法连接到 {API_URL}")
        print(f"提示: 请确保API服务已启动")
        return 1
    except FileNotFoundError as e:
        print(f"\n✗ 文件未找到: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
