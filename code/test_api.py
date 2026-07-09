#!/usr/bin/env python3
"""测试API：传入floor_display2的图片和检测框"""
import json
import bson
import requests
from pathlib import Path

# 配置
API_URL = "http://localhost:8000/api"
IMAGE_DIR = Path("../imdata/floor_display2/images")
DETECTION_PATH = Path("../imdata/floor_display2/detections_results")  # 可以是目录或单个JSON文件

def load_data():
    """加载图片和检测结果（兼容目录和单个JSON文件）"""
    images = []

    # 按文件名排序（支持大小写）
    image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.JPG")))

    # 读取所有图片
    for img_file in image_files:
        with open(img_file, "rb") as f:
            images.append(f.read())

    # 读取检测结果（兼容单个JSON文件或目录）
    if DETECTION_PATH.is_file() and DETECTION_PATH.suffix == '.json':
        # 单个JSON文件：作为字符串传递给API
        print(f"检测到单个JSON文件: {DETECTION_PATH}")
        with open(DETECTION_PATH, "r") as f:
            skus = f.read()
    elif DETECTION_PATH.is_dir():
        # 目录格式：每张图片对应一个JSON文件
        print(f"检测到目录格式: {DETECTION_PATH}")
        skus = []
        for img_file in image_files:
            det_file = DETECTION_PATH / f"{img_file.stem}.json"
            with open(det_file, "r") as f:
                skus.append(json.load(f))
    else:
        raise FileNotFoundError(f"检测路径不存在或格式不正确: {DETECTION_PATH}")

    return {"images": images, "skus": skus}

def main():
    print(f"加载数据...")
    data = load_data()
    print(f"图片数量: {len(data['images'])}")
    print(f"检测结果数量: {len(data['skus'])}")

    print(f"\n发送请求到 {API_URL}...")
    # 使用BSON编码
    bson_data = bson.dumps(data)
    response = requests.post(API_URL, data=bson_data, headers={"Content-Type": "application/bson"})

    if response.status_code == 200:
        # 解析BSON响应
        result = bson.loads(response.content)
        print(f"\n成功! 返回结果:")
        print(f"- detection_with_global_id: {len(result.get('detection_with_global_id', []))} 条")
        print(f"- global_mapping: {len(result.get('global_mapping', {}))} 个全局ID")

        # 保存结果
        output_file = Path("api_result.json")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    else:
        print(f"\n失败! 状态码: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
