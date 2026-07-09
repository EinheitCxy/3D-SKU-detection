#!/usr/bin/env python3
"""测试读取 sku_detection.json 格式"""
import json
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_parse_skus():
    """测试解析 sku_detection.json"""
    json_path = Path(__file__).parent.parent / "sku_detection.json"

    print(f"读取文件: {json_path}")
    if not json_path.exists():
        print(f"❌ 文件不存在: {json_path}")
        return False

    # 读取JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        skus_raw = f.read()

    print(f"✓ 文件大小: {len(skus_raw)} 字节")

    # 模拟 processor.py 的解析逻辑
    try:
        # 解析JSON字符串
        skus_data = json.loads(skus_raw)
        print(f"✓ JSON解析成功")

        if not isinstance(skus_data, list):
            print(f"❌ 应为列表，实际类型: {type(skus_data)}")
            return False

        print(f"✓ 数据类型: list，包含 {len(skus_data)} 个元素")

        # 检查格式
        if len(skus_data) == 1 and isinstance(skus_data[0], dict):
            if "skus" in skus_data[0] and isinstance(skus_data[0]["skus"], list):
                print(f"⚠ 检测到包装格式 {{'skus': [...]}}, 需要展开")
                skus_data = skus_data[0]["skus"]
                print(f"✓ 展开后包含 {len(skus_data)} 张图片")

        # 验证每个元素
        valid_count = 0
        for idx, item in enumerate(skus_data):
            if not isinstance(item, dict):
                print(f"❌ skus[{idx}] 应为字典，实际: {type(item)}")
                continue

            if "objects" not in item:
                print(f"⚠ skus[{idx}] 缺少 'objects' 字段")
                continue

            if "classes" not in item:
                print(f"⚠ skus[{idx}] 缺少 'classes' 字段")

            obj_count = len(item.get("objects", []))
            valid_count += 1

            if idx < 3:  # 只显示前3张
                print(f"  图片 {idx}: {obj_count} 个检测框")

        print(f"\n✓ 验证完成: {valid_count}/{len(skus_data)} 张图片格式正确")

        # 统计总检测框数
        total_boxes = sum(len(item.get("objects", [])) for item in skus_data)
        print(f"✓ 总检测框数: {total_boxes}")

        return True

    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parse_skus()
    sys.exit(0 if success else 1)
