import os
import torch

#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ----------------- 配置区 -----------------
IMAGE_PATH = "/home/chenxingyu/3D_Recognization/imdata/floor_display11/images/1.JPG"
OUTPUT_ROOT = "outputs" 

# only accepts single concept
# TEXT_PROMPT = "bottles, cans, jars, boxes and bags" 
TEXT_PROMPT = "boxes"
CONFIDENCE_THRESHOLD = 0.5
# MAX_SIZE = 2000  # 仅在原图过大时缩放，保持宽高比
# ---------------------------------------

# 创建输出目录（按图片名再建子目录，方便管理多张图）
image_basename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
output_dir = os.path.join(OUTPUT_ROOT, image_basename)
os.makedirs(output_dir, exist_ok=True)

# Load the model
model = build_sam3_image_model(checkpoint_path="checkpoints/sam3.pt", load_from_HF=False)
processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
# print("mean abs weight:", next(model.parameters()).abs().mean())

# Load an image
image = Image.open(IMAGE_PATH)
print(f"原始图片尺寸: {image.size}")

# # 缩小图片以节省显存（保持宽高比）
# if max(image.size) > MAX_SIZE:
#     ratio = MAX_SIZE / max(image.size)
#     new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
#     image = image.resize(new_size, Image.Resampling.LANCZOS)
#     print(f"缩放后图片尺寸: {image.size}")

inference_state = processor.set_image(image)

# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

#################################### Visualize Masks ####################################
import numpy as np
import cv2
from sam3.visualization_utils import plot_results
import matplotlib.pyplot as plt

# 方法1: 使用官方plot_results函数（推荐）
if len(boxes) > 0:
    print("\n正在生成可视化结果...")
    plot_results(image, output)
    det_path = os.path.join(output_dir, f"{image_basename}_detection_result.jpg")
    plt.savefig(det_path, dpi=150, bbox_inches="tight")
    print(f"✓ 已保存可视化结果到 {det_path}")
    plt.close()

    print(f"\n✓ 已保存 {len(masks)} 个mask图片到 {output_dir}")
    print(f"✓ 已保存 {len(masks)} 个叠加结果到 {output_dir}")
else:
    print("\n⚠ 未检测到任何对象，请检查:")
    print("  1. checkpoint文件是否正确")
    print("  2. text prompt是否合适（建议使用简单的单词，如 'bottle'）")
    print("  3. 图片中是否包含目标对象")
    print("  4. 尝试降低confidence_threshold或添加bfloat16精度")
