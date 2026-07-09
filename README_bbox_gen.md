# bbox_gen.py

SKU 检测 CLI：扫描图片目录或单张图片，用 YOLO 检测，输出**可视化图 + 下游兼容 JSON**。

## 用法

```bash
# 目录输入（递归扫描 .jpg/.jpeg/.png）
uv run python bbox_gen.py imdata/floor_display1/images -o output

# 单张图片输入
uv run python bbox_gen.py imdata/floor_display1/images/1.jpg -o output

# 指定 GPU / 阈值 / 模型
CUDA_VISIBLE_DEVICES=1 uv run python bbox_gen.py imdata/floor_display1/images -o output \
    --device cuda:0 --conf 0.4 --iou 0.3 --model checkpoints/best.pt
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `input`（位置参数）| - | 图片目录或单张图片路径（.jpg/.jpeg/.png） |
| `-o, --output` | `output` | 输出目录 |
| `-m, --model` | `checkpoints/best.pt` | YOLO 权重路径 |
| `--imgsz` | `640` | 推理输入尺寸 |
| `--conf` | `0.35` | 置信度阈值 |
| `--iou` | `0.25` | NMS IoU 阈值 |
| `--device` | 自动 | 推理设备，如 `cuda:0` / `cpu` |

## 输出结构

```
output/
├── images/                 # 带检测框的可视化图（.jpg）
│   ├── 1.jpg
│   └── ...
└── detections_results/      # 检测结果 JSON（与下游 code/Global-ID-Mapping 对齐）
    ├── 1.json
    └── ...
```

### JSON 格式

```json
{
  "skus": [
    {
      "classes": { "det": ["8926^bottle"] },
      "objects": [
        {
          "position": [x1, y1, x2, y2],   // 整数像素
          "classes": { "det": 0 },          // 类别索引
          "confidences": { "det": 0.9388 } // 置信度
        }
      ]
    }
  ]
}
```

与 `imdata/floor_display*/detections_results/*.json` 格式一致，可直接作为 `code/main.py --mode pipeline` 的 `detections_results/` 输入，下游 `utils/data_utils.py::load_detections` / `extract_bboxes_from_detections` 可直接解析。

## GPU 注意事项

`torch 2.7.0+cu126` + `cuDNN 9.2` 在本机可能遇到 `CUDNN_STATUS_NOT_INITIALIZED`。规避方式：

- 指定空闲卡：`CUDA_VISIBLE_DEVICES=1 uv run python bbox_gen.py ... --device cuda:0`
- 或退回 CPU：`--device cpu`（慢但可用）
