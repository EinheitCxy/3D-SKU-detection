# 3D货架重建与SKU匹配（含顺序去重）

[English Version](README_EN.md) | [Code README](code/README.md)

项目聚焦两部分核心能力：
- 跨图片 SKU 匹配（点追踪与 3D→2D 投影两种算法）
- 顺序去重与全局 ID 聚合（global_mapping）

## 功能特性

- 🎯 跨图像 SKU 匹配：点追踪与 3D 投影两套实现，可独立或同时运行
- 🔄 顺序去重：按顺序移除重复检测，生成统一的全局 ID 映射
- 👀 可视化与摘要：检出框可视化、匹配日志解析与汇总
- 📊 批量评估：与标注对照的准确性评估与汇总
- 🧩 模块化：`modules/` 可执行脚本 + `utils/` 复用库，便于拓展与集成

## 项目结构（精简）

```
3D_SKU_Detection/
├── code/                          # 主要代码目录
│   ├── main.py                    # 统一 CLI（interactive/pipeline/concise/analyzer/dedup）
│   ├── modules/                   # 可执行脚本（匹配/去重/可视化/评估/3D重建）
│   ├── utils/                     # 复用库模块（算法/工具/可视化等）
│   ├── batch_run_inference.sh     # 批量匹配
│   └── batch_accuracy_evaluation.sh # 批量评估
├── imdata/                        # 数据集（images/ + detections_results/）
├── vggt-main/                     # VGGT 依赖（已 vendor，可选）
├── ultralytics/                   # 其他 vendor 依赖（可选）
├── requirements.txt               # 根依赖
└── README.md, README_EN.md        # 顶层说明
```

## 安装依赖

推荐使用 uv：

```bash
# 根依赖（如有）
uv pip install -r requirements.txt

# 进入 code/ 并同步项目依赖
cd code && uv sync
```

## 快速开始（统一 CLI）

```bash
# 顺序去重（默认同名输出），并生成全局ID映射
uv run python code/main.py --mode dedup --dataset imdata/floor_display2 --save_root ./Output

# 完整流水线（校验→可视化→匹配→分析→顺序去重→评估）
uv run python code/main.py --mode pipeline --dataset imdata/floor_display2 --save_root ./Output

# 仅匹配（点追踪+3D 两者）
uv run python code/main.py --mode concise --dataset imdata/floor_display2 --algorithm both --save_root ./Output

# 批量匹配（参考索引 0..N）
bash code/batch_run_inference.sh floor_display2 4
```

更多细节（参数、输出路径、测试命令）参考 `code/README.md` 与 `code/README_EN.md`。

## 日志

- 每次运行仅生成一个日志文件，位于 `<save_root>/run_YYYYMMDD_HHMMSS.log`
- 使用 `--save_root` 统一控制日志与所有产物的根目录
- 控制台输出与文件内容一致，便于实时查看与追溯

## 使用方法

更多命令与参数请见 `code/README.md`；核心脚本位于 `code/modules/`，算法与工具位于 `code/utils/`。

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image_dir` | `imdata` | 图片目录路径 |
| `--detection_file` | `sku_detection.json` | 检测结果JSON文件路径 |
| `--output_dir` | `output` | 输出目录路径 |
| `--eps` | `0.5` | DBSCAN聚类距离阈值 (米) |
| `--min_samples` | `2` | DBSCAN最小样本数 |
| `--no_viz` | - | 跳过3D可视化 |

## 检测数据格式

检测结果JSON文件应包含以下格式：

```json
[
  {
    "classes": {
      "det": ["8926^bottle"]
    },
    "objects": [
      {
        "position": [x1, y1, x2, y2],
        "classes": {"det": 0},
        "confidences": {"det": 0.93}
      }
    ]
  }
]
```

## 输出文件

### 📦 GLB/GLTF 3D文件 (主要格式)

所有工具自动输出GLB格式文件到 `output` 目录：

**基础可视化**：
- `point_cloud.glb`: 3D点云文件
- `detection_scene.glb`: 完整检测场景

**高级重建**：
- `shelf_point_cloud.glb`: 货架点云
- `detection_points.glb`: 检测点云
- `complete_scene.glb`: 完整3D场景

**SKU聚类分析**：
- `sku_clusters_3d_YYYYMMDD_HHMMSS.glb`: 聚类结果3D可视化
- `sku_cluster_analysis_YYYYMMDD_HHMMSS.json`: 详细分析数据
- `sku_analysis_report_YYYYMMDD_HHMMSS.txt`: 人类可读报告
- `sku_centers_3d_YYYYMMDD_HHMMSS.json`: 3D坐标数据
- `clusters_3d_viz_YYYYMMDD_HHMMSS.png`: 聚类可视化图

### 📄 其他输出文件

- `3d_visualization.png`: matplotlib 3D可视化图片
- `detection_report.json`: 检测统计报告
- `camera_poses.json`: 相机姿态数据

> 💡 **GLB文件兼容性**: 可在Blender、Three.js、Unity、GLTF Viewer等3D软件中直接打开

## 技术实现

### 1. 🏗️ 3D重建流程

1. **Fast3R重建** (推荐): 使用CVPR 2025技术，支持1000+图片
2. **VGGT重建** (备选): Meta AI的Visual Geometry Grounded Transformer
3. **COLMAP回退**: 传统SfM方法作为最后备选

### 2. 🎯 SKU聚类算法

**核心约束**: 同一图片内的物体绝不会被聚类

```python
# 聚类后处理，确保跨图片聚类
for cluster_id, images_dict in cluster_to_images.items():
    images_with_multiple_objects = []
    for img_idx, point_indices in images_dict.items():
        if len(point_indices) > 1:  # 同一图片多个物体
            # 只保留第一个物体，其余标记为噪声点
            for point_idx in point_indices[1:]:
                new_cluster_labels[point_idx] = -1
```

### 3. 🚀 设备优化

智能设备选择优先级：
1. **CUDA**: NVIDIA GPU加速
2. **MPS**: Apple Silicon GPU加速  
3. **CPU**: 通用处理器回退

## 使用示例

### 典型工作流程

```bash
# 1. 将图片放入imdata目录
# 2. 确保有sku_detection.json检测结果
# 3. 运行SKU聚类分析
uv run python sku_cluster_analyzer.py

# 4. 查看results目录的GLB文件和报告
ls output/
```

### 测试数据结果

在包含13张图片和446个SKU检测的测试中：
- ✅ 识别15个SKU类型 (跨图片聚类)
- ✅ 412个独立SKU正确标记
- ✅ 无同图片内物体错误聚类
- ✅ 平均聚类距离0.6-1.6米

## 开发日志

### 2025-01-29 重大更新
- ✅ 删除GUI交互式功能，简化项目结构
- ✅ 实现跨图片SKU聚类，确保同图片内物体不聚类
- ✅ 所有工具统一输出GLB格式到output目录
- ✅ 智能设备选择 (CUDA → MPS → CPU)
- ✅ 完整的命令行工具和报告系统

### 2024-12-19 初始版本
- 创建基础3D检测可视化脚本
- 实现2D到3D映射功能
- 添加多种可视化方法
- 集成Fast3R和VGGT模型

## 故障排除

### 常见问题

1. **Fast3R导入错误**: 
   ```bash
   pip install -r fast3r/requirements.txt
   ```

2. **GLB导出失败**: 
   ```bash
   uv pip install trimesh pygltflib
   ```

3. **CUDA不可用**: 
   - 系统会自动回退到MPS (Apple Silicon) 或CPU
   - 无需手动干预

4. **内存不足**: 
   - 减少图片数量或使用 `--no_viz` 跳过可视化

### 性能优化

- **GPU加速**: 确保安装CUDA/MPS支持
- **大数据集**: 使用 `--no_viz` 跳过可视化
- **聚类参数**: 调整 `--eps` 和 `--min_samples` 参数
