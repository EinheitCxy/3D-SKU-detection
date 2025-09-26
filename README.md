# 3D货架重建与SKU聚类分析（含顺序去重）

[English Version](README_EN.md) | [Code README](code/README.md)

这个项目实现了将多张货架图片进行3D重建，并在生成的3D点云上显示物体检测的中心点，支持跨图片SKU聚类分析。

## 功能特性

- 🏗️ **高质量3D重建**: 使用Fast3R进行真实3D重建，自动回退到COLMAP
- 🎯 **物体检测映射**: 将2D检测结果精确映射到3D空间
- 🔍 **智能SKU聚类**: 仅对不同图片中的物体进行聚类，识别相同SKU
- 📦 **GLB/GLTF导出**: 自动导出GLB格式3D文件，兼容Blender、Three.js等
- 🚀 **智能设备选择**: 自动选择最优计算设备 (CUDA → MPS → CPU)
- 📊 **详细报告**: 生成完整的分析报告和3D可视化

## 项目结构

```
3D_SKU_Detection/
├── imdata/                         # 货架图片目录
├── sku_count/                      # 主要技术文件目录
│   ├── sku_cluster_analyzer.py     # SKU聚类分析 (主要工具)
│   ├── 3d_detection_visualization.py # 基础3D可视化
│   ├── advanced_3d_reconstruction.py # 高级3D重建
│   ├── device_utils.py             # 智能设备选择
│   ├── gltf_export_utils.py        # GLB/GLTF导出工具
│   └── ...                         # 其他技术文件
├── fast3r/                         # Fast3R 3D重建模型
├── vggt-main/                      # VGGT模型 (备选)
├── requirements.txt                # 依赖文件
└── README.md                       # 说明文档
```

## 安装依赖

使用uv安装依赖：

```bash
# 安装主要依赖
uv pip install -r requirements.txt

# 安装Fast3R依赖 (可选，用于高质量3D重建)
pip install -r fast3r/requirements.txt

# 安装GLB导出依赖 (可选，用于3D文件导出)
uv pip install trimesh pygltflib
```

## 快速使用（统一 CLI）

```bash
# 顺序去重（默认同名输出），并生成全局ID映射
uv run python code/main.py --mode dedup --dataset imdata/floor_display2 --save_root ./Output

# 完整流水线（校验→可视化→匹配→分析→顺序去重→评估）
uv run python code/main.py --mode pipeline --dataset imdata/floor_display2 --save_root ./Output
```

更多说明见 `code/README.md` 与 `README_EN.md`。

## 使用方法

### 🎯 SKU聚类分析 (推荐)

**主要功能**，用于识别不同图片中的相同SKU：

```bash
# 进入sku_count目录
cd sku_count

# 基本使用
uv run python sku_cluster_analyzer.py

# 自定义参数
uv run python sku_cluster_analyzer.py --eps 0.8 --min_samples 3 --output_dir ../results

# 跳过可视化 (加快处理速度)
uv run python sku_cluster_analyzer.py --no_viz
```

**关键特性**：

- ✅ 只对不同图片中的物体进行聚类
- ❌ 同一图片内的物体绝不会被聚类
- 📍 基于3D空间距离识别相同SKU

### 🏗️ 基础3D可视化

简单的3D重建和可视化：

```bash
cd sku_count
uv run python 3d_detection_visualization.py --image_dir ../imdata --detection_file ../sku_detection.json --output_dir ../output
```

### 🚀 高级3D重建

使用Fast3R进行高质量3D重建：

```bash
cd sku_count
uv run python advanced_3d_reconstruction.py --image_dir ../imdata --detection_file ../sku_detection.json --output_dir ../output
```

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
