# 3D货架重建与SKU匹配（含顺序去重）

[English Version](README_EN.md) | [Code README](code/README.md)

项目聚焦两部分核心能力：
- 跨图片 SKU 匹配（点追踪与 3D→2D 投影两种算法）
- 顺序去重与全局 ID 聚合（global_mapping）

## 功能特性

- **跨图像SKU匹配**：点追踪与3D投影两套算法，可独立或同时运行
- **智能去重系统**：顺序去重，自动生成全局ID映射，支持批量处理
- **交互式3D可视化**：基于Viser的实时3D viewer，支持GPU加速和进度条显示
- **准确性评估**：与人工标注对照，计算Precision/Recall/F1指标
- **模块化架构**：统一CLI接口，模块化设计便于拓展与集成
- **性能优化**：FAISS GPU/CPU加速，智能缓存管理，自动降级策略
- **代码质量**：统一数据加载接口，消除重复逻辑，完善错误处理

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

# 3D交互式可视化（Viewer模式）
uv run python code/main.py --mode viewer --dataset imdata/floor_display2 --save_root ./Output

# 批量匹配（参考索引 0..N）
bash code/batch_run_inference.sh floor_display2 4
```

**Viewer模式说明**：
- 启动基于Viser的交互式3D可视化服务器
- 默认端口：8080，浏览器访问 `http://localhost:8080`
- 自动推导路径：无需手动指定global_mapping、reconstruction等参数
- 支持GPU/CPU加速的KNN搜索和点云下采样
- 智能缓存：首次构建后自动检测文件变更，按需重建

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

### GLB/GLTF 3D文件 (主要格式)

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

### 其他输出文件

- `3d_visualization.png`: matplotlib 3D可视化图片
- `detection_report.json`: 检测统计报告
- `camera_poses.json`: 相机姿态数据

> **GLB文件兼容性**: 可在Blender、Three.js、Unity、GLTF Viewer等3D软件中直接打开

## 技术实现

### 1. 3D重建流程

**支持的重建方法**：
1. **VGGT** (Visual Geometry Grounded Transformer): Meta AI的前馈式3D重建模型
2. **COLMAP**: 传统SfM方法作为备选

**路径注入策略**：
- 统一由 `utils/__init__.py` 管理VGGT路径注入
- 其他模块通过 `import utils` 或 `from utils import get_vggt_root` 触发路径配置
- 避免重复的sys.path操作，确保导入一致性

### 2. SKU聚类与匹配

**核心约束**: 同一图片内的物体绝不会被聚类

```python
# 聚类后处理，确保跨图片聚类
for cluster_id, images_dict in cluster_to_images.items():
    for img_idx, point_indices in images_dict.items():
        if len(point_indices) > 1:  # 同一图片多个物体
            # 只保留第一个物体，其余标记为噪声点
            for point_idx in point_indices[1:]:
                new_cluster_labels[point_idx] = -1
```

**统一数据加载**：
- 所有检测文件加载使用 `utils.data_utils.load_detections` 作为唯一标准源
- 支持 `return_index_map=True` 获取 `[(文件编号, 检测数据)]` 格式
- 自动处理 floor_display1 和 floor_display2 两种JSON格式
- 自动过滤空objects，确保数据质量

### 3. 性能优化

**智能设备选择优先级**：
1. **CUDA**: NVIDIA GPU加速（优先）
2. **MPS**: Apple Silicon GPU加速
3. **CPU**: 通用处理器回退

**FAISS加速**：
- GPU版本：`faiss-gpu` (Python 3.8-3.10) + `cupy-cuda12x`
- CPU版本：`faiss-cpu` (Python 3.8-3.12，推荐)
- 自动降级：GPU不可用时自动切换CPU
- KNN搜索性能提升：3-10倍

**3D Viewer缓存系统**：
- **智能缓存管理**：基于文件mtime/md5的自动失效检测，无需手动清理
- **4阶段进度条**：点云构建→颜色映射→索引建立→KDTree构建，实时显示进度
- **GPU加速下采样**：使用CuPy进行点云下采样，FAISS-GPU加速最近邻搜索
- **持久化缓存**：支持 `--cache-dir` 指定缓存目录，跨会话复用
- **性能基准**（1M点云）：
  - CPU模式：缓存构建约120s，KDTree构建约2.5s
  - GPU模式：缓存构建约30s，KDTree构建约0.3s（使用FAISS-GPU）

**Viewer交互优化**：
- **事件注册封装**：统一的try/except和按钮过滤逻辑，减少80行重复代码
- **参数自动推导**：从 `--save_root` 和 `--dataset` 自动推导所有路径
- **错误容错**：防御式编程，兼容多种viser API版本

### 4. 代码质量保障

**消除重复逻辑**：
- 统一检测文件扫描：3处重复 → 1个标准接口（减少57行）
- 封装viewer事件注册：4处重复try/except → 1个注册方法（减少80行）
- 优化索引枚举：手写扫描 → 复用load_detections（减少6行）

**错误处理**：
- 分层异常处理：FileNotFoundError、ValueError、ImportError
- 防御式编程：try/except包装外部API调用
- 清晰的日志级别：INFO（关键事件）、DEBUG（详细信息）

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
- 识别15个SKU类型 (跨图片聚类)
- 412个独立SKU正确标记
- 无同图片内物体错误聚类
- 平均聚类距离0.6-1.6米

## 故障排除

### 常见问题

1. **VGGT导入错误**:
   ```bash
   pip install -r vggt-main/requirements.txt
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