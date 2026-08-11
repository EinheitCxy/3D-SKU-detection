# SKU匹配与顺序去重（模块化版本）

重构后的 SKU 匹配与顺序去重系统，提供统一 CLI 入口、鲁棒的匹配日志解析、序列去重与全局 ID 聚合（global_mapping）。

## 📁 项目结构（已对目录调整后）

```
code/
├── main.py                        # 统一 CLI 入口（interactive/pipeline/concise/analyzer/dedup）
├── modules/                       # 可执行/流水线脚本集合
│   ├── inference.py               # 匹配引擎（点追踪/3D-2D/两者）
│   ├── draw_detection_boxes.py    # 检出框可视化
│   ├── improved_sku_analyzer.py   # 一对多/多对一过滤（最佳一对一）
│   ├── deduplicate_detections.py  # 顺序去重 + 全局ID聚合（robust 解析）
│   ├── analyze_accuracy_metrics.py# 批量准确性指标分析
│   ├── reconstructor_base.py      # 3D重建抽象基类 + 后端注册表
│   ├── pi3_3d_reconstructor.py    # Pi3 后端（缓存式，快速批量）
│   ├── da3_3d_reconstructor.py    # Depth-Anything-3 后端（高精度多视角）
│   ├── da3_runner.py              # DA3 推理脚本（subprocess，在 Depth-Anything-3/.venv 中运行）
│   └── vggt_3d_reconstructor.py   # VGGT 后端（实时重建，当前已注释）
├── utils/                         # 可复用库模块
│   ├── config.py, data_utils.py, transforms.py, point_utils.py
│   ├── geometry_3d.py, matching_algorithms.py, visualization.py
│   ├── sku_matching_system.py, bbox_utils.py
│   └── process_image_orientation.py
├── scripts/batch.sh                # 批量匹配脚本（参考索引 0..N）
├── batch_accuracy_evaluation.sh   # 批量评估脚本
├── output_viz/, output_logs/, output_dedup/    # 运行产物（默认目录）
├── pyproject.toml, uv.lock
└── README.md
```

## 🚀 使用方法

### CLI 模式
```bash
# 交互模式
uv run main.py --mode interactive

# 完整流水线（校验→可视化→匹配→分析→顺序去重→评估）
uv run main.py --mode pipeline --dataset imdata/floor_display2 --save_root ./Output

# 使用 DA3 后端重建 + 匹配
uv run main.py --mode pipeline --dataset imdata/floor_display2 \
  --recon_backend da3 --match_backend da3 --algorithm 3d --save_root ./Output

# 并行处理参考图（pi3/da3 后端推荐，4 线程）
uv run main.py --mode concise --dataset imdata/floor_display2 \
  --match_backend pi3 --algorithm 3d --parallel_refs 4 --save_root ./Output

# 仅匹配（默认：对每张图片都作为参考图跑一遍）
uv run main.py --mode concise --dataset imdata/floor_display2 --algorithm both --save_root ./Output

# 仅跑单个参考图（直接调用引擎）
uv run modules/inference.py --algorithm point_tracking \
  --image_folder imdata/floor_display2/images \
  --detection_dir imdata/floor_display2/detections_results \
  --output_dir imdata/floor_display2 \
  --reference_idx 0 --max_images 50 --device cuda --save_json

# 仅分析（一对一过滤报告）
uv run main.py --mode analyzer --dataset imdata/floor_display2 --save_root ./Output

# 仅顺序去重（默认同名输出为 1.json..X.json）
uv run main.py --mode dedup --dataset imdata/floor_display2 --save_root ./Output

# 地堆 bbox 面积（读取已有检测和去重结果；每个global_id只累计一次）
uv run python main.py --mode ground-stack-area \
  --dataset ../imdata/my_stack --save_root ../Output \
  --area-anchor-frame 0 --area-anchor-object 3 \
  --area-anchor-width-cm 32.0 --area-anchor-height-cm 24.0

# 批量匹配（参考索引 0..N，等价于 main.py concise）
bash scripts/batch.sh floor_display2 4
```

### 重要参数
```bash
# 匹配参数透传（由 modules/inference.py 接收）
uv run main.py \
  --mode concise \
  --dataset imdata/floor_display2 \
  --algorithm both \
  --reference_idx 0 \
  --max_images 20 \
  --device cuda \
  --save_json \
  --save_root ./Output
```

### 查看帮助
```bash
uv run main.py --help
```

## 📦 模块说明（utils/）

| 模块 | 功能 | 依赖 |
|------|------|------|
| `config.py` | 配置参数管理 | 无 |
| `data_utils.py` | 数据加载和处理 | 基础Python |
| `transforms.py` | VGGT坐标变换 | PIL, numpy |
| `point_utils.py` | 点采样工具 | numpy, torch |
| `geometry_3d.py` | 3D几何处理 | torch |
| `matching_algorithms.py` | 匹配算法核心（point_tracking 需 VGGT；3d 投影读 pi3/da3 缓存） | torch |
| `visualization.py` | 结果可视化 | opencv, numpy |
| `sku_matching_system.py` | 系统封装（pi3/da3/vggt 后端） | torch |
| `bbox_utils.py` | 检出框工具 | 基础Python |
| `process_image_orientation.py` | 图像方向修复 | PIL |

附：可执行脚本位于 `modules/` 目录，例如 `modules/inference.py`、`modules/draw_detection_boxes.py`、`modules/deduplicate_detections.py` 等。

## 🔧 依赖管理

系统采用**渐进式依赖**设计：

- **基础功能**：配置管理、坐标变换、数据处理等功能随时可用
- **完整功能**：3D 匹配需 Pi3/DA3 重建缓存（`pi3_cache/` / `da3_cache/`，无需 VGGT）；仅 `point_tracking` 算法需要 VGGT 环境

检查依赖状态：
```python
from utils import check_dependencies
deps = check_dependencies()
print(deps)  # {'vggt_modules': False, 'visualization': True}
```

## 🎯 支持的算法

### 1. 传统点追踪匹配
- 基于VGGT点追踪的物体匹配
- 输出目录：`<dataset>/output_pt/<ref_idx>/`

### 2. 3D-2D投影匹配  
- 基于3D几何重建的投影匹配
- 包含3D几何验证
- 输出目录：`<dataset>/output_3dmapping/<ref_idx>/`

## 📊 输出结果（关键）

- 匹配可视化与摘要：生成在 `dataset/output_pt/<ref_idx>/`（由匹配引擎写入）
- 可视化导出：`output_viz/<dataset_name>/`（或 `--save_root/output_viz/<dataset_name>/`）
- 改进分析报告：`output_reports/<dataset_name>/report_*.txt`（或 `--save_root/output_reports/<dataset_name>/`）
- 顺序去重（同名输出）：`<save_root>/<dataset_name>/1.json..X.json`
- 全局ID聚合：`<save_root>/<dataset_name>/global_mapping.json`
- 地堆 bbox 面积：`<save_root>/<dataset_name>/ground_stack_area/{measurement_report.json,selected_instances.json,annotated_frames/}`

### 地堆 bbox 面积的定义与限制

`--mode ground-stack-area` 读取既有 `detections_results/` 和 `dedup_detections/global_mapping.json`，使用一个已知正面宽高的检测框作为标定锚点，只将锚点帧内每个物理 `global_id` 的有效且完整位于源图像内的 bbox 换算为 `cm²`，再得到总 `m²`。anchor 的 frame、object_id 和 bbox 必须和 mapping 中的记录一致；其他帧的观测不会参与面积选择。缺少锚点帧观测、越界/截断或索引非整数的 global ID 会被拒绝，以避免相机运动造成的跨帧尺度误差与边界框偏差。多个 `skus` 组使用同一稳定对象索引参与匹配、去重与计量。必须显式提供锚点帧号、对象索引、宽度和高度；没有任何隐式锚点默认值。

结果是**bbox 物理等效面积的算术和**，不是 bbox 并集、SAM3 mask 面积、包装表面积或地面占地面积。它假设被测包装正面与锚点近似共面，不估计未检测/被遮挡商品，并且不会改写检测 JSON 或 `global_mapping.json`。`measurement_report.json` 会记录状态、标定和拒绝原因；`selected_instances.json` 与标注帧用于审查每一项贡献。

## 🧾 日志输出（统一）

- 每次运行生成一个日志文件：`<save_root>/run_YYYYMMDD_HHMMSS.log`
- 控制台输出 INFO 级简报；文件记录 DEBUG 级诊断（更详细）
- 每个高阶步骤都有成对日志：
  - `START visualization|matching|improved_analysis|evaluation|dedup_sequence|reconstruct`
  - `END <stage> duration=Ns result=ok|fail [output=…|count=…]`
- 匹配阶段（modules/inference.py）额外输出：
  - 开始：`stage=match algo=point_tracking|3d ref=R images=N detections=M`
  - 结束：`matched_total=T saved_json=True|False output_dir=… duration=…s`
- 调试细节（仅在日志文件中）：
  - 检测加载汇总：`load_detections summary: skipped_non_numeric=K empty_objects=L`
  - BBox 过滤：`bbox_filter image_idx=… total=… below_det_conf=… below_min_area=… kept=… max_bboxes=…`
  - 每个 target 聚合：`target=t matched_objs=X skipped_low_overlap=Y top_hit=Z(hit=0.82)`
  
示例：
  - `tail -f ./Output/run_20250101_101530.log`

## 🔄 顺序去重与 Global ID

- 解析器采用“Found N matches in image T”锚点，回溯分配最近 N 条 `Matched`，适配“匹配在前、分组在后”的日志格式
- 去重规则（双向）：
  - 目标视角：删除第 i 张中被任一前序参考图命中的 target_id
  - 参考视角：删除第 i 张作为参考图时命中“更早目标图”的 ref_id
- global_mapping：为所有检出框（包括被去重的）分配全局唯一ID（并查集连通），条目标注 `removed: true/false`

## ⚡ 性能优化

- 模块化设计减少了内存占用
- 延迟导入避免不必要的依赖检查
- 保持了原有的所有性能优化

## 🧪 测试/验证

```bash
# 基础导入
uv run -c "from utils import SKUMatchingConfig; print('ok')"

# 快速烟测
uv run modules/inference.py --algorithm both --max_images 2
```

## ⚙️ 使用 YAML 配置（可选）

- 在 `code/config.yaml` 中集中管理参数（示例结构）：
```
main:
  dataset: imdata/floor_display2
  mode: concise
  algorithm: both
  save_root: ./Output
  device: cuda
  reference_idx: 0
  max_images: 20
  save_json: true

matching:
  algorithm: point_tracking  # 或 3d / 3d_projection
  max_points_per_bbox: 50
  confidence_threshold: 0.5
  min_confident_points: 7
  correspondence_threshold: 0.5
  output_dir: ./Output/floor_display2

reconstruction:
  conf_thres: 50.0
  output: reconstruction.glb
  model_path: /path/to/model.pt  # 无网环境建议指定
```

- 读取与使用（示例）：
```python
from utils import load_yaml_config, build_matching_config_from_yaml, extract_main_settings

cfg = load_yaml_config('code/config.yaml')
main_args = extract_main_settings(cfg)
matching_cfg = build_matching_config_from_yaml('code/config.yaml', algorithm=main_args.get('algorithm'))
print(main_args)
print(matching_cfg)
```

- 主入口自动加载（CLI 参数优先覆盖）：
  - 如果存在 `code/config.yaml`，`code/main.py` 会自动使用其中的 `main` 和 `reconstruction` 参数作为默认值；命令行显式传入的参数会覆盖 YAML 值。
  - 也可通过 `--config /path/to/config.yaml` 指定其它配置文件。

## 🔧 3D 重建后端对比

| 后端 | 速度 | 精度 | 缓存 | 适用场景 |
|------|------|------|------|----------|
| `vggt` | 慢（每次推理）| 高 | 无 | 单次调试 |
| `pi3` | 快（读缓存）| 高 | `pi3_cache/` | 批量生产（推荐）|
| `da3` | 中（读缓存）| 更高 | `da3_cache/` | 高精度场景 |

新增后端只需：① 继承 `ReconstructorBase` ② 用 `@register_reconstructor("name")` 装饰 ③ 在 `modules/__init__.py` 导入即可，无需修改 `main.py`。

### DA3 后端的 subprocess 隔离

Depth-Anything-3 依赖 `omegaconf/addict/e3nn/evo` 等 `code/` 未安装的包（code/ 与 DA3 均为 numpy<2，无 numpy 冲突）。因此 `da3_3d_reconstructor.py` **不 in-process 加载 DA3**，而是通过 **subprocess** 调用 `Depth-Anything-3/.venv/bin/python modules/da3_runner.py`（自包含脚本，不 import `code/`）：

- `da3_runner.py`：在 DA3 venv 中加载 `depth-anything/DA3NESTED-GIANT-LARGE`（6.3GB，米制，CC BY-NC 4.0），多视图批量推理，输出 `da3_cache/predictions.npz`（depth/extrinsics(w2c)/intrinsics，并反投影出 `world_points`，schema 与 `pi3_cache` 完全一致）。
- `da3_3d_reconstructor.py`：`load_model` 为 no-op（仅校验 venv/runner 存在），`run_inference` 调 subprocess 生成 npz 后读回，`export_glb` 跳过（SKU matching 仅需 npz）。
- 前置条件：`Depth-Anything-3/.venv` 必须存在且已装 DA3 依赖；HF 权重首次运行自动下载（6.3GB）。

## 📝 开发说明

- `main.py` - 命令行入口，保持在根目录
- `modules/` - 可执行/流水线脚本集合（调用 `utils/` 提供的库能力）
- `utils/` - 可复用功能模块（算法/工具/可视化等）
- 向后兼容原有API，支持渐进式功能启用
