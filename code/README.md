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
│   ├── da3_footprint_stage.py     # DA3 地堆 footprint 面积阶段（支撑平面 OBB 并集）
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

# 若在 Git worktree 中运行，复用已有 DA3 环境（不创建新的 .venv）
DA3_VENV_PYTHON=/home/xingyu/3D_Recognization/Depth-Anything-3/.venv/bin/python \
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

# DA3 地堆 footprint 面积（要求 DA3 cache、global_mapping.json 与本地 SAM3 checkpoint）
uv run python main.py --mode ground-stack-area \
  --dataset imdata/my_stack --save_root ./Output

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
- 地堆 footprint 面积：先解析 `<save_root>/<dataset_name>/ground_stack_footprint/CURRENT`，再读取其指向的不可变 `runs/<run_id>/{measurement_report.json,footprints.geojson,top_down_footprint.png,manifest.json}`

### 地堆 footprint 面积的定义与限制

`--mode ground-stack-area` 是锚点无关的只读计量阶段：它读取 schema-v2 `da3_cache/predictions.npz` 的 metric `world_points`、`world_points_conf`、逐帧原图→处理网格 affine 与缓存原图尺寸，从 `dedup_detections/global_mapping.json` 的全部观测中为每个物理 `global_id` 聚合其有效 3D 点，并用本地 SAM3 checkpoint 生成每个检测框的 mask（mask 决定对象点与背景排除）。背景以 12 mm RANSAC 产生候选平面，再最多三次 SVD 精修、每轮剔除残差超过 10 mm 的点；精修后仍须保留至少 10,000 点及原始背景的 10%，且 P95 残差不超过 10 mm。对每一 global ID，沿拟合支撑平面法向投影全部有效观测的 3D 点并恢复 OBB；最终取所有 carton OBB 投影的**多边形并集面积**（`m²`），指标 `da3_ground_footprint_union`。若任一 global ID 几何不完整（缺 mask、有效点不足、OBB 退化等），整体拒绝并输出 `status: rejected` 与 `value_m2: null`，不会以部分结果冒充总面积。旧 runner cache 不满足该 schema/provenance 合约，必须先用 DA3 reconstruction 重新生成 cache。DA3 尺度是模型估计，现场 reference 可用于 QA，而非必需输入。

结果是**每个 carton OBB 投影到支撑平面的多边形并集**，不是 bbox 面积算术和、SAM3 mask 面积、包装表面积、正面/接触面积或地面接触面积，并且不估计未检测/被遮挡商品。要求现有 DA3 cache、global mapping 与本地 SAM3 checkpoint，缺少任一即拒绝；不会改写检测 JSON 或 `global_mapping.json`。每次运行把 report/GeoJSON/PNG/manifest 写入并 fsync 到不可变 `ground_stack_footprint/runs/<run_id>/`，完整后才原子更新 `CURRENT`；读取方须从 `CURRENT` 解析路径，发布失败时仍只会看到旧的完整 generation 或没有 generation，绝不会看到 mixed artifacts。report 仅保存 generation-relative 产物名；既有 generation 不会被覆写。`measurement_report.json` 记录状态、尺度来源、支撑平面门与拒绝原因；每个候选的 `ransac.trial_count` 与 `ransac.early_exit` 是性能审计字段，不放宽任何门，也不改变 m² 定义；`footprints.geojson` 与 `top_down_footprint.png` 用于审查每项贡献。

公开命令在完成全部 DA3/image/detection/mapping 验证后使用 `<save_root>/<dataset_name>/sam3_mask_cache/v1/` 的逐源帧不可变 bundle。key 覆盖 image ID、原图 bytes/size、完整有序 object ID + 精确 binary64 bbox prompts、真实 checkpoint SHA-256、stage/cache/SAM3 code fingerprint、Python/NumPy/PyTorch/SAM3/CUDA/cuDNN/device/precision runtime fingerprint、完整 `predict_inst` contract 与 source-mask shape/dtype；任一项变化都会失效并重算。report 的 `sam3_mask_cache.frames[]` 记录逐帧 key、payload/checkpoint digest、code provenance 和有序 events：`miss,written`、`hit`、损坏隔离到 `corrupt/` 后的 `invalid,written`，以及非致命写失败的 `miss,cache_write_failed`。SAM3 helper 在模型加载前后验证 checkpoint bytes，in-process model cache 以 digest、canonical concrete device 和固定 inference-contract fingerprint 为键。缓存损坏只会隔离并重算，cache reuse 绝不允许 bbox fallback；空/无效 mask 仍使正式总量 `rejected`/`null`，不会发布 partial total。

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
