# SKU匹配与顺序去重（模块化版本）

重构后的 SKU 匹配与顺序去重系统，提供统一 CLI 入口、鲁棒的匹配日志解析、序列去重与全局 ID 聚合（global_mapping）。

## 📁 项目结构（已对目录调整后）

```
code/
├── main.py                        # 统一 CLI 入口（含 viewer-web 静态 bundle 导出）
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

# 导出 TypeScript/Three.js 静态 viewer bundle（只读已有正式产物）
uv run python main.py --mode viewer-web \
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

## TypeScript/Three.js 静态 viewer

Python 是唯一的数据与证据生产端，负责 DA3/SAM3、匹配、去重、正式 ground footprint 面积和 provenance；TypeScript 只负责严格加载 bundle、渲染点云/正式 footprint 与交互式审查，不重新计算面积。导出时在 web-export 阶段自动进行**场景点云过滤**（`utils/pointcloud_filter.py`：有限值/零点掩码 -> SOR 统计离群点移除 -> DBSCAN 保留主要簇 -> 带双侧护栏的 RANSAC 地面剔除，`export_web_viewer_bundle(filter_config=...)` 可覆盖），随后再做 voxel 去重。实例标签/染色的范围与保护范围同源：导出时读取 `<save_root>/<dataset>/sam3_mask_cache/v1/`（与 `ground-stack-area` 共用同一不可变 cache），按 DA3 affine 把处理网格点映射回源图像素、做最近邻掩码采样，mask 未覆盖即不标签；入口缺失/schema 不符/bbox 与 global_mapping 不一致（canonical binary64 逐位相等）均 fail closed。**仅 `labels >= 0` 的有效 SAM3 mask 点硬保护**：它们必须仍通过有限/非零基础有效性，但绕过 SOR、DBSCAN、ground-plane 与 sky-line；bbox 或 mask 外点不受保护，继续作为背景过滤。voxel 内先按 `(voxel, instance label)` 保留每个保护实例的最高 confidence 代表，并淘汰同 voxel 的未标注竞争点；只有无保护实例的 voxel 才保留其最高 confidence 背景代表。`max_points` 会先完整保留全部保护代表，若保护代表本身超限则 fail closed，绝不静默删除 SKU。导出还会计算**朝向校正矩阵** `world_to_view = T_center @ R_level @ M_flip`（行主序写入 manifest）：`M_flip = diag(1,-1,-1,1)` 做 CV->glTF 翻转；`R_level` 在**过滤前**有效点集（过滤会剔除地面，故不能用过滤后点集）上以 RANSAC 迭代拟合地平面（地堆/货架场景最大平面常是竖直墙/货架面，故最多剔除 8 个已拟合平面再试），候选平面须同时满足内点比 >= 0.05、法向与 -Y 夹角 <= 60°、**地板性门**（< 15% 点位于平面下方 0.1 m 之外--地面从下方支撑场景，斜穿主体的平面必然有大比例点在其下方）；法向定向到相机一侧后取到 +Y 的最短弧旋转，无合格平面时不摆平仅翻转；`T_center` 把翻转+摆平后过滤点云的逐轴 median 移到原点。摆平至关重要：DA3 world 锚定首帧相机姿态，俯拍 ~20° 的地堆场景里地面在世界系倾斜 ~20°，不摆平会把真实周边货架/商品点渲染到"空中"（fd6 实测 58k 点因此显示为天空噪点）。在过滤之后还做**天空线裁剪**：摆平坐标系中以实例标签点高度的 p99.9 + 0.15 m 为界，裁掉界以上的未保护点（天花板/天空薄片与主簇稠密相连、SOR/DBSCAN 剔不干净），并有 30% 最大裁剪比例护栏。同时为 `objects.json` 每个实例写入 `point_index_range: [start, end)`（体素采样后按实例稳定排序，使各实例点集连续），供前端选中 3D 包围框使用。导出还会为**每个实例（含 removed）**生成缩略图：从 `<dataset>/images/<image_id>.JPG` 按 bbox（四周加 10% padding 并 clamp 到图内）裁剪、长边缩到 <=256px、JPEG q85，写入 `runs/<run_id>/thumbs/<globalId>_<instanceIndex>.jpg`，并把相对路径写进 `objects.json` 每个实例的 `thumbnail` 字段（前端必填契约，缺失即 fail closed 提示重新导出）。源图目录按 `da3_runner.py` 同一约定解析（文件名 stem 数字 = image_id），且逐文件 SHA-256 必须与 DA3 cache 的 `source_image_sha256` 一致、bbox 必须落在源图尺寸内，否则 fail closed；字节级一致保证缩略图坐标系（raw 未转置像素空间，与 bbox/affine 相同）与 DA3 推理时读到的图像一致（EXIF 方向已验证：fd6 全部 11 张 EXIF orientation=1，裁剪区域平均色与 cache 低清图 affine 映射区域逐通道差 <=0.9/255）。默认密度参数为 `--viewer-web-voxel-size 0.005` / `--viewer-web-max-points 1500000`。导出与开发分两步：

```bash
# 第一步：导出到默认 viewer-web/public/data/，也可传 --viewer-web-output
uv run python main.py --mode viewer-web \
  --dataset imdata/my_stack --save_root ./Output

# 第二步：在 viewer-web/ 中启动已有 Vite 脚本
cd ../viewer-web
npm run dev
```

每个新 bundle 的 `manifest.source.export` 还冻结了实际 `filter_config`、导出器源码 SHA-256、读取时两次校验的 `global_mapping.json` SHA-256，以及每个实际使用的 SAM3 cache entry 的 `image_id`、content-addressed `key` 与 `masks.npz` SHA-256。导出会严格验证该 entry 的 canonical key payload、payload digest、逐 mask digest/true-pixel count 和 canonical bbox clipping；历史同图像 entry 可并存，但只有唯一精确覆盖当前 mapping 的 entry 能被使用，歧义或任一篡改均 fail closed。前端 contract 对这些字段使用 exact-key 校验，旧 bundle 必须重新导出。

`viewer-web` 默认从 `<save_root>/<dataset_name>/da3_cache/predictions.npz`、`dedup_detections/global_mapping.json`、`ground_stack_footprint/` 读取正式产物，并额外要求 `--dataset` 指向的数据集存在 `<dataset>/images/` 源图目录（用于实例缩略图）。bundle 使用不可变 `CURRENT -> runs/<run_id>/` 布局；exporter 对最终 `positions` 精确计算每轴 p01/p99，写入 manifest 必填 `display_bounds: [min_x,min_y,min_z,max_x,max_y,max_z]`（bundle/source 坐标、六个有限数且逐轴 min<=max）。前端严格校验 schema、provenance、数组 byte length、`display_bounds`、`world_to_view`（必需 16 个有限 float）与 `normals` int8 数组；缺失任一字段均提示用最新导出器重新导出，且不做前端全量点排序 fallback。
exporter 在任何临时 generation 创建前还验证从 mapping 得到的 object index：global ID 是 canonical decimal string（前端以 `BigInt` 排序）；只有 image/object ID 必须是 JavaScript safe integer（排除 bool，`abs(id) <= 2**53-1`）。此外 removed 必须为 bool、bbox 必须为四个 finite 且有序数，derived images/objects 和 active/removed/total counts 必须与 instances 完全一致；否则 fail closed，绝不发布浏览器会拒绝的 bundle。RANSAC 若返回零/非有限法向则记录并跳过该退化候选，绝不执行归一化除零。

Vite 本地 `/data/` 只直接对应默认的 `viewer-web/public/data/`。默认 output 导出成功后，CLI 只打印、不执行绝对路径命令 `npm --prefix <repo>/viewer-web run dev`，实际输出使用当前 checkout 的绝对路径；使用 `--viewer-web-output <custom-output>` 时不会打印该默认 npm 命令，custom output 必须在前端启动前部署，或挂载/serve 到浏览器 URL `/data/`。

正式 report 绑定生成时读取的 raw `global_mapping.json` 字节快照 SHA-256（`global_mapping_sha256`）。exporter 会在构建 object index 前后校验该 digest；mapping 不同或在导出期间变化会 fail closed，且 `accepted` generation 的 object ID 集与 footprint geometry ID 集不一致也会拒绝发布。没有 `global_mapping_sha256` 的历史 formal generation 必须先重新运行 `--mode ground-stack-area`，再执行 viewer export；不提供 fallback。

`accepted` 的数值只表示正式 `da3_ground_footprint_union`；`rejected` 或 `value_m2: null` 表示 unavailable，界面显示 `—`，绝不显示为 `0 m²`。实验性 front-facing area 不接入 v1，青色保留给未来该指标，正式 ground footprint 使用琥珀色。

> **注意（2026-08 缓存修复）**：旧版 `save_predictions_cache` 把 `source_model` 写成 object dtype，严格加载（`allow_pickle=False`）会拒绝。此类历史 `da3_cache/predictions.npz` 必须重新运行重建生成后才能导出 viewer-web bundle；新写入器已统一为 unicode 标量。bundle 数组为 `positions.f32.bin` / `colors.u8.bin` / `normals.i8.bin`（不再携带前端未消费的 `confidences` / `frame_ids`；`normals.i8.bin` 只在 voxel/protected/max-points 完成后的最终代表点集合上估计，再随实例排序与 colors 同步置换，供前端 half-Lambert 光照，缺失时前端 fail-closed 提示重新导出）。
>
> **DA3 缓存 schema v3（米制硬门）**：`da3_runner.py` 推理后强制校验 `prediction.is_metric == 1`，非 metric 模型（如 DA3-GIANT-1.1，输出相对尺度点云）直接 fail-closed 不写缓存；`is_metric` 与 `scale_factor` 写入 npz，ground footprint 阶段与 viewer-web 导出均要求 schema 恰好为 3 且 `is_metric == 1`。所有 v2 及更早缓存需重新运行 `--mode reconstruct` 生成。默认 checkpoint 已切换为修复训练问题的 `-1.1`（`depth-anything/DA3NESTED-GIANT-LARGE-1.1`，2026-08-20 起；全量 fd2–12 A/B：micro F1 78.62%→80.51%，详见 `.research`/memory）。`--recon_model_path` 仍可覆盖任意兼容 checkpoint；旧默认 `DA3NESTED-GIANT-LARGE` 的缓存与 1.1 缓存不要混用（provenance 的 `source_model` 字段区分）。

前端开发验证：

```bash
cd ../viewer-web
npm test -- --run
npm run build
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
| `pointcloud_filter.py` | 场景点云过滤（SOR/主簇/地面剔除，web viewer 导出使用） | numpy, open3d |
| `da3_cache_validation.py` | DA3 缓存标量/仿射契约校验（web 导出与 footprint 阶段共用） | numpy |

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

结果是**每个 carton OBB 投影到支撑平面的多边形并集**，不是 bbox 面积算术和、SAM3 mask 面积、包装表面积、正面/接触面积或地面接触面积，并且不估计未检测/被遮挡商品。要求现有 DA3 cache、global mapping 与本地 SAM3 checkpoint，缺少任一即拒绝；不会改写检测 JSON 或 `global_mapping.json`。每次运行把 report/GeoJSON/PNG/manifest 写入并 fsync 到不可变 `ground_stack_footprint/runs/<run_id>/`；generation 创建可并行，但更新 `CURRENT` 与 publisher 按自身 expected run ID 解析路径会在固定 `locks/publication.lock` 上持有 exclusive `flock`，reader 持 shared lock。expected run 不符报 `OSError`，因此 publisher 不会返回竞争者 generation，reader 只会看到一个完整旧或新 generation。replace 前的 write/fsync/rename/temp/replace 失败会报错并保留旧的完整 generation 或不创建 CURRENT；成功 replace 是逻辑发布点，之后 output-root directory fsync 失败只记录 durability warning，新 generation 仍成功返回且不回滚。report 仅保存 generation-relative 产物名；既有 generation 不会被覆写。`measurement_report.json` 记录状态、尺度来源、支撑平面门与拒绝原因；每个候选的 `ransac.trial_count` 与 `ransac.early_exit` 是性能审计字段。`performance.stages_seconds` 用 `time.monotonic` 记录 validation/I/O、SAM3 source masks、plane selection、per-ID OBB/union、shadow evidence 和 artifact creation，未进入项为 `null`；`total_seconds_pre_publication` 从 stage 入口计至 GeoJSON/PNG 已创建且正式 report 唯一一次序列化之前，不含 report/manifest fsync、generation rename、publication lock 等待或 CURRENT replace。所有 timing 都是 additive diagnostics，不改变 accepted/rejected 或 m² 定义；`footprints.geojson` 与 `top_down_footprint.png` 用于审查每项贡献。

公开命令在完成全部 DA3/image/detection/mapping 验证后使用 `<save_root>/<dataset_name>/sam3_mask_cache/v1/` 的逐源帧不可变 bundle。key 覆盖 image ID、原图 bytes/size、完整有序 object ID + 精确 binary64 bbox prompts、真实 checkpoint SHA-256、stage/cache/SAM3 code fingerprint、Python/NumPy/PyTorch/SAM3/CUDA/cuDNN/device/precision runtime fingerprint、完整 `predict_inst` contract 与 source-mask shape/dtype；任一项变化都会失效并重算。producer 与 hit validation 直接调用 SAM3 正式 `clip_mask_to_bbox`，包括 fractional、边界和四侧完全越界 prompt 的 canonical 边界像素；semantic schema 已升级，旧 clip 语义 entry 会 miss，不存在 cache 私有 clip 或 bbox fallback。stage 在一次运行入口和全部 frame cache access 后各做一次 checkpoint checksum；真实 SAM3 miss 仍保留 producer 与 model-load 的 before/after checks。report 的 `sam3_mask_cache.frames[]` 记录逐帧 key、payload/checkpoint digest、code provenance 和唯一有序字段 `cache_events`（无 `events` compatibility alias）：`miss,written`、`hit`、损坏隔离到 `corrupt/` 后的 `invalid,written`，以及非致命写失败的 `miss,cache_write_failed`。缓存损坏只会隔离并重算，cache reuse 绝不允许 bbox fallback；空/无效 mask 仍使正式总量 `rejected`/`null`，不会发布 partial total。

若 cache `entries/` 目录初始化或临时 bundle 目录创建发生 `OSError`，已验证且已 canonical clipping 的 fresh boolean masks 仍供该次 stage 使用：每帧记录 `miss,cache_write_failed`（重建损坏 bundle 时为 `invalid,cache_write_failed`），`payload_sha256` 为 `null`，既有 final entry 不会被删除。

`ground-stack-area` 会在正式 status/value/plane/per-ID diagnostics/polygons/union/rejection reason 完全冻结后，调用内部 `utils/footprint_evidence.py` 附加 `evidence.mode=shadow`。每个 observation 保存真实 source mask、正式 affine 与 processed mask，分别报告 source/processed pixel count。camera 分支单独验证 optional DA3 camera tensors，并计算 source-world reconstruction residual、最多 512 个确定性采样点的双向 occlusion-aware reprojection，以及固定正式支撑平面的 per-ID leave-one-observation-out OBB diagnostics。独立 `mask_robustness` 不依赖 depth/K/E：先在 source grid 上做 3×3、1 iteration 的 erosion/original/dilation，再与正式 stage 调用同一个 nearest CV2 affine warp，随后严格复用冻结 plane、32 points/observation、64 elevated points/ID、5 mm voxel、component OBB、0.1 mm union 与 all-ID 规则。variant 失败只报告 `value_m2: null`、reason/rejection transition，不暴露 partial polygons；只有三种都 accepted 才输出经验性 `area_interval_m2`，从不替换正式 mask 或正式面积。双向 pair 只比较不同 source image：投影点比 target depth 远超过 20 mm 记为 occluded（遮挡中性），近超过 20 mm 记为 foreground conflict，中间才记为 visible-consistent；20 mm 只是 diagnostic tolerance，不是正式 threshold。formal plane 不存在时，`evidence.status` 与 `evidence.mask_robustness.status` 均为 `unavailable_no_formal_geometry`；只有一个 distinct source image 时标记 `single_observation_insufficient_cross_view_evidence`。camera fields 缺失可使 camera 顶层状态 unavailable，同时 mask robustness 仍实际运行；malformed/算法失败只产生 additive evidence 状态或 variant rejection，不会导致面积拒绝或改变已冻结 generation geometry。shadow evidence 只表示 internal consistency/sensitivity，不是 calibrated uncertainty、accuracy claim、confidence interval 或 error bound；把它提升为 hard gate/error bound 前，必须先在独立 calibration set 上完成校准与验证。

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

- `da3_runner.py`：在 DA3 venv 中加载 `depth-anything/DA3NESTED-GIANT-LARGE-1.1`（6.3GB，米制，CC BY-NC 4.0；`DEFAULT_HF_REPO` 常量，可被 `--recon_model_path` 覆盖），多视图批量推理，输出 `da3_cache/predictions.npz`（depth/extrinsics(w2c)/intrinsics，并反投影出 `world_points`，schema 与 `pi3_cache` 完全一致）。
- `da3_3d_reconstructor.py`：`load_model` 为 no-op（仅校验 venv/runner 存在），`run_inference` 调 subprocess 生成 npz 后读回，`export_glb` 跳过（SKU matching 仅需 npz）。
- 前置条件：`Depth-Anything-3/.venv` 必须存在且已装 DA3 依赖；HF 权重首次运行自动下载（6.3GB）。

## 📝 开发说明

- `main.py` - 命令行入口，保持在根目录
- `modules/` - 可执行/流水线脚本集合（调用 `utils/` 提供的库能力）
- `utils/` - 可复用功能模块（算法/工具/可视化等）
- 向后兼容原有API，支持渐进式功能启用
