# 3D SKU Recognition

面向货架与地堆 SKU 的 DA3 三维重建、跨图匹配、去重、ground-footprint 计量和静态 Web Viewer。Python 负责生成可审计产物；Three.js 只加载、校验和交互展示已经发布的 bundle。

详细的核心契约见 [docs/3d_core.md](docs/3d_core.md)，当前端到端测量结果见 [perf/runs/20260826T084815Z/FINAL_REPORT.md](perf/runs/20260826T084815Z/FINAL_REPORT.md)。

`perf/benchmark.py` 只采集 one-shot cold 数据：fd2–4 各自使用全新的隔离输出运行一次，
浏览器也只导航一次；不调度、不保存、不汇总 warm-start case。classification 与
reconstruction/matching 并行，并在 dedup 前 join；端到端统计使用真实 case wall time。

## 当前布局

```text
3D_Recognization/
├── main.py, config.yaml, pyproject.toml, uv.lock  # DA3 核心 CLI 与依赖
├── src/                                           # 流水线阶段与重建后端
├── utils/                                         # 几何、匹配、cache、过滤等共享库
├── tests/                                         # 核心回归测试
├── modules/
│   ├── sku_detector/                              # YOLO 检测器与独立 uv 项目
│   ├── personalcare_classifier/                   # source/ + canonical model.bin
│   ├── viewer_web/                                # TypeScript / Three.js 前端
│   └── video_to_dedup/                            # 视频到 DA3 去重入口
├── scripts/3d/{pipeline,evaluation,tuning,ops}/   # 端到端 pipeline 与维护工具
├── Output/                                        # 忽略的用户可见 pipeline 产物
├── runtime/                                       # 忽略的迁移环境与工具 cache
│   ├── 3d-core/.venv/
│   └── video_to_dedup/
├── perf/                                          # 可复现的性能采集与报告
└── frame_sampler/                                 # 保持为外部嵌套 Git 仓库
```

`Output/` 与 `runtime/` 都不是源码，也不应提交。迁移前的 `code/.venv` 保留在 `runtime/3d-core/`；日常核心开发环境从仓库根通过 `uv sync` 重建。

## 环境

所有 Python 命令从仓库根执行，并使用 `uv`：

```bash
uv sync --extra dev
```

验收并切换候选环境后，根 `.venv` 是 core、DA3、SAM3 和 BSON API 的唯一宿主环境：Python 3.11、NumPy
1.26.4、Torch 2.7.1、TorchVision 0.22.1 与 xFormers 0.0.31 由根 `uv.lock` 固定。
DA3/SAM3 仍保留为仓库内源码，DA3 subprocess 默认执行 `.venv/bin/python`；仅在诊断或
显式测试时才用 `DA3_VENV_PYTHON` 覆盖。SKU detector 保持自己的
`modules/sku_detector/pyproject.toml`；其 `runtime/sku_detector/.venv` 固定为 NumPy
1.26.4 与 OpenCV 4.11，视频工作流默认复用该环境。

需要重建候选统一环境时，先提供一个不存在的目标目录；脚本不会更改当前根 `.venv` 或
`Depth-Anything-3/.venv`。候选以 uv 的 relocatable venv 创建，因此验收后可安全重命名
为根 `.venv`；脚本随后执行锁文件同步、依赖检查与 DA3/SAM3 import smoke：

```bash
scripts/3d/ops/build_unified_env.sh /tmp/3d-recognition-unified-env
```

候选环境的完整测试、主环境切换和 GPU 等价性验证由维护流程在验收后执行；不要在构建
候选时覆盖任一现有环境。

若历史候选在未使用 relocatable 选项时已被重命名，`uv sync --frozen` 不会修复其旧的
console-script shebang。维护者应在停掉项目 Python 进程后重新创建根环境（不要在运行中的
环境上执行）：

```bash
uv venv --relocatable --clear .venv --python 3.11
uv sync --frozen --extra dev
```

统一宿主环境只支持本流水线的 image-only SAM3 推理，不安装 `decord`。官方 SAM3 将它
归入可选 notebook 依赖；本项目的视频入口先用 OpenCV 抽帧，再向 SAM3 传入图像目录。
直接让 SAM3 读取 `.mp4` 不属于该宿主环境的支持范围，并会明确要求该可选依赖。

## 常用命令

```bash
# DA3 重建、完整 batch-all-refs 3D matching 和去重；默认输出 Output/。
# matching 是唯一会运行 SAM3 self-exemplar 并发布 v2 processed-mask cache 的阶段。
CUDA_VISIBLE_DEVICES=2 uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d \
  --recon_backend da3 --match_backend da3 \
  --classifier-device cuda:0

# 复用外部已补全 classification 的检测结果，不启动本地分类器
uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d --no-classifier

# 导出静态 minimal schema 3.0.0 Web bundle（不运行 ground-stack-area）
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2

# 启动前端；默认 /data/ 映射 modules/viewer_web/public/data/
npm --prefix modules/viewer_web run dev

# 一条命令从视频生成去重、Viewer bundle，并启动 Web Viewer
bash scripts/3d/pipeline/video_to_viewer.sh \
  --video /path/to/video.mp4 --fps 2.0 --gpu 2 \
  --classifier-device cuda:0 --serve
```

`--save_root` 可以覆盖输出目录；相对值始终相对仓库根解析。默认 bundle 位于 `modules/viewer_web/public/data/`，自定义 bundle 必须在前端启动前挂载或 serve 到浏览器的 `/data/`。

完整视频入口的参数、阶段顺序和输出路径见 [scripts/3d/pipeline/README.md](scripts/3d/pipeline/README.md)。其中 `--gpu 2` 设置物理 GPU mask，进程内分类器继续使用 `--classifier-device cuda:0`；`--detections-dir` 可复用已有逐帧检测 JSON。脚本在 dedup 后直接导出 minimal schema 3.0.0 bundle，默认只导出 bundle，增加 `--serve` 才会以前台进程启动 Vite。独立 `ground-stack-area` 仍可按需运行，但不再是该 Viewer 入口的前置阶段。

上例的 `CUDA_VISIBLE_DEVICES=2` 把物理 GPU 2 映射为进程内的 `cuda:0`；`--classifier-device` 必须是一个显式可用的 CUDA device。分类器不接受 CPU 或替代模型 fallback。若不使用 GPU mask，可直接传入实际可见设备号，例如 `--classifier-device cuda:2`。

## Canonical SAM3 processed-mask workflow

matching 是唯一的 SAM3 producer。默认 `enable_sam3_mask_sampling: true` 时，它只使用 self-exemplar，并为每个 detection frame 一次性发布完整的 `sam3_mask_cache/v2` processed-space bool mask；payload 用 little-endian `np.packbits` 无损保存。`enable_sam3_mask_sampling: false` 仅保留既有 bbox sampling，且不会发布 cache，因此 footprint 与 viewer export 会 fail closed，提示先运行 matching。

运行顺序是硬约束：先完成 **完整 batch-all-refs matching**；`ground-stack-area` 是独立的后端计量阶段，`viewer-web` 可在 dedup 后直接发布产品 bundle。Viewer exporter 不导入、加载或推理 SAM3；matching 的 processed-mask cache 仍是点标签发布的输入，cache miss 会 fail closed。

`sam3_mask_cache/v1` 与 v2 不兼容：v1 从不被读取、迁移、复制或删除。v2 的 formal metric 是 `da3_self_exemplar_ground_footprint_union`，viewer bundle schema 是 `3.0.0`；它们与旧面积输出和 bundle `1.0.0`、`2.0.0` 不可数值比较，旧 bundle 必须按上述顺序重新生成。

DA3 bbox 的 source→processed 映射唯一权威是 `predictions.npz` 中每帧完整的 `source_to_processed_affine`（2×3）及其 processed grid。该 affine 可能同时编码 resize、pixel-center、patch rounding 与 crop offset；`x'=sx*x+(sx-1)/2`、`y'=sy*y+(sy-1)/2` 仅是没有额外 crop 的 simple-resize 例子，绝不能据此重算或替代 cache。matching 缺少显式 cache affine/shape 会 fail closed；旧 scale-only 或含 raw out-of-grid bbox 的 v2 entries 均不能命中，须完整重跑 matching → footprint → export。

## Minimal Viewer 与点云策略

Web bundle 使用不可变 `CURRENT -> runs/<run_id>/` 发布。`CURRENT` 只包含 `run_id`；run 内的 `manifest.json` 固定为 schema `3.0.0`，包含轻量 `backend: "DA3"`、真实 `dataset_name`、`frame_count`、六维 `display_bounds` 和 16 维 `world_to_view`，不携带 source model 或 provenance。固定二进制文件为 `positions.f32.bin`、`colors.u8.bin`、`normals.i8.bin`，`point_count` 由 positions 长度推导。导出器从 dataset `images/` 中按数字文件名解析原图，为每个 active 与 removed observation 按 bbox 写入 `thumbs/*.jpg`（最长边 256px）；`objects.json` 只包含每个 global ID 的 `ordered_skus`、`point_ranges` 和 observations 的 `image_id`、`object_id`、`removed`、`thumbnail`。

canonical “其他品类”是 `sku_id=56642`、`sku_name=其他品类`。只要存在任一具体 SKU，具体 SKU 按既有 confidence/support 顺序排在 56642 之前；只有全部有效观测都是其他品类时，56642 才能排在首位。Viewer 只消费已排序的 SKU ID/名称，不接收或显示 confidence。

产品界面的 Backend badge 直接显示 manifest 的 `backend`；对象与 SKU counts 由前端读取 `objects.json` 的 observations 派生，而非额外后端聚合字段。默认 `Select by SKU`，与 `Select by Global ID` 互斥，切换会清除上一选择。SKU 选择保留完整场景并批量 magenta 高亮；canvas pick 自动切换为 Global ID。`View Controls` 默认折叠，展开后只有 Fit、Top、Iso 和 Point size。右栏为 `Selected Object`，只显示 Global ID 与按发布顺序排列的 SKU。

Viewer bundle 不包含 footprint、evidence、hash/provenance、source digest、confidence 或其他审计型 rich-contract 元数据。它只恢复产品缩略图所需的 observation 标识和相对 JPEG 路径。点云过滤仍对所有点统一执行，选择和 Focus 通过 `point_ranges` 增量更新现有 geometry，不复制点云。

## Personalcare classification in viewer objects

pipeline 默认启用 `--classifier`：在 dataset validation 后立即异步提交独立的 personalcare 分类 subprocess；它可与 DA3 reconstruction 或 matching 并行。分类进程每个 dataset 仅加载一次原始 MobileNetV3 模型，按原始 object 顺序批量分类；原始 `detections_results/` 永不改写。实际是否存在 classifier/matching 的时间重叠取决于 cache、可视化与运行 receipt：本次 fd6 cache-reuse receipt 中分类在 matching 开始前已完成。matching 完成后 pipeline 才 join 分类结果；两者都成功才执行 dedup，因此 dedup 显式读取本次已发布的 enriched detections。

传入 `--no-classifier` 时，pipeline 不启动本地分类器，也不创建分类发布指针或兼容副本；它会在任何重建或 matching 前同步校验 `<dataset>/detections_results/` 中每个数字命名 JSON 的每个 object `classification`，然后将该输入目录显式传给 dedup。外部检测必须符合现有 personalcare classification schema。

分类产物在 `<save_root>/<dataset>/personalcare_classification/CURRENT -> runs/<time_ns>-<pid>/`：run 内有 `detections/<frame>.json` 和 `result.json`。有效 object 同时保留 raw `classes.cls`/`confidences.cls` 与规范化 `classification`；无效 bbox 保留原始数据并以 `status: unavailable`、`reason: invalid_bbox` 发布。此发布只用完整 run 的原子 `CURRENT` 指针，不产生分类 hash、signature、encryption、feature vector 或内容指纹。

`global_mapping.json` 的每个 observation（含 `removed: true`）都保存该 classification。导出阶段按 `(sku_id, sku_name)` 聚合并确定性排序；具体 SKU 优先于 canonical other `56642/其他品类`，所有 candidate 仍可在后端产物中审计。Minimal Viewer 的 `objects.json` 只发布排序后的 `ordered_skus` 和 `point_ranges`，Selected Object 只显示 Global ID 与 SKU 名称，不发布或渲染 confidence。厂商/品牌/品类为禁用的“主数据待接入”，POSM/价签/空缺位为禁用的“检测能力待接入”；V1 不根据名称推断这些字段。分类不会改变 SAM3 processed masks、point ranges、点云过滤或 formal metric。

分类器可单独运行（`--output-root` 是分类产物根，不会修改输入 detections）：

```bash
CUDA_VISIBLE_DEVICES=2 uv run --project modules/personalcare_classifier python \
  modules/personalcare_classifier/source/classify_dataset.py \
  --dataset imdata/floor_display6 \
  --output-root /tmp/personalcare-classifier-smoke \
  --device cuda:0
```

## Ground footprint

`ground-stack-area` 从 matching 发布的 processed-space self-exemplar masks 为每个去重 `global_id` 取多视图 metric 点云，投影到推断支撑平面后以 OBB polygon union 计算 `da3_self_exemplar_ground_footprint_union`（m²）。它不是包装表面积、正面面积、SAM3 mask 面积或 bbox 面积。任何对象缺少足够几何时整次结果为 `rejected` 与 `value_m2: null`，不会发布部分总量；本指标是新 baseline，不可与旧 `da3_ground_footprint_union` 面积直接比较。

## 验证

```bash
PYTHONPATH=. VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv \
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --active --no-project python -m pytest -q tests
(cd modules/viewer_web && npm test -- --run && npm run build)
bash -n modules/video_to_dedup/*.sh scripts/3d/{evaluation,ops,pipeline,tuning}/*.sh
```

上面的 Python 命令是已验证的 owned gate。仓库根的裸 `uv run --offline pytest -q` 会收集未跟踪 nested checkout、`frame_sampler` 的 BSON client 与 legacy SAM3 tests，不能当作成功门。

个人护理分类器的精简使用说明见 [modules/personalcare_classifier/README.md](modules/personalcare_classifier/README.md)。
