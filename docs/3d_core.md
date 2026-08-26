# DA3 Core Contract

本文件描述当前根目录 DA3 核心的运行边界。历史设计和实验记录保留在 `docs/superpowers/`，其中的 `code/` 路径仅代表当时布局，不是当前命令。

## 责任边界

| 区域 | 责任 | 是否写入运行数据 |
| --- | --- | --- |
| `main.py` | 唯一 Python CLI；组装重建、匹配、去重、footprint、viewer export | 通过 `--save_root` |
| `src/` | DA3 runner、重建器、匹配/去重/导出阶段 | 否，除调用产生的明确产物 |
| `utils/` | 共享几何、cache、匹配和点云过滤 | 否 |
| `modules/` | 可独立使用的检测器、分类器、viewer 和视频 workflow | viewer bundle 例外 |
| `runtime/` | 迁移环境、工具 cache 和临时 workflow 数据 | 是；Git 忽略 |

根 `pyproject.toml` 与 `uv.lock` 定义 core、DA3、SAM3、Open3D 和 BSON API 的单一宿主
依赖契约：Python 3.11、NumPy 1.26.4、Torch 2.7.1、TorchVision 0.22.1 和 xFormers
0.0.31 必须保持锁定。`Depth-Anything-3/` 与 `sam3/` 是仓库内源码，不在候选环境中
重复安装 distribution。`modules/sku_detector/pyproject.toml` 是独立的 YOLO 依赖，二者不合并。

## 默认路径与后端

- 默认 dataset：`imdata/floor_display2`。
- 默认重建/匹配 backend：`da3`。
- 默认输出：根 `Output/`。
- 默认 viewer bundle：`modules/viewer_web/public/data`。
- 所有相对 `--save_root` 值按仓库根解析，而非调用终端的当前目录。
- DA3 runner 默认使用根 `.venv/bin/python`；可用 `DA3_VENV_PYTHON` 覆盖。

候选宿主环境必须由以下命令创建，且 `OUTPUT_DIR` 必须不存在。脚本使用冻结的根锁文件，
不会修改根 `.venv` 或 `Depth-Anything-3/.venv`，然后执行 `uv pip check` 和 DA3/SAM3
源码 import smoke：

```bash
scripts/3d/ops/build_unified_env.sh OUTPUT_DIR
```

候选测试通过后才可由维护者执行候选优先的环境切换；切换后保留原根环境为有界备份，直到
导入、聚焦测试和 GPU 等价性验证全部验收。

完整视频工作流使用：

```bash
bash scripts/3d/pipeline/video_to_viewer.sh \
  --video /path/to/video.mp4 --fps 2.0 --gpu 2 \
  --classifier-device cuda:0 --serve
```

该入口复用 `modules/video_to_dedup/run.sh` 完成抽帧、检测、pipeline 与 dedup，再直接调用 `viewer-web` 发布 minimal bundle；`ground-stack-area` 仍可独立运行但不是该入口的前置阶段。详细参数见 [scripts/3d/pipeline/README.md](../scripts/3d/pipeline/README.md)。

```bash
uv sync --extra dev
CUDA_VISIBLE_DEVICES=2 uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d \
  --recon_backend da3 --match_backend da3 \
  --classifier-device cuda:0
```

运行输出按数据集隔离：

```text
Output/<dataset>/
├── da3_cache/predictions.npz
├── output_3dmapping_da3/
├── dedup_detections/global_mapping.json
├── personalcare_classification/CURRENT -> runs/<time_ns>-<pid>/
│   ├── detections/<frame>.json
│   └── result.json
├── sam3_mask_cache/v2/
│   ├── entries/<image_id>/{manifest.json,masks.npz}
│   ├── locks/
│   └── corrupt/
└── ground_stack_footprint/CURRENT -> runs/<run_id>/
```

`CUDA_VISIBLE_DEVICES=2` 时，进程内 `cuda:0` 就是物理 GPU 2；没有该 mask 时请将 `--classifier-device` 改为实际可见 CUDA index。分类器必须使用显式 CUDA device，模型/CUDA 故障即为该阶段失败，绝不回退 CPU 或替代模型。

## Personalcare classification and publication

`--mode pipeline` 在 input validation 后立即异步提交一个独立 classifier subprocess；它可与 DA3 reconstruction 或 matching 并行。每个 dataset run 只加载一次原始 classifier 模型；分类按 object 原顺序以 batch（最多 32 个有效 crop）运行。实际时间重叠取决于 cache、可视化与运行 receipt：本次 fd6 cache-reuse receipt 中分类在 matching 开始前已完成。原始 `<dataset>/detections_results/` 是不可变输入，classifier 仅向上述 `personalcare_classification/runs/` 写 enriched copy；完整 frame/object count 校验后才原子替换 `CURRENT`。

每个有效 object 保留 raw `classes.cls` 和 `confidences.cls`，并加上规范化 `classification`（SKU ID、名称、confidence、`master_data_pending` metadata）。无效 bbox 保留原 object，发布 `status: unavailable` 与 `reason: invalid_bbox`，不合成替代 crop。此 V1 不生成 classification hash、signature、encryption、feature payload 或 content fingerprint。

matching 完成后 orchestrator 才 join classifier future；matching 与 classifier 都成功后，dedup 才显式接收本 run 的 enriched detection directory。若分类失败，已有 reconstruction/matching artifact 可以留作诊断，但 dedup、global mapping、footprint 与 viewer publication 不会继续。

`global_mapping.json` 的每个 observation（包括 removed observation）保留 classification。后端聚合在 `global_mapping.json` / classification 数据中保留 `confidence`、支持数和最大 confidence，用于对同一 global ID 的 `(sku_id, sku_name)` candidates 确定性排序；这些聚合值不属于 Viewer bundle。resolved 有一个 candidate，conflict 保留全部 candidates，unavailable 没有 candidate；首项 primary 是 Total/SKU facet 的唯一计数来源，避免一个物理对象重复计数。minimal Viewer 的 `objects.json` 只投影排序后的 `ordered_skus` 与 `point_ranges`，不发布或渲染 confidence。

## Footprint 与 SAM3 cache

运行顺序是 canonical contract：matching 必须完成 **全部** `batch_all_refs` references；`--mode ground-stack-area` 是独立的后端计量阶段，`--mode viewer-web` 可在 dedup 后直接发布产品 bundle。matching 是唯一的 SAM3 producer；它在默认 `enable_sam3_mask_sampling: true` 下以 self-exemplar 生成每个 frame 的完整 processed-space masks。master gate 为 false 时 matching 走既有 bbox sampling 且不发布 cache，任何需要实例点标签的 export 必须 fail closed。

v2 cache 的每个 payload 是 processed DA3 grid 上 `(object_count, height, width)` bool masks，以 little-endian `np.packbits` 无损打包到 `masks.npz`；manifest 以 `object_id` 绑定 mask，而非依赖数组位置。partial、mismatched 或 corrupt frame 不能命中。`sam3_mask_cache/v1` 与 v2 不兼容，v1 从不读取、迁移、复制、fallback 或自动删除。

`--mode ground-stack-area` 只读取 matching 已发布的 v2 cache、metric DA3 cache 与去重映射，不导入、加载或推理 SAM3，也不会因 cache miss 重算。它为每个 `global_id` 从全部有效观测重建 OBB，并在支撑平面上取 polygon union，得到 `da3_self_exemplar_ground_footprint_union`（m²）。缺少任何 canonical mask 或必要几何会发布 `rejected`/`null`，不会伪造部分结果。该 metric 是新 baseline，不能与旧 `da3_ground_footprint_union` 面积比较。

matching producer 与所有 v2 consumer 都只读取 `predictions.npz` 的逐帧完整 `source_to_processed_affine`（2×3）及 processed shape 来映射并裁剪 bbox。该 affine 可包含 resize、pixel-center、patch rounding 和 crop offset；`x'=sx*x+(sx-1)/2`、`y'=sy*y+(sy-1)/2` 仅是无额外 crop 的 simple-resize 例子，不能从 `process_res` 重算并替代 cache。缺 affine/shape 即 fail closed；此前 scale-only 或 raw out-of-grid bbox 的 entries 不能命中，必须重跑完整顺序。

cache 不再是 Web Viewer 的 protection mask。Viewer export 也只读 v2 processed masks，绝不加载或推理 SAM3；它在常规点云过滤之后传播实例标签。任何点都不会因带有 SAM3 标签而跳过点云去噪、地面或天空过滤。

## Minimal Web Viewer

```bash
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2
npm --prefix modules/viewer_web run dev
```

exporter 消费已发布的 DA3、去重和 matching 点标签输入，并从 dataset `images/` 按数字 stem 解析原图，写入不可变 `CURRENT -> runs/<run_id>/` minimal schema `3.0.0` bundle。`CURRENT` 只包含 `run_id`；manifest 必须包含轻量固定 `backend: "DA3"`、由 dataset path basename 提供的 `dataset_name`、`frame_count`、六维 `display_bounds` 和 16 维 `world_to_view`，不包含 source model 或 provenance。固定二进制文件为 `positions.f32.bin`、`colors.u8.bin`、`normals.i8.bin`，`point_count` 由 positions 长度推导。每个 run 的 `thumbs/` 包含 active 与 removed observation 的 bbox crop，编码为最长边 256px 的 JPEG；`objects.json` 只包含每个 global ID 的 `ordered_skus`、`point_ranges` 和 observations 的 `image_id`、`object_id`、`removed`、`thumbnail`。

canonical “其他品类”是 `sku_id=56642`、`sku_name=其他品类`。只要存在任一具体 SKU，具体 SKU 按既有 confidence/support 顺序排在 56642 之前；只有全部有效观测都是其他品类时，56642 才能排在首位。Viewer 只消费排序后的 SKU ID/名称，不接收或验证 confidence。

默认 `/data/` 由 `modules/viewer_web/public/data/` 提供。自定义 `--viewer-web-output` 不会自动被 Vite 服务，必须由部署层挂载到 `/data/`。

Viewer 的 Backend badge 直接显示 manifest 的 `backend`；objects 与 SKU counts 由前端读取 `objects.json` 的 observations 派生。左侧默认展开 `Select by SKU`，与 `Select by Global ID` 互斥，切换会清除上一选择；SKU 选择保留完整场景并批量 magenta 高亮，canvas 点选自动切换为 Global ID。`View Controls` 默认折叠，展开后只有 Fit、Top、Iso 和 Point size。右栏为 `Selected Object`，只显示 Global ID 和按发布顺序排列的 SKU。所有选择和 Focus 复用 `point_ranges`，不复制 point geometry。

Minimal Viewer 不包含 footprint、evidence、hash/provenance、source digest 或 confidence 字段；它只恢复产品缩略图所必需的 observation 标识和相对 thumbnail 路径，也不依赖旧的审计型 rich-contract 元数据。

## 性能基线

性能采集器只接受 one-shot cold 口径：fd2–4 各自从空的隔离 `save_root` 完整运行一次，
classification 与 reconstruction/matching 并行并在 dedup 前 join，浏览器只执行一次
cache-disabled 导航。当前 [20260826T084815Z](../perf/runs/20260826T084815Z/FINAL_REPORT.md)
三个 case 全部完成，平均真实 wall time 为 396.943s；footprint 平均 239.435s，是当前主瓶颈。

## 回归验证

```bash
PYTHONPATH=. VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv \
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --active --no-project python -m pytest -q tests
(CUDA_VISIBLE_DEVICES=2 uv run --project modules/personalcare_classifier python \
  modules/personalcare_classifier/source/classify_dataset.py \
  --dataset imdata/floor_display6 --output-root /tmp/personalcare-classifier-smoke \
  --device cuda:0)
(cd modules/viewer_web && npm test -- --run && npm run build)
bash -n modules/video_to_dedup/*.sh scripts/3d/{evaluation,ops,pipeline,tuning}/*.sh
```

该 Python 命令是已验证的 owned gate。仓库根的裸 `uv run --offline pytest -q` 会因未跟踪 nested checkout、`frame_sampler` BSON client 与 legacy SAM3 tests 的 collection 污染而失败，不能表示项目测试结果。
