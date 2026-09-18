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
源码 import smoke。候选 venv 使用 uv 的 relocatable 模式，验收后才可安全重命名为
根 `.venv`：

```bash
scripts/3d/ops/build_unified_env.sh OUTPUT_DIR
```

候选测试通过后才可由维护者执行候选优先的环境切换；切换后保留原根环境为有界备份，直到
导入、聚焦测试和 GPU 等价性验证全部验收。

历史上以非 relocatable 模式创建且已重命名的环境，其 `pytest`、`uvicorn` 等
console-script shebang 仍会指向旧候选路径；`uv sync --frozen` 不会改写它们。停止项目
Python 进程后，维护者应重建当前根环境，而不是尝试手改 shebang：

```bash
uv venv --relocatable --clear .venv --python 3.11
uv sync --frozen --extra dev
```

根宿主只承诺 image-only SAM3 推理，故不安装官方 notebook-only 的 `decord`。视频
workflow 在进入 SAM3 前由 OpenCV 抽取图像帧；直接 `.mp4` loader 仍显式依赖 `decord`，
不会在统一环境中回退到其他解码实现。

根依赖精确固定 `opencv-python-headless==4.11.0.86`，不允许并存 GUI `opencv-python`，以避免
无显示 Docker 镜像导入 `cv2` 时要求 `libGL`。首次将官方 Linux x86_64 wheel 放到
`docker/wheels/`（或创建被 Git 忽略的同名本地链接）后，候选 builder 从这个项目相对 flat index
保持 `--offline`；Docker builder 以 `OPENCV_WHEEL_DIR` 指定的外部目录只读覆盖相同路径。

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

## DA3 默认推理分辨率（当前）

生产默认长边为 **504**；只做等比缩放，并通过 resize 结果对齐模型的 14 像素网格，不裁边、不补边。16:9 横屏得到 **504×280**（宽×高），16:9 竖屏得到 **280×504**；4:3 横屏得到 **504×378**，4:3 竖屏得到 **378×504**。

同一任务混合横屏与竖屏，或缩放后得到不一致的网格，会在模型加载前直接报错；请按方向和网格分别提交。输入按原比例只缩放，不裁边、不补边。输入应使用与检测框一致的已正确朝向的图像像素；流程不额外旋转照片或改变检测框坐标。

完整 pipeline 可复用同一 504 网格的旧缓存；旧 896 缓存必须重建。独立 runner 仍可用 `--process_res` 做指定分辨率实验；`viewer-web` 仅导出已有缓存，不会升级旧包。已生成的本地/COS可视化需按新分辨率重新推理并导出，刷新网页不会提高深度网格。

当前 504 网格的分辨率对照固定最多 200 万商品点和 50 万背景点；预算只影响可视化导出。

## 静默辅助输出

`uv run python main.py --mode pipeline --dataset <目录> --algorithm 3d --quiet_outputs` 关闭原始/去重检测框图片、匹配示意图、独立分析与准确率报告、可选 correspondences.json。匹配摘要、重建缓存、SAM masks、去重 JSON 与最终 Viewer 数据仍按流程需要生成；日志保留。

Docker `run_mapping_request` 默认启用此选项。请求的必要临时数据仍由原有 `TemporaryDirectory` 在结束时清理；不新增持久 outputs。普通本地 pipeline 默认保留调试输出。

## 商品点预算更新（2026-09-10）

当前固定保留 **最多 200 万商品点 + 最多 50 万背景点**，取代此前商品 mask 全量输出规则。SAM 内有效点先受几何过滤保护；之后按非空 `global_id` 均分商品预算，将同一商品的多帧观测合并抽样。小组实际点数不足配额时回收未用配额并分配给其他组；组内使用固定随机种子、不放回均匀抽样。总商品点不足 200 万时不抽样。空 SAM 掩码不能产生商品点。分辨率对照固定上述预算，预算仅影响可视化导出，不改变 mapping 匹配输入。

位置、颜色、Surfel U/V、源帧和点击范围使用相同点索引。此上限由导出端控制，Viewer 不再独立截断点数组；旧包需重新导出才生效。

## 可选的 DA3 几何优化模块

`src/da3_geometry_refinement.py` 是独立的实验入口，默认 pipeline / Docker 不启用。它读取已完成的 DA3 schema-v3 cache，在当前预处理网格上提取独立图像特征，对相机位姿与可选的逐帧深度尺度做小幅修正。输入预处理必须与当前默认一致，避免后续 pipeline 将优化缓存判为旧缓存并重新推理覆盖。

```bash
uv run python -m src.da3_geometry_refinement \
  --cache Output/<数据集名>/da3_cache/predictions.npz \
  --output-root runtime/refined-output \
  --mode pose
```

`--output-root` 必须不存在。通过验收后，缓存写入 `<output-root>/<数据集名>/da3_cache/predictions.npz`；原始缓存保留。需要同时优化深度时用 `--mode pose-scale`。第一版只有每帧一个正深度尺度，没有自由逐像素深度形变，也没有重训练 DA3。

- 独立对应：OpenCV SIFT，双向 ratio/mutual 筛选、图像坐标 RANSAC、空间覆盖筛选与相机图连通检查；不把已有 global_id 或 DA3 自投影当作精确对应。
- 优化：固定首相机、首帧深度尺度与全部内参。对其余帧优化旋转/平移小增量，可选深度正尺度；残差包含双向重投影、深度一致性及初始化先验，使用鲁棒损失。
- 深度采样：跳过无效点以及局部 2×2 深度跨度超过中值 5% 的边缘，原有效/无效掩码保持。旋转每分量限制 ±5°，平移每分量限制为样本中值深度的 ±10%；深度尺度范围约 0.909–1.1。
- 验收：每帧对确定性留出约 20% 对应，不参与优化。除了收敛与有效性，留出像素分量 RMSE 必须 ≤3 px、归一化深度 RMSE ≤0.05，同时分别不得相对恶化超过 1%/5%（深度有 1e-4 数值容差）。归一化深度误差的分母是所采样深度的场景中值，不是每点自身深度。这些是初版筛查阈值，不是商品匹配准确率保证。
- 发布：通过后同步重算 depth、extrinsic、world_points 并保留帧顺序/affine。先校验临时 NPZ，再原子发布。拒绝时 CLI 返回 2，保存 `geometry_refinement.json`，不生成供下游消费的缓存。组件异常保留原异常并记录失败阶段。
- 置信度与尺度：原 DA3 置信度不重新估计；原 `scale_factor` 只保留来源语义，逐帧修正在 `geometry_refinement` 元数据中记录。固定尺度锚不等于外部验证了真实米制尺寸。

通过验收后，用同一数据集及新的保存根目录运行完整流程，让 SAM 采样、匹配、去重及 Viewer 导出重新读取修正后的几何：

```bash
uv run python main.py --mode pipeline --dataset <原数据集目录> \
  --algorithm 3d --match_backend da3 --recon_backend da3 \
  --save_root runtime/refined-output

uv run python main.py --mode viewer-web --dataset <原数据集目录> \
  --save_root runtime/refined-output \
  --viewer-web-output modules/viewer_web/public/data-refined \
  --viewer-web-sku-masterdata-csv runtime/sku_masterdata.csv
```

不要把旧 `matching_summary`、global mapping 或 Viewer bundle 复制到新根目录。`--frame-count 8` 仅用于取现有联合预测中的前 8 帧做优化测试，不表示重新做了 8 帧 DA3 推理；继续下游时图像及分类必须是相同子集。完整数据集不传该参数。

当前限制：重复包装可能造成错误像素对应，反光/弱纹理可能导致连通失败；留出对应与训练来自同一场景/特征提取器，并非人工真值或独立视频。较大的位姿错误和局部深度形变可能超出初版能力，不通过时不会自动改阈值、切算法或回退发布。

如果保存根目录内存在几何优化报告但结果被拒绝、缓存未发布或已不符合当前预处理，完整 pipeline 会直接报错，不会自动重新推理覆盖该优化目录。首轮本地 8 帧实验两种模式均未通过绝对验收，详见 `runtime/geometry-refinement-review/README.md`；模块实现通过测试不等于已证明实际场景效果改善。

## Ray pose 独立对照

`src/da3_runner.py` 支持实验参数 `--use-ray-pose`，默认关闭；`--seed 42` 可固定对照的随机种子，未指定时保持原随机行为。例如：

```bash
uv run python src/da3_runner.py --input_dir <同一组图片> \
  --output_npz <全新实验目录>/predictions.npz \
  --model_path <本地DA3权重> --process_res 504 --seed 42 --use-ray-pose
```

输出缓存用 bool 标量 `use_ray_pose` 标记分支。默认 pipeline 不复用标记为 true 的实验缓存；旧默认缓存缺少该字段仍按原合同处理。实验应保存在独立目录，并显式执行匹配和 Viewer 导出；不要在候选目录运行默认完整 pipeline，否则会按默认分支重新推理。

Ray 分支同时恢复外参和内参；nested 的尺度对齐可能间接改变最终深度。因此对照固定权重、图片、分辨率、参考视图策略与种子，只改变此开关，不同时加入共享内参或几何优化。没有修改生产/Docker 默认。
