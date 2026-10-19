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

## MapAnything provisioning

MapAnything 源码以 `map-anything/` submodule 提供。首次 clone 使用
`git clone --recurse-submodules <repo>`；已有 checkout 运行 `git submodule update --init`，再执行
`uv sync --extra mapanything`。该 backend 不下载模型、没有默认模型缓存路径；使用 root CLI 时必须
显式提供本地快照：`--recon_backend mapanything --recon_model_path /path/to/local-model`。适配器代码为
Apache-2.0；权重 `facebook/map-anything` 为 CC-BY-NC 4.0，
`facebook/map-anything-apache` 为 Apache-2.0。

## 默认路径与后端

- 默认 dataset：`imdata/floor_display2`。
- 默认重建/匹配 backend：`da3`。
- 重建后端：`da3`、`pi3`、`pi3x`、`mapanything`；VGGT 保留用于 point tracking。
- 默认输出：根 `Output/`。
- 默认 viewer bundle：`modules/viewer_web/public/data`。
- 所有相对 `--save_root` 值按仓库根解析，而非调用终端的当前目录。
- DA3 runner 默认使用根 `.venv/bin/python`；可用 `DA3_VENV_PYTHON` 覆盖。
- CLI 每次重建起一个 `da3_runner.py` 子进程（加载模型 → 推理 → 退出）。长驻服务调用
  `DA33DReconstructor.keep_standby_worker()` 后，同 `(python, model_path, device)` 的重建改用
  预加载待命 worker（`src/da3_worker.py`）：模型已在 CPU 上加载（不建 CUDA context，0 MiB
  GPU），请求时搬到 GPU 跑 `da3_runner.main(argv, model=...)`，完成后进程退出释放全部显存；
  取用时后台预加载下一个待命 worker。失败抛 `DA3WorkerError`（含 traceback / stderr 末尾），
  不回退一次性路径。
- `predictions.npz` 用未压缩 `np.savez` 写出（fd12 写入 0.18s/78.5 MiB，压缩版 5.3s/67.6 MiB）。

DA3普通runner和CPU standby共用`src/da3_model_loading.py`：构造期间仅跳过元素数不少于65536的Linear/Conv weight随机Kaiming填充，保留bias、token、norm与buffer初始化。加载前以safetensors元数据检查key/shape/dtype以及真实Parameter身份别名，未覆盖或不匹配即拒绝；不使用meta、不改变模型精度、不静默改走另一种loader。实际Nested checkpoint有1383个state键/1377个独立参数，6个缺省键是共享LayerNorm别名而非漏载；1个nonpersistent buffer由构造器初始化。逐项覆盖及video3验证见[优化报告](../reports/latency-optimization-20261018/README.md)。

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

去重先构建并校验最终 global mapping，再生成逐帧去重 JSON 和全帧 `global_skus`。
逐帧 JSON 按原始对象顺序保留 `removed=false` 的对象；`global_skus` 保留所有原始对象，
其 `is_deduplicated` 与同一映射的 `removed` 一致。同帧冲突被拒绝的匹配边不参与删除，
零匹配时每个原始对象各自保留独立 ID。映射或全帧数据校验失败会停止本次逐帧写出；
单个逐帧文件写入失败仍按现有规则记录并跳过，全局 pair 发布失败仍清理两个全局文件。

Docker BSON 服务通过 Starlette thread pool 执行同步处理：解码、处理、请求清理和编码
都在线程内请求锁的保护下。单 worker 保证同一时间只有一个映射请求执行；事件循环可以
继续接收请求、等待工作线程或响应 `/openapi.json`。等待协程的取消不会释放正在执行的
线程所持有的锁。成功 BSON 和 HTTP 500 traceback 的接口约定保持一致。

## Footprint 与 SAM3 cache

运行顺序是 canonical contract：matching 必须完成 **全部** `batch_all_refs` references；`--mode ground-stack-area` 是独立的后端计量阶段，`--mode viewer-web` 可在 dedup 后直接发布产品 bundle。matching 是唯一的 SAM3 producer；它在默认 `enable_sam3_mask_sampling: true` 下以 SAM3.1 PVS 生成每个 frame 的完整 processed-space masks。master gate 为 false 时 matching 走既有 bbox sampling 且不发布 cache，任何需要实例点标签的 export 必须 fail closed。

v2 cache 的每个 payload 是 processed DA3 grid 上 `(object_count, height, width)` bool masks，以 little-endian `np.packbits` 无损打包到 `masks.npz`；manifest 以 `object_id` 绑定 mask，而非依赖数组位置。partial、mismatched 或 corrupt frame 不能命中。`sam3_mask_cache/v1` 与 v2 不兼容，v1 从不读取、迁移、复制、fallback 或自动删除。

`--mode ground-stack-area` 只读取 matching 已发布的 v2 cache、metric DA3 cache 与去重映射，不导入、加载或推理 SAM3，也不会因 cache miss 重算。它为每个 `global_id` 从全部有效观测重建 OBB，并在支撑平面上取 polygon union，得到 `da3_pvs_ground_footprint_union`（m²）。缺少任何 canonical mask 或必要几何会发布 `rejected`/`null`，不会伪造部分结果。该 metric 是新 baseline，不能与旧 `da3_ground_footprint_union` 面积比较。

PVS 使用现有 SAM3.1 multiplex 权重的 native tracker / interactive neck；整图 encoder 为 1008×1008，固定 batch=4，FP32 weights / BF16 autocast，只支持 CUDA。框为输入 processed 图像空间 XYXY，输出恢复到该图像尺寸并按 round 边界裁框。保留原生 presence/stability selection，不以空 mask 重试或填满 bbox。matching 与 warmup 共用同一 PVS model cache。

PVS loader默认跳过checkpoint将覆盖的、至少1048576元素的Linear/Conv weight随机填充及对应ViT重复trunc-normal；保留token、bias、norm、buffer和普通CUDA位置cache。构造后按真实Parameter身份/别名及key/shape/dtype审核，每个被跳过weight都必须被完整strict load覆盖，否则直接报错。显式`skip_random_init=False`用于原生对照。真实checkpoint与video3完整输出对照见[八项加速记录](../reports/gpu-acceleration-20261019/README.md)；Matching同时复用同一次投影的bbox hit counts，采样RNG、候选顺序和0.4m商品进深判定不变。

Viewer默认准备线程在同一完整float64 base点域建一个SciPy cKDTree，分块查询包含自身的KNN20，同时计算第二邻居距离和SOR均距；保留顺序归约、样本标准差、零距离及严格阈值。原商品保护、20% guard、DBSCAN和ground未改。底层`prepare_scene_filter(..., shared_neighbors=False)`保留原Open3D显式控制。具体范围与两视频对照见同一加速记录。

PVS cache schema 为 `sam31_pvs_processed_mask_cache_v1`，与旧 self-exemplar schema 不兼容；默认新目录保留旧缓存。面积报告同步使用 `da3_pvs_ground_footprint_union` 与 `mask_contract.source=sam3_pvs`，不能把 mask 来源改变当作面积准确率提升。

面积入口使用统一默认目录 `sam3_mask_cache_sam31_pvs/v2`，也接收 `--sam3_mask_cache_root` 指定 matching 实际发布的目录；显式目录不完整时直接拒绝，不尝试默认目录。DA3 输入及报告中的 cache schema 均为 3。

DA3 Viewer 和面积阶段共用 `utils/scene_orientation.py`，在实际 `predictions.npz` 旁保存 `scene_orientation.json`。方向命中直接复用；文件重建后失效重算，格式错误则报错。方向读取与 NPZ 文件状态绑定，避免跨重建坐标复用；不扫描其他 Viewer bundle，也不以 Viewer 的居中平移定义地面。面积用这个方向约束水平面高度搜索，并复用 Surfel 网格切线排除墙面。报告分别记录 `orientation.cache_event` 和 `plane.method`；没有可靠拟合方向时拒绝计量。最低可见水平面不等于已识别真实地板。规则与限制见 [面积算法说明](ground_stack_footprint_algorithm.md)。

matching producer 与所有 v2 consumer 都只读取 `predictions.npz` 的逐帧完整 `source_to_processed_affine`（2×3）及 processed shape 来映射并裁剪 bbox。该 affine 可包含 resize、pixel-center、patch rounding 和 crop offset；`x'=sx*x+(sx-1)/2`、`y'=sy*y+(sy-1)/2` 仅是无额外 crop 的 simple-resize 例子，不能从 `process_res` 重算并替代 cache。缺 affine/shape 即 fail closed；此前 scale-only 或 raw out-of-grid bbox 的 entries 不能命中，必须重跑完整顺序。

Viewer export 只读 v2 processed masks，不加载或推理 SAM3。它在几何过滤前给有效点建立实例标签；商品点参与 NN/SOR、聚类和地面估计，但应用各阶段删除结果时受 SAM 标签保护。非有限、全零和非正 Surfel 深度等基础无效点仍会删除。随后才按既有商品与背景预算采样。

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

唯一总耗时及阶段组成见[统一 profiling](profiling_breakdown.md)：包含2026-10-18 GPU2六视频同次细分，以及后续固定video3优化前后对照。计时从帧就绪至本地ZIP/GLB写入返回；不同仪器模式和输入不混算提速率。独立组件诊断不与主跑相加，准备线程的时间重叠不等于净收益。

## 回归验证

去重产物一致性与 API 并发行为可分别在根目录运行以下 CPU 测试。API 测试使用 processor
stub，不加载模型、不访问 COS；`--confcutdir` 避免收集独立 checkout 的包初始化代码。
运行环境需允许 asyncio 跨线程唤醒使用的本地 socket 写入。

```bash
uv run --no-sync python -B -m pytest -q test/test_classification_aggregation.py \
  -k 'conflicting_ring or successful_match_and_isolated or zero_matches'
uv run --no-sync python -B -m pytest -q \
  --confcutdir=runtime/worktrees/mapping-docker-rebuild/test \
  runtime/worktrees/mapping-docker-rebuild/test/test_mapping_api.py
```

```bash
PYTHONPATH=. VIRTUAL_ENV=/home/xingyu/3D_Recognization/.venv \
UV_CACHE_DIR=/tmp/3d-recognition-uv-cache \
uv run --active --no-project python -m pytest -q test
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

本地 pipeline、concise 与 matching 入口默认启用 `quiet_outputs`，不保存原始/去重检测框图片、匹配示意图、独立分析与准确率报告、可选 correspondences.json。匹配摘要、重建缓存、SAM masks、去重 JSON 与最终 Viewer 数据仍按流程需要生成；源帧、Viewer 纹理和商品缩略图是必要数据，也会保留。

需要恢复辅助输出时显式传 `--no-quiet_outputs`；正向 `--quiet_outputs` 仍可使用。主入口优先级是 CLI、YAML `main.quiet_outputs`、`inference.quiet_outputs`、默认 True；直接 inference 入口也允许 CLI 覆盖 YAML 的 True/False。可选 correspondences JSON 还需要 `--save_json`。Docker `run_mapping_request` 继续默认启用静默输出。请求的必要临时数据仍由原有 `TemporaryDirectory` 在结束时清理；不删除已有历史图片。

## 完整 Surfel 提前准备（2026-10-16）

`run_complete_pipeline(..., surfel_output_dir=...)` 在 DA3 缓存就绪后启动单个 CPU worker：加载缓存、计算原始 U/V 和深度、编码 JPEG 与相机元数据。2026-10-17 的源码扩展还会在同一个有效点域提前计算 NN 中位数和 SOR 原始保留 mask；这两项不依赖商品标签。matching/dedup 继续运行；dedup 完成后才合并最终 labels、应用 SOR 商品保护与保留比例 guard、执行聚类和地面过滤、预算采样、生成 point ranges 并发布 `CURRENT`。缓存复用也走同一路径；准备结果绑定同一缓存、源图目录、纹理尺寸、tuning、有效点顺序和过滤配置，错配直接报错。

`prepare_web_viewer_geometry(..., filter_config=...)` 与 `export_web_viewer_bundle(..., filter_config=..., prepared=...)` 提供对应 Python 接口，准备和最终阶段须使用相同过滤配置。独立 exporter 仍可串行执行；`prepare_filter=False` 显式推迟 NN/SOR。方向拟合、商品保护、DBSCAN 和地面删除仍在最终阶段；默认全量估计域及原有保护门槛不变。提前准备会增加原始网格、点云和纹理的驻留时间。新增同域复用、保护与输出对齐等功能测试已通过；真实全场景输出对照、matching 重叠收益与峰值 RSS 尚待验证。

2026-10-18新增显式`PointCloudFilterConfig(ground_proposal_max_points=50_000)`候选抽样实验，最终仍在完整当前域验证，默认`0`不变。该实验改变平面选择，不能作为等价加速；video3发现候选与摆正地面近乎垂直，尚未满足推广依据。point ranges已经改为同dtype批量查询，保留原稳定排序、空区间及JSON结构。[本轮记录](../reports/latency-optimization-20261018/README.md)区分已采用与实验选项。

完整循环、各个门槛、Open3D 内部邻域计算，以及提前抽样会改变哪些判断，见 [Viewer 导出算法详解](viewer_export_algorithm.md)。本轮实现、预算和验收状态见 [2026-10-17 计划](../reports/viewer-latency-20261017/PLAN.md)；不以历史组件耗时预测此次实际收益。

CLI 通过 `--mode pipeline --algorithm 3d --viewer-web-output <目录>` 启用；视频到 Viewer 脚本自动接入。

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

## Viewer背景清理更新（2026-10-19）

默认`PointCloudFilterConfig.cluster_proxy_voxel_scale=2.0`，在代理上聚类后将标签传回原始点。`remove_ground`实际是平面背景清理：仅未保护背景参与拟合，`ground_min_inlier_ratio`与全局8%门槛已删除；保留商品保护、双侧15%安全检查和每阶段20%guard。该步骤不承诺识别真实地板，未标注商品仍可能被视为背景。

此默认变更不修改共享方向、footprint、Viewer预算或Surfel长边限制。详细循环见[Viewer算法第4节](viewer_export_algorithm.md#4-过滤如何保护商品)，三视频缓存导出与商品保留验证见[profiling第17节](profiling_breakdown.md#17-默认proxy2与背景平面清理2026-10-19)。前述完整端到端计时属于旧过滤策略。
