# 3D SKU Recognition

面向货架与地堆 SKU 的 DA3 三维重建、跨图匹配、去重、ground-footprint 计量和静态 Web Viewer。Python 负责生成可审计产物；Three.js 只加载、校验和交互展示已经发布的 bundle。

详细的核心契约见 [docs/3d_core.md](docs/3d_core.md)，当前端到端测量结果见 [perf/runs/20260826T084815Z/FINAL_REPORT.md](perf/runs/20260826T084815Z/FINAL_REPORT.md)。

`perf/benchmark.py` 只采集 one-shot cold 数据：fd2–4 各自使用全新的隔离输出运行一次，
浏览器也只导航一次；不调度、不保存、不汇总 warm-start case。classification 与
reconstruction/matching 并行，并在 dedup 前 join；端到端统计使用真实 case wall time。

## 性能与显存

- 检测、匹配、分析、重建、分类、可视化与评估阶段失败直接抛出异常，CLI 非零退出，后续阶段停止；
  成功调用仍返回产物路径和耗时。批量参考帧失败不再继续串行处理其他帧，也不自动串行重跑；
  已提交的并行任务在退出前完成资源清理。CLI 与交互菜单共用 concise 流程。
  批量准确率评估在首个失败时非零退出；没有可比较的人工标注直接报错，有标注但零预测仍正常计分。
- 检测器仅在推理成功且无目标时写空检测；坏检测 JSON 和缺失帧直接报错，不再自动取交集丢帧。
  DA3/Pi3/Pi3X 匹配按 image ID 对齐，场景内存缓存按帧 ID 顺序区分；旧缓存缺少 ID 时需重新重建。
  Pi3 后处理、内参拟合和 transforms 保存失败均停止，不再使用默认 K。
- SKU 分析逐目标图解析 summary，保留不同图片对中相同 object ID 的匹配；零匹配可正常生成报告。
  3D 匹配每张目标图只变换一次目标框，所有参考物体复用，不改变采样和匹配规则。
- 本轮最小验证：`uv run --no-sync python -m pytest -q test/test_fail_fast.py
  test/test_main_pipeline.py test/test_matching_sam3_cache.py test/test_pi3x_export.py
  test/test_reconstructor_base.py -k 'not cuda'`；CPU stub 覆盖失败传播与正常数据路径，不代表 GPU 性能测量。

- 重建生命周期由 `ReconstructorBase` 统一调度：推理后准备导出数据，再保存 NPZ、导出 GLB，
  `finally` 清理单次运行资源；缓存或导出失败直接向调用方报错。Pi3X 将导出所需张量一次转到
  CPU，共享置信度和深度视图，供两种导出复用。直接复用同一 reconstructor 时模型保留，
  `close()` 或退出 `with` 时释放模型；pipeline 使用 `with` 管理单次重建实例。
  DA3 保留子进程推理和 partial 缓存原子发布。

- ground-stack support-plane 的 deterministic RANSAC 以 16 个候选为一批计算距离，仍按原始
  seed、triplet 顺序、严格 tie、trial count 与 gate 串行决定结果；批化只增加有限 CPU 临时空间，
  不使用 GPU。
- DA3、Pi3/Pi3X matching 使用零存储 image descriptor；Pi3/Pi3X 按帧顺序、后端和 pixel limit 缓存 transforms，RGB 仅在生成匹配可视化时在 CPU 解码（Pi3/Pi3X 保留 LANCZOS resize）。2D box
  hit 以单次向量化计算复用给 Top-K，诊断性的 host 拷贝只在 DEBUG 开启。
- DA3 runner 先写 `da3_cache/predictions.npz.partial`，父进程验证 schema-v3 metric 和 matcher
  字段后以原子 replace 发布 `predictions.npz`；失败不会覆盖现有 cache。

## 当前布局

`fd-videos/` 保存原始 HDR 测试视频；最终 SDR/H.264 压缩版位于 `fd-videos-sdr/30fps/`，不覆盖原片。保留 1080p 和竖屏方向，帧率按 Rick 确认降为 30 fps；直接从原片先降帧、再做色调映射与 CRF 23 编码，参数见 [视频说明](fd-videos-sdr/README.md)。SDR 压缩版适合上传/流程测试，不等同于原片画质或重建精度基准。60 fps 中间视频已按要求删除，原片和 30 fps 最终版本保留。

```text
3D_Recognization/
├── main.py, config.yaml, pyproject.toml, uv.lock  # DA3 核心 CLI 与依赖
├── src/                                           # 流水线阶段与重建后端
├── utils/                                         # 几何、匹配、cache、过滤等共享库
├── test/                                         # 核心回归测试
├── modules/
│   ├── sku_detector/                              # YOLO 检测器与独立 uv 项目
│   ├── personalcare_classifier/                   # source/ + canonical model.bin
│   ├── viewer_web/                                # TypeScript / Three.js 前端
│   └── video_to_dedup/                            # 视频到 DA3 去重入口
├── scripts/3d/{pipeline,evaluation,tuning,ops}/   # 端到端 pipeline 与维护工具
├── Output/                                        # 忽略的用户可见 pipeline 产物
├── runtime/                                       # 忽略的迁移环境与工具 cache
│   ├── sku_detector/.venv/                     # detector 独立运行环境
│   └── video_to_dedup/
├── perf/                                          # 可复现的性能采集与报告
└── frame_sampler/                                 # 保持为外部嵌套 Git 仓库
```

`Output/` 与 `runtime/` 都不是源码，也不应提交。根 `.venv` 是 core、DA3 和 SAM3 的统一环境；`runtime/sku_detector/.venv` 仅供独立 detector 使用。

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

根环境固定使用 `opencv-python-headless==4.11.0.86`，不能同时安装 GUI `opencv-python`，从而避免
最小容器导入 `cv2` 时依赖 `libGL`。一次性准备官方 Linux x86_64 wheel 后，在
独立 `origin/docker` checkout 的 `docker/wheels/` 建立被 Git 忽略的本地链接（或手动放置同名
wheel）；根项目以该相对 flat index 离线锁定和构建，不提交 wheel：

```bash
ln -s /path/to/offline-wheels/opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
  docker/wheels/opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
scripts/3d/ops/build_unified_env.sh /tmp/3d-recognition-unified-env
OPENCV_WHEEL_DIR=/path/to/offline-wheels bash docker/build.sh
```

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

## 离线 Global-ID Mapping Docker 服务

Docker 包装代码由独立 `origin/docker` 分支维护，父仓库不跟踪并忽略 `/docker/`。首次建立标准
工作区时，在 main checkout 根目录创建一个追踪该远端分支的嵌套 worktree：

```bash
git fetch origin docker
git worktree add --track -b docker docker origin/docker
```

此后 `/home/xingyu/3D_Recognization` 追踪 `main`，其 `docker/` 子目录独立追踪 `docker`。
Docker wrapper 提供只消费外部 classifier 结果的 DA3/SAM3 BSON 映射服务，使用本地 base
image、冻结的 root lock、完整 DA3 Hugging Face cache 与本地 SAM3 checkpoint 离线构建；镜像
不包含 detector/classifier、Pi3、VGGT、输入数据或运行输出。

运行时采用直接同步链路 `docker/api.py -> docker/processor.py:process()`：单 worker 配合请求锁
串行处理 fd，不创建 multiprocessing child 或 Pipe。processor 不会根据 pipeline summary 伪造
stage 异常；API 成功返回 BSON，未捕获的 pipeline、输入或导出异常直接以 HTTP 500 traceback 返回。每个 fd 完成后清除按临时路径持有的 DA3
request cache，SAM3 model cache 保留并跨请求复用。

客户端发送以下 BSON 输入，`taskID` 标识本次识别任务，`images` 为非空 bytes list，`skus` 为同帧数的 classifier JSON-string
list：

```text
{taskID: "<recognition-task-id>", images: [<numeric-frame image bytes>, ...], skus: ["{classes: {det, cls}, objects: [...]}", ...]}
```

顶层 `features`、`project_id` 和其他上游透传字段均被 Docker adapter 忽略，不会解析、校验、复制或
落盘；adapter 固定以 personalcare domain `51` 构建 object-level `classification`。object 内的
`features` 仍会被拒绝。成功 BSON 响应严格只有 `global_skus`（逐帧 JSON-string list），每帧为
`{classes, objects}`。返回的 object 保留原始字段和 `global_id`、`is_deduplicated`，不返回内部
`classification`。调用方通过帧级 `classes.cls[object.classes.cls]` 读取 `sku_id^sku_name`，
通过 `object.confidences.cls` 读取分类置信度；内部分类聚合和 Viewer 导出仍使用内部分类数据。
Viewer bundle 按 `taskID` 上传 COS，不随 BSON 返回。它是扁平 ZIP：根目录含 `manifest.json`、
`positions.f32.bin`、`colors.u8.bin`、`normals.i8.bin`、`objects.json`，缩略图为
`thumbs/*.jpg`；它不包含发布器内部的 `CURRENT` 或 `runs/<run_id>/` 路径。

```bash
bash docker/build.sh
docker run --rm --gpus all -p 8011:80 global-id-mapping:da3-self-contained
uv run python docker/test/test_api.py --dataset <path> --classifier-result <path>
```

详细的离线前提、named contexts、输入 shape 与客户端输出见
[独立 Docker 分支 README](https://github.com/EinheitCxy/3D-SKU-detection/blob/docker/README.md)。

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
# 将 61 MiB SKU Excel 转为构建期窄 CSV；浏览器不会加载此全量文件
uv run python scripts/convert_sku_maindata.py \
  --input sku-maindata.xlsx --output runtime/sku_masterdata.csv

uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2 \
  --viewer-web-sku-masterdata-csv runtime/sku_masterdata.csv

# 启动前端；默认 /data/ 映射 modules/viewer_web/public/data/
npm --prefix modules/viewer_web run dev

# 一条命令从视频生成去重、Viewer bundle，并启动 Web Viewer
bash scripts/3d/pipeline/video_to_viewer.sh \
  --video /path/to/video.mp4 --fps 2.0 --gpu 2 \
  --classifier-device cuda:0 --serve
```

`--save_root` 可以覆盖输出目录；相对值始终相对仓库根解析。默认 bundle 位于 `modules/viewer_web/public/data/`，自定义 bundle 必须在前端启动前挂载或 serve 到浏览器的 `/data/`。`runtime/sku_masterdata.csv` 是从 Excel 提取的 `sku_id`、厂商（`group_name`）、品牌（`brand_name`）、品类（`category_name`）和 POSM（`type_id == 27`）窄表；导出时只把当前 bundle 使用的 SKU 主数据写入该 run 的 `sku_masterdata.json`。

已有完整 Viewer bundle 只需补充主数据时，可避免重跑点云导出：

```bash
uv run python scripts/publish_viewer_masterdata.py \
  --viewer-output modules/viewer_web/public/data \
  --sku-masterdata-csv runtime/sku_masterdata.csv
```

该命令会复制 `CURRENT` 指向的完整 run，在新 run 写入 `sku_masterdata.json` 后原子更新 `CURRENT`；旧 run 保持不变。

### Pi3X 后端（实验性）

`--recon_backend pi3x --match_backend pi3x` 启用 Pi3X（`yyfz233/Pi3X`，CC BY-NC 4.0）重建与匹配。权重为本地目录 `runtime/models/pi3x/`（model.safetensors + config.json），离线加载。Pi3X 产出与 Pi3 相同的 `pi3x_cache/predictions.npz` schema-v3 契约（sigmoid conf、`local_points` 深度、`camera_poses` 取逆为 w2c extrinsic），匹配侧复用 pi3 图像加载路径；SAM3 v2 mask 缓存需要的 `source_to_processed_affine` 由纯缩放 transform 显式合成。Pi3X 是 approximate metric（尺度逐 batch 估计），不接 `is_metric==1` 硬门，ground-stack-area 等米制计量阶段不对其启用；RoPE2D CUDA kernel 未编译时自动走 PyTorch 慢速路径。fd4–8 与 DA3 1.1 的准确率/性能对比见 [docs/accuracy_da3_vs_pi3x.md](docs/accuracy_da3_vs_pi3x.md)（总 F1 86.8% vs 85.0%）。

完整视频入口的参数、阶段顺序和输出路径见 [scripts/3d/pipeline/README.md](scripts/3d/pipeline/README.md)。其中 `--gpu 2` 设置物理 GPU mask，进程内分类器继续使用 `--classifier-device cuda:0`；`--detections-dir` 可复用已有逐帧检测 JSON。脚本在 dedup 后直接导出 minimal schema 3.0.0 bundle，默认只导出 bundle，增加 `--serve` 才会以前台进程启动 Vite。独立 `ground-stack-area` 仍可按需运行，但不再是该 Viewer 入口的前置阶段。

上例的 `CUDA_VISIBLE_DEVICES=2` 把物理 GPU 2 映射为进程内的 `cuda:0`；`--classifier-device` 必须是一个显式可用的 CUDA device。分类器不接受 CPU 或替代模型 fallback。若不使用 GPU mask，可直接传入实际可见设备号，例如 `--classifier-device cuda:2`。

## Canonical SAM3 processed-mask workflow

matching 是唯一的 SAM3 producer。默认 `enable_sam3_mask_sampling: true` 时，它只使用 self-exemplar，并为每个 detection frame 一次性发布完整的 `sam3_mask_cache/v2` processed-space bool mask；payload 用 little-endian `np.packbits` 无损保存。`enable_sam3_mask_sampling: false` 仅保留既有 bbox sampling，且不会发布 cache，因此 footprint 与 viewer export 会 fail closed，提示先运行 matching。

运行顺序是硬约束：先完成 **完整 batch-all-refs matching**；`ground-stack-area` 是独立的后端计量阶段，`viewer-web` 可在 dedup 后直接发布产品 bundle。Viewer exporter 不导入、加载或推理 SAM3；matching 的 processed-mask cache 仍是点标签发布的输入，cache miss 会 fail closed。

`sam3_mask_cache/v1` 与 v2 不兼容：v1 从不被读取、迁移、复制或删除。v2 的 formal metric 是 `da3_self_exemplar_ground_footprint_union`，viewer bundle schema 是 `3.0.0`；它们与旧面积输出和 bundle `1.0.0`、`2.0.0` 不可数值比较，旧 bundle 必须按上述顺序重新生成。

DA3 bbox 的 source→processed 映射唯一权威是 `predictions.npz` 中每帧完整的 `source_to_processed_affine`（2×3）及其 processed grid。该 affine 可能同时编码 resize、pixel-center、patch rounding 与 crop offset；`x'=sx*x+(sx-1)/2`、`y'=sy*y+(sy-1)/2` 仅是没有额外 crop 的 simple-resize 例子，绝不能据此重算或替代 cache。matching 缺少显式 cache affine/shape 会 fail closed；旧 scale-only 或含 raw out-of-grid bbox 的 v2 entries 均不能命中，须完整重跑 matching → footprint → export。

## Minimal Viewer 与点云策略

Docker 服务端调用 `run_complete_pipeline(..., evaluate_accuracy=False)`，线上请求完成计数后直接导出 Surfel，不依赖人工 benchmark；本地流水线默认仍执行准确率评估。未运行的评估步骤不会在摘要中标为成功。

Surfel 使用已有 DA3 缓存、原图和同网格 SAM3 mask 导出独立纹理表面数据；浏览器通过 `/?data=/data-surfel/&render=surfel` 显式启用，默认 points 入口保持独立。后端导出、Float16 sidecar、逐片元投影与深度融合、商品交互和按需重绘的代码说明见 [Surfel 集成说明](docs/surfel_implementation.md)，命令与限制见 [Viewer README](modules/viewer_web/README.md#深度约束-surfel)。Docker 服务端已启用 Surfel 导出和 COS ZIP 打包；`docker/viewer` 默认以 Surfel 加载新任务。

点云导出默认使用 **5 mm 背景体素、最多 50 万背景点**；商品点最多 200 万。商品点先由 SAM mask 过滤保护，避免有效实例点被几何过滤误删，再按非空 `global_id` 均分商品预算；小组实际点数不足时回收未用配额并分配给其他组。背景上限为固定导出预算，`--viewer-web-voxel-size` 仍可调整背景体素尺寸。分辨率对照固定这两项预算；预算只影响可视化导出点数，不改变 mapping 匹配输入。已有 bundle 需要重新导出才能改变点数。

Web bundle 使用不可变 `CURRENT -> runs/<run_id>/` 发布。`CURRENT` 只包含 `run_id`；run 内的 `manifest.json` 固定为 schema `3.0.0`，包含轻量 `backend: "DA3"`、真实 `dataset_name`、`frame_count`、六维 `display_bounds` 和 16 维 `world_to_view`，不携带 source model 或 provenance。固定二进制文件为 `positions.f32.bin`、`colors.u8.bin`、`normals.i8.bin`，`point_count` 由 positions 长度推导。导出器从 dataset `images/` 中按数字文件名解析原图，为每个 active 与 removed observation 按 bbox（保留 10% padding）写入 `thumbs/*.jpg`：JPEG 始终为精确 `128×128`，crop 等比缩放并居中补深色背景，不拉伸或中心裁掉商品；`objects.json` 只包含每个 global ID 的 `ordered_skus`、`point_ranges` 和 observations 的 `image_id`、`object_id`、`removed`、`thumbnail`。

canonical “其他品类”是 `sku_id=56642`、`sku_name=其他品类`。只要存在任一具体 SKU，具体 SKU 按既有 confidence/support 顺序排在 56642 之前；只有全部有效观测都是其他品类时，56642 才能排在首位。Viewer 只消费已排序的 SKU ID/名称，不接收或显示 confidence。

产品界面的 Backend badge 直接显示 manifest 的 `backend`；对象与 SKU counts 由前端读取 `objects.json` 的 observations 派生，而非额外后端聚合字段。默认 `Select by SKU`，与 `Select by Global ID` 互斥，切换会清除上一选择。SKU 选择保留完整场景并批量 magenta 高亮；canvas pick 自动切换为 Global ID。`View Controls` 默认折叠，展开后只有 Fit、Top、Iso 和 Point size。右栏为 `Selected Object`，只显示 Global ID 与按发布顺序排列的 SKU；observation 缩略图以紧凑三列优先网格显示，caption 与 removed 灰化语义保持不变。

Viewer bundle 不包含 footprint、evidence、hash/provenance、source digest、confidence 或其他审计型 rich-contract 元数据。它只恢复产品缩略图所需的 observation 标识和相对 JPEG 路径。导出先读取 SAM3 mask 标注有效点，再将 mask 内的点传入 `protect_mask`，防止离群、小簇和平面过滤误删商品；之后商品点按非空 `global_id` 均分最多 200 万点预算，同一商品的多帧观测合并抽样，小组不足配额时回收未用配额。背景保持原有过滤并独立使用最多 50 万点预算；商品点不占用背景预算。组内抽样使用固定随机种子、不放回均匀抽样，空 SAM 掩码不能产生商品点。分辨率对照固定这两项预算，预算仅影响可视化导出，不改变 mapping 匹配输入。原始点云与 Surfel 共用这组导出点。选择和 Focus 通过 `point_ranges` 增量更新现有 geometry，不复制点云。已有 COS ZIP 不会自动恢复被删点，需要用修改后的后端重新导出。

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

### Viewer backend 对比

画布右下角显示实际 backend。已有 Pi3X 缓存与匹配结果可运行 `uv run python scripts/export_pi3x_viewer.py --dataset imdata/floor_display6`，生成独立 Pi3X bundle；本地 Viewer 使用 `/?data=/data-pi3x/` 查看，默认 `/` 保留 DA3。详见 [Viewer README](modules/viewer_web/README.md)。

本地测试统一保存在 `test/` 并由 `.gitignore` 忽略，不再纳入 Git；新克隆不包含测试文件。
核心测试运行 `uv run --no-sync pytest test/`；性能测试在 `test/perf/`，Viewer 测试在 `test/viewer_web/`，SAM3 测试及资源在 `test/sam3/`。Docker 独立工作目录的测试在 `docker/test/`。

Docker Viewer 导出不传入主数据参数，跳过主数据读取和文件生成，镜像构建不复制 CSV；主数据由 `visualization` 分支的 `viewer/masterdata.json` 提供。本地 Viewer 导出仍传入 CSV 路径以生成任务所需的 `sku_masterdata.json`。

### 匹配、导出与点击的一致性

匹配计算内部使用数组下标，`matching_summary.txt` 发布真实图片文件 ID（包括零匹配摘要的 `Reference image file ID`）。去重和计数分析读取摘要中的文件编号，不用输出目录编号代替，也不猜测 0/1 偏移；旧摘要需重新生成。人工 benchmark 延续现有“源文件 ID + 1”的展示编号约定，评估参考编号从摘要头读取。

每个原始检测都进入 global mapping；没有可信匹配边的检测保留独立 ID。逐帧去重 JSON 和 `global_skus` 均依据最终 global mapping 的 removed 标志，避免冲突匹配剔除后商品丢失或多个输出计数矛盾。不存在的帧/对象或不完整匹配分组直接报错。

原始点云点击遇到可见无归属前景点即停止，不继续选择后面的商品。射线点选仍有交互容差；Surfel 使用可见面片的 ID pass。重叠 SAM mask 仍按已有顺序确定单一点击归属，零面积 U/V 点可保留在原始点云中但不产生 Surfel 像素；导出点保留不等于任意视角都无遮挡。

当前 3D 匹配、身份去重、SAM 点保护、背景预算和 Surfel 点击链路的完整说明见 [3D 去重流程详图](docs/3d_dedup_flowcharts.md)，可直接打开 [离线交互流程图](docs/figures/dedup-flowcharts/index.html)。

## DA3 默认推理分辨率（当前）

生产默认长边为 **504**；只做等比缩放，并通过 resize 结果对齐模型的 14 像素网格，不裁边、不补边。16:9 横屏得到 **504×280**（宽×高），16:9 竖屏得到 **280×504**；4:3 横屏得到 **504×378**，4:3 竖屏得到 **378×504**。

同一任务混合横屏与竖屏，或缩放后得到不一致的网格，会在模型加载前直接报错；请按方向和网格分别提交。输入按原比例只缩放，不裁边、不补边。输入应使用与检测框一致的已正确朝向的图像像素；流程不额外旋转照片或改变检测框坐标。

完整 pipeline 可复用同一 504 网格的旧缓存；旧 896 缓存必须重建。独立 runner 仍可用 `--process_res` 做指定分辨率实验；`viewer-web` 仅导出已有缓存，不会升级旧包。已生成的本地/COS可视化需按新分辨率重新推理并导出，刷新网页不会提高深度网格。

当前 504 网格的分辨率对照固定最多 200 万商品点和 50 万背景点；预算只影响可视化导出。

历史部署验证：`global-id-mapping:viewer-896-20260910` 曾运行于 `global-id-mapping-local`；该记录保留用于追溯，不代表当前镜像或运行服务仍使用 896 默认值。

## 静默辅助输出

`uv run python main.py --mode pipeline --dataset <目录> --algorithm 3d --quiet_outputs` 关闭原始/去重检测框图片、匹配示意图、独立分析与准确率报告、可选 correspondences.json。匹配摘要、重建缓存、SAM masks、去重 JSON 与最终 Viewer 数据仍按流程需要生成；日志保留。

Docker `run_mapping_request` 默认启用此选项。请求的必要临时数据仍由原有 `TemporaryDirectory` 在结束时清理；不新增持久 outputs。普通本地 pipeline 默认保留调试输出。

## 商品点预算更新（2026-09-10）

当前固定保留 **最多 200 万商品点 + 最多 50 万背景点**，取代此前商品 mask 全量输出规则。SAM 内有效点先受几何过滤保护；之后按非空 `global_id` 均分商品预算，将同一商品的多帧观测合并抽样。小组实际点数不足配额时回收未用配额并分配给其他组；组内使用固定随机种子、不放回均匀抽样。总商品点不足 200 万时不抽样。空 SAM 掩码不能产生商品点。分辨率对照固定上述预算，预算仅影响可视化导出，不改变 mapping 匹配输入。

位置、颜色、Surfel U/V、源帧和点击范围使用相同点索引。此上限由导出端控制，Viewer 不再独立截断点数组；旧包需重新导出才生效。

### DA3 位姿与深度尺度优化（实验）

新增独立命令 `uv run python -m src.da3_geometry_refinement --cache Output/<数据集名>/da3_cache/predictions.npz --output-root runtime/refined-output --mode pose`。`pose-scale` 模式同时优化逐帧正深度尺度。输出根目录必须是新目录；只有留出几何验证通过才发布缓存，之后以同一 `--save_root` 重新运行完整匹配与 Viewer 导出。生产和 Docker 默认不变。参数、验收与限制见 [3D core 文档](docs/3d_core.md#可选的-da3-几何优化模块)。
