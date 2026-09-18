# 离线 Global-ID Mapping 服务

此目录将当前 DA3/SAM3/global-ID/Minimal Viewer 流水线封装为同步 BSON
`POST /api` 服务。镜像不包含 detector、personalcare classifier、Pi3、VGGT、任何
输入数据或运行产物；调用方必须提供每帧的原图和外部 classifier 结果。

请求链路保持为 `api.py -> processor.process()`：API 只做 BSON 解码/编码并用一个锁
串行执行请求，processor 在临时目录中直接运行 pipeline、Viewer export 并上传两个结果到 COS。服务不再创建
request 子进程或 Pipe，也不会根据 pipeline summary 伪造 stage 异常；未捕获的处理失败会以 HTTP 500 直接返回 Python traceback。每次请求结束都会
清理按临时路径缓存的 DA3 image/transform/scene tensor，SAM3 model cache 则留在进程内供
下一次请求复用。

## 离线构建

`build.sh` 将包装代码与核心运行代码分开定位：Dockerfile、`build.sh`、
`__init__.py`、`api.py` 和 `processor.py` 始终从脚本所在的
`SCRIPT_DIR` 读取；pipeline 核心源码、`pyproject.toml`、`uv.lock`、DA3/SAM3.1 源码，
以及 builder 的 `/workspace` 挂载均从 `CORE_REPO_ROOT` 读取；SAM3.1 checkpoint 从 `SAM3_CHECKPOINT_DIR` 读取。
无论使用默认值还是显式传入值，`CORE_REPO_ROOT`（包括相对路径）都会立即规范化为
绝对且存在的路径。

在完整核心 checkout 内构建时，`CORE_REPO_ROOT` 默认是 `docker/` 的父目录：

```bash
cd /path/to/3D_Recognization
bash docker/build.sh
```

## 更新核心与服务代码

依赖、模型和 vendored 模型源码没有变化时，可从现有运行镜像派生代码更新镜像，
复制根 `main.py`、`config.yaml`、`src/`、`utils/` 与服务三个包装器，不重新装配
DA3 权重或 Python 环境。`src/`、`utils/` 先移除旧目录，避免保留已经删除的模块：

```bash
bash docker/build_code_update.sh
```

默认输出为 `global-id-mapping:4.0-traceback`，原 `4.0` 镜像和运行中的容器不会改变。
可通过 `BASE_IMAGE` 与 `IMAGE_TAG` 覆盖输入和输出 tag。若依赖、DA3 或 SAM3 源码
也有改动，必须改用完整的 `build.sh`。在独立 worktree 中必须显式设置
`CORE_REPO_ROOT=/path/to/3D_Recognization`；不能依赖其父目录默认值。

在由 `git subtree split --prefix=docker` 得到的 standalone checkout 中，必须显式
指向完整核心 checkout：

```bash
cd /path/to/3D_Recognization-docker
CORE_REPO_ROOT=/path/to/3D_Recognization bash build.sh
```

构建主机必须已经拥有下列本地内容：

- `harbor-cn.lingmouai.com/alg/sku-classifier-base:0.0.4`（linux/amd64、Ubuntu
  22.04）；
- `/home/xingyu/.local/bin/uv` 和完整 uv cache；
- 官方 `opencv-python-headless==4.11.0.86` Linux x86_64 wheel：
  `/data/www/comfyui/3d-recognition-build/runtime/wheels/opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`；
- Ubuntu 22.04 amd64 runtime `.deb` 目录，默认是
  `/data/www/comfyui/3d-recognition-build/runtime/system-debs/ubuntu-22.04-amd64`，其中必须含有
  `libx11-6_*.deb` 与 `libgl1_*.deb`；
- DA3 Hugging Face cache（`refs/`、`blobs/` 和 snapshot
  `b2359bdf726fb44ef62acca04d629dcf158053e7`）；
- SAM3.1 checkpoint `$SAM3_CHECKPOINT_DIR/sam3.1_multiplex.pt`（默认 modelscope 快照 `/home/xingyu/.cache/modelscope/models/facebook--sam3.1/snapshots/master`），代码树取自 `$CORE_REPO_ROOT/sam31/sam3`；根 `config.yaml` 必须使用相对路径 `sam31/checkpoints/sam3.1_multiplex.pt`，镜像内对应 `/app/sam31/checkpoints/`。镜像不再包含 SAM3.0 代码或权重。
- Docker 服务端不配置或处理 SKU 主数据，不读取 CSV，也不生成主数据 JSON；镜像不包含 CSV。SKU 主数据由 `visualization` 分支的 `viewer/masterdata.json` 提供。

默认 DA3 cache 是
`/home/xingyu/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1`。
路径不同则在调用时显式覆盖 `DA3_MODEL_CACHE`；同理可覆盖 `CORE_REPO_ROOT`、`HOST_UV`、
`UV_CACHE_DIR`、`IMAGE_TAG`、`BUILD_WORK_ROOT`、`OPENCV_WHEEL_DIR` 和 `SYSTEM_DEB_DIR`。
`OPENCV_WHEEL_DIR` 必须指向包含上述官方 wheel 的现有目录；`SYSTEM_DEB_DIR` 必须指向准备好的
Ubuntu `.deb` 目录。`build.sh` 会先把它们规范化为绝对路径，再在创建临时 context 前检查精确文件名
是否存在，并以只读挂载覆盖 builder 的 `/workspace/docker/wheels` 相对 flat index。默认临时构建目录是
`/data/www/comfyui/3d-recognition-build`，必须已经存在且当前用户可写。构建需要约 6 GB 的 DA3
named context，且不会下载、push、prune 或使用网络。

首次在可联网环境准备 Open3D 所需的 Ubuntu runtime packages；该命令复用已存在的 Mapping
base image，并在下载前离线验证它是 Ubuntu 22.04 amd64。它将下载 `libx11-6`、`libgl1` 及其
必需依赖到本地目录，然后退出而不构建最终镜像。准备时只清理该目录顶层已有 `.deb`、`lock` 和
`partial`，并以调用者 uid/gid 回写下载结果：

```bash
PREPARE_SYSTEM_DEPS_ONLY=1 bash docker/build.sh
```

之后的正常构建只使用该本地目录和其他已准备输入，仍保持 `--network=none`：

```bash
bash docker/build.sh
docker run --rm --gpus all -p 8011:80 \
  --mount type=bind,src="$(pwd)/docker/.env",dst=/app/.env,readonly \
  global-id-mapping:da3-self-contained
```

`build.sh` 会先用离线 base container 从根 `uv.lock` 生成可搬迁的统一 `.venv`，只安装
root base dependencies（不含 dev extras，且不把 editable 项目路径写进 venv），再用
BuildKit named contexts 装配最终镜像。运行时固定一份 `/app/.venv`、
`DA3_VENV_PYTHON=/app/.venv/bin/python`、离线 Hugging Face/Transformers，并且 Uvicorn
仅启动一个 worker；API 内的请求锁保证同一时刻只运行一个 fd。

COS 上传使用官方 `cos-python-sdk-v5` 的 `CosS3Client.put_object`，请求超时设为 120 秒（`CosConfig(Timeout=120)`，不是整个流水线的总时限）。该依赖已由根
`uv.lock` 锁定。`build_code_update.sh` 默认从本机高重叠的
`global-id-mapping:4.0-traceback` 派生，并离线复制根 `.venv` 中的 COS SDK 运行时包；可用
`BASE_IMAGE`、`CORE_REPO_ROOT`、`COS_SITE_PACKAGES` 和 `IMAGE_TAG` 覆盖这些输入。

`libX11` 与 `libGL` 仅用于满足 Open3D 的动态链接依赖；Docker 容器不会显示任何 UI。首次准备
wheel 和上述 system `.deb` 后，后续构建始终使用 `--network=none`、冻结 lock 和项目
`[tool.uv].find-links` flat index，不需要网络。`OPENCV_WHEEL_DIR` 与 `SYSTEM_DEB_DIR` 均可覆盖，
且必须为现有目录；脚本会在使用前把它们规范化为绝对路径：

```bash
OPENCV_WHEEL_DIR=/path/to/offline-wheels \
SYSTEM_DEB_DIR=/path/to/ubuntu-22.04-amd64-debs \
bash docker/build.sh
```

## BSON 输入与客户端

服务只读取如下 BSON 文档中的必需输入：

```text
{
  taskID: "<taskID>",
  images: [<numeric-frame image bytes>, ...],
  skus: ["{classes: {det, cls}, objects: [...]}", ...]
}
```

`taskID` 必须是非空的 `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` 字符串；它只能作为 COS
对象 key 的前缀使用。`images` 必须是非空 bytes list；`skus` 必须是同帧数的 JSON-string list。顶层的
`features`、`project_id` 及其他上游透传字段会被忽略：服务不会解析、校验、复制或落盘它们。
Adapter 固定以 personalcare domain `51` 构建 object-level `classification`，请求不能改变该值。
`skus[i]` 是 classifier 的逐帧 JSON，保留 `{classes, objects}`；每个 object 必须包含 det/cls
索引、det/cls confidence 和 bbox；object 内的 `features` 仍会被拒绝。服务将它规范化为当前
pipeline 所需的 object-level `classification`，不会在容器内执行分类器。

在 pipeline 和 Viewer 成功后，服务只读取容器内 `/app/.env`。部署主机的
`docker/.env` 必须显式提供 `COS_SECRET_ID`、`COS_SECRET_KEY`、`COS_BUCKET`、`COS_REGION`
和 `COS_KEY_PREFIX`，并以只读 bind mount 挂入；该文件受 `docker/.gitignore` 与
`docker/Dockerfile.dockerignore` 排除，绝不能提交、复制进镜像或上传到 GitHub/Gitee。
无法读取该文件时，服务保留底层的原生 OS 异常。

当前 `COS_KEY_PREFIX=global-id-mapping`，因此上传 key 固定为
`global-id-mapping/<taskID>/viewer_bundle.zip`，不会落在 bucket 根目录。成功 BSON 响应只返回
`global_skus`；Viewer 依据请求中的 `taskID` 直接从 COS 定位 ZIP。
`global_skus` 仍是逐帧 JSON 字符串数组，每帧保留 `{classes, objects}`。返回的 object
保留原始字段及 `global_id`、`is_deduplicated`，不返回内部生成的 `classification`。
调用方通过帧级 `classes.cls` 和 object 的 `classes.cls` 索引读取原始分类标签，通过
`confidences.cls` 读取分类置信度。此响应投影不修改内部结果文件或 Viewer ZIP。

客户端从 `<dataset>/images/` 和 `--classifier-result` 中读取相同数字 frame ID 的文件，
POST 到本机服务，并将响应中的 `global_skus` 写为 `global_skus.json`。Viewer ZIP 不经 BSON 返回，
由独立 Viewer 依据 `taskID` 从 COS 下载。

## Viewer Bundle

服务端显式调用 `run_complete_pipeline(..., evaluate_accuracy=False)`，不对没有人工标注的线上任务执行离线准确率评估；核心研究流水线默认仍评估。

服务端默认调用 `export_web_viewer_bundle(..., surfel_texture_edge=1920)`，生成 Surfel v2 纹理数据。ZIP 包含 `manifest.json`、`positions.f32.bin`、`normals.i8.bin`、`objects.json`、`surfel.json`、`surfel-u.f16.bin`、`surfel-v.f16.bin`、`surfel-frame.u8.bin`、`surfel-depth.f16.bin`、metadata 引用的全部 `surfel-texture-N.jpg` 和商品 `thumbs/*.jpg`，包含 `colors.u8.bin`。引用纹理缺失时打包直接失败，不上传不完整包。

原图等比缩小到最长边 1920、短边 1080 的上限；请求格式仍是 `taskID/images/skus`，响应仍仅有 `global_skus`，COS key 保持不变。前端 visualization 分支默认以 Surfel 加载，因此 `/?recognition_task_id=<taskID>` 即可查看新任务。旧普通点云任务需显式使用 `&render=points`；服务端不自动生成两份数据。Surfel 支持 1..32 个来源帧，GPU 资源上限仍取决于查看设备。


此 Docker 服务不构建、携带或托管可视化页面。它只生成平铺、非加密 `ZIP_STORED` schema 3.0.0 的
`viewer_bundle.zip` 并上传 COS；`global_skus` 只保留在 BSON 成功响应中，不写入 COS。独立 Viewer
依据 `taskID` 直接定位并下载该 ZIP。

可视化代码位于独立的 `visualization` 分支，且该分支只包含 `viewer/`。Viewer 从页面 URL 读取
`recognition_task_id`，以它定位对应的 COS `viewer_bundle.zip` 后在浏览器渲染；其 `viewer/.env` 只配置
公开的 COS 基址，不包含也不应包含 `COS_SECRET_ID` 或 `COS_SECRET_KEY`。Docker 镜像和该服务运行时均不依赖
该前端目录或其配置。

```bash
uv run python docker/test/test_api.py \
  --dataset /path/to/dataset \
  --classifier-result /path/to/classifier/detections \
  --taskID task-01
```

默认输出目录是 `<dataset>/docker_mapping_response/`；可用 `--output-dir` 指定其他路径。

本地测试统一保存在 `test/` 并由 `.gitignore` 忽略，不再纳入 Git；新克隆不包含测试文件。

## 本次完整重建验证（2026-09-08）

使用当前核心工作区源码与服务端 docker 分支代码完整构建 `global-id-mapping:da3-self-contained`，冻结 lock 离线安装 162 个包且依赖检查通过。构建缺少的锁定包先补齐到标准 uv cache，没有复制其他虚拟环境的 site-packages。

镜像内导入、SKU 主数据存在性、服务端调用与当前 exporter 签名的适配检查通过。CPU API smoke 覆盖坏 BSON、缺少请求字段的 500 traceback，以及 stub 成功响应仅含 global_skus 的 BSON 封装。未执行真实 GPU mapping 或 COS 上传，未替换运行中的容器；该次重建时服务端仍生成普通 points ZIP；本节为启用 Surfel 前的历史验证。

端到端验证（2026-09-08）：两张真实图片经 API 推理、Surfel 导出和 COS 上传成功，API/COS 均返回 200；结果为 247,673 点、56 个全局id，ZIP 9,677,757 bytes。Chromium 直接下载该任务，无 render 参数即可显示；全局id选择、中文界面和静止停绘通过，未见页面/控制台错误。浏览器使用软件 WebGL，此结果不代表硬件 GPU 帧率。

### 2026-09-08 移除 Docker CSV 后重建

已完整构建 `global-id-mapping:da3-surfel`（image ID 前缀 `470c38f17704`）。镜像检查确认无 `/app/runtime/sku_masterdata.csv`，服务适配器不传主数据参数，共用导出器默认跳过主数据。构建日志位于主仓库 `runtime/verification/docker-no-masterdata/build.log`。本次仅重建镜像，未替换已有容器；构建后执行 `docker builder prune --force`，释放 18.19GB 构建缓存。

### 原始点云与 Surfel 切换

默认显示 Surfel，视图控制旁的“切换到原始点云 / 切换到 Surfel”按钮即时切换，不重新下载数据，保留相机、筛选和高亮。`render=points` 可选择初始点云模式。原始点云使用导出时保留的同一组点及逐点 RGB，不是未过滤的全量重建点。两种渲染共享点索引与商品归属。

新导出及 ZIP 必须同时包含 `colors.u8.bin` 和 Surfel v2 数据；之前省略颜色的数据包需要重新导出。颜色额外占每点 3 字节，切换功能也会增加点云几何与渲染目标的显存占用。修改源码不会自动更新已运行的 Docker 服务，服务需要重新构建部署。

### SAM 商品点保护

核心 `src/web_viewer_export.py` 在几何过滤前读取 SAM3 mask，并通过 `protect_mask` 保护 mask 内的有效点，避免整件商品被小簇、离群或平面过滤删除。保护后的商品点按非空 `global_id` 均分最多 200 万预算，同一商品的多帧观测合并抽样，小组不足配额时回收未用配额；mask 外继续过滤并降采样，独立使用最多 50 万背景点预算。商品点不占用背景预算，原始点云与 Surfel 共用结果。分辨率对照固定上述预算，预算仅影响可视化导出，不改变 mapping 匹配输入。

构建脚本从 `CORE_REPO_ROOT/src` 复制导出代码；在此 worktree 构建时需显式设置 `CORE_REPO_ROOT=/home/xingyu/3D_Recognization`。修改核心源码不会自动更新正在运行的容器或已有 COS ZIP，需要重新构建、部署并重新导出任务。

背景点上限固定为最多 50 万点，不接受命令行或函数参数覆盖；商品预算固定为最多 200 万点，按 `global_id` 均分并回收小组未用配额，SAM 有效点先受过滤保护。Viewer 按导出点和 `point_ranges` 渲染，不在浏览器二次删点。Docker 后端构建从 `CORE_REPO_ROOT/src` 复制同一核心实现，旧容器和旧 COS ZIP 需要更新后端并重新导出才生效。

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

当前正式规则固定保留 **最多 200 万商品点 + 最多 50 万背景点**，取代此前商品 mask 全量输出规则。SAM 内有效点先受几何过滤保护；之后按非空 `global_id` 均分商品预算，将同一商品的多帧观测合并抽样。小组实际点数不足配额时回收未用配额并分配给其他组；组内使用固定随机种子、不放回均匀抽样。总商品点不足 200 万时不抽样。空 SAM 掩码不能产生商品点。分辨率对照固定上述预算，预算仅影响可视化导出，不改变 mapping 匹配输入。

位置、颜色、Surfel U/V、源帧和点击范围使用相同点索引。此上限由导出端控制，Viewer 不再独立截断点数组；旧包需重新导出才生效。

历史部署记录：`global-id-mapping:viewer-896-2m-quiet-20260910` 曾替换 8011 服务；当时镜像入口检查确认 896 长边、商品 200 万/背景 80 万及 `quiet_outputs=True`。该记录仅描述当时的历史镜像和容器，不代表当前镜像或运行服务已按当前 504 默认和 200 万/50 万规则重建；现有 COS 对象也未覆盖。

历史部署记录（2026-09-11）：曾构建并部署 `global-id-mapping:viewer-896-1500k-500k-20260911` 到 `global-id-mapping-local`（8011）。该记录不代表当前镜像或运行服务已更新到 504 默认和 200 万/50 万预算。

2026-09-11 最终本地部署：`global-id-mapping:viewer-504-2m-500k-20260911` 已用于8011服务。镜像内检查确认默认长边504、商品2000000点和背景500000点；16:9输出504×280、4:3输出504×378，竖屏交换宽高。同批次不能统一网格时拒绝运行，不自动裁切边缘。接口HTTP200。旧容器保留为`global-id-mapping-local-before-504-2m-20260911`，历史对照包未重导出。

## SAM3.1 部署（2026-09-18）

`build_code_update.sh` 以 `global-id-mapping:surfel-256-20260911` 为 `BASE_IMAGE` 增量构建 `global-id-mapping:sam31-20260918`：替换核心代码，删除 `/app/sam3`，加入 `/app/sam31/sam3` 代码树与 `/app/sam31/checkpoints/sam3.1_multiplex.pt`，`PYTHONPATH` 改指 `/app/sam31`。镜像内不再使用 SAM3.0；增量构建仍继承基础镜像层的体积，完整构建 `build.sh` 同样只打包 SAM3.1。

```bash
CORE_REPO_ROOT=/home/xingyu/3D_Recognization \
BASE_IMAGE=global-id-mapping:surfel-256-20260911 \
IMAGE_TAG=global-id-mapping:sam31-20260918 \
bash docker/build_code_update.sh
docker run -d --name global-id-mapping-local --gpus '"device=2"' -p 8011:80 \
  --env-file docker/.env \
  --mount type=bind,src="$(pwd)/docker/.env",dst=/app/.env,readonly \
  global-id-mapping:sam31-20260918
```

8011 服务固定 GPU2：GPU0 被其他进程占用约 30GB，`--gpus all` 默认落在 GPU0 会在匹配阶段 CUDA OOM。验证：video3 前 6 帧真实请求 HTTP 200（约 82 秒），SAM3 从 `/app/sam31` 加载，186 个对象归并为 125 个 `global_id`，COS `global-id-mapping/smoke-sam31-20260918/viewer_bundle.zip` 可下载（27.4MB）；8011 与临时 8012 容器返回的 `global_skus` 完全一致。旧容器保留为 `global-id-mapping-local-before-sam31-20260918`（已停止）。
