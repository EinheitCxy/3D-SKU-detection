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
`SCRIPT_DIR` 读取；pipeline 核心源码、`pyproject.toml`、`uv.lock`、DA3/SAM3 源码、
SAM3 checkpoint，以及 builder 的 `/workspace` 挂载均从 `CORE_REPO_ROOT` 读取。
无论使用默认值还是显式传入值，`CORE_REPO_ROOT`（包括相对路径）都会立即规范化为
绝对且存在的路径。

在完整核心 checkout 内构建时，`CORE_REPO_ROOT` 默认是 `docker/` 的父目录：

```bash
cd /path/to/3D_Recognization
bash docker/build.sh
```

## 仅更新 processor 代码

如果本地已存在 `harbor-cn.lingmouai.com/asu/global-id-mapping:4.0`，且改动仅为
`docker/processor.py`，可从该镜像派生一个小层镜像，不会重新装配 DA3 权重或 Python
环境：

```bash
bash docker/build_code_update.sh
```

默认输出为 `global-id-mapping:4.0-traceback`，原 `4.0` 镜像和运行中的容器不会改变。
可通过 `BASE_IMAGE` 与 `IMAGE_TAG` 覆盖输入和输出 tag。若 `main.py`、`src/`、`utils/`、
DA3 或 SAM3 源码也有改动，必须改用完整的 `build.sh`。本派生镜像同时更新
`processor.py` 与 `cos_upload.py`。

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
- `$CORE_REPO_ROOT/sam3/checkpoints/sam3.pt`。

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

COS 上传使用官方 `cos-python-sdk-v5` 的 `CosS3Client.put_object`，该依赖已由根
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

客户端从 `<dataset>/images/` 和 `--classifier-result` 中读取相同数字 frame ID 的文件，
POST 到本机服务，并将响应中的 `global_skus` 写为 `global_skus.json`。Viewer ZIP 不经 BSON 返回，
由独立 Viewer 依据 `taskID` 从 COS 下载。

## Viewer Bundle

此 Docker 服务不构建、携带或托管可视化页面。它只生成平铺、非加密 `ZIP_STORED` schema 3.0.0 的
`viewer_bundle.zip` 并上传 COS；`global_skus` 只保留在 BSON 成功响应中，不写入 COS。独立 Viewer
依据 `taskID` 直接定位并下载该 ZIP。

可视化代码位于独立的 `visualization` 分支，且该分支只包含 `viewer/`。Viewer 从页面 URL 读取
`recognition_task_id`，以它定位对应的 COS `viewer_bundle.zip` 后在浏览器渲染；其 `viewer/.env` 只配置
公开的 COS 基址，不包含也不应包含 `COS_SECRET_ID` 或 `COS_SECRET_KEY`。Docker 镜像和该服务运行时均不依赖
该前端目录或其配置。

```bash
uv run python docker/test_api.py \
  --dataset /path/to/dataset \
  --classifier-result /path/to/classifier/detections \
  --taskID task-01
```

默认输出目录是 `<dataset>/docker_mapping_response/`；可用 `--output-dir` 指定其他路径。
