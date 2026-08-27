# 离线 Global-ID Mapping 服务

此目录将当前 DA3/SAM3/global-ID/Minimal Viewer 流水线封装为同步 BSON
`POST /api` 服务。镜像不包含 detector、personalcare classifier、Pi3、VGGT、任何
输入数据或运行产物；调用方必须提供每帧的原图和外部 classifier 结果。

## 离线构建

`build.sh` 将包装代码与核心运行代码分开定位：Dockerfile、`build.sh`、
`__init__.py`、`api.py`、`processor.py` 和 `request_runner.py` 始终从脚本所在的
`SCRIPT_DIR` 读取；pipeline 核心源码、`pyproject.toml`、`uv.lock`、DA3/SAM3 源码、
SAM3 checkpoint，以及 builder 的 `/workspace` 挂载均从 `CORE_REPO_ROOT` 读取。
无论使用默认值还是显式传入值，`CORE_REPO_ROOT`（包括相对路径）都会立即规范化为
绝对且存在的路径。

在完整核心 checkout 内构建时，`CORE_REPO_ROOT` 默认是 `docker/` 的父目录：

```bash
cd /path/to/3D_Recognization
bash docker/build.sh
```

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
docker run --rm --gpus all -p 8011:80 global-id-mapping:da3-self-contained
```

`build.sh` 会先用离线 base container 从根 `uv.lock` 生成可搬迁的统一 `.venv`，只安装
root base dependencies（不含 dev extras，且不把 editable 项目路径写进 venv），再用
BuildKit named contexts 装配最终镜像。运行时固定一份 `/app/.venv`、
`DA3_VENV_PYTHON=/app/.venv/bin/python`、离线 Hugging Face/Transformers，并且 Uvicorn
仅启动一个 worker。

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

服务接收如下 BSON 文档，`project_id` 必须为 `51`：

```text
{
  images: [<numeric-frame image bytes>, ...],
  skus: ["{classes: {det, cls}, objects: [...]}", ...],
  project_id: 51,
}
```

`skus[i]` 是 classifier 的逐帧 JSON，保留 `{classes, objects}`；每个 object 必须包含
det/cls 索引、det/cls confidence 和 bbox。服务将它规范化为当前 pipeline 所需的
object-level `classification`，不会在容器内执行分类器。

客户端从 `<dataset>/images/` 和 `--classifier-result` 中读取相同数字 frame ID 的文件，
POST 到本机服务，并将响应写为 `global_skus.json` 与 `viewer_bundle.zip`。它会验证 BSON
成功响应仅包含这两个字段，并验证 Viewer ZIP 的 `CURRENT` 以及固定 run members。

```bash
uv run python docker/test_api.py \
  --dataset /path/to/dataset \
  --classifier-result /path/to/classifier/detections
```

默认输出目录是 `<dataset>/docker_mapping_response/`；可用 `--output-dir` 指定其他路径。
