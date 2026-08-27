# 离线 Global-ID Mapping 服务

此目录将当前 DA3/SAM3/global-ID/Minimal Viewer 流水线封装为同步 BSON
`POST /api` 服务。镜像不包含 detector、personalcare classifier、Pi3、VGGT、任何
输入数据或运行产物；调用方必须提供每帧的原图和外部 classifier 结果。

## 离线构建

构建主机必须已经拥有下列本地内容：

- `harbor-cn.lingmouai.com/alg/sku-classifier-base:0.0.4`（linux/amd64）；
- `/home/xingyu/.local/bin/uv` 和完整 uv cache；
- 官方 `opencv-python-headless==4.11.0.86` Linux x86_64 wheel：
  `/data/www/comfyui/3d-recognition-build/runtime/wheels/opencv_python_headless-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`；
- DA3 Hugging Face cache（`refs/`、`blobs/` 和 snapshot
  `b2359bdf726fb44ef62acca04d629dcf158053e7`）；
- `sam3/checkpoints/sam3.pt`。

默认 DA3 cache 是
`/home/xingyu/.cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1`。
路径不同则在调用时显式覆盖 `DA3_MODEL_CACHE`；同理可覆盖 `HOST_UV`、
`UV_CACHE_DIR`、`IMAGE_TAG`、`BUILD_WORK_ROOT` 和 `OPENCV_WHEEL_DIR`。后者必须指向包含上述
官方 wheel 的现有目录；`build.sh` 会先把它规范化为绝对路径，再在创建临时 context 前检查精确文件名
是否存在，并以只读挂载覆盖 builder 的 `/workspace/docker/wheels` 相对 flat index。默认临时构建目录是
`/data/www/comfyui/3d-recognition-build`，必须已经存在且当前用户可写。构建需要约 6 GB
的 DA3 named context，且不会下载、
push、prune 或使用网络。

```bash
bash docker/build.sh
docker run --rm --gpus all -p 8011:80 global-id-mapping:da3-self-contained
```

`build.sh` 会先用离线 base container 从根 `uv.lock` 生成可搬迁的统一 `.venv`，只安装
root base dependencies（不含 dev extras，且不把 editable 项目路径写进 venv），再用
BuildKit named contexts 装配最终镜像。运行时固定一份 `/app/.venv`、
`DA3_VENV_PYTHON=/app/.venv/bin/python`、离线 Hugging Face/Transformers，并且 Uvicorn
仅启动一个 worker。

首次准备 wheel 后，后续构建始终使用 `--network=none`、冻结 lock 和项目 `[tool.uv].find-links`
flat index，不需要网络。`OPENCV_WHEEL_DIR` 必须为现有目录，脚本会在挂载前把它规范化为绝对路径：

```bash
OPENCV_WHEEL_DIR=/path/to/offline-wheels bash docker/build.sh
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
