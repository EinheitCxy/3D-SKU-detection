# SKU Global ID Mapping 服务

## 概述

这是一个 **HTTP API 服务**（FastAPI），对同一零售场景的多张图片执行跨图 SKU 匹配、去重，并为每个物理商品分配跨图唯一的 `global_id`。

服务以 **BSON** 编码通过 `POST /api` 收发数据，内部驱动 `code/` 系统完成三步流程：

1. **Pi3 3D 重建**：多视图场景重建（点云 + 相机位姿）
2. **3D 投影匹配**：mask 引导采样 + 3D→2D 投影，确定跨图同一物理对象
3. **并查集去重**：传递性匹配聚类，每个连通分量 = 一个 `global_id`

> ⚠️ README 早期版本描述的 `docker run ... --images ... --skus ... --output ...` CLI 模式**已不存在**。当前仅提供 HTTP API。本文档已与 `api.py` / `processor.py` / `build.sh` 对齐。

## 运行要求

- **GPU 必需**（NVIDIA 驱动 + nvidia-container-toolkit）。`processor.py` 中 `device='cuda'` 硬编码。
- 基础镜像为私有 Harbor 镜像 `harbor-cn.lingmouai.com/asu/pricetag_ocr_recognition:...`（CUDA 12.1 + PyTorch），需内网访问。
- `processor.py` 硬编码 Pi3 权重路径 `/app/Pi3/checkpoints/snapshots/.../model.safetensors`；SAM3 路径来自 `config.yaml`。迁移部署位置需同步更新。

## API 契约

### `POST /api`

- **Content-Type**：`application/bson`（或任意，body 当作 BSON 解析）
- **请求体**（BSON）：

```jsonc
{
  "images": [<bytes>, <bytes>, ...],   // 图片二进制，每张一个元素
  "skus":   ["<json_str>", "<json_str>", ...]  // 每张图对应的检测结果 JSON 字符串
}
```

约束：
- `len(images)` **必须** == `len(skus)`，否则返回 500。
- 每个 `skus[i]` 是一个 JSON 字符串，解析后为 dict，期望含 `objects` 字段。
- `skus[i]` 接受 `code/utils/data_utils.py::load_detections` 支持的三种格式：
  - `{"skus": [{"classes": {...}, "objects": [{"position":[x1,y1,x2,y2], "classes":{"det":..}, "confidences":{"det":..}}]}]}`（推荐，与 `imdata/floor_display*/detections_results` 一致）
  - `{"objects": [...]}`（直接 dict）
  - `[{...}]`（列表，取首元素）
- 文件名编号约定：服务内部按 `0,1,2,...` 顺序保存图片与检测，因此**顺序即索引**，需与图片顺序对齐。

- **响应体**（BSON，HTTP 200）：

```jsonc
{
  "global_skus": ["<json_str>", "<json_str>", ...]  // 每张图一个 JSON 字符串
}
```

每个 `global_skus[i]` JSON 字符串含：
- `global_id`：跨图唯一 ID
- `is_deduplicated`：`true` 表示该物体在更早图片中已出现（被去重）

- **错误**：HTTP 500，响应体为 Python traceback 文本。

### 静态前端

`/` 挂载 `static/` 目录（`StaticFiles(html=True)`），提供一个简单的 Web UI。

## 构建与运行

### Docker（生产）

`build.sh` 三步：构建镜像 → 启动容器 → 推送 Harbor：

```bash
bash build.sh
# 等价于：
# docker build -t harbor-cn.lingmouai.com/asu/global-id-mapping:3.1.0 .
# docker run -d --name global-id-mapping -p 8011:80 --gpus all harbor-cn.lingmouai.com/asu/global-id-mapping:3.1.0
# docker push harbor-cn.lingmouai.com/asu/global-id-mapping:3.1.0
```

服务在**容器 80 端口**监听，host 映射 **8011**。

### 本地开发

```bash
uv run python api.py   # uvicorn 0.0.0.0:8010
```

### 端口速查（容易搞错）

| 入口 | 端口 |
|---|---|
| `api.py` `__main__`（本地开发） | 8010 |
| `build.sh` host 映射 | 8011 → 容器 80 |
| `Dockerfile` CMD | 80 |
| `test_api.py` 目标 | 8011 |
| `test_connection.py` 目标 | 8010 |

## 测试客户端

```bash
uv run python test_api.py          # 打容器（localhost:8011），返回 global_skus
uv run python test_connection.py   # 打本地开发（localhost:8010）
```

## 处理流程（`processor.py::process`）

1. 解析 BSON 输入；校验 `len(images)==len(skus)`，每个 `skus[i]` 为 dict。
2. 将图片 `cv2.imdecode` 后保存为 `<idx>.jpg`，检测 JSON 保存为 `<idx>.json` 到临时 `dataset/{images,detections_results}/`。
3. `SKUDetectionMain.run_reconstruction(backend='pi3', device='cuda')` → Pi3 重建。
4. `run_sku_matching(algorithm='3d', batch_all_refs=True, backend='pi3')` → 3D 匹配。
5. `run_dedup_sequence()` → 并查集 `global_id` 分配，输出 `dedup_detections/{global_skus.json, global_mapping.json}`。
6. 读取 `global_skus.json` 作为返回；清理临时目录。

## GPU / 故障排查

- **GPU 必需**：`--gpus all` 启动容器；本地需可见 CUDA 设备。
- **cuDNN 初始化失败**（`CUDNN_STATUS_NOT_INITIALIZED`）：多为选到被占用的卡。指定空闲卡：`CUDA_VISIBLE_DEVICES=<n> ...`。
- **500 + traceback**：常见为图片数量与 skus 数量不一致、skus JSON 非法、或 Pi3 权重路径缺失。
- **权重缺失**：确认容器内 `/app/Pi3/checkpoints/.../model.safetensors` 存在（镜像内已 COPY）。

## 说明

- `processor.py` 通过 `sys.path` 插入 `Global-ID-Mapping/code/` 后 `from main import SKUDetectionMain`，**进程内**驱动 `code/` 系统（非子进程）。
- `Global-ID-Mapping/code/` 是根 `code/` 的副本（Docker 自包含所需），变更需手动同步。
