# 3D Pipeline Shell Entrypoints

## 视频到 Viewer

`video_to_viewer.sh` 是从原始视频到产品 Viewer 的 canonical shell 入口：

```text
video
  -> 0-based frames
  -> SKU detector
  -> DA3 reconstruction + SAM3 matching + personalcare classifier
  -> dedup/global ID
  -> minimal schema 3.0.0 viewer bundle
  -> optional Vite server
```

基本用法：

```bash
bash scripts/3d/pipeline/video_to_viewer.sh \
  --video /path/to/video.mp4 \
  --fps 2.0 \
  --gpu 2 \
  --classifier-device cuda:0 \
  --serve \
  --host 127.0.0.1 \
  --port 5173
```

参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--video PATH` | 必填 | 输入视频；相对路径按仓库根解析 |
| `--fps FPS` | `2.0` | 每秒目标抽帧数 |
| `--gpu INDEX` | `0` | 写入 `CUDA_VISIBLE_DEVICES` 的物理 GPU 编号 |
| `--detections-dir DIR` | 无 | 复用已有逐帧 JSON；省略时运行 detector |
| `--detector-device DEVICE` | `cpu` | SKU detector 的设备 |
| `--classifier-device DEVICE` | `cuda:0` | GPU mask 后 classifier 使用的设备 |
| `--save-root DIR` | `Output` | reconstruction、matching、classification、dedup 的输出根 |
| `--viewer-output DIR` | `modules/viewer_web/public/data` | Viewer bundle 发布目录 |
| `--serve` | 关闭 | 导出后以前台 Vite 进程提供页面 |
| `--host HOST` | `127.0.0.1` | Vite 监听地址；对外暴露需显式改为 `0.0.0.0` |
| `--port PORT` | `5173` | Vite 监听端口 |
| `--help` | - | 输出内置参数说明 |

`--gpu 2 --classifier-device cuda:0` 表示把物理 GPU 2 映射成进程内的第一张卡。若指定 `--serve`，`--viewer-output` 必须使用默认目录，因为 Vite 只把 `modules/viewer_web/public/data` 映射到浏览器 `/data/`；自定义目录需要部署层另行挂载。

脚本复用 `modules/video_to_dedup/run.sh` 完成抽帧、检测、DA3 pipeline 和 dedup，然后直接调用 `main.py --mode viewer-web` 发布 minimal schema 3.0.0 bundle。独立的 `main.py --mode ground-stack-area` 仍可按需运行，但不是视频到 Viewer 的前置阶段。

只导出、不启动服务：

```bash
bash scripts/3d/pipeline/video_to_viewer.sh \
  --video /path/to/video.mp4 --fps 2.0 --gpu 2

npm --prefix modules/viewer_web run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

已有检测结果：

```bash
bash scripts/3d/pipeline/video_to_viewer.sh \
  --video /path/to/video.mp4 \
  --detections-dir /path/to/detections_results \
  --gpu 2 --serve
```

## Minimal Viewer bundle

发布目录使用不可变 `CURRENT -> runs/<run_id>/`。`CURRENT` 仅包含 `run_id`；run 内的 `manifest.json` 固定为 schema `3.0.0`，包含真实 `dataset_name`、`frame_count`、`display_bounds` 和 `world_to_view`。固定二进制文件为 `positions.f32.bin`、`colors.u8.bin`、`normals.i8.bin`，`point_count` 由 positions 长度推导。

`objects.json` 按 global ID 提供 `ordered_skus`、`point_ranges` 与必填 `observations`。每个
observation 只包含 `image_id`、`object_id`、`removed` 和 bbox crop 的相对 `thumbnail` 路径；
缩略图最长边为 256px。canonical “其他品类”是 `sku_id=56642`、`sku_name=其他品类`；只要存在
任一具体 SKU，其他品类排在具体 SKU 之后，只有全为其他品类时才排在首位。Viewer 不接收或
显示 confidence。

产品界面显示 `Dataset`（`dataset_name · frame_count frames`）和点数；默认 `Select by SKU`，与
`Select by Global ID` 互斥，切换会清除上一选择。SKU 选择保留完整场景并批量 magenta 高亮；
canvas pick 自动切换为 Global ID。`View Controls` 默认折叠，展开后只有 Fit、Top、Iso 和
Point size。右栏 `Selected Object` 显示 Global ID、按发布顺序排列的 SKU、observation
Active/Removed 汇总和 thumbnail grid。

Viewer bundle 不包含或依赖 footprint、evidence、hash/provenance、source digest、confidence
字段或其他审计型 rich-contract 元数据；只保留产品缩略图所需的最小 observation 标识。

## Dataset 批处理

- `batch_run_pipeline.sh`：对配置范围内的 `floor_display*` 运行默认 DA3 pipeline。
- `batch_pipeline_backend.sh`：显式选择 backend、dataset 范围、GPU 和隔离的 save root。

这两个脚本接收已经准备好的 dataset，不负责抽帧、检测、Viewer export 或 Web 服务。`ground-stack-area` 仍是可独立调用的后端计量阶段。
