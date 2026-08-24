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

根 `pyproject.toml` 是 DA3/SAM3/Open3D 核心依赖。`modules/sku_detector/pyproject.toml` 是独立的 YOLO 依赖，二者不合并。

## 默认路径与后端

- 默认 dataset：`imdata/floor_display2`。
- 默认重建/匹配 backend：`da3`。
- 默认输出：根 `Output/`。
- 默认 viewer bundle：`modules/viewer_web/public/data`。
- 所有相对 `--save_root` 值按仓库根解析，而非调用终端的当前目录。
- DA3 runner 使用 `Depth-Anything-3/.venv/bin/python`；可用 `DA3_VENV_PYTHON` 覆盖。

```bash
uv sync --extra dev
uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d \
  --recon_backend da3 --match_backend da3
```

运行输出按数据集隔离：

```text
Output/<dataset>/
├── da3_cache/predictions.npz
├── output_3dmapping_da3/
├── dedup_detections/global_mapping.json
├── sam3_mask_cache/v1/
└── ground_stack_footprint/CURRENT -> runs/<run_id>/
```

## Footprint 与 SAM3 cache

`--mode ground-stack-area` 需要 metric DA3 cache、去重映射与可用 SAM3 checkpoint。每个 `global_id` 从全部有效观测重建 OBB，并在支撑平面上取 polygon union，得到 `da3_ground_footprint_union`（m²）。缺少任意必要几何会发布 `rejected`/`null`，不会伪造部分结果。

`sam3_mask_cache/v1` 是不可变分割 bundle cache；key 包含输入图、bbox、checkpoint 与运行/代码指纹。cache 失效或损坏会重算并记录事件；它从不退回为 bbox mask。

cache 不再是 Web Viewer 的 protection mask。Viewer export 不会因为某点带有 SAM3 标签而跳过点云去噪、地面或天空过滤；常规过滤对所有点一致。SAM3 仍用于 footprint 输入、实例关联和审计 provenance。

## Web Viewer

```bash
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2
npm --prefix modules/viewer_web run dev
```

exporter 只消费已发布的 DA3、去重和 footprint 产物，写入 `CURRENT -> runs/<run_id>/` bundle。前端严格校验 manifest、provenance、二进制数组长度、`world_to_view`、缩略图和 footprint 状态；契约不满足时 fail closed 并提示重新导出。

默认 `/data/` 由 `modules/viewer_web/public/data/` 提供。自定义 `--viewer-web-output` 不会自动被 Vite 服务，必须由部署层挂载到 `/data/`。

## 性能基线

完整 fd2–4 cold/warm 结果在 [perf/runs/20260824T032553Z/FINAL_REPORT.md](../perf/runs/20260824T032553Z/FINAL_REPORT.md)：GPU 1 的冷启动均值 752.03 s、warm 均值 309.65 s；footprint/SAM3 是主要瓶颈。GPU 2（24 GiB）出现容量 OOM，因此不混入均值。

## 回归验证

```bash
uv run --offline pytest tests/test_root_layout.py tests/test_da3_3d_reconstructor.py \
  tests/test_web_viewer_export.py tests/test_da3_import_isolation.py -q
(cd modules/viewer_web && npm test -- --run && npm run build)
bash -n modules/video_to_dedup/*.sh scripts/3d/{evaluation,ops,pipeline,tuning}/*.sh
```
