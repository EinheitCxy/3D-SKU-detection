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
├── sam3_mask_cache/v2/
│   ├── entries/<image_id>/{manifest.json,masks.npz}
│   ├── locks/
│   └── corrupt/
└── ground_stack_footprint/CURRENT -> runs/<run_id>/
```

## Footprint 与 SAM3 cache

运行顺序是 canonical contract：先让 matching 完成 **全部** `batch_all_refs` references，再运行 `--mode ground-stack-area`，最后运行 `--mode viewer-web`。matching 是唯一的 SAM3 producer；它在默认 `enable_sam3_mask_sampling: true` 下以 self-exemplar 生成每个 frame 的完整 processed-space masks。master gate 为 false 时 matching 走既有 bbox sampling 且不发布 cache，两个 consumer 必须 fail closed。

v2 cache 的每个 payload 是 processed DA3 grid 上 `(object_count, height, width)` bool masks，以 little-endian `np.packbits` 无损打包到 `masks.npz`；manifest 以 `object_id` 绑定 mask，而非依赖数组位置。partial、mismatched 或 corrupt frame 不能命中。`sam3_mask_cache/v1` 与 v2 不兼容，v1 从不读取、迁移、复制、fallback 或自动删除。

`--mode ground-stack-area` 只读取 matching 已发布的 v2 cache、metric DA3 cache 与去重映射，不导入、加载或推理 SAM3，也不会因 cache miss 重算。它为每个 `global_id` 从全部有效观测重建 OBB，并在支撑平面上取 polygon union，得到 `da3_self_exemplar_ground_footprint_union`（m²）。缺少任何 canonical mask 或必要几何会发布 `rejected`/`null`，不会伪造部分结果。该 metric 是新 baseline，不能与旧 `da3_ground_footprint_union` 面积比较。

cache 不再是 Web Viewer 的 protection mask。Viewer export 也只读 v2 processed masks，绝不加载或推理 SAM3；它在常规点云过滤之后传播实例标签。任何点都不会因带有 SAM3 标签而跳过点云去噪、地面或天空过滤。

## Web Viewer

```bash
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2
npm --prefix modules/viewer_web run dev
```

exporter 只消费已发布的 DA3、去重、matching v2 cache 和 footprint 产物，写入 `CURRENT -> runs/<run_id>/` schema `2.0.0` bundle。前端严格校验 manifest、provenance、二进制数组长度、`world_to_view`、缩略图和 footprint 状态；bundle `1.0.0` 与旧 metric 被拒绝，必须按 matching → footprint → export 的顺序重新生成。

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
