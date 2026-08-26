# Web Viewer

这是静态 TypeScript/Three.js 产品 Viewer。它不运行 DA3、SAM3 或 Python pipeline，只加载 Python 已发布的 minimal schema `3.0.0` bundle。

## 导出与启动

从仓库根执行：

```bash
# 先完成 DA3 reconstruction、完整 batch-all-refs matching 和 dedup
CUDA_VISIBLE_DEVICES=2 uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d \
  --recon_backend da3 --match_backend da3 \
  --classifier-device cuda:0

# 直接导出 minimal Viewer bundle；不需要先运行 ground-stack-area
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2

npm --prefix modules/viewer_web run dev
```

默认 bundle 写入 `modules/viewer_web/public/data/`，Vite 的 `/data/` 直接对应这里。使用 `--viewer-web-output <dir>` 时，部署层必须把该目录挂载或 serve 到 `/data/`；Vite 不会自动发现自定义路径。

从视频开始的一条命令入口：

```bash
bash scripts/3d/pipeline/video_to_viewer.sh \
  --video /path/to/video.mp4 --fps 2.0 --gpu 2 \
  --classifier-device cuda:0 --serve
```

该脚本在 dedup 后直接 export，成功后才启动 Vite。独立的 `main.py --mode ground-stack-area` 仍可按需运行，但不是视频到 Viewer 的前置阶段。

## Minimal schema 3.0.0

发布采用不可变 `CURRENT -> runs/<run_id>/`：

- `CURRENT` 只包含非空 `run_id`。
- `manifest.json` 只包含 schema `3.0.0` 所需的 `dataset_name`、非空 `backend`、`frame_count`、`display_bounds` 和 `world_to_view`。
- 固定二进制文件为 `positions.f32.bin`、`colors.u8.bin`、`normals.i8.bin`；`point_count` 由 positions 长度推导。
- `objects.json` 每个 global ID 只包含 `ordered_skus`、`point_ranges` 与必填 `observations`；每个 observation 只有 `image_id`、`object_id`、`removed` 与 `thumbnail`，未知扩展字段会被忽略。

canonical “其他品类”是 `sku_id=56642`、`sku_name=其他品类`。只要存在任一具体 SKU，具体 SKU 按既有 confidence/support 顺序排在其他品类之前；只有全部有效观测都是其他品类时，56642 才能排在首位。Viewer 只接收排序后的 SKU ID/名称，不接收或显示 confidence。

导出仍从匹配产物传播实例点标签并执行统一点云过滤；matching 的 processed-mask cache 缺失时 export 会 fail closed，但 Viewer bundle 自身不包含该 cache。

## 产品交互

- 顶部 `Dataset` 显示 `dataset_name · frame_count frames`，同时显示点数。
- 顶部显示 `Backend · DA3`（值来自 manifest 的非空 backend 字段）。
- 默认 `Select by SKU`；它与 `Select by Global ID` 互斥，切换会清除上一选择。
- 选择栏保留 Total / Visible，并显示从 observations 派生的全局 Observations / Active / Removed；厂商、品牌、品类、POSM、价签、空缺位仅作为禁用占位，不推断数据。
- SKU 选择保留完整场景，并批量以 magenta 高亮匹配的 Global ID；canvas 点选自动切换到 Global ID。
- `View Controls` 默认折叠；展开后只提供 Fit、Top、Iso 和 Point size，折叠时不会遮挡 canvas。
- 左栏显示从 object index 与当前 scene 可见集合推导的 `Total` / `Visible`。
- 右栏标题为 `Selected Object`，显示 Global ID、该对象由 observations 派生的 Observations / Active / Removed、按发布顺序排列的 SKU，以及 observation thumbnail grid；卡片显示 image/object ID，removed 卡片灰化。
- Focus 始终使用对象的 `point_ranges`，不会复制点云 geometry。

Viewer 不包含 footprint、source provenance、hash/filter metadata 或 confidence 字段；也不加载这些旧 rich-contract 产物。

## 验证

```bash
npm test -- --run
npm run build
```
