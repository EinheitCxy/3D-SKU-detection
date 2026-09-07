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

# 先将大型 Excel 转为构建期窄 CSV；该文件不由浏览器加载
uv run python scripts/convert_sku_maindata.py \
  --input sku-maindata.xlsx --output runtime/sku_masterdata.csv

# 直接导出 minimal Viewer bundle；不需要先运行 ground-stack-area
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2 \
  --viewer-web-sku-masterdata-csv runtime/sku_masterdata.csv

npm --prefix modules/viewer_web run dev
```

默认 bundle 写入 `modules/viewer_web/public/data/`，Vite 的 `/data/` 直接对应这里。使用 `--viewer-web-output <dir>` 时，部署层必须把该目录挂载或 serve 到 `/data/`；Vite 不会自动发现自定义路径。

已有 bundle 可用下列命令补充主数据，不重跑点云导出：

```bash
uv run python scripts/publish_viewer_masterdata.py \
  --viewer-output modules/viewer_web/public/data \
  --sku-masterdata-csv runtime/sku_masterdata.csv
```

脚本复制当前 run 后写入 `sku_masterdata.json`，仅在新 run 完整就绪后才更新 `CURRENT`。

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
- `sku_masterdata.json` 只包含本 run 的 SKU ID。每项提供 `manufacturer`、`brand`、`category`（可为 null）和布尔 `is_posm`；导出会拒绝缺少任一 bundle SKU 的窄 CSV。

canonical “其他品类”是 `sku_id=56642`、`sku_name=其他品类`。只要存在任一具体 SKU，具体 SKU 按既有 confidence/support 顺序排在其他品类之前；只有全部有效观测都是其他品类时，56642 才能排在首位。Viewer 只接收排序后的 SKU ID/名称，不接收或显示 confidence。

导出仍从匹配产物传播实例点标签并执行统一点云过滤；matching 的 processed-mask cache 缺失时 export 会 fail closed，但 Viewer bundle 自身不包含该 cache。

## 产品交互

- 界面保留英文，采用浅灰侧栏、白色内容卡片和橙色选中态；筛选列表名称与数量分列，固定自然行高，右侧 SKU 名称、编号、数量分级呈现。键盘焦点清晰可见，并尊重减少动画的系统设置。
- 页面不显示 Dataset、Backend 或 Points 顶部摘要；页面采用白色场景背景且不显示地面网格。
- 默认 `Select by SKU`；它与 `Select by Global ID` 互斥，切换会清除上一选择。
- 选择栏只用更小的统计卡片显示 Total 和从 observations 派生的 Removed。
- 选择模式提供 Manufacturer、Brand、Category、SKU 和 Global ID。前四类按对象的主 SKU 聚合计数并高亮全部匹配 Global ID；canvas 点选自动切换到 Global ID。
- 点击 Manufacturer、Brand、Category 或 SKU 的某一项后，右栏显示该维度和值、匹配 SKU/Global ID 总数，以及每个主 SKU 的名称、独立次级编号和对应 Global ID 数量；维度、已选值、汇总和 SKU 行使用不同层级的文字与计数标记。
- `View Controls` 默认折叠；展开后只提供 Fit、Top、Iso 和 Point size，默认点大小为 `0.005`，可按 `0.001` 调整，范围为 `0.004–0.070`；折叠时不会遮挡 canvas。
- Global ID 面板的搜索、匹配计数、列表与导航控件使用较小字号；左栏只显示 `Total` / `Removed`。Selected Object 为主 SKU 显示厂商、品牌、品类，POSM 固定显示 `N/A`。
- 右栏标题为 `Selected Object`，显示 Global ID、该对象由 observations 派生的 Observations / Active / Removed、按发布顺序排列的 SKU，以及紧凑的三列优先 observation thumbnail grid；每张图片都是导出器生成的精确 `128×128` 等比补深色背景 JPEG，卡片仍显示 image/object ID，removed 卡片灰化。
- Focus 始终使用对象的 `point_ranges`，不会复制点云 geometry。

Viewer 不包含 footprint、source provenance、hash/filter metadata 或 confidence 字段；也不加载这些旧 rich-contract 产物。

## 验证

```bash
npm test -- --run
npm run build
```

Manufacturer / Brand display normalization: `冲劲`, `沖劲`, `100冲劲`, and `100沖劲` are grouped as `其他` when loading master data, consistently across filters, counts, and object details. Source spreadsheet and bundle files remain unchanged.

SKU Breakdown rows are clickable: expand a SKU to show its Global IDs in numeric ascending order within the selected Manufacturer / Brand / Category. Click an ID to highlight it in orange and select its scene points while keeping the SKU breakdown visible. Clicking the expanded SKU again collapses it.

View controls: Fit frames the scene from the front at eye level, Top looks straight down along +Y, and Iso views from a diagonal side at 45° elevation.
