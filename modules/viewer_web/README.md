# Web Viewer

这是静态 TypeScript/Three.js Viewer。它不运行 DA3、SAM3 或 Python pipeline；它只严格加载 Python 已发布的 schema `2.0.0` bundle。

## 导出与启动

从仓库根执行：

```bash
# 1. 先完成全量 batch-all-refs matching（matching 是唯一 SAM3 producer）
uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d \
  --recon_backend da3 --match_backend da3

# 2. 再生成 formal footprint（只读 v2 masks；不运行 SAM3）
uv run python main.py --mode ground-stack-area \
  --dataset imdata/floor_display2

# 3. 最后导出 Viewer bundle（只读 v2 masks；不运行 SAM3）
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2
npm --prefix modules/viewer_web run dev
```

默认 bundle 写入 `modules/viewer_web/public/data/`，Vite 的 `/data/` 直接对应这里。使用 `--viewer-web-output <dir>` 时，部署层必须把该目录挂载或 serve 到 `/data/`；Vite 不会自动发现自定义路径。

导出需要同一 dataset 已有：

- `Output/<dataset>/da3_cache/predictions.npz`
- `Output/<dataset>/dedup_detections/global_mapping.json`
- `Output/<dataset>/sam3_mask_cache/v2/`（matching 已完整发布的 processed-space self-exemplar masks）
- `Output/<dataset>/ground_stack_footprint/CURRENT`
- `<dataset>/images/`

matching 的 master gate `enable_sam3_mask_sampling` 默认 true；此时 only self-exemplar，mask 用 little-endian `np.packbits` 无损保存。gate 为 false 时没有 canonical cache，因此 footprint/export 必须 fail closed。Viewer exporter 不会在 cache miss 时加载或推理 SAM3。

bundle 使用不可变 `CURRENT -> runs/<run_id>/`，schema 固定为 `2.0.0`，formal metric 固定为 `da3_self_exemplar_ground_footprint_union`。loader 对 manifest、provenance、二进制数组长度、坐标变换、缩略图与 footprint status 严格校验；旧 schema `1.0.0`、旧面积结果、`sam3_mask_cache/v1` 均不兼容，既不读、迁移、复制也不删除，必须从完整 matching 重新按上述顺序生成。

## 点云过滤

Viewer 不使用 SAM3 protection mask。exporter 从 matching 的 v2 processed-space masks 直接赋实例标签，但不会让任何点绕过离群、地面或天空过滤；同一过滤规则适用于所有展示点。

## SKU 浏览与点选

左侧显示 `Total`/`Visible`，SKU facet 只按每个 global ID 的 primary candidate 计数；`显示所有` 可清除 SKU facet，搜索框仍按 global ID 过滤。厂商、品牌、品类显示 `主数据待接入`，POSM、价签、空缺位显示 `检测能力待接入`，这些占位项不可用。右侧选中详情按发布顺序显示所有 `sku_id · sku_name` candidate，不显示 confidence。

点云和 footprint 共用一个 global-ID pick handler。点云通过按 `point_index_range` 排序的二分查找解析归属；footprint 优先，隐藏 ID 不会被选中。过滤只更新一个运行时 `Uint8` visibility attribute 并同步 footprint `visible`，不会复制点云 geometry；shader 会丢弃隐藏点，选中点继续使用现有 magenta 高亮。

## 验证

```bash
npm test -- --run
npm run build
```
