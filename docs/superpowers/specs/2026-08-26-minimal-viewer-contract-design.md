# Minimal Viewer Product Contract

## Goal

将 Web Viewer 从审计型 evidence bundle 收缩为产品交互 bundle，只保留点云、真实数据集名、SKU/Global ID 选择、magenta 高亮、Focus、Selected Object 和折叠 View Controls。

## Bundle schema 3.0.0

`CURRENT`：

```json
{"run_id": "<non-empty string>"}
```

`manifest.json`：

```json
{
  "schema_version": "3.0.0",
  "dataset_name": "floor_display6",
  "frame_count": 11,
  "display_bounds": [0, 0, 0, 1, 1, 1],
  "world_to_view": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
}
```

`objects.json`：

```json
{
  "1": {
    "ordered_skus": [{"sku_id": "123", "sku_name": "产品"}],
    "point_ranges": [[0, 20]]
  }
}
```

固定二进制文件：`positions.f32.bin`、`colors.u8.bin`、`normals.i8.bin`。`point_count` 从 positions 长度推导。

## Minimal validation

仅保留避免运行时错误的检查：schema version、非空 dataset/run ID、frame count、六维有限 bounds、16 维有限矩阵、三组二进制 shape 一致、数字 global ID、SKU 字符串、point ranges 安全整数/边界/全局不重叠。忽略未知 JSON 字段，不做 exact-key、hash、provenance、source、filter、footprint 或 aggregate 重算。

## Classification rule

canonical other 是 `(sku_id="56642", sku_name="其他品类")`。Python 聚合保留所有 candidate；存在任一非 other 时，所有非 other 先按既有 confidence/support 排序，other 最后。全部有效观测都是 other 时才允许 other primary。Viewer 只接收已排序的 `sku_id/sku_name`，不接收或验证 confidence。

## UI

- Dataset：`floor_display6 · 11 frames`。
- 默认 `Select by SKU`，与 `Select by Global ID` 互斥；切换清除上一选择。
- SKU 选择保留完整场景并批量 magenta；canvas pick 自动切换到 Global ID。
- View Controls 默认折叠，只保留 trigger；展开显示 Fit/Top/Iso 和 Point size。
- 右栏改为 `Selected Object`，只显示 Global ID 与有序 SKU；无 thumbnails、formal footprint 或 evidence。
- Focus 永远使用 point ranges。

## Removed

删除 footprint 文件加载/mesh/picking/opacity、Evidence summary、thumbnails、instance observation、confidence fields、active/removed counts、source provenance、hash/filter/SAM3/DA3 contract 字段和重复 TypeScript aggregation comparator。后端 footprint stage 可独立保留，但 Viewer 不依赖它；video-to-viewer 脚本不再必须运行 footprint。

## Constraints

- 不新增测试文件，只修改现有测试。
- 不做旧 schema fallback 或兼容层。
- 不复制点云 geometry；继续使用 point ranges 增量更新颜色与可见性。
- 不修改原始 classification observations 或 global mapping。
