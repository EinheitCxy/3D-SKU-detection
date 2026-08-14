# 3D 展示优化研究建议

日期：2026-08-14
基线：`main@39dc91f`（`origin/area_prediction`）
研究方式：本地源码审计 + 3 个 `gpt-5.6-sol` 只读 agents + 官方网页/源码资料

## 结论先行

研究阶段原推荐“Viser 增量 + 2D/3D 证据抽屉”。Rick 在 2026-08-14 审阅后，明确选择进入 **Python artifact producer + TypeScript/Three.js viewer** 的混合实施路线，以获得更强的产品化布局、交互和长期前端可维护性。

原风险判断仍然成立，因此实施顺序不是直接重写 UI：先由 Python 建立严格、版本化、fail-closed 的静态 bundle，再由 TypeScript 读取。这样保留 DA3/schema-v2、formal footprint generation 和 provenance 的唯一权威，同时避免把旧 Viser cache 中的 VGGT 路径假设带入新 viewer。

### Rick 确认后的实施决策

1. Python 继续拥有 DA3、SAM3、匹配、去重、面积计算、CURRENT/manifest 校验和 provenance；不在浏览器重算面积。
2. TypeScript + Three.js 拥有点云渲染、布局、selection、Raycaster、相机、状态栏和 evidence drawer。
3. 双方通过静态 bundle v1 通信，不引入 live WebSocket/Python callback；详细契约见 `typescript-viewer-implementation-plan.md`。
4. 首版只接入正式 ground footprint；未跟踪 front-facing area 草案不进入 KPI。
5. Potree 继续保留为超大点云需要 octree/LOD 后的备选，而不是首版依赖。

推荐体验组合：

1. 3D 审查驾驶舱：中性 RGB 全景作为主视图，选中 SKU 后高亮其点云/轮廓，其他对象降饱和或隐藏。
2. 证据抽屉：选中后显示 global ID、观测图片/object、面积指标、formal status、shadow evidence status、run/generation/provenance。
3. 轻量总览卡：显示数据集、重建 backend、点数、global ID 数、正式 footprint status；不把实验性 facing area 当正式 KPI。
4. 相机聚焦：从对象 AABB/包围球计算 look-at 和距离，平滑移动；普通 orbit/pan/zoom 始终可用。
5. 证据颜色固定：青色代表 front-facing projected area，琥珀色代表 ground footprint；不共用排行榜、不相加。

## 当前系统的阻塞点

### Viewer 输入链不统一

当前主链是：

```text
main.py
  -> modules.viewer_runner.run_viewer
  -> build_viewer_cache
  -> load GLB/NPZ + downsample
  -> assign global IDs
  -> pcd_gid.npz / global_object_index.json
  -> ViserViewer.start
```

但 CLI、交互入口、cache builder 和 ID assignment 对 `vggt_cache`、`pi3_cache`、`da3_cache` 的假设不一致；DA3 schema-v2 使用 `world_points_conf`，viewer 在线分配路径强制取 `conf`。因此必须先加入 `ViewerInputResolver` 和 backend-neutral `SceneBundle`。

### 正式面积结果还没有进入 viewer

正式 footprint 使用不可变 generation、`CURRENT`、manifest、`measurement_report.json` 和 `footprints.geojson`，并且 fail-closed；viewer 当前不解析这些产物。建议加入只读 `MeasurementOverlayAdapter`，仅负责解析和坐标变换，不重算几何。

### front-facing area 目前不能作为正式 KPI

未跟踪的 `facing_area` 草案会写 `facing_area_m2/facing_share`，但没有正式 status/provenance/immutable publication/geometry；并且当前草案构造 transforms 后没有传入 bbox 3D extractor，存在原图 bbox 与 processed grid 错位风险。因此首阶段只能把它显示为 `experimental/unverified`，或暂不接入正式指标卡。

## 候选路线比较

评分为 1-5，越高越好；成本和风险列为反向评分。视觉质量和长期成本属于方案判断，不是 benchmark 结果。

| 路线 | 视觉质量 | 操作性 | 当前项目匹配 | 实现成本 | 数据契约风险 | 建议 |
|---|---:|---:|---:|---:|---:|---|
| Viser 增量 + 证据抽屉 | 4 | 5 | 5 | 4 | 4 | 研究原推荐，保留为调试工具 |
| Python bundle + Three.js 前端 | 5 | 5 | 5 | 3 | 4 | Rick 确认的实施路线 |
| Potree 大点云路线 | 3 | 5 | 3 | 2 | 2 | 规模超出 Viser 后再评估 |

### 路线 A：Viser 增量

优点：复用 Python 状态、已有 Viser GUI、global ID 和点云缓存；选择、标签、相机、可见性可渐进增强。缺点：高级布局和品牌化能力受 Viser GUI 限制，频繁全量点云更新会受 WebSocket/浏览器反序列化影响。

### 路线 B：Three.js 独立前端

优点：GLB、scene graph、Raycaster、SelectionBox、clipping 和自定义 CSS UI 都更灵活。缺点：必须新建前端 schema、资源服务、状态同步、打包和测试体系；它只提供渲染工具，不提供本项目的 evidence/provenance 语义。

### 路线 C：Potree

优点：octree、point budget、LOD、clipping、measurement 和 annotation 适合超大点云。缺点：需要转换格式、独立 JS viewer 和 Python↔前端协议；对当前 SKU 对象级证据展示来说过重。

## 推荐的实现-ready 边界

以下边界已被收敛为实施计划，具体任务见 `typescript-viewer-implementation-plan.md`：

```text
Python WebViewerBundleExporter
  ├─ validate schema-v2 DA3 cache and formal CURRENT generation
  ├─ deterministic voxel sampling and little-endian arrays
  ├─ export object index + formal footprint evidence
  └─ publish bundle v1 without geometry recomputation

TypeScript BundleLoader
  ├─ validate manifest/version/byte lengths
  ├─ load typed arrays + objects + footprint evidence
  └─ fail closed on malformed or mismatched data

Three.js Viewer
  ├─ RGB point cloud + OrbitControls + camera framing
  ├─ footprint meshes + Raycaster + global ID selection
  └─ status bar + object list + evidence drawer
```

## 交互流程

### 首屏

- 中性 RGB 点云和相机/坐标参考。
- 顶部状态：dataset、backend、点数、global IDs、formal artifact status。
- 左侧或 GUI folder：Global ID 搜索/排序、显示模式、点云预算/采样、相机开关。
- 不默认用彩色面积热力图，避免用户把颜色当作置信度或真值。

### 选中 SKU

- 通过 dropdown、Prev/Next、Pick 或 rect-select 进入。
- 选中对象使用高亮色；其他点云淡化或切换 visibility。
- 相机平滑聚焦到对象中心。
- evidence drawer 显示：global ID、观测数、image/object IDs、active/removed、面积指标、单位、formal status、shadow status、run/generation。

### 进入证据

证据链必须能从数字回到对象，再回到观测：

```text
value
  -> metric / unit / formal status
  -> run or generation / manifest
  -> polygon or object points
  -> global_id
  -> image_id / object_id / bbox / mask
  -> source image and digest
```

`null` 显示为 unavailable/rejected，不显示为 `0 m²`。shadow evidence 只能作为附加诊断，不能改变正式面积状态。

## 首阶段不做

- 不引入 React、Potree、WebSocket 或新的 Python live 服务端；Three.js 是已确认的首版渲染层。
- 不在 viewer 中重新拟合平面、重新计算 convex hull/union 或修补缺失数据。
- 不把未跟踪 `facing_area` 草案直接写成正式 KPI。
- 不在 GUI callback 中同步执行 Poisson meshing 或其他长任务。
- 不为每个点/每条证据建立大量 scene nodes。
- 不在没有点云规模证据时引入 octree/格式转换。

## 最小验证矩阵

| 层 | 验证 | 失败含义 |
|---|---|---|
| 输入解析 | Pi3/DA3/可用 VGGT 路径、字段别名、非连续 image IDs、missing artifact | viewer 不应猜路径或静默回退错误数据 |
| SceneBundle | points/colors/conf/frame shape、global ID 对齐、坐标系和单位 | 拒绝启动或显示明确错误 |
| Overlay | CURRENT、manifest、accepted/rejected、GeoJSON 局部平面到世界坐标 round-trip | 不显示 partial/不可信轮廓 |
| 选择 | click、rect-select、visible-only/through-selection、global ID 聚合 | 不得把点选择误当对象真值 |
| 相机 | AABB/包围球 look-at、相机平滑聚焦、普通 orbit 不被抢占 | 交互回归 |
| 性能 | 合成 10k/100k 点、稳定 chunk/LOD、筛选不重复全量上传 | 决定是否需要 Potree/前端路线 |
| CLI | `main.py --mode viewer --help`、路径矩阵、无 GPU smoke | 验证编排而非重建质量 |
| 证据 | rejected/unavailable/shadow 状态和 provenance 可见 | 防止 KPI 误读 |

## 外部来源

详细来源见 [sources/official-viewers-2026-08-14.md](sources/official-viewers-2026-08-14.md) 和三份 sol agent receipts。主要官方来源：

- <https://viser.studio/main/api/core/scene_api/>
- <https://viser.studio/main/examples/interaction/scene_pointer/>
- <https://viser.studio/main/performance_tips/>
- <https://docs.nerf.studio/developer_guides/viewer/viewer_control.html>
- <https://github.com/potree/potree>
- <https://threejs.org/docs/pages/Raycaster.html>

## 下一步决策

研究阶段已完成，Rick 已确认混合 TypeScript 路线并授权在当前 `main` 上实施。当前执行依据为 `typescript-viewer-implementation-plan.md`；首版只接入正式 footprint，并以精简 contract/build 验证优先。
