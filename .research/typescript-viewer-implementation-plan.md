# TypeScript 3D Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 DA3、SAM3、匹配、去重和面积算法的前提下，把 schema-v2 DA3 场景、global ID 对象索引和正式 ground footprint generation 导出为严格静态 bundle，并用 TypeScript + Three.js 提供美观、可操作、可审查的 3D Web viewer。

**Architecture:** Python 是唯一数据/证据生产端，读取并验证现有正式产物后生成浏览器友好的静态 bundle；TypeScript 是纯展示端，只解析 bundle、渲染点云与 footprint、维护选择/相机/UI 状态，不重新计算面积。前后端以版本化 `manifest.json` 为边界，不使用 Python live callback，也不复用带 VGGT 假设的旧 `viewer_cache`。

**Tech Stack:** Python 3.11、NumPy、pytest、TypeScript、Vite、Three.js、Vitest、原生 HTML/CSS。

**Spec:** `.research/recommendation.md` 中“Rick 确认后的实施决策”与本计划的 Global Constraints。

## Global Constraints

- Python 命令必须通过 `uv` 执行；本地环境验证统一使用 `UV_CACHE_DIR=/tmp/uv-cache-3d-viewer uv run --offline --no-sync ...`，不得触发模型、数据或 GPU 下载。
- 不修改 DA3、SAM3、匹配、去重、正式 footprint 计算和 accepted/rejected 判定；viewer 只读正式 generation。
- `da3_ground_footprint_union` 的 `rejected`/`unavailable` 必须显示为 `null`/“—”，绝不能显示为 `0 m²`。
- 首版不接入未跟踪 `facing_area` 草案；青色仅保留为未来 front-facing projected area 的视觉语义，琥珀色专用于正式 ground footprint。
- 不添加 backend alias、旧 schema fallback、polyfill 或兼容分支；输入不满足 schema-v2/bundle-v1 时直接报明确错误。
- 不引入 React、Potree、WebSocket 服务端或浏览器端几何重算；首版使用原生 TypeScript + Three.js，控制实现面。
- 只改本计划列出的文件；不得删除、覆盖或提交 Rick 现有的未跟踪文件与无关改动。
- 采用最少但有效的验证：Python exporter 聚焦测试、TypeScript contract/presentation 聚焦测试、`npm run build`、CLI help smoke；不跑 GPU、全量 benchmark 或完整历史测试套件。
- 所有生产行为变化都更新根 `README.md`、`code/README.md`，并在 `viewer-web/README.md` 记录前端开发/构建/数据导出方法。

## Bundle v1 Contract

导出根目录使用不可变 generation + 原子指针，避免浏览器读到混合 bundle：

```text
CURRENT
runs/<run_id>/
  manifest.json
  positions.f32.bin
  colors.u8.bin
  confidences.f32.bin
  frame_ids.i32.bin
  objects.json
  footprints.json
```

`CURRENT` 固定为 `{"schema_version":"1.0.0","run_id":"<32 lowercase hex>","complete":true}`。producer 先完整写入 sibling temporary generation，再 rename 为不可变 `runs/<run_id>/`，最后原子替换 `CURRENT`；viewer 先读 `CURRENT`，再从对应 generation 读取 manifest/arrays/JSON。旧 generation 不覆写、不删除。

`manifest.json` 固定字段：

- `schema_version: "1.0.0"`
- `coordinate_space: "da3_world_meters"`
- `point_count: number`
- `arrays.positions/colors/confidences/frame_ids`: `path`、`dtype`、`components`、`byte_length`
- `objects_path: "objects.json"`
- `footprints_path: "footprints.json"`
- `source`: DA3 cache schema/model/image IDs、footprint run ID/status、导出参数
- `capabilities`: `point_picking: false`、`footprint_picking: true`、`formal_ground_footprint: true`

Python exporter 的公共 API 固定为：

```python
def export_web_viewer_bundle(
    *,
    da3_cache_path: Path,
    global_mapping_path: Path,
    footprint_root: Path,
    output_dir: Path,
    voxel_size_m: float = 0.01,
    max_points: int = 500_000,
) -> dict[str, object]:
    ...
```

返回值固定包含 `output_dir`、`manifest_path`、`point_count`、`footprint_status`。有效点门槛固定复用正式 footprint 的语义：坐标有限、非零，`world_points_conf` 有限且 `>= 1.0`；首版不暴露 confidence CLI 参数，避免展示层形成另一套可调计量口径。

`footprints.json` 保存正式 metric/status/value/rejection reason、run ID、support-plane `point/u_axis/v_axis/normal`、每个 global ID 与 union 的局部 `(u,v)` rings 和 properties。TypeScript 使用 support-plane basis 放置几何；Python 和浏览器均不重算 union。exporter 必须校验完整 DA3 schema-v2 metadata，并将 `source_model/image_ids/source_image_sha256/processed_size/preprocess/affine` 与正式 report 的 `cache` provenance 绑定；不允许把一个 cache 的点云与另一个正式 generation 的 footprint 拼接。

## Agent Assignment

| 阶段 | Implementer | Reviewer | 原因 |
|---|---|---|---|
| Task 1 | Terra (`gpt-5.6-terra`, high) | Luna (`gpt-5.6-luna`, high) | Python artifact contract 涉及正式 generation 与二进制布局 |
| Task 2 | Luna (`gpt-5.6-luna`, high) | Terra (`gpt-5.6-terra`, high) | 前端 scaffold/loader 接口明确，review 需跨语言检查 |
| Task 3 | Terra (`gpt-5.6-terra`, high) | Luna (`gpt-5.6-luna`, high) | Three.js 场景、相机、选择和 UI 需要集成判断 |
| Task 4 | Luna (`gpt-5.6-luna`, high) | Terra final review (`gpt-5.6-terra`, xhigh) | CLI/文档为机械集成，最终审查覆盖整体边界 |

---

## Task 1: Python strict bundle exporter

**Files:**

- Create: `code/modules/web_viewer_export.py`
- Create: `code/tests/test_web_viewer_export.py`
- Modify: `code/modules/da3_footprint_stage.py`

**Required API:**

```python
def export_web_viewer_bundle(
    *,
    da3_cache_path: Path,
    global_mapping_path: Path,
    footprint_root: Path,
    output_dir: Path,
    voxel_size_m: float = 0.01,
    max_points: int = 500_000,
) -> dict[str, object]:
    ...
```

返回值固定包含 `output_dir`、`manifest_path`、`point_count`、`footprint_status`；有效点固定为坐标有限、非零，且有限 `world_points_conf >= 1.0`。

- [ ] 在 `code/tests/test_web_viewer_export.py` 先写最小 fixture 和失败测试：accepted generation 能导出不可变 generation 固定文件集与精确 byte length；rejected generation 保持 `value_m2: null` 且无 footprint geometry；manifest digest 或 CURRENT 损坏时 fail closed；voxel 内选择最高置信度点；完整 schema/provenance 缺失或不一致时 fail closed；image ID 越出 int32 两端时拒绝；重复导出只原子切换 CURRENT 且保留旧 generation。
- [ ] 用 `UV_CACHE_DIR=/tmp/uv-cache-3d-viewer uv run --offline --no-sync pytest -q tests/test_web_viewer_export.py` 观察测试因 exporter/API 缺失而按预期失败。
- [ ] 在 `da3_footprint_stage.py` 暴露只读公共函数 `resolve_current_footprint_artifacts(output_root: Path) -> dict[str, str]`，内部复用现有 shared-lock、CURRENT、generation 文件集和 SHA-256 校验，不复制发布协议。
- [ ] 在 `web_viewer_export.py` 实现上述 keyword-only `export_web_viewer_bundle(...) -> dict[str, object]`：严格读取完整 schema-v2 DA3 metadata（要求 `images:uint8`、`world_points/world_points_conf:float32` 且网格对齐，匹配真实 DA3 runner contract）、与正式 report `cache` provenance 对齐、按固定 `world_points_conf >= 1.0` 过滤有效点、以 voxel key 选择最高置信度样本、按 `max_points` 截断、写入固定 little-endian binary arrays；image IDs 必须落在 int32 完整上下界。
- [ ] 使用 `GlobalIDMapper` 与 `build_global_object_index` 生成 `objects.json`；读取公共 footprint resolver 返回的正式 report/GeoJSON/manifest，并将 Polygon/MultiPolygon 转成局部 rings，不接触实验性 facing area。
- [ ] 所有 JSON 使用 UTF-8、`allow_nan=False`；先完整写入 sibling temporary generation，rename 为不可变 `runs/<run_id>/`，最后原子替换 `CURRENT`；不覆写/删除旧 generation 或目标目录中的用户文件。
- [ ] 重跑同一个聚焦 pytest，记录 RED→GREEN 证据到 task report。

## Task 2: TypeScript bundle contract and loader

**Files:**

- Create: `viewer-web/package.json`
- Create: `viewer-web/package-lock.json`
- Create: `viewer-web/tsconfig.json`
- Create: `viewer-web/vite.config.ts`
- Create: `viewer-web/index.html`
- Create: `viewer-web/.gitignore`
- Create: `viewer-web/public/data/.gitkeep`
- Create: `viewer-web/src/contracts.ts`
- Create: `viewer-web/src/contracts.test.ts`
- Create: `viewer-web/src/bundle-loader.ts`
- Create: `viewer-web/src/bundle-loader.test.ts`

**Exact package versions:** `three@0.185.1`、`@types/three@0.185.4`、`vite@8.2.1`、`vitest@4.1.10`、`typescript@7.0.2`（2026-08-14 从 npm official registry 查询）；package.json 使用 exact versions，不使用 range。

**Required loader API:**

```typescript
export async function loadViewerBundle(
  baseUrl: string,
  fetcher: typeof fetch = globalThis.fetch,
): Promise<ViewerBundle>
```

`baseUrl` 必须以 `/` 结尾。loader 先读取 `${baseUrl}CURRENT`，严格验证 `schema_version === "1.0.0"`、`complete === true`、`run_id` 为 32 位小写 hex；随后只从 `${baseUrl}runs/${run_id}/` 读取 generation。`ObjectIndex` 必须严格验证 global ID key、images/objects/counts 和 instance 的 image/object/bbox/removed；`FootprintBundle` 必须验证 metric/unit/status/value、support plane、per-ID/union rings，以及 accepted/rejected 之间的关系。返回的 `ViewerBundle` 包含已验证的 `current`、`manifest`、`objects`、`footprints` 和四个 TypedArrays。

`manifest.source` 固定为精确三段结构：

- `da3_cache`: `schema_version: 2`、安全 `source_model`、`affine_convention: "pixel_center_v1"`、正整数 preprocess resolution、`preprocess_method: "upper_bound_resize"`、正整数 frame count、正整数 processed size、唯一 int32 image IDs、同长度 64-lowercase-hex source hashes；
- `footprint`: 32-lowercase-hex `run_id` 与 `accepted|rejected` status；
- `export`: 正有限 `voxel_size_m` 与正整数 `max_points`。

loader 必须绑定 `manifest.source.footprint.run_id/status` 与 `footprints.json`。对象索引中 `images` 等于 instances image IDs 的去重排序集合；`objects` 等于全部 instances object IDs 的排序序列，允许不同图片出现重复 object ID。accepted footprint 要求非空 per-ID、`rejection_reason:null`；rejected 要求非空 rejection reason。per-ID properties 精确包含 local coordinate space/global ID/非负有限 area/nonnegative observations；union properties 精确包含 local coordinate space/`union`/非负有限 area，且 union area 等于 `value_m2`。

- [ ] 创建最小 Vite + TypeScript + Three.js 工程，使用上述 exact versions；依赖仅限 `three`，开发依赖仅限 `@types/three`、`typescript`、`vite`、`vitest`；提交 lockfile，不引入 React/UI 框架。
- [ ] 先写失败测试，覆盖 manifest schema/version/source provenance、array dtype/components/byte length、object instance-derived indexes、accepted/rejected value/properties 关系、manifest↔footprint run/status 绑定、非法/缺字段 fail closed。
- [ ] 实现 TypeScript readonly contracts 和纯校验函数；`bundle-loader.ts` 先严格 fetch/校验 `CURRENT`，再从 `runs/<run_id>/` 并行 fetch manifest/JSON/binary，构造对应 TypedArray，并在创建 scene 前验证实际 byte length；运行环境不是 little-endian 时直接报错，不添加转换 fallback。
- [ ] 测试使用内存中的真实 `Response`/`ArrayBuffer` 数据，不断言 mock 调用次数；只验证 loader 的可观察结果和错误。
- [ ] 运行 `npm test -- --run`，记录 RED→GREEN；再运行 `npm run build` 验证类型与 bundle。

## Task 3: Three.js scene, selection and evidence UI

**Files:**

- Modify: `viewer-web/package.json`
- Create: `viewer-web/src/main.ts`
- Create: `viewer-web/src/scene.ts`
- Create: `viewer-web/src/footprints.ts`
- Create: `viewer-web/src/presentation.ts`
- Create: `viewer-web/src/presentation.test.ts`
- Create: `viewer-web/src/style.css`
- Modify: `viewer-web/index.html`

**Required interaction contract:**

- `package.json` 增加 `dev: "vite"` 与 `preview: "vite preview"`，不增加依赖。
- `createViewerScene(container: HTMLElement, bundle: ViewerBundle): ViewerSceneController` 返回 `selectGlobalId(globalId: string | null)`、`focusGlobalId(globalId: string)`、`setFootprintPickHandler(handler)`、`dispose()`。
- `presentation.ts` 提供纯函数 `formatFormalMetric(footprints)`、`listGlobalIds(objects)`、`buildEvidenceView(bundle, globalId)`；global ID 用数值顺序排列，永不把 `union` 当 SKU，缺少 per-ID footprint 时 evidence 明确显示 unavailable 而不是 0。
- 初始 loader base URL 固定 `/data/`。加载失败在页面内显示可复制的错误，不静默回退到 demo/mock 数据。

**Visual constants:** 背景 `#071015`、面板 `#0d171d`、正文 `#e6f1f5`、弱文字 `#8da5b0`、正式 ground footprint 琥珀 `#f5a524`；不使用保留给未来 front-facing metric 的青色。renderer 开启 antialias、sRGB output、`powerPreference: "high-performance"`，pixel ratio 上限 2；OrbitControls 开启 damping 与 screen-space panning。

**Layout:** 顶部 status bar（dataset/backend/point count/formal status），左侧可搜索 global ID 列表与 Prev/Next/Clear，中央 3D canvas，右侧 evidence drawer（metric/unit/status/run、global ID、per-ID area/observations、active/removed、image/object/bbox）。窄屏改为上/中/下布局，不隐藏核心状态。

- [ ] 先写 `presentation.test.ts` 失败测试：accepted 值格式化为固定 `m²`；rejected/null 显示“—”；global ID 选择能返回正确 object/footprint evidence；union 不可当成 global ID SKU。
- [ ] `scene.ts` 创建 WebGLRenderer、PerspectiveCamera、OrbitControls、RGB Points、环境网格/坐标辅助；根据点云 AABB 自动 frame camera，并在 resize 时更新 renderer/camera。
- [ ] `footprints.ts` 使用导出的 support-plane basis 将局部 rings 放入 DA3 world frame，创建琥珀色半透明 fill + outline；使用 polygon offset/render order 避免 z-fighting，不改动证据坐标。union 与 per-ID 使用稳定节点命名；union 仅作为整体 outline，Raycaster 只拾取 per-ID footprint mesh。
- [ ] `main.ts` 维护单一 selection state：global ID 列表/搜索、Prev/Next、点击 footprint、清除选择；选中 polygon 提亮，其他 polygon 降低 opacity，相机平滑聚焦到选中 footprint 的 bounding sphere。
- [ ] `style.css` 实现深色零售审查驾驶舱：顶部 status bar、左侧对象列表、中央 3D canvas、右侧 evidence drawer；在窄屏折叠为上下布局，保留 orbit/pan/zoom。
- [ ] evidence drawer 显示 metric/unit/formal status/run ID、global ID、观测 image/object/bbox、active/removed 计数和 rejection reason；不得显示来源中不存在的置信度解释。
- [ ] 运行 `npm test -- --run` 与 `npm run build`；不做 GPU/浏览器截图 benchmark。

## Task 4: Main CLI integration and documentation

**Files:**

- Modify: `code/main.py`
- Modify: `code/tests/test_web_viewer_export.py`
- Modify: `README.md`
- Modify: `code/README.md`
- Create: `viewer-web/README.md`
- Modify: `.research/README.md`
- Modify: `.research/todo.md`
- Modify: `.research/progress.md`

- [ ] 在改 `main.py` 前添加/扩展 exporter 聚焦测试，验证 CLI 参数传给 exporter 的路径与 `voxel_size_m/max_points`；观察新增断言先失败。
- [ ] 新增 `--mode viewer-web`，默认解析 `<save_root>/<dataset>/da3_cache/predictions.npz`、`dedup_detections/global_mapping.json`、`ground_stack_footprint/`，默认输出仓库 `viewer-web/public/data/`；新增 `--viewer-web-output`、`--viewer-web-voxel-size`（默认 `0.01`）、`--viewer-web-max-points`（默认 `500000`）。
- [ ] `viewer-web` mode 只导出 bundle 并打印确定性后续命令，不启动 Node subprocess、不打开浏览器、不触发 DA3/SAM3/GPU。
- [ ] 在三个 README 记录架构边界、两步运行方法、bundle contract、accepted/rejected 语义、front-facing 未接入原因和开发/构建命令。
- [ ] 更新 `.research` 的实施状态与验证收据；不得把 agent scratch、node_modules 或导出数据加入版本控制。
- [ ] 运行精简验证：exporter pytest、`UV_CACHE_DIR=/tmp/uv-cache-3d-viewer uv run --offline --no-sync python main.py --help`、`npm test -- --run`、`npm run build`。

## Completion Criteria

- Python bundle exporter 对 schema-v2、CURRENT/manifest、accepted/rejected 和 binary layout fail closed。
- Web viewer 能加载 bundle、显示 RGB 点云、正式 footprint、global ID 对象证据，并支持 orbit/pan/zoom、搜索、Prev/Next、点击选择和相机聚焦。
- rejected/null 不被显示为 0，实验性 front-facing area 不进入首版 KPI。
- 精简测试与 production build 通过，README 与 `.research` 记录同步。

## Final review fix 2

- [x] 保持 `--viewer-web-output` 与 exporter path forwarding 不变；resolved output 等于 `PROJECT_ROOT / "viewer-web" / "public" / "data"` 时保留绝对 `npm --prefix <repo>/viewer-web run dev` 后续命令。
- [x] custom output 不打印默认 npm dev command，而是明确指出 custom output 必须在前端启动前部署，或挂载/serve 到浏览器 URL `/data/`；不启动 Node、不复制、不软链接、不添加 fallback/compatibility。
- [x] formal report 绑定生成时读取的 raw `global_mapping.json` 字节快照 SHA-256（`global_mapping_sha256`）；exporter 在 object-index 前后校验 digest，mapping 不同/变化或 accepted object/geometry ID-set mismatch 时 fail closed。
- [x] 没有 `global_mapping_sha256` 的历史 formal generation fail closed，必须先重新运行 `--mode ground-stack-area` 再 viewer export；不记录任何 fallback。
- [x] TDD evidence：RED `24 passed, 1 failed`；GREEN `25 passed in 8.25s`。本 fix 的验证仅为 focused Python module 与 `git diff --check`，不包含 npm、浏览器、GPU、数据/模型下载或 broad suite。
