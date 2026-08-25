# 3D SKU Recognition

面向货架与地堆 SKU 的 DA3 三维重建、跨图匹配、去重、ground-footprint 计量和静态 Web Viewer。Python 负责生成可审计产物；Three.js 只加载、校验和交互展示已经发布的 bundle。

详细的核心契约见 [docs/3d_core.md](docs/3d_core.md)，端到端测量结果见 [perf/runs/20260824T032553Z/FINAL_REPORT.md](perf/runs/20260824T032553Z/FINAL_REPORT.md)。

## 当前布局

```text
3D_Recognization/
├── main.py, config.yaml, pyproject.toml, uv.lock  # DA3 核心 CLI 与依赖
├── src/                                           # 流水线阶段与重建后端
├── utils/                                         # 几何、匹配、cache、过滤等共享库
├── tests/                                         # 核心回归测试
├── modules/
│   ├── sku_detector/                              # YOLO 检测器与独立 uv 项目
│   ├── personalcare_classifier/                   # source/ + canonical model.bin
│   ├── viewer_web/                                # TypeScript / Three.js 前端
│   └── video_to_dedup/                            # 视频到 DA3 去重入口
├── scripts/3d/{pipeline,evaluation,tuning,ops}/   # 项目维护的 shell 工具
├── Output/                                        # 忽略的用户可见 pipeline 产物
├── runtime/                                       # 忽略的迁移环境与工具 cache
│   ├── 3d-core/.venv/
│   └── video_to_dedup/
├── perf/                                          # 可复现的性能采集与报告
└── frame_sampler/                                 # 保持为外部嵌套 Git 仓库
```

`Output/` 与 `runtime/` 都不是源码，也不应提交。迁移前的 `code/.venv` 保留在 `runtime/3d-core/`；日常核心开发环境从仓库根通过 `uv sync` 重建。

## 环境

所有 Python 命令从仓库根执行，并使用 `uv`：

```bash
uv sync --extra dev
```

DA3 仍通过 `Depth-Anything-3/.venv/bin/python` 的隔离 subprocess 运行；需要时可用 `DA3_VENV_PYTHON` 指向它。SKU detector 保持自己的 `modules/sku_detector/pyproject.toml`；其 `runtime/sku_detector/.venv` 固定为 NumPy 1.26.4 与 OpenCV 4.11，视频工作流默认复用该环境。

## 常用命令

```bash
# DA3 重建、完整 batch-all-refs 3D matching 和去重；默认输出 Output/。
# matching 是唯一会运行 SAM3 self-exemplar 并发布 v2 processed-mask cache 的阶段。
uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d \
  --recon_backend da3 --match_backend da3

# 必须先完成完整 batch-all-refs matching，才可运行正式 footprint。
# 默认 batch_all_refs=true；不要用单个 reference 的 cache 代替完整 cache。
# 正式 footprint 只读取 matching 已发布的 v2 cache；不会加载/推理 SAM3。
uv run python main.py --mode ground-stack-area \
  --dataset imdata/floor_display2

# 导出静态 Web bundle（必须在 footprint 之后；同样只读取 v2 cache，不运行 SAM3）
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2

# 启动前端；默认 /data/ 映射 modules/viewer_web/public/data/
npm --prefix modules/viewer_web run dev

# 视频 -> 抽帧 -> 检测 -> DA3 pipeline -> 去重结果
bash modules/video_to_dedup/quickstart.sh <video> <fps> <gpu>
```

`--save_root` 可以覆盖输出目录；相对值始终相对仓库根解析。默认 bundle 位于 `modules/viewer_web/public/data/`，自定义 bundle 必须在前端启动前挂载或 serve 到浏览器的 `/data/`。

## Canonical SAM3 processed-mask workflow

matching 是唯一的 SAM3 producer。默认 `enable_sam3_mask_sampling: true` 时，它只使用 self-exemplar，并为每个 detection frame 一次性发布完整的 `sam3_mask_cache/v2` processed-space bool mask；payload 用 little-endian `np.packbits` 无损保存。`enable_sam3_mask_sampling: false` 仅保留既有 bbox sampling，且不会发布 cache，因此 footprint 与 viewer export 会 fail closed，提示先运行 matching。

运行顺序是硬约束：先完成 **完整 batch-all-refs matching**，再运行 `ground-stack-area`，最后运行 `viewer-web`。footprint 与 export 是只读 consumer：二者均不导入、加载或推理 SAM3，cache miss 也不会补算。

`sam3_mask_cache/v1` 与 v2 不兼容：v1 从不被读取、迁移、复制或删除。v2 的 formal metric 是 `da3_self_exemplar_ground_footprint_union`，viewer bundle schema 是 `2.0.0`；它们与旧面积输出和 bundle `1.0.0` 不可数值比较，旧 bundle 必须按上述顺序重新生成。

DA3 bbox 的 source→processed 映射唯一权威是 `predictions.npz` 中每帧完整的 `source_to_processed_affine`（2×3）及其 processed grid。该 affine 可能同时编码 resize、pixel-center、patch rounding 与 crop offset；`x'=sx*x+(sx-1)/2`、`y'=sy*y+(sy-1)/2` 仅是没有额外 crop 的 simple-resize 例子，绝不能据此重算或替代 cache。matching 缺少显式 cache affine/shape 会 fail closed；旧 scale-only 或含 raw out-of-grid bbox 的 v2 entries 均不能命中，须完整重跑 matching → footprint → export。

## Viewer 与点云策略

Web bundle 使用不可变 `CURRENT -> runs/<run_id>/` 发布。导出器验证 DA3 cache、去重映射、v2 footprint generation、processed-mask cache 与源图，然后写入 schema `2.0.0` manifest、二进制点云、缩略图和正式 footprint。

Viewer **不使用 SAM3 protection mask**。SAM3 cache 仍作为 ground-footprint 和可审计实例信息的输入，但不会绕过点云的常规噪声、地面或天空过滤；所有展示点遵循同一过滤策略。这样不会因高置信门控而静默丢失 SKU 点，也不会把 cache mask 变成永久保留区。

## Ground footprint

`ground-stack-area` 从 matching 发布的 processed-space self-exemplar masks 为每个去重 `global_id` 取多视图 metric 点云，投影到推断支撑平面后以 OBB polygon union 计算 `da3_self_exemplar_ground_footprint_union`（m²）。它不是包装表面积、正面面积、SAM3 mask 面积或 bbox 面积。任何对象缺少足够几何时整次结果为 `rejected` 与 `value_m2: null`，不会发布部分总量；本指标是新 baseline，不可与旧 `da3_ground_footprint_union` 面积直接比较。

## 验证

```bash
uv run --offline pytest -q
(cd modules/viewer_web && npm test -- --run && npm run build)
bash -n modules/video_to_dedup/*.sh scripts/3d/{evaluation,ops,pipeline,tuning}/*.sh
```

个人护理分类器的精简使用说明见 [modules/personalcare_classifier/README.md](modules/personalcare_classifier/README.md)。
