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
# DA3 重建、3D 匹配和去重；默认输出 Output/
uv run python main.py --mode pipeline \
  --dataset imdata/floor_display2 --algorithm 3d \
  --recon_backend da3 --match_backend da3

# 单独运行正式 footprint 计量（要求已有 DA3 cache 与 global_mapping）
uv run python main.py --mode ground-stack-area \
  --dataset imdata/floor_display2

# 导出静态 Web bundle（要求已有 DA3 cache、去重、footprint 和源图）
uv run python main.py --mode viewer-web \
  --dataset imdata/floor_display2

# 启动前端；默认 /data/ 映射 modules/viewer_web/public/data/
npm --prefix modules/viewer_web run dev

# 视频 -> 抽帧 -> 检测 -> DA3 pipeline -> 去重结果
bash modules/video_to_dedup/quickstart.sh <video> <fps> <gpu>
```

`--save_root` 可以覆盖输出目录；相对值始终相对仓库根解析。默认 bundle 位于 `modules/viewer_web/public/data/`，自定义 bundle 必须在前端启动前挂载或 serve 到浏览器的 `/data/`。

## Viewer 与点云策略

Web bundle 使用不可变 `CURRENT -> runs/<run_id>/` 发布。导出器验证 DA3 cache、去重映射、footprint generation 与源图，然后写入 manifest、二进制点云、缩略图和正式 footprint。

Viewer **不使用 SAM3 protection mask**。SAM3 cache 仍作为 ground-footprint 和可审计实例信息的输入，但不会绕过点云的常规噪声、地面或天空过滤；所有展示点遵循同一过滤策略。这样不会因高置信门控而静默丢失 SKU 点，也不会把 cache mask 变成永久保留区。

## Ground footprint

`ground-stack-area` 将每个去重 `global_id` 的多视图 metric 点云投影到推断支撑平面，以 OBB polygon union 计算 `da3_ground_footprint_union`（m²）。它不是包装表面积、正面面积、SAM3 mask 面积或 bbox 面积。任何对象缺少足够几何时整次结果为 `rejected` 与 `value_m2: null`，不会发布部分总量。

## 验证

```bash
uv run --offline pytest -q
(cd modules/viewer_web && npm test -- --run && npm run build)
bash -n modules/video_to_dedup/*.sh scripts/3d/{evaluation,ops,pipeline,tuning}/*.sh
```

个人护理分类器的精简使用说明见 [modules/personalcare_classifier/README.md](modules/personalcare_classifier/README.md)。
