# 3D 展示优化研究 Todo

## P0：研究准备

- [x] 核验当前 checkout、分支和工作区状态。
- [x] 确认 `main` 已从 `origin/area_prediction` 建立，旧 `main` 已备份。
- [x] 建立 `.research/` 记录目录和只读研究边界。
- [x] 记录本地 viewer 的真实数据流、输入输出和依赖。

## P1：外部优秀案例调查

- [x] 调查 Viser 的场景树、点云、标签、点击/框选、GUI 和相机控制模式。
- [x] 调查 Nerfstudio viewer 的信息架构、控制面板和交互回调模式。
- [x] 调查 Potree / Three.js 等作品在点云性能、测量、裁剪、标注和 GLB 展示方面的模式。
- [x] 至少为每个可迁移结论记录官方 URL、代码入口或文档段落。

## P2：项目匹配分析

- [x] 明确 front-facing area 与 ground-stack footprint 的语义边界。
- [x] 对比 2-3 个实现路线：Viser 增量、Three.js 独立前端、Potree/大点云路线。
- [x] 评估美观性、可操作性、数据契约风险、实现成本和长期维护成本。
- [x] 形成“首阶段必须做 / 暂不做 / 需要用户确认”的清单。

## P3：研究交付

- [x] 形成推荐方案和实现-ready 的模块边界。
- [x] 形成最小验证矩阵：解析、交互、证据状态、性能和回归。
- [x] Rick 已审阅研究结论并确认进入 TypeScript/Three.js 混合 implementation phase。

## P4：混合 Web viewer 实施

- [x] Task 1：Python strict bundle exporter 与聚焦测试。
- [x] Task 2：TypeScript bundle contract/loader 与构建。
- [x] Task 3：Three.js 点云、footprint、选择、相机和 evidence UI。
- [x] Task 4：`main.py` CLI、README 和精简整体验证；验证收据见 [progress.md](progress.md) 的「2026-08-14 Task 4」章节。
- [x] Final review fix 2：custom viewer output 不再打印默认 npm 命令；默认 output 保留可直接运行命令，并补齐 `/data/` 挂载与 strict mapping digest/regeneration 文档。
- [x] Final scoped Terra review：mapping provenance、跨语言 footprint contract 和 custom-output 修复复审 `CLEAN`，无 residual finding。

## 实施边界

- Python 是 DA3/SAM3、匹配、去重、正式 `da3_ground_footprint_union` 和 provenance 的唯一生产端；TypeScript 只做严格加载、渲染和交互。
- bundle 采用不可变 `CURRENT -> runs/<run_id>/`，schema/provenance/数组长度不符时 fail closed。
- `accepted` 为正式 ground footprint；`rejected`/`null` 显示 unavailable（`—`），不显示零；front-facing area v1 不接入，青色保留。
- 已完成的 npm build 可能有 non-fatal Three.js chunk-size warning；它不等同于浏览器或性能验证。
- Final review fix 2 的验证边界严格限定为 `code/tests/test_web_viewer_export.py` 与 `git diff --check`；不运行 npm、浏览器、GPU、数据/模型下载或 broad test suite。
