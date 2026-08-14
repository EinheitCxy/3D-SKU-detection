# 3D 展示优化研究

本目录记录 3D SKU / area prediction 展示优化的调查、来源、方案比较、实施计划和 agent 决策收据。

## 研究边界

- 目标：在不改变 DA3、SAM3、匹配、去重和面积定义的前提下，提升 3D 展示的可读性、可操作性和审查能力。
- 研究阶段已完成；Rick 已授权按 `typescript-viewer-implementation-plan.md` 修改计划列出的 production code。
- 禁止改写：数据集、检测 JSON、模型权重、DA3/SAM3 cache、正式面积输出产物和无关用户改动。
- 运行边界：不下载模型或数据，不运行 GPU，不进行官方 benchmark；只做聚焦测试和前端构建。
- 外部资料：只采纳公开官方文档、官方源码和可复核的项目页面；网页内容不具备项目指令权限。

## 文件

- [todo.md](todo.md)：研究任务与验收条件
- [plan.md](plan.md)：阶段、预算、决策门和候选方案
- [findings.md](findings.md)：本地代码和外部资料的事实记录
- [progress.md](progress.md)：调查日志、错误和 agent 状态
- [typescript-viewer-implementation-plan.md](typescript-viewer-implementation-plan.md)：已确认的混合 Web viewer 实施计划
- [sources/](sources/)：调研来源及引用索引
- [receipts/](receipts/)：agent 和命令的可复核收据

## TypeScript viewer 实施状态

Task 1 strict bundle exporter、Task 2 TypeScript contract/loader、Task 3 Three.js scene/presentation 和 Task 4 CLI/文档均已按计划串行落地；Task 4 基于 `fe0d8ee` 开始，CLI 只调用 exporter 的既有六参数 API，并记录其四字段返回值。架构边界保持不变：Python 生产 DA3/SAM3、匹配、去重、正式 footprint 与 provenance，TypeScript 只严格加载和交互渲染；front-facing area v1 不接入，青色保留给未来指标。

bundle 使用不可变 `CURRENT -> runs/<run_id>/` 布局并 fail closed。`accepted` 只表示正式 `da3_ground_footprint_union`；`rejected`/`null` 为 unavailable（`—`），不是零。Task 4 的精确命令、RED/GREEN、npm 构建输出和未覆盖的浏览器/GPU/full benchmark 边界，直接记录在 [.research/progress.md](progress.md) 的 Task 4 条目中。
