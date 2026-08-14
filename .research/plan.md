# 3D 展示优化研究计划

## 目标

研究如何把现有的 3D 重建、global ID、area prediction 和 footprint 结果，组织成美观、可操作、可审查的展示体验；研究结果必须匹配当前项目真实数据契约，不得把展示层重新变成一个几何计量器。

## 当前基线

- checkout：`/home/xingyu/3D_Recognization`
- branch：`main`
- `main` HEAD：`39dc91f`，跟踪 `origin/area_prediction`
- 备份 branch：`main-backup-20260814`，HEAD `33c5498`
- 工作区：存在 Rick 之前留下的未跟踪文件；研究期间不处理、不删除、不覆盖。
- 项目入口：`code/main.py`
- 现有 viewer：`code/viewer/` + `code/modules/viewer_runner.py`

## 研究预算边界（已完成）

- 最多 3 个 `gpt-5.6-sol` subagents。
- 只读本地源码和公开网页/官方源码；不下载模型、数据或大文件。
- 不启动 GPU、不运行 DA3/SAM3/重建/匹配/正式评估。
- 只允许新增或更新 `.research/` 内的 markdown、文本收据和来源索引。
- 研究时间：以完成可审阅的方案为止；若连续 3 次遇到同一外部访问问题，则记录并切换到官方网页搜索。

## 阶段

### Phase 0：准备（已完成）

核验分支、工作区、项目 README/TODO、既有研究目录和 API 可用性。

### Phase 1：本地架构审计

阅读 viewer runtime、datasource、cache、indexer、CLI 和面积产物 schema，输出真实数据流、缺口和不可触碰的科学契约。

### Phase 2：外部案例审计

按交互模式而不是按“看起来漂亮”收集 Viser、Nerfstudio、Potree、Three.js 等官方资料，记录可迁移的控件、场景组织、标注、选择、相机和性能策略。

### Phase 3：方案比较

比较 Viser 增量、Three.js 前端和 Potree/大点云路线，采用项目适配度、交互完整性、视觉质量、实现成本、依赖风险和数据契约风险评分。

### Phase 4：推荐与审阅门

产出推荐方案、模块边界、验证矩阵和不做清单，已写入 `recommendation.md`。没有 Rick 的明确确认，不进入 production code 实现；研究记录本身可以继续完善。

## 当前实施决策

Rick 已确认采用 Python 静态 bundle producer + TypeScript/Three.js viewer。Python 保留算法、正式产物验证和 provenance；TypeScript 负责渲染与交互。实施按 `typescript-viewer-implementation-plan.md` 串行执行，front-facing area 在正式 schema/provenance/geometry 修复前不接入首版 KPI。

## 决策门

1. 不把 `facing_area_m2`、`facing_share` 和 `da3_ground_footprint_union` 混为同一指标。
2. rejected/unavailable 证据必须 fail-closed 展示，不能渲染成可信数字。
3. viewer 读取现有正式产物；不在 viewer 中偷偷重算面积。
4. 任何引入新前端、格式转换或大文件处理的方案，必须说明维护和回归成本。
5. 研究结论需由本地代码证据和可访问的外部来源共同支持。

## 研究错误

| 错误 | 尝试 | 处理 |
|---|---:|---|
| `research-lookup` 所需 `PARALLEL_API_KEY` / `OPENROUTER_API_KEY` 缺失 | 1 | 不调用该 API，改用官方网页/源码调研并记录来源 |
| 推荐报告首轮 patch 上下文不匹配 | 1 | 未写入部分文件；改用分步 patch 并按实时内容更新 |
