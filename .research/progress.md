# 3D 展示优化研究进度

## 2026-08-14

- [完成] 确认当前 checkout 为 `/home/xingyu/3D_Recognization`。
- [完成] 确认新 `main` 为 `39dc91f`，旧 `main` 已备份为 `main-backup-20260814`。
- [完成] 读取根 `TODO.md`、`README.md`、`code/README.md` 和现有 viewer 结构。
- [完成] 确认根 `.research/` 尚不存在；已有 Track4World `.research` 属于另一个受控研究项目，不复用。
- [完成] 建立本研究 `.research/`、`sources/` 和 `receipts/`。
- [完成] 发现 `research-lookup` API key 缺失；切换到官方网页/源码搜索，不执行 API 重试。
- [完成] 完成第一轮 Viser、Nerfstudio、Potree、Three.js 官方资料调查并写入 `sources/official-viewers-2026-08-14.md`。
- [完成] architecture-advisor sol agent 返回报告，收据已写入 `receipts/`。
- [完成] explorer sol agent 返回本地 viewer 审计，收据已写入 `receipts/`。
- [完成] research-innovator sol agent 返回零售 UX 和产品匹配分析，收据已写入 `receipts/`。
- [完成] 汇总三份 sol 报告，形成 `recommendation.md`、方案比较、模块边界和最小验证矩阵。
- [完成] Rick 确认 Python bundle + TypeScript/Three.js 混合路线，并授权在当前 `main` 上进入 implementation phase。
- [完成] 固化 `typescript-viewer-implementation-plan.md`，定义 bundle v1、任务边界、agent 顺序与精简验证矩阵。
- [完成] Tasks 1-4 已按计划实现；最终 coordinator/Terra review 待完成。

## Agent 记录

本轮将为每个 agent 保存：agent id、模型、任务边界、返回摘要、引用来源和是否修改文件。所有 agents 都被要求只读调查，不修改 production code。

## 验证记录

- 在 Task 4 开始前尚未运行 Python、GPU 推理或正式 benchmark，也尚未修改 production code 或 README。

## 2026-08-14 Task 4

- [完成] 从基线 `fe0d8ee` 开始；保留旧 `viewer` 分支，只新增独立 `viewer-web` CLI 路由与三个参数：output、voxel size、max points。
- [完成] TDD RED：先添加 `test_viewer_web_cli_routes_exporter_arguments`；在 `code/` 项目环境运行聚焦 pytest 得到 `16 passed, 1 failed`，失败原因为 argparse 尚未接受 `viewer-web`。
- [完成] GREEN：CLI 按已解析 save root 与 dataset name 推导 DA3 cache、global mapping、formal footprint root；调用既有 exporter 六参数 API，记录其四字段返回值，并只打印 CWD-independent 的 `npm --prefix /home/xingyu/3D_Recognization/viewer-web run dev` 后续命令。
- [完成] 更新根、code、viewer-web README 与研究状态，记录 Python/TypeScript 架构边界、immutable bundle、fail-closed 语义、accepted/rejected/null 和 front-facing/cyan 边界。
- [收据] focused Python pytest、CLI help、Vitest 与 Vite build 的精确输出写入 Task 4 report；build 中若出现 non-fatal Three.js chunk-size warning 仅作 warning 记录。
- [收据] coordinator 使用 `uv run --project code --offline --no-sync` 从仓库根复核 Python suite：`17 passed`；CLI help、Vitest `18 passed` 与 production build 均再次通过。
- [边界] 未启动浏览器、未运行 GPU/DA3/SAM3、未下载依赖/数据、未运行 full benchmark；未 stage/commit。

## 2026-08-14 Final review fix 2

- [完成] 基于 `main` commit `eb6a769`，只拥有 brief 列出的 CLI、focused test、三份 README、四份 `.research` 文件和 scratch report；保留所有其它用户文件，不 stage/commit，不修改 exporter、formal stage、TypeScript 或 Node 行为。
- [完成] TDD RED：先扩展 `test_viewer_web_cli_routes_exporter_arguments`，新增 default-output 路由断言；focused pytest 得到 `24 passed, 1 failed`，custom output 失败于仍包含 `Next step: npm --prefix ... run dev`。
- [完成] TDD GREEN：`main.py` 仅在 resolved output 等于 `PROJECT_ROOT / viewer-web / public / data` 时打印绝对 npm 命令；custom output 改为打印明确的 output 路径及必须 serve/mount 到浏览器 `/data/` 的提示。focused pytest 得到 `25 passed in 8.25s`。
- [完成] 三份用户 README 与研究记录同步 `/data/` mounting semantics、raw mapping SHA-256 binding、导出前后 digest/变化拒绝、accepted object/geometry ID-set mismatch、历史缺 digest generation 需重跑 `--mode ground-stack-area` 且无 fallback。
- [边界] 本 fix 只运行 focused Python module 和 `git diff --check`；不运行 npm、浏览器、GPU、DA3/SAM3、数据/模型下载或 broad suite。精确 RED/GREEN、diff-check、changed paths、自检和 concerns 已记录在本节。
