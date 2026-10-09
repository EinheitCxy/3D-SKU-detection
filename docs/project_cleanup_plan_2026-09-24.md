# 项目文件清理计划（2026-09-24）

## 范围与现状

仅处理 `/home/xingyu/3D_Recognization`。清理前项目约 53 GiB；所在文件系统 2.0 TiB、使用率 99%，可用约 31 GiB。清理后项目约 32 GiB，文件系统可用约 47 GiB（98%）。同机有其他任务在写入，因此项目目录缩减量与 `df` 可用空间变化不必相等。

现有根目录 `task_plan.md`、`findings.md`、`progress.md` 属于另一项未完成工作；本计划不改动它们。四个当前 Git checkout（根 main、映射 docker、中英文 Viewer）与各自 upstream 同步，但多个未跟踪文件和实验资料仍需保留。

## 原计划与执行边界

| 阶段 | 具体对象 | 预计可回收 | 执行条件 |
|---|---|---:|---|
| 0. 锁定清单 | 为每批列出精确路径、大小、Git 状态和当前进程/服务引用 | 0 | Rick 审阅目标清单后才执行删除；执行前重新核对路径与活跃写入 |
| 1. 明确的派生物 | `runtime/models/pi3x/.cache/huggingface/download/` 中 9/4 遗留的三个 `.incomplete` 文件（其中两个为空，另一个为 2,631,925,760 B）；`modules/viewer_web/dist/` 构建输出 | 约 2.5 GiB + 4.9 GiB | 确认完整 `model.safetensors` 存在、没有下载进程持有残片；确认本地静态服务未直接读取 `dist/`。`dist/` 可由 `modules/viewer_web` 的 Vite build 重建 |
| 2. 工具环境 | `perf/.playwright/`、`perf/node_modules/` | 约 0.66 GiB | 仅在近期不需要浏览器基准，且按 `perf/README.md` 可重装时执行 |
| 3. 按数据集筛选 | `Output/floor_display*/` 中 DA3、Pi3X、MapAnything、SAM3 缓存 | 合计约 2.0 GiB | 逐数据集保留正在比较的基线和重算代价高的缓存；核对 `Output_ab/` 链接与报告引用；不整删 `Output/` |
| 4. 实验归档 | `perf/runs/`、旧 `runtime/` 实验目录、Viewer `public/` 中的历史 bundle | 待逐项测算 | 先确定正式基线、当前演示和可复现输入，保留报告/运行回执；对大型结果逐项决定保留、外部归档或删除 |

## 2026-09-24 执行回执

- 已删除 Pi3X 下载目录中三份 9/4 的 `.incomplete`（两份为空、一份 2,631,925,760 B），保留完整 `model.safetensors`。
- 已删除根 Viewer 的 `dist/`（原约 4.9 GiB）。当前中英文 Docker Viewer 均未挂载该目录；下次需要本地构建时在 `modules/viewer_web` 执行 `npm run build`。
- 已删除 `perf/runs/`（原约 1.3 GiB）。将其唯一被跟踪的正式报告摘要移至 [perf_baseline_20260826.md](perf_baseline_20260826.md)，同步修正根 README、核心文档与 perf README 的链接；原始日志、遥测及 run receipts 不再可从本地复核。
- 已删除 `modules/personalcare_classifier/.venv/`（原约 5.3 GiB）。它只属于 classifier 子项目；`source/`、模型、`pyproject.toml` 和 `uv.lock` 保留。下次默认 pipeline 使用 classifier 时，`uv` 需重建该环境，依赖缓存不足时可能需要网络。
- 已删除 17 个旧 `runtime/` 对照实验目录（原合计约 3.2 GiB）：`da3-video1-2s*`、`pi3x-video1-comparison`、`pi3x-video1-2s`、`pi3x-density-audit`、`surfel-precision`、`surfel-coverage-relax`、`surfel-shape-fix`、`keyframe-review-20260914`、`ray-pose-review-20260913`、`fig8-investigation`、`viewer-chain-audit`、`video14-diagnosis`、`task-XUa2-*`、`video1-task-896`、`video1-31frames-896`。这些目录内的原始截图、脚本和结果也随之移除。
- 已删除 Viewer `public/` 中 21 个旧 `data-*` bundle，以及 `data/runs/` 中 26 个、`data-surfel/runs/` 中 3 个非当前 run。两处 `CURRENT` 及其指向的 run 都保留；`data-pi3x/` 也保留。
- `perf/.playwright/`、`perf/node_modules/` 和 `Output/` 缓存未在本次授权目标中执行删除。

## 保护清单

- `modules/viewer_web/public/` 清理后约 1.4 GiB，仍是真实 Viewer 数据目录；保留默认 `data/` 当前 run、`data-surfel/` 当前 run、`data-pi3x/`、各 `comparison*/`。`data-video1-1fps/` 被未跟踪的 `comparison-capture.mjs` 直接引用，故也保留。
- `runtime/models/pi3x/model.safetensors`（约 5.4 GiB）、`sam3/checkpoints/sam3.pt`（约 3.3 GiB）是模型权重；`runtime/sku_detector/.venv/`（约 4.8 GiB）是独立 detector 环境。
- `runtime/surfel-coverage-20260924/`、`runtime/roi-fusion-video3-gid3-v2/`、`runtime/video3-resolution-review/`、`runtime/surfel-comparison/` 含当前 Surfel/TSDF 比较、部署回执、上游缓存或 GS checkpoint；现阶段不列入删除。
- `runtime/fd-videos-20260911/` 仍被 `public/` 中的图片链接引用；`runtime/reproduce-video1-1fps/` 仍是实验输入。`runtime/worktrees/`、`docker/` 和已注册的 `/tmp` Git worktree 是 checkout，不按普通临时目录清理；后者占用很小。
- 根目录及 `docker/` 的未跟踪研究文件、配置、测试、`.env`、SKU 表格、当前计划文件均按用户资料保护。

## 验收与边界

删除前核对了精确路径、当前 `CURRENT`、Git 状态、当前 Docker 容器挂载和主机进程；主机 `lsof` 对 Docker overlay 有可见性警告，因此不能把空结果当作完整句柄证明。删除后确认两处当前 run、保留的实验输入和模型权重仍在，`git diff --check` 通过。没有执行 GPU pipeline 或重新构建 classifier 环境。若需要从根本上解决 2 TiB 文件系统的空间压力，应另行盘点项目外占用。
