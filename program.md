# Program: da3 SKU Matching 自动优化

## Objective
自动优化 3D SKU 跨图匹配系统（`code/`），提升 da3 后端在 floor_display2..12 全量数据集上的匹配准确率。优化目标是 **Recall**（主指标，当前 71.5%）和 **Precision**（次指标，当前 85.5%），二者均不可显著回退。Agent 在 fixed contract 约束下迭代试验匹配算法/阈值/采样策略，每轮跑全量评估，keep/discard 决策推进 current best。

## Setup Checklist
- run tag: `da3-opt-{cycle:03d}`（如 da3-opt-001）
- branch: `deep-anything-reconstructor`（当前），或新建 `da3-opt-loop` 分支隔离试验
- environment: `uv`（code/.venv, Python 3.11）；命令一律 `cd code && uv run python ...`
- dataset check: `imdata/floor_display2..12/`（images/ + detections_results/，只读）
- benchmark: `imdata/picture_mapping_benchmark.csv`（人工标注 GT，只读）
- Initial baseline: da3 Recall 71.5% / Precision 85.5% (1505/2106 TP, 2026-07-16)
- Current best: 同 Initial baseline（首 cycle 前）
- time budget: 每 cycle ≤ 10 分钟 GPU（matching，复用 cache）；单日 ≤ 4 小时 GPU；无 train_agent.py，budget 由 cycle 数 + 超时控制
- seed: 42（config 默认，随机采样用）

## Project Snapshot
- root: `/home/xingyu/3D_Recognization`
- environment: `uv`（code/.venv）；DA3 子进程用 `Depth-Anything-3/.venv/bin/python`
- 匹配入口: `code/main.py --mode concise --dataset ../imdata/floor_displayN --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3`
- 重建入口: `code/main.py --mode pipeline ...`（cache 命中则跳过重建）
- 评估入口: `bash code/batch_accuracy_evaluation.sh --backend da3 --start 2 --end 12 --save-root ../Output/da3`
- 评估器: `code/accuracy_annotation.py`（解析 matching_summary.txt，算 Recall/Precision，只读）
- cache: `Output/da3/floor_displayN/da3_cache/predictions.npz`（可复用避免重建）
- 输出: `Output/da3/floor_displayN/output_3dmapping_da3/`（matching_summary.txt per ref）
- 结果: `docs/accuracy_da3_vs_pi3.md`（当前对比表）
- 模型库: `Pi3/`, `Depth-Anything-3/`, `sam3/`（只读，权重在 checkpoints/）

## File Modification Boundary
| Category | Files or directories | Rule |
|---|---|---|
| Editable | `code/utils/config.py`（SKUMatchingConfig 阈值/参数） | 阈值、字段默认值、backend-aware 分支可改 |
| Editable | `code/utils/geometry_3d.py` | 采样函数、投影、几何验证（find_best_matching_bbox_with_3d_validation）可改 |
| Editable | `code/utils/matching_algorithms.py` | find_correspondences_3d_mapping 主流程、Top-K、评分、唯一性约束可改 |
| Editable | `code/utils/sku_matching_system.py` | process_images 编排、pairing 策略可改 |
| Editable | `code/config.yaml` | inference 段阈值覆盖可改 |
| Editable | `code/modules/da3_runner.py`, `da3_3d_reconstructor.py` | da3 cache 后处理可改（不重跑重建） |
| Append-only | `progress.md`, `results.tsv` | append/update，不覆盖历史 |
| Append-only | `docs/accuracy_da3_vs_pi3.md` | 评估后可更新 |
| Read-only | `imdata/`, `imdata/picture_mapping_benchmark.csv` | 数据集 + GT，禁改 |
| Read-only | `code/accuracy_annotation.py` | 评估器逻辑，禁改（保证评估一致） |
| Read-only | `code/batch_accuracy_evaluation.sh`, `code/accuracy_evaluation.sh` | 评估脚本，禁改 |
| Read-only | `Pi3/`, `Depth-Anything-3/`, `sam3/`, `*/checkpoints/` | 模型库 + 权重，禁改 |
| Requires Rick approval | 引入新依赖（如 scipy.optimize/g2o） | P3 pose graph 等需新库 |
| Requires Rick approval | 改 da3 cache 生成（da3_runner 推理逻辑） | 重跑重建耗时长 |
| Requires Rick approval | 删 cache/checkpoint、`git reset --hard` | 破坏性 |

## Fixed Experiment Contract
- dataset/splits: `imdata/floor_display2..12`（images/ + detections_results/，固定）
- evaluator: `code/accuracy_annotation.py` + `batch_accuracy_evaluation.sh`（固定）
- metrics: Recall=TP/GT, Precision=correct/common（accuracy_annotation 定义，固定）
- cache 复用: da3_cache/predictions.npz 存在则 matching 跳过重建（固定，保证评估快）
- 匹配模式: `--mode concise --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3`（batch_all_refs=true，每图作参考）
- GPU: CUDA_VISIBLE_DEVICES=2（da3 用 GPU2，pi3 对比用 GPU1）
- random seed: 42（config 默认）
- Initial baseline: da3 Recall 71.5% / Precision 85.5%（2026-07-16, commit 3635c8f）
- Current best: 同 Initial baseline（首 cycle）；每 keep 推进
- comparison rule: 首个官方候选比 Initial baseline；后续比 Current best

## Editable Surface
试验可改（无需批准）：
- **阈值参数**（config.py + config.yaml）：max_3d_distance, projection_match_threshold, min_3d_sample_points, plane_normal_alignment_threshold, depth_confidence_threshold, point_3d_confidence_threshold, min_depth, max_depth, pairing_3d, min_hit_ratio, max_3d_validation_candidates, max_3d_points_per_bbox
- **评分权重**（geometry_3d.py combined_score）：match_ratio/geometry/coplanar 的权重
- **几何验证逻辑**（geometry_3d.py）：Top-K 预筛数量、best_match 选择、候选框数量
- **采样策略**（sam3_utils.py + geometry_3d.py）：SAM3 mask 采样、非重叠采样、采样点数
- **唯一性约束**（matching_algorithms.py apply_uniqueness_constraint）：一对多允许度
- **CLI 参数**：--max_3d_distance, --plane_normal_alignment_threshold, --pairing_3d, --min_3d_sample_points 等（已支持）

## Forbidden Changes Without Rick's Approval
- 改 `imdata/` 数据或 benchmark CSV
- 改 `accuracy_annotation.py` 评估器或评估脚本（metric 定义）
- 改 `Pi3/`/`Depth-Anything-3/`/`sam3/` 模型库或权重
- 改 da3_runner 推理逻辑（导致需重跑重建，cache 失效）
- 引入新依赖（scipy.optimize, g2o, GTSAM 等）
- 删 cache/checkpoint/ledger/logs
- 一个官方 run 混多个不相关假设
- 用不同 seed/image 数/budget 比

## Commands
### Setup
```bash
cd /home/xingyu/3D_Recognization/code
uv sync  # 确保依赖
```

### Smoke Check（单数据集快速验证，结果不入 ledger）
```bash
cd /home/xingyu/3D_Recognization/code
rm -rf ../Output/da3/floor_display2/output_3dmapping_da3 ../Output/da3/floor_display2/accuracy_evaluation_da3
CUDA_VISIBLE_DEVICES=2 uv run python main.py --mode concise --dataset ../imdata/floor_display2 \
  --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3
bash accuracy_evaluation.sh floor_display2 --backend da3 --save-root ../Output/da3
# 看 Output/da3/floor_display2/accuracy_evaluation_da3/summary.txt
```

### Official Matching（全量 fd2-12，复用 cache）
```bash
cd /home/xingyu/3D_Recognization/code
for i in 2 3 4 5 6 7 8 9 10 11 12; do
  rm -rf ../Output/da3/floor_display${i}/output_3dmapping_da3 ../Output/da3/floor_display${i}/accuracy_evaluation_da3
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python main.py --mode concise \
    --dataset ../imdata/floor_display${i} --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3
done
```

### Official Evaluation（全量，独立于 matching）
```bash
cd /home/xingyu/3D_Recognization/code
bash batch_accuracy_evaluation.sh --backend da3 --start 2 --end 12 --save-root ../Output/da3
```

### Log Inspection
```bash
# 最新 run log
ls -t /home/xingyu/3D_Recognization/Output/da3/run_*.log | head -1
# 某数据集 summary
cat /home/xingyu/3D_Recognization/Output/da3/floor_displayN/accuracy_evaluation_da3/summary.txt
```

## Time Budget and Runner Control
- official time budget: 单 cycle（matching+评估全量）≤ 10 分钟 GPU；单日 ≤ 4 小时 GPU
- training wrapper: 无（非训练，是 matching 跑批）；budget 由 cycle 超时控制
- budget parameter: 无 CLI flag；由 Coordinator 计时，超 10 分钟 cycle 标 crash
- enforcement: Coordinator 监控 `time` 命令；cache 必须复用（否则重建 +数分钟）
- blocked if missing: 若 cache 不全（da3_cache 缺失），先跑 pipeline 重建（需批准，耗时长）

## Runner and Ledger Ownership
- matching runner: `main.py --mode concise` 产 matching_summary.txt
- evaluator: `batch_accuracy_evaluation.sh` + `accuracy_annotation.py` 产 summary.txt
- ledger writer: 仅 Archivist 追加 results.tsv；runner 不改 ledger
- summary blocks: per-dataset summary.txt + 全量 docs/accuracy_da3_vs_pi3.md

## Planner and Critic Exploration
- planner scope: 匹配算法、采样策略、阈值、评分权重、几何验证、唯一性约束、pairing。**已有候选方向**（基于之前诊断）：
  1. Top-3 几何重排序（best_match 保留 Top-3 候选，质心距离定夺，救"投影命中非GT但几何对"）
  2. 评分权重调整（投影命中率 0.5->0.6，救"hit高被低hit抢"）
  3. SAM3 采样改进（mask 质量过滤 / bbox 中心区域采样 / 多区域采样）
  4. pairing_3d="all" + 唯一性全局排序（救闭环传递断裂）
  5. 多帧可见性投票（跨 ref 聚合，抑制 spurious）
  6. 平面约束残差 gating（残差小跳过法向，救圆柱商品）
- critic scope: 挑战假设（如"Top-3 重排序是否降 Precision"）、找 confound、提替代
- web/literature search: 可搜多视图匹配/点云配准/SLAM loop closure 论文（网络允许时）
- source recording: 每假设记录来源（代码分析/论文/诊断数据）

## Integrated Paper Workflows
- arXiv/Paper Reader: 读 multi-view 3D matching / point cloud registration 论文（网络允许）
- literature-to-code mapping: 论文方法 → 可改文件 → 预期指标变化 → smoke 命令 → 风险 → rollback
- 优先用已有诊断数据驱动（fd2/fd12 漏检分析已存），论文为辅

## Failure Recovery
- failure classes: BFloat16/dtype 错（da3 子进程）、OOM（GPU 抢占）、cache 缺失、SOCKS 代理（网络）、代码 bug、超时
- log inspection: 读 summary.txt + run_*.log 尾部 50 行
- allowed fixes: 改 editable surface 内的 bug；cache 缺失则跑 pipeline 重建（需批准）
- smoke rerun policy: 改代码后先 smoke（fd2 单数据集）再 official
- crash policy: 恢复触及 read-only 或改 contract → 标 crash，记 progress.md
- recovery record: progress.md 记 failure class + fix + rerun command + status

## Metrics and Decision Rule
- primary metric: **Recall**（TP/GT，accuracy_annotation 输出，全量聚合）
- secondary metric: **Precision**（correct/common）
- noise floor: ±1pt（随机性，主要来自 SAM3 采样 + GPU 非确定性）
- Initial baseline: Recall 71.5% / Precision 85.5%
- Current best: 同上（首 cycle）
- comparison target: Current best（首个比 Initial baseline）
- **keep**: Recall 提升 ≥1pt 且 Precision 回退 ≤1pt（或 Precision 提升 ≥1pt 且 Recall 回退 ≤1pt）
- **discard**: Recall/Precision 均持平或回退，或 confound
- **crash**: 失败且不可在 editable surface 修复
- **needs_review**: Recall 升但 Precision 降超 1pt（或反之），需 Rick 判断

## Reproducibility Requirements
- seed: 42
- starting commit: 3635c8f（记录每 cycle 改动）
- command signature: 完整 CLI（含 --save_root, --match_backend 等）
- changed files/flags: 记录改了哪些文件/CLI 参数
- cache path: 确认 da3_cache 完整（fd2-12）
- evaluation count: 11 数据集（fd2-12）
- GPU id: 2（da3）
- logs: Output/da3/run_*.log + accuracy_evaluation_da3/summary.txt

## State Files
- progress: `/home/xingyu/3D_Recognization/progress.md`（durable loop notes + queue）
- results ledger: `/home/xingyu/3D_Recognization/results.tsv`（append-only）
- ledger policy: append-only；列：cycle, hypothesis, status, seed, commit, changed_files, command, recall, precision, tp_gt, verdict, reason

## Role Protocol
| Role | Responsibility | Required output |
|---|---|---|
| Coordinator | 协议合规、状态、决策、budget 计时 | cycle status + next action |
| Protocol Auditor | 检查改动是否在 editable surface | allow/reject/needs_user |
| Planner | 探索匹配算法/采样/阈值/评分/几何，提一个可测假设 | hypothesis + evidence + alternatives + allowed changes + exact commands |
| Executor | 应用改动、跑 smoke+official matching、failure recovery | changed files + command receipt + train summary |
| Evaluator | 跑 batch_accuracy_evaluation | metric summary + keep/discard 建议 |
| Critic | 挑战假设、找 confound、提替代 | vetoes + risk notes + better options |
| Archivist | 更新 progress.md + 追加 results.tsv | ledger row + progress entry |

## Cycle Protocol
1. Load program.md, progress.md, results.tsv
2. Planner 选一个假设（只改 editable surface，优先已有候选方向）
3. Protocol Auditor 检查边界（allow/reject）
4. Executor 应用改动 → smoke check（fd2 单数据集，crash/debug 用，不入 ledger）
5. 若 smoke 通过，跑 official matching（全量 fd2-12，复用 cache）
6. Evaluator 跑 batch_accuracy_evaluation（独立于 matching）
7. 比较 Recall/Precision vs Current best（首 cycle 比 Initial baseline）
8. Archivist 追加 results.tsv + 更新 progress.md
9. keep → 推进 Current best；discard/crash → 保留旧 best，revert 改动（或保留作 reference）

## Git and File Hygiene
- branch: 新建 `da3-opt-loop`（或沿用 deep-anything-reconstructor）
- 每 keep 提交一次（commit message: `opt(da3): cycle N {hypothesis} R{x}/P{x}`）
- discard 可保留改动在 worktree 供参考，或 git checkout 还原
- `git reset --hard` 需 Rick 批准
- 永不删 results.tsv/progress.md/cache/datasets/logs

## Open Questions
1. 是否允许改 da3_runner 推理逻辑（重跑重建）？默认禁，需 Rick 批准
2. 是否允许引入 scipy.optimize（P3 pose graph）？默认需批准
3. budget 按 cycle 数还是 GPU 小时？当前按 cycle（≤10min/cycle）
4. 是否新建 da3-opt-loop 分支隔离试验？默认沿用当前分支
5. pi3 是否同步重跑作公平对比？pi3 当前用旧评估（image_ids 修复前），但 pi3 不受该 bug 影响

## 已知优化候选（Planner 优先考虑，按预期收益）
| 方向 | 预期收益 | 复杂度 | 风险 |
|---|---|---|---|
| Top-3 几何重排序 | Recall +3-5pt | 中 | 可能降 Precision |
| 评分权重调整(投影0.5->0.6) | Recall +1-2pt | 低 | Precision 微降 |
| SAM3 采样改进(mask质量) | Recall +3-5pt | 中-高 | 采样改复杂 |
| pairing_3d=all + 全局排序 | Recall +2-4pt | 低 | matching 时间 N倍 |
| 多帧可见性投票 | Recall +2-3pt | 高 | 新聚合层 |
| 平面残差 gating | Recall +1-2pt | 低 | 放过误匹配 |
| P3 pose graph(治drift) | Recall +5-8pt | 高 | 需新依赖+批准 |
