# Program: da3 SKU Matching 速度优化（autoresearch loop）

## Objective
最小化 3D SKU 跨图匹配系统（`code/`）的**运行速度成本**：在 `--mode concise` 匹配（**fd5/fd6/fd7 三个数据集**，复用 da3_cache，不含重建）的 wall-clock 下，迭代降低耗时，**同时保持匹配结果等价性**（Recall/Precision 相对三数据集 baseline 在 ±0.5pt 噪声容差内）。Agent 在 fixed contract 约束下迭代性能优化（缓存/并行/架构重构/向量化/I/O），每轮跑三数据集 matching 计时 + 等价性评估，keep/discard 决策推进 current best（最快且等价）。

**数据集选择理由**：fd5(33M cache)/fd6(46M)/fd7(67M) 覆盖小/中/大三种规模，代表性足够；用 3 个数据集（替代全量 11 个 fd2-12）使每轮 cycle 时间从 ~10min 降到 ~3min，加速迭代。全量 fd2-12 的最终验证仅在 Cycle 收尾或 Rick 要求时跑一次。

与 `program.md`（精度优化，已收尾 R73.27%）的关系：本协议是**速度轴**的正交优化，不改算法语义，只改执行路径。算法正确性由等价性 gate 守护。

## Setup Checklist
- run tag: `da3-speed-{cycle:03d}`（如 da3-speed-001）
- branch: 沿用当前 `deep-anything-reconstructor`（速度改动经等价性验证后直接 commit）
- environment: `uv`（code/.venv, Python 3.11）；命令一律 `cd code && uv run python ...`；DA3 子进程用 `Depth-Anything-3/.venv/bin/python`
- dataset check: `imdata/floor_display5..7/`（images/ + detections_results/，只读）+ cache `Output/da3/floor_display5..7/da3_cache/predictions.npz`（已确认齐全）
- benchmark: `imdata/picture_mapping_benchmark.csv`（人工标注 GT，只读，等价性验证用）
- Equivalence baseline: **R84.01% / P94.06%**（fd5/6/7 聚合，pairing_3d=next，TP=557/GT=663，correct=554/common=589，commit 255f4f2，Phase 0 实测 2026-07-17）
- Speed baseline: **720s**（fd5/6/7 concise matching wall-clock，复用 cache，Phase 0 可靠重测：fd5=141s/fd6=215s/fd7=364s，refs=33。注: wall-clock 须同口径对比，GPU 热节流/负载噪声大）
- Current best: 同 Speed/Equiv baseline（首 cycle 前）
- time budget: 每 cycle（fd5/6/7 matching+评估）≤ 5 分钟；单日 ≤ 4 小时 GPU
- seed: 42（config 默认，随机采样用，禁改）
- GPU: CUDA_VISIBLE_DEVICES=2（da3，已确认空闲 24G）

## Project Snapshot
- root: `/home/xingyu/3D_Recognization`
- environment: `uv`（code/.venv）；DA3 子进程 `Depth-Anything-3/.venv/bin/python`
- matching 入口: `code/main.py --mode concise --dataset ../imdata/floor_displayN --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3`
- 重建入口: `code/main.py --mode pipeline ...`（cache 命中则跳过；**速度优化不碰重建**）
- 评估入口: `bash code/batch_accuracy_evaluation.sh --backend da3 --start 5 --end 7 --save-root ../Output/da3`
- 评估器: `code/accuracy_annotation.py`（解析 matching_summary.txt，算 Recall/Precision，只读，**等价性 gate**）
- cache: `Output/da3/floor_displayN/da3_cache/predictions.npz`（fd5/6/7 已确认齐全：33M/46M/67M）
- 输出: `Output/da3/floor_displayN/output_3dmapping_da3/`（matching_summary.txt per ref）
- 模型库: `Pi3/`, `Depth-Anything-3/`, `sam3/`（只读，权重在 checkpoints/）

## Pipeline 耗时结构（代码分析，待 profiling 证实）
`--mode concise` batch_all_refs=True 串行链路（main.py `_run_matching_inference` -> 每 ref 独立 `inference_main()` -> 新建 `SKUMatchingSystem` -> `process_images`）：

```
_run_matching_inference (main.py:257)           # N refs 串行循环
└─ per ref: _run_single_matching -> inference_main -> run_3d_mapping
   └─ SKUMatchingSystem.process_images (sku_matching_system.py:99)
      ├─ _load_data (image+detection 加载+对齐)        # [疑似瓶颈] 每 ref 重复
      ├─ da3 image load+resize (sku_matching_system.py:146-164)  # [疑似瓶颈] 每 ref 逐张 PIL+resize，N ref 重复 N 次
      └─ _run_matching -> find_correspondences_3d_mapping (matching_algorithms.py:279)
         ├─ cache npz load (matching_algorithms.py:308-357)      # 已有 PI3_SCENE_CACHE 模块级缓存 ✅ 仅首 ref
         ├─ SAM3 mask 生成 maybe_run_sam3_for_reference (line 395)  # 模型已缓存(_SAM3_PREDICT_INST_CACHE)，开销=推理本身
         └─ 三重 Python 循环 (line 424+):
            ├─ target_img × ref_bbox × target_bbox
            ├─   sample_3d_points_from_mask/non_overlap (line 446/456)  # [疑似瓶颈] 点采样
            ├─   project_3d_to_2d (line 472)                              # [疑似瓶颈] 逐点投影
            └─   find_best_matching_bbox_with_3d_validation + apply_uniqueness_constraint  # KD-tree/最近邻+贪心
```

**已有优化（不可回退）**：`PI3_SCENE_CACHE` 模块级缓存（npz 每 ref 复用）；`_SAM3_PREDICT_INST_CACHE`/`_SAM3_BATCH_API_CACHE`（SAM3 模型不每 ref reload，且已对 ref 所有 bbox batch）；`parallel_refs` ThreadPoolExecutor 框架（main.py:282，默认 1，>1 启用跨 ref 线程并行，pi3/da3 cache 只读线程安全）。

## File Modification Boundary
| Category | Files or directories | Rule |
|---|---|---|
| Editable（性能层+架构重构） | `code/utils/profiling.py`（新）、`code/utils/sku_matching_system.py`、`code/utils/matching_algorithms.py`、`code/utils/geometry_3d.py`（仅执行路径）、`code/utils/sam3_utils.py`（仅加载/缓存路径）、`code/utils/data_utils.py`、`code/utils/transforms.py`、`code/modules/inference.py`、`code/main.py`（仅 dispatch/计时）、`code/config.yaml`（性能开关） | 改执行路径不改算法语义；架构重构需过等价性 gate |
| Append-only | `results_time.tsv`、`progress_time.md` | append-only，不覆盖 |
| Read-only（算法语义） | `code/utils/config.py` 阈值默认、`code/utils/geometry_3d.py` combined_score 权重与验证逻辑、`code/utils/matching_algorithms.py` 匹配判定逻辑、`code/accuracy_annotation.py`、`code/batch_accuracy_evaluation.sh` | 禁改（改则破坏等价性 gate） |
| Read-only（基础设施） | `imdata/`、`picture_mapping_benchmark.csv`、`Pi3/`、`Depth-Anything-3/`、`sam3/`、`code/modules/da3_runner.py`、da3_cache/*.npz | 禁改/禁删 |

## Fixed Experiment Contract
- dataset/splits: `imdata/floor_display5, floor_display6, floor_display7`（固定，3 个数据集）
- evaluator: `code/accuracy_annotation.py` + `batch_accuracy_evaluation.sh --start 5 --end 7`（固定，等价性 gate）
- metrics: **primary=wall-clock**（fd5/6/7 concise matching，复用 cache，秒）；**equivalence=Recall/Precision**（accuracy_annotation 定义，固定）
- cache 复用: da3_cache/predictions.npz 必须存在，matching 跳过重建（固定，保证计时只测匹配）
- 匹配模式: `--mode concise --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3`（batch_all_refs=true，每图作参考）
- GPU: CUDA_VISIBLE_DEVICES=2（da3）
- random seed: 42（禁改）
- Equivalence baseline: **R84.01% / P94.06%**（fd5/6/7 聚合，pairing_3d=next，commit 255f4f2，Phase 0 实测）
- Speed baseline: **543s**（fd5/6/7 wall-clock，Phase 0 实测，refs=33）
- Current best: 同 Speed/Equiv baseline（首 cycle）；每 keep 推进
- comparison rule: 首个官方候选比 Speed baseline；后续比 Current best（wall-clock）
- equivalence tolerance: **R/P 相对三数据集 Equivalence baseline (R84.01%/P94.06%) 在 ±0.5pt 内**（batch/CUDA 浮点非确定容许）

## Editable Surface
速度试验可改（无需批准，但须过等价性 gate）：
- **计时工具**（`utils/profiling.py`）：StageTimer context manager + 累加器 + json 输出，纯加法不改逻辑
- **缓存只读数据**：模块级缓存图像 tensor / KD-tree / projection 矩阵（跨 ref 复用，类似 PI3_SCENE_CACHE 模式）。SAM3 模型已缓存，无需重复。
- **架构重构**：`process_images` 跨 ref 共享只读数据（cache/图像），改 SKUMatchingSystem 生命周期，**不改 find_correspondences_3d_mapping 的匹配判定与评分**
- **并行化**：`parallel_refs` 默认值、ThreadPoolExecutor 配置、target 投影并行、bbox 间并行
- **batch/向量化**：CUDA batch matmul、批量投影、torch.no_grad 聚合（浮点累加顺序变 -> 强制走等价性 gate）
- **数据结构**：KD-tree ↔ FAISS 切换、预计算 target 点云索引、numpy 向量化替代 Python loop
- **I/O**：图像并行加载、npz 内存映射(mmap_mode)、避免重复 PIL open
- **性能开关**（config.yaml）：`parallel_refs`、`enable_profiling`、`disable_visualization`（可选可视化关闭省时）

## Forbidden Changes Without Rick's Approval
- 改算法语义：阈值参数（max_3d_distance/projection_match_threshold/min_3d_sample_points/pairing_3d/max_3d_validation_candidates 等）、combined_score 权重、几何验证逻辑、采样点数、唯一性约束判定
- 改 `accuracy_annotation.py` 评估器或 metric 定义（等价性 gate 失效）
- 改 `imdata/` 数据或 benchmark CSV
- 改 `Pi3/`/`Depth-Anything-3/`/`sam3/` 模型库或权重
- 改 `da3_runner.py` 推理逻辑（cache 失效，重跑重建）
- 引入新依赖（numba/cython/taichi/g2o 等）-- 仅 torch/numpy/scipy(已有) 内优化
- 删 cache/checkpoint/ledger/logs
- 一个官方 run 混多个不相关假设
- 用不同 seed/image 数/budget/GPU 比

## Profiling Protocol（计时注入规范）⭐
**目标**：得到 per-stage wall-clock breakdown，定位 top-3 瓶颈。计时是纯加法，不改算法逻辑（StageTimer 只 perf_counter 累加 + log/json 输出）。

### 计时工具 `utils/profiling.py`（新建）
```python
# StageTimer: context manager + 模块级累加器 + json dump
# 用法:
#   with StageTimer("cache_npz_load"): data = np.load(...)
#   with StageTimer("sam3_mask"): masks = maybe_run_sam3_for_reference(...)
# 输出: 累加到 STAGE_TIMES dict + 每次调用 logger.info(f"[PROF] stage=X dur=Ys")
# 结束: dump 到 Output/da3/<dataset>/profiling_<timestamp>.json
```
- 用 `time.perf_counter()`（非 time.time()，更精确）
- 支持嵌套（parent stage）
- per-call + per-stage-total 两级
- 受 `config.enable_profiling` gate（默认 False，profiling run 时 True）

### 注入点（必须覆盖）
| Stage | 位置 | 说明 |
|---|---|---|
| `batch_all_refs_total` | main.py `_run_matching_inference` | N refs 循环总（已有 duration，复用） |
| `per_ref_total` | inference.py `run_3d_mapping` | 每 ref process_images（已有，复用） |
| `process_images` | sku_matching_system.py:99 | 总 |
| `load_data` | sku_matching_system.py `_load_data` | 图像+detection 加载+对齐 |
| `image_load_resize` | sku_matching_system.py:146-164 | da3 逐张 PIL+resize（疑似瓶颈） |
| `cache_npz_load` | matching_algorithms.py:308-357 | npz load（仅首 ref，验证缓存命中） |
| `sam3_mask` | matching_algorithms.py:395 | SAM3 推理 |
| `ref_point_sampling` | matching_algorithms.py:446/456 | sample_3d_points_from_mask/non_overlap |
| `projection_3d_to_2d` | matching_algorithms.py:472 | project_3d_to_2d（疑似瓶颈） |
| `target_bbox_match` | matching_algorithms.py:496 | find_best_matching_bbox_with_3d_validation |
| `uniqueness_constraint` | geometry_3d.py apply_uniqueness_constraint | 贪心 fallback 分配 |
| `post_process` | sku_matching_system.py `_post_process_results` | 可视化+保存 |

### Profiling Run（Phase 0）
- 单数据集 fd5（小）+ fd7（大）对比，看规模 scaling（fd6 作 official baseline 用，不重复 profiling）
- 跑 2 次取 min（消除 GPU 调度波动）
- 输出 `Output/da3/floor_displayN/profiling_*.json` + 汇总到 `docs/profiling_breakdown.md`
- **定位 top-3 瓶颈**，回填本文件 Speed baseline + 候选优先级

## Commands

### Phase 0: Profiling（计时后跑，定位瓶颈）
```bash
cd /home/xingyu/3D_Recognization/code
# fd5 小数据集
rm -rf ../Output/da3/floor_display5/output_3dmapping_da3
CUDA_VISIBLE_DEVICES=2 uv run python main.py --mode concise --dataset ../imdata/floor_display5 \
  --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3 \
  --enable_profiling 2>&1 | tee /tmp/prof_fd5.log
# fd7 大数据集（看 scaling）
rm -rf ../Output/da3/floor_display7/output_3dmapping_da3
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python main.py --mode concise \
  --dataset ../imdata/floor_display7 --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3 \
  --enable_profiling 2>&1 | tee /tmp/prof_fd7.log
# 汇总 profiling json -> docs/profiling_breakdown.md
```

### Phase 0: 三数据集 Baseline（无 profiling，测 wall-clock + R/P）
```bash
cd /home/xingyu/3D_Recognization/code
T0=$(date +%s)
for i in 5 6 7; do
  rm -rf ../Output/da3/floor_display${i}/output_3dmapping_da3 ../Output/da3/floor_display${i}/accuracy_evaluation_da3
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python main.py --mode concise \
    --dataset ../imdata/floor_display${i} --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3
done
T1=$(date +%s); echo "BASELINE_WALL_CLOCK=$((T1-T0))s"
bash batch_accuracy_evaluation.sh --backend da3 --start 5 --end 7 --save-root ../Output/da3
# 看 docs/accuracy_da3_vs_pi3.md 的 fd5/6/7 R/P = 三数据集 Equivalence baseline
```

### Smoke Check（单数据集速度快筛，不入 ledger）
```bash
cd /home/xingyu/3D_Recognization/code
rm -rf ../Output/da3/floor_display6/output_3dmapping_da3 ../Output/da3/floor_display6/accuracy_evaluation_da3
CUDA_VISIBLE_DEVICES=2 /usr/bin/time -v uv run python main.py --mode concise --dataset ../imdata/floor_display6 \
  --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3 2>&1 | tee /tmp/smoke_speed.log
bash accuracy_evaluation.sh floor_display6 --backend da3 --save-root ../Output/da3
# 看 wall-clock（/usr/bin/time 的 real）+ summary.txt 的 R/P（等价性）
```

### Official Matching（fd5/6/7，复用 cache，计时）
```bash
cd /home/xingyu/3D_Recognization/code
T0=$(date +%s)
for i in 5 6 7; do
  rm -rf ../Output/da3/floor_display${i}/output_3dmapping_da3 ../Output/da3/floor_display${i}/accuracy_evaluation_da3
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python main.py --mode concise \
    --dataset ../imdata/floor_display${i} --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3
done
T1=$(date +%s); echo "OFFICIAL_WALL_CLOCK=$((T1-T0))s"
```

### Official Evaluation（fd5/6/7 等价性验证）
```bash
cd /home/xingyu/3D_Recognization/code
bash batch_accuracy_evaluation.sh --backend da3 --start 5 --end 7 --save-root ../Output/da3
# 看 docs/accuracy_da3_vs_pi3.md fd5/6/7 R/P，必须 ±0.5pt 内
```

### Final Verification（全量 fd2-12，仅收尾或 Rick 要求时跑一次）
```bash
cd /home/xingyu/3D_Recognization/code
for i in 2 3 4 5 6 7 8 9 10 11 12; do
  rm -rf ../Output/da3/floor_display${i}/output_3dmapping_da3 ../Output/da3/floor_display${i}/accuracy_evaluation_da3
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python main.py --mode concise \
    --dataset ../imdata/floor_display${i} --algorithm 3d --match_backend da3 --recon_backend da3 --save_root ../Output/da3
done
bash batch_accuracy_evaluation.sh --backend da3 --start 2 --end 12 --save-root ../Output/da3
```

### Log Inspection
```bash
# profiling json
ls -t /home/xingyu/3D_Recognization/Output/da3/floor_displayN/profiling_*.json | head -1
# 瓶颈汇总
cat /home/xingyu/3D_Recognization/docs/profiling_breakdown.md
```

## Time Budget and Runner Control
- official time budget: 单 cycle（fd5/6/7 matching+评估）≤ 5 分钟；单日 ≤ 4 小时 GPU
- training wrapper: 无（非训练，matching 跑批+评估）
- budget parameter: 无 CLI flag；Coordinator 用 `/usr/bin/time` + `date +%s` 计时
- enforcement: Coordinator 监控 wall-clock；cache 必须复用；超 5 分钟 cycle 标 crash
- blocked if missing: da3_cache 不全 -> 先跑 pipeline 重建（需批准，不计入速度基线）

## Runner and Ledger Ownership
- matching runner: `main.py --mode concise` 产 matching_summary.txt + profiling json
- evaluator: `batch_accuracy_evaluation.sh` + `accuracy_annotation.py` 产 summary.txt（等价性 gate）
- ledger writer: 仅 Archivist 追加 `results_time.tsv`
- summary blocks: per-dataset summary.txt + `docs/accuracy_da3_vs_pi3.md`（fd5/6/7 行）+ `docs/profiling_breakdown.md`

## Planner and Critic Exploration
- planner scope: 缓存策略、并行化、架构重构（跨 ref 共享只读数据）、batch/向量化、数据结构、I/O。**已有候选方向**（基于代码分析，待 profiling 证实优先级）：
  1. **跨 ref 共享图像 tensor**（最大结构性机会）：当前每 ref process_images 重新 load+resize 全部图像（sku_matching_system.py:146-164 逐张 PIL），N ref 重复 N 次。模块级缓存（类似 PI3_SCENE_CACHE）-> 预期省 N×image_load。不改算法（图像只读）。
  2. **`parallel_refs` 默认开启**：ThreadPoolExecutor 跨 ref 并行（cache/SAM3 线程安全已验证）。torch CUDA 操作释放 GIL -> GPU-bound（SAM3 forward/投影 on CUDA）可并行；纯 Python 三重循环受 GIL 限制（需向量化解决）。
  3. **投影向量化**：project_3d_to_2d 逐点循环 -> batch matmul（浮点累加顺序变，过等价性 gate）。
  4. **target 点云 KD-tree 预计算共享**：跨 ref 共享 target 索引（若 target 不随 ref 变）。
  5. **关闭可选可视化**：post_process 的 visualize_results 在 concise 模式可能不必要 -> config 开关。
  6. **npz mmap**：da3_cache npz 用 mmap_mode 加载，避免全量进内存。
- critic scope: 挑战"是否破坏等价性"（如 batch matmul 浮点非确定 -> R/P 漂移）、找 confound（wall-clock 波动、GPU 抢占、cache 冷热）、提替代（更简单的等价优化）
- web/literature search: **WebSearch 工具当前故障（返回空，已 8 轮验证），但网络通**；可用 `curl` 抓取 duckduckgo/stackoverflow 替代（Bash 分类器恢复后）。搜索方向：torch CUDA batch/profiling、点云匹配 KD-tree/FAISS 加速、Python GIL、PIL 批量加载、numpy 向量化。
- source recording: 每假设记录来源（代码 file:line / profiling json / 论文）
- **基于 profiling 事实推测**：每个假设必须先看 profiling breakdown 证实瓶颈占比，再提改动。禁止未验证猜测。

## Failure Recovery
- failure classes: 等价性破坏（R/P 超 ±0.5pt）、OOM（GPU 抢占/并行争用）、cache 缺失、代码 bug、超时、batch/CUDA 浮点漂移
- log inspection: profiling json + summary.txt + run_*.log 尾部 50 行
- allowed fixes: 改 editable surface 内的执行路径；等价性破坏 -> revert + 降级为非 batch 路径
- smoke rerun policy: 改代码后先 smoke（fd6 单数据集，看 wall-clock + 等价性）再 official
- crash policy: 等价性破坏且不可在 editable surface 修复 -> 标 crash，revert，记 progress_time.md
- recovery record: progress_time.md 记 failure class + fix + rerun command + status

## Metrics and Decision Rule
- primary metric: **wall-clock**（fd5/6/7 concise matching，复用 cache，秒，跑 1 次 official）
- equivalence metric: **Recall/Precision**（accuracy_annotation，fd5/6/7 聚合）
- noise floor: wall-clock ±5%（GPU 调度/系统负载波动，单次运行）；R/P ±0.5pt（batch/CUDA 浮点非确定）
- Equivalence baseline: R84.01% / P94.06%（fd5/6/7 聚合，commit 255f4f2，Phase 0 实测）
- Speed baseline: **720s**（fd5/6/7 wall-clock，Phase 0 可靠重测，refs=33。注: 须同口径对比）
- Current best: 同 Speed/Equiv baseline（首 cycle）
- comparison target: Current best wall-clock（首个比 Speed baseline）
- **keep**: wall-clock 下降 ≥5% **且** R/P 相对三数据集 Equivalence baseline 在 ±0.5pt 内
- **discard**: wall-clock 持平/上升，或 R/P 等价但无收益
- **crash**: R/P 超 ±0.5pt（破坏等价性，不可在 editable surface 修复），或失败不可恢复
- **needs_review**: wall-clock 下降但 R/P 微变在 ±0.5pt 边界（需 Rick 判断是否接受）

## Reproducibility Requirements
- seed: 42
- starting commit: 255f4f2（R73.27% 等价基线，全量；三数据集 baseline 同 commit）
- command signature: 完整 CLI（含 --save_root, --match_backend, --enable_profiling 等）
- changed files/flags: 记录改了哪些文件/CLI 参数
- cache path: 确认 da3_cache 完整（fd5/6/7）
- evaluation count: 3 数据集（fd5/6/7）
- GPU id: 2（da3）
- logs: Output/da3/run_*.log + accuracy_evaluation_da3/summary.txt + profiling_*.json
- wall-clock: `/usr/bin/time -v` real 或 `date +%s` 差值，记 fd5/6/7 official

## State Files
- progress: `/home/xingyu/3D_Recognization/progress_time.md`（durable loop notes + queue）
- results ledger: `/home/xingyu/3D_Recognization/results_time.tsv`（append-only）
- ledger policy: append-only；列：cycle, hypothesis, status, seed, commit, changed_files, command, wall_clock_s, recall, precision, verdict, reason
- profiling: `docs/profiling_breakdown.md`（Phase 0 汇总 + 每 cycle 更新）

## Role Protocol
- **思考类角色用 Opus（深度推理）**，执行类角色用 Sonnet（代码生成/运行）。
- Protocol Auditor 合并进 Planner：Planner 自审边界（尤其等价性），不单独设 Auditor。

| Role | Model | Responsibility | Required output |
|---|---|---|---|
| Coordinator | Opus | 协议合规、状态、决策、wall-clock 计时 | cycle status + next action |
| Planner | Opus | 探索缓存/并行/架构/向量化/I/O，**自审等价性边界（allow/reject/needs_user）**，提一个可测假设 | hypothesis + profiling 证据 + 等价性判定 + allowed changes + exact commands |
| Executor | Sonnet | 应用改动、跑 profiling+smoke+official matching、failure recovery | changed files + command receipt + wall-clock + profiling summary |
| Evaluator | Sonnet | 跑 batch_accuracy_evaluation（等价性 gate） | R/P summary + keep/discard 建议 |
| Critic | Opus | 挑战等价性破坏风险、找 wall-clock confound、提替代 | vetoes + risk notes + better options |
| Archivist | Sonnet | 更新 progress_time.md + 追加 results_time.tsv + profiling_breakdown.md | ledger row + progress entry |

### Planner 探索原则
- **鼓励多探索方向**：每 cycle 探索 2-3 个不同方向（缓存/并行/架构/向量化/数据结构/I/O）再收敛到 1 个可测假设。
- **基于 profiling 事实**：每个假设必须先读 profiling breakdown 证实瓶颈占比，再设计改动。禁止未验证猜测。
- **自审等价性**：Planner 提假设时同步判定是否破坏算法语义（改阈值/采样/评分/验证逻辑 = reject；改执行路径/缓存/并行 = allow，须过等价性 gate；架构重构 = allow 但 needs_review 标注风险）。
- 记录 evidence 来源（profiling json stage:Xs / 代码 file:line / 论文）。

## Cycle Protocol
1. Load program_time.md, progress_time.md, results_time.tsv, profiling_breakdown.md
2. Planner 探索 2-3 个方向（基于 profiling 事实），自审等价性边界，选一个可测假设
3. Executor 应用改动 -> smoke check（fd6 单数据集，看 wall-clock + 等价性，crash/debug 用，不入 ledger）
4. 若 smoke 通过（wall-clock 不升 + R/P ±0.5pt），跑 official matching（fd5/6/7，计时）
5. Evaluator 跑 batch_accuracy_evaluation --start 5 --end 7（等价性 gate）
6. 比较 wall-clock vs Current best + R/P vs 三数据集 Equivalence baseline（首 cycle 比 Speed baseline）
7. Archivist 追加 results_time.tsv + 更新 progress_time.md + profiling_breakdown.md
8. keep -> 推进 Current best；discard/crash -> 保留旧 best，revert 改动（或保留作 reference）

## Git and File Hygiene
- branch: 沿用 `deep-anything-reconstructor`
- 每 keep 提交一次（commit message: `perf(da3): cycle N {hypothesis} wall_clock {x}s R{x}/P{x}`）
- discard 时候 git checkout 还原
- `git reset --hard` 需 Rick 批准
- 永不删 results_time.tsv/progress_time.md/cache/datasets/logs/profiling json

## Open Questions
1. 三数据集 Speed/Equiv baseline wall-clock + R/P 具体值？-> Phase 0 测量
2. `parallel_refs` 默认开启是否引入 GPU 争用 OOM？-> smoke 验证（fd6）
3. 跨 ref 共享图像 tensor 是否破坏 batch_all_refs 独立性（某 ref 改 images）？-> 代码审查 images 只读
4. 是否允许关可视化（concise 模式 visualize_results 是否必要）？-> smoke 看等价性
5. batch matmul 浮点非确定是否真导致 R/P 漂移？-> 等价性 gate 实测
6. fd5/6/7 结果与全量 fd2-12 73.27% 差距多大？-> Phase 0 三数据集 baseline 揭示代表性

## 已知优化候选（Planner 优先考虑，待 profiling 证优先级）
| 方向 | 预期 wall-clock 收益 | 复杂度 | 等价性风险 |
|---|---|---|---|
| 跨 ref 共享图像 tensor | 大（省 N×image_load） | 中 | 低（图像只读） |
| parallel_refs 默认开启 | 中-大（N×并行） | 低 | 中（GPU 争用 OOM） |
| 投影向量化 batch | 中（GPU 并行） | 中-高 | 高（浮点非确定，须过 gate） |
| target KD-tree 预计算共享 | 中 | 中 | 低（target 只读） |
| 关闭可选可视化 | 小-中 | 低 | 低（看 concise 是否必要） |
| npz mmap | 小 | 低 | 低 |
