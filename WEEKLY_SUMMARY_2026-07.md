# 周工作总结：DA3 重建后端集成与自动优化（2026-07-09 ~ 2026-07-17）

> 分支：`deep-anything-reconstructor` · 起始 commit `5e64dee` → 终点 commit `255f4f2`
> 最终成果：da3 **Recall 71.50% → 73.27%（+1.77pt）**，Precision 85.54% → 85.09%（-0.45pt，可接受）

本周工作围绕 **Depth-Anything-3（DA3）** 这一新 3D 重建后端的集成、调优、与自动化探索展开，分三个部分：①DA3 实现；②工程改进；③基于 autoresearch 协议的自动优化。

---

## 一、DA3 的实现

### 1.1 背景：为什么引入 DA3

原系统支持 `pi3`（cache 化、快、批量友好）与 `vggt`（实时、每轮重推理）两个 3D 重建后端。DA3（`depth-anything/DA3NESTED-GIANT-LARGE`，6.3GB，metric 米制深度，CC BY-NC 4.0）是精度更高的多视图重建模型，作为第三个后端接入，目标是在货架跨图 SKU 匹配任务上提升重建/投影质量。

### 1.2 依赖冲突与 subprocess 隔离架构（commit `3635c8f`）

DA3 依赖 `numpy<2` + `omegaconf`/`addict`/`e3nn`/`evo` 等，与 `code/` 的 venv 依赖集冲突（注：非 numpy 版本冲突，code/ 实际 numpy 1.26.1，是**依赖集差异**）。解决方案是**子进程隔离**，而非引入新依赖或污染主 venv：

```
code/ (numpy 1.26.1 主 venv)
  └─ modules/da3_3d_reconstructor.py   # 主进程：ReconstructorBase 子类，subprocess 调用
        │ subprocess: Depth-Anything-3/.venv/bin/python modules/da3_runner.py
        ▼
code/modules/da3_runner.py             # 子进程：独立运行于 DA3 venv，不 import code/
        └─ da3_cache/predictions.npz   # 输出 Pi3 兼容 schema
```

- **`modules/da3_runner.py`（新）**：独立 DA3 推理脚本，在 `Depth-Anything-3/.venv`（numpy<2）下运行。多视图批量推理，将 depth + extrinsics(w2c) + intrinsics 反投影为 `world_points`，写入 `da3_cache/predictions.npz`，schema 与 Pi3 完全兼容（`depth` N,H,W,1；`world_points` N,H,W,3；`extrinsic` 支持 3,4 与 4,4）。
- **`modules/da3_3d_reconstructor.py`**：从 in-process 重写为 subprocess 架构。`load_model`→noop（校验 DA3 venv/runner），`run_inference`→subprocess 后读 npz，`export_glb`→skip（SKU 匹配只需 npz）。修正 extrinsics `(3,4)→(4,4)` 反投影补齐。

### 1.3 后端注册机制（commit `f38285b` + `9d9503f`）

引入 registry 模式替代硬编码 if/else，便于扩展新后端：

```python
RECONSTRUCTOR_REGISTRY = {}
def register_reconstructor(name): ...           # 装饰器注册
def get_reconstructor(name): ...                 # 按名实例化
@register_reconstructor("da3")                   # da3 / pi3 / vggt 均用此注册
class DA33DReconstructor(ReconstructorBase): ...
```

`da3` 接入匹配流水线：`matching_algorithms.py`（`{backend}_cache` 路径泛化）、`sku_matching_system.py`（da3 加入 cache-only 后端列表）、`config.py`（`model_type`/`transform_kwargs`/`preprocess_mode` 支持 da3）、`inference.py`/`main.py`（`--backend` / `--recon_backend` / `--match_backend` 新增 da3 选项）。

### 1.4 端到端验证

`main.py --mode reconstruct --recon_backend da3`（13 图，39.6s）→ `da3_cache/predictions.npz`；`--mode concise --match_backend da3` → `matching_summary.txt` + `correspondences.json`（ref0 24 matches，hit ratio 最高 0.94）。

---

## 二、改进（工程调优 + Bug 修复）

### 2.1 关键 Bug 修复（驱动 da3 Recall 从 22.8% → 71.5%）

da3 全量 Recall 提升轨迹：`22.8%（尺寸bug）→ 29.6%（image_ids 错位暴露）→ 71.5%（image_ids 修复）`

- **image_ids 字典序排序 bug**（`reconstructor_base.py`）：原 `sorted()` 用字典序，≥10 图数据集（fd4/6/11/12）的 `image_ids` 排成 `[1,10,11,2,3,...]` 而非 `[1,2,3,...,10,11]`，导致这些数据集 Recall≈0%。修复为按数字排序键。
- **`allow_pickle` 回归 bug**：`da3_runner` 存 `source_model` 为 object 数组，`np.load` 默认 `allow_pickle=False` 报 "Object arrays cannot be loaded"。修复 `da3_3d_reconstructor.py:143` 与 `matching_algorithms.py:317` 两处 load 点加 `allow_pickle=True`。

### 2.2 da3 后端专属阈值标定（commit `9d9503f`）

da3 与 pi3 的深度/conf 语义不同，需独立标定（`config.py` `for_3d_mapping` da3 分支）：

| 参数 | da3 值 | pi3 值 | 说明 |
|---|---|---|---|
| `min_depth` / `max_depth` | 0.3 / 8.0 m | 0.1 / 3.0 | da3 米制深度，货架 1-5m + 过道纵深 |
| `depth_confidence_threshold` | 1.5 | 0.05 | da3 conf 原始 `[1,25.4]`，非 pi3 sigmoid `[0,1]` |
| `point_3d_confidence_threshold` | 1.5 | 0.05 | 同上 |
| `max_3d_distance` | 0.5 m | 0.8 | 同物体跨视角中心应 <0.1m，0.5 容采样抖动 |
| `depth_consistency_threshold` | 0.3 m | 0.5 | 采样端 cache 自洽性容差 |

### 2.3 评分公式去冗余 + plane 评分

`geometry_3d.py` 的 `combined_score` 精简为三因素（删除无效的 `depth_consistency` 维度）：`match_ratio*0.5 + geometry_score*0.2 + coplanar_score*0.3`（有参考平面时）；无平面时退化为 `match_ratio*0.6 + geometry_score*0.4`。平面共面约束（法向对齐 × 残差分）参与软评分，针对货架同层板场景抑制跨层误匹配。

### 2.4 threshold-scan CLI + 批量脚本（commit `9d9503f`）

- `main.py`：6 个新 3D 阈值 CLI flag 用于网格扫描，`pipeline --recon_model_path` 透传，viewer `--no_confidence`/`--no_class`。
- `bbox_gen.py`：argparse CLI + 校验（conf 范围、imgsz%32、相对模型路径）。
- 新脚本：`batch_pipeline_backend.sh`、`scan_thresholds.sh`、`video2dedup.sh`。

### 2.5 并行 batch_all_refs（commit `f38285b`）

`run_sku_matching()` 新增 `parallel_refs` 参数（`--parallel_refs` CLI，默认 1，传如 4 启用），对 pi3/da3 这类 cache 后端用 `ThreadPoolExecutor` 并行处理多参考图。

### 2.6 仓库治理（commit `9d9503f`）

按 GitHub 治理清理：移除 Docker 服务（Global-ID-Mapping、Dockered_GlobalIDMapping、docker_template）、omni-test 图片、PDF 论文、重复 scripts、EN README、根 requirements*.txt。

---

## 三、用 autoresearch 提升的部分

### 3.1 autoresearch 协议（commit `7e80008`）

参考 `/home/xingyu/agent-skills/my_skills/autoresearch-loop` skill，编写 `program.md` 作为 agent 自动优化 da3 SKU matching 的协议。核心设计：

- **角色分工（Opus 思考 / Sonnet 执行）**：Coordinator/Planner/Critic 用 Opus（深度推理），Executor/Evaluator/Archivist 用 Sonnet（代码生成运行）。
- **Protocol Auditor 合并进 Planner**：Planner 提假设时自审边界（allow/reject/needs_user），不再单独设 Auditor 角色。
- **事实驱动探索**：每假设必须先读诊断数据/代码实证，禁止未验证的猜测性改动。
- **固定实验契约**：dataset `fd2-12`、评估器 `accuracy_annotation.py`、metrics Recall=TP/GT & Precision=correct/common、seed 42、cache 复用。
- **keep/discard 规则**：Recall ≥1pt 提升且 Precision 回退 ≤1pt → keep；否则 discard 并 revert。

### 3.2 自动优化循环执行（7 cycle/诊断，1 成功 → R73.27%）

| Cycle | 方向 | 结果 | 关键证据 |
|---|---|---|---|
| C1 | gaussian 坐标 bug 修复 | discard | 采样分布非主因，R-0.3pt 在 noise floor |
| C2 | GT-free SE2 漂移矫正 | discard | GT-free 估计在密集货架失败（consistency 0.40-0.60） |
| **C3** | **唯一性 fallback 分配** | **KEEP ✅** | **R+1.77pt，救回 223 类竞争淘汰漏检，+38 TP** |
| C4c | 空 mask 回退 non-overlap | discard | P 崩（65.9%→21.3%，产 +651 冗余 FP） |
| C5 | 全局禁用 SAM3（A/B 测试） | discard | fd6 R-2pt，SAM3 保密集覆盖有正价值 |
| C6 | da3 外参修正诊断 | 反转前提 | da3 实优 pi3，外参无系统性偏差可修 |

### 3.3 唯一成功改动：Cycle 3 唯一性 fallback 分配（commit `255f4f2`）

**问题实证**：诊断 fd2+fd12 共 1280 ref-obj 发现，618 个投影命中正确框中 **227 漏，其中 223（98%）是唯一性竞争淘汰**——多个 ref 投影落进同一 target 框，原 `apply_uniqueness_constraint` 只保留最高分、丢弃输者且无 fallback。

**改动**（`code/utils/geometry_3d.py`）：
- `find_best_matching_bbox_with_3d_validation`：返回最佳 match 的同时携带 `validated_candidates`（全部过 projection_match_threshold + spatial_distance 门的候选，按 combined_score 降序）。
- `apply_uniqueness_constraint`：重写为贪心分配——所有 (ref,candidate) 全局按 score 降序，高分 ref 优先占框；被淘汰的 ref 取其候选链中**次优非冲突框**（已被占用的跳过），而非直接丢弃。

**结果**：全量 fd2-12 Recall 71.50%→73.27%（+1.77pt），Precision 85.54%→85.09%（-0.45pt），TP 1505→1543（+38），所有数据集 R 持平或升、无回退。

### 3.4 自动优化沉淀的关键认知

1. **剩余漏检主因是 da3 投影精度**（per-object 深度噪声 + per-frame 独立残差），matching 层（score/分配/prefilter/采样）杠杆已实证用尽。
2. **da3 整体优于 pi3**（da3 R71.5% > pi3 R67.4%，跨帧重投影中位 68px < pi3 92px），非此前假设的"da3 更差"——"pi3 2-3x 优"是 per-ref 错框率被误读为全局。
3. **R73.27% 是当前架构合理上限**：6 个方向经实证否决（pairing all / GT-free SE2 / gap 门控 / Top-K prefilter / mask 回退 / 禁用 SAM3 / 外参修正），继续提升需重建层质量改进。
4. **方法论教训**：离线复现≠pipeline 质量；单点投影精度指标误导；局部现象勿推全局。

---

## 四、速度优化（autoresearch 速度轴，与精度正交）

精度优化收尾后，启动**速度轴**正交优化（`program_time.md` 协议）：在保持 R/P 等价（gate ±0.5pt）的前提下最小化 `--mode concise` 匹配 wall-clock。**fd5/6/7 全量 720s -> 180s（-75%），R/P 全程等价（R83.56%/P93.71%）**。

### 4.1 协议与基线（commit `255f4f2` 起）

- 数据集：fd5/fd6/fd7（小/中/大规模，3 数据集替代全量 11 个，加速迭代）。
- 等价性 gate：R/P 相对三数据集 baseline 在 ±0.5pt 内（batch/CUDA 浮点非确定容许）。
- 可靠 baseline：**720s**（fd5:141+fd6:215+fd7:364）+ R84.01%/P94.06%（wall-clock 须同口径对比，原 543s 偏低 32% 不可靠，GPU 热节流噪声大）。
- 工具：新建 `utils/profiling.py`（StageTimer，纯加法计时，`--enable_profiling` 零开销 no-op，**带/不带 flag 的 matching_summary 字节级完全一致**证不破坏算法）。

### 4.2 profiling 定位瓶颈（推翻假设）

profiling 证实 **sam3_mask 占 75.4%**（fd2，57.6s/76.3s）是唯一显著瓶颈，每 ref ~11.5s SAM3 forward 串行。**证伪**了"三重 Python 循环/投影是瓶颈"的假设（projection 仅 0.2%）。Cycle 4 补细化计时又定位 `build_transforms`（build_da3_transforms 对每图 `Image.open().convert("RGB")` 仅读尺寸却触发完整 JPEG 解码，未缓存）占残余 95%。

### 4.3 自动优化循环执行（6 cycle，4 keep + 1 诊断 + 1 收尾，commit `10c7abc`~`e3ed126`）

| Cycle | 假设/改动 | 结果 | 判定 |
|---|---|---|---|
| C1 | `--parallel_refs 4` 跨 ref 线程并行 SAM3（+`_BATCH_QUERY_LOCK` 原子化） | fd6 wall 191s(+22%)，R+2.92pt 超 gate | **discard** |
| C2 | `sam3_max_batch_size` 5->32（maybe_run_sam3 未传该值用默认5，>5 bbox/ref 时 N×forward） | 565s，sam3 -38%，R-0.45/P-0.35pt | **keep** |
| C3 | `_DA3_IMAGE_CACHE` 模块级缓存图像 tensor（da3 每 ref 逐张 PIL+resize 重复 N 次） | 402s，image_load -90%，R/P **0pt 位级** | **keep** |
| C4 | 补 StageTimer 诊断（定位 build_transforms 36s 真凶） | R/P 与 C3 完全一致（证纯加法） | 诊断 |
| C5 | `_DA3_TRANSFORMS_CACHE` 模块级缓存 transforms_info | 240s，build_transforms -85%，R/P **0pt 位级** | **keep** |
| C6 | build_da3 `.size` 懒读 + 移除每 ref `PI3_SCENE_CACHE.clear()` + [DIAG] 降 debug | 180s，R/P **0pt 位级**，GPU 16.6GB 安全 | **keep** |

### 4.4 核心技术：模块级只读缓存模式

PI3_SCENE_CACHE 模式推广到 4 处只读数据跨 ref 复用：`_DA3_IMAGE_CACHE`（图像 tensor）、`_DA3_TRANSFORMS_CACHE`（transforms_info）、`PI3_SCENE_CACHE`（scene_data，移除每 ref clear）、`sam3_max_batch_size`（避免 N×forward）。**只读数据缓存应位级等价**--C3/C5/C6 三次"0pt 位级一致"铁证（TP/correct 逐数字相同）。

### 4.5 决定性结论与架构上限

1. **`--parallel_refs` 对 SAM3 无效**：SAM3 是 GPU-bound，CUDA kernel 跨线程串行化（4 线程每 call 变慢 4.1x，有效并行 3.86x 被抵消），且多线程 RNG 非确定（`np.random.choice`/`torch.randperm` 全局状态交错）致 R/P 漂移。
2. **跨 ref SAM3 batching 不可行**：SAM3 单图 API 限制（`datapoint.images` 单元素）。
3. **剩余瓶颈 sam3_mask 占 64% 是 GPU compute-bound 不可破**（单次 forward/ref，Cycle2 已优化 batch）--这是速度轴的架构上限（与精度轴 R73.27% 上限正交）。

### 4.6 真实 pipeline 验证（6-1_frames，24 图）

`small_fd_video/video-test/6-1_frames` 完整 pipeline（da3 重建+匹配+分析+去重+可视化）**总 98s**：重建 42s（DA3 推理 35.7s + cache 5.6s）+ 匹配 45s（24 ref，首 ref 17.3s 含首次缓存填充，后续 ref 平均 1.19s）+ 其他 11s。缓存按预期生效（首 ref 付一次性成本，后续 ref 几乎只跑 SAM3）。

---

## 总结

本周完成 DA3 后端从集成（subprocess 隔离 + registry）到调优（bug 修复 + 阈值标定）到自动优化（autoresearch 协议）的全链路，覆盖**精度**与**速度**两个正交轴：

- **精度轴**：7 cycle/诊断迭代（1 成功），da3 达 **Recall 73.27% / Precision 85.09%**，比 baseline 净提升 +1.77pt R，且整体优于 pi3。R73.27% 为当前架构合理上限（6 方向实证否决），生产代码仅保留 Cycle 3 成功 commit（`255f4f2`）。
- **速度轴**：6 cycle（4 keep + 1 诊断 + 1 收尾），fd5/6/7 匹配 **720s -> 180s（-75%）**，R/P 全程等价（R83.56%/P93.71%，3 次位级一致铁证）。核心是 4 处模块级只读缓存（PI3_SCENE_CACHE 模式推广）+ profiling 驱动定位真瓶颈。剩余 sam3_mask 64% 是 GPU compute-bound 不可破（速度轴架构上限）。真实 pipeline（6-1_frames 24 图）总 98s 验证缓存生效。

成果文件：`progress.md`/`progress_time.md`（循环状态）、`results.tsv`/`results_time.tsv`（append-only 实验台账）、`docs/accuracy_da3_vs_pi3.md`（精度对比）、`docs/profiling_breakdown.md`（速度瓶颈分解）、`program_time.md`（速度协议）。commits：精度 `5e64dee`→`255f4f2`，速度 `10c7abc`→`e3ed126`+文档 `d7109eb`。
