# da3 SKU Matching 速度优化 - 循环状态

## 目标
最小化 fd5/6/7 concise matching wall-clock，保持等价性（三数据集 baseline R/P ±0.5pt）。
协议: program_time.md

## Phase 0: Profiling ✅ 完成
- 计时 instrumentation: utils/profiling.py + 12 stage 注入，字节级 diff 证纯加法（带/不带 --enable_profiling matching_summary 完全一致）
- fd2 profiling（5 ref, 76.3s, pairing_3d=next）:
  - **sam3_mask 57.6s = 75.4%** ⭐ 唯一显著瓶颈（每 ref ~11.5s SAM3 forward）
  - image_load_resize 5.2s (6.9%), 三重循环合计 2.5s (3.3%), projection 0.145s (0.2%), 其余 <0.1s
  - 证伪: "三重循环/投影是瓶颈"假设 -> projection 仅 0.2%
- 详见 docs/profiling_breakdown.md
- 已有优化（不可回退）: PI3_SCENE_CACHE（npz 复用）、_SAM3_PREDICT_INST_CACHE（模型不 reload）、SAM3 已对 ref bbox batch、parallel_refs 框架已存在（CLI flag）

## Phase 0: 三数据集 Baseline ✅ 完成
- config: pairing_3d=next（当前默认，不改），enable_sam3_mask_sampling=true，commit 255f4f2
- **Speed baseline: 720s**（fd5=141s + fd6=215s + fd7=364s，refs=33，**可靠重测**）
  - 注: 原 543s 偏低不可靠(GPU热节流/负载噪声), 720s 为可靠基线; wall-clock 须同口径对比
- **Equivalence baseline: R84.01% / P94.06%**（fd5/6/7 聚合 TP=557/GT=663, correct=554/common=589）
  - fd5: R83.33% P90.48% (8 ref)
  - fd6: R87.88% P96.09% (9 ref)
  - fd7: R82.05% P94.10% (16 ref)
- 等价性 gate: cycle 间比 R/P ±0.5pt 内（基线 R84.01%/P94.06%）

## Cycle 2 [KEEP ✅] max_batch_size 5->32
- 假设: max_batch_size 5->32 使每 ref 1次forward替代⌈N/5⌉次,省batch overhead(set_image+collate+launch重复)
- 事实: maybe_run_sam3 未传 max_batch_size 用默认5; fd5/6/7每ref bbox 12-102, 99bbox需20次forward+20次set_image
- 改动: config.py 加 sam3_max_batch_size(默认5兼容,yaml设32); sam3_utils.py maybe_run_sam3 传参; 128/64 OOM,32通过
- 结果(可靠baseline5 720s口径): wall 720->565s(**-21.5%**), sam3 fd6 91.8->56.8s(**-38%**)
- 等价性: official R 84.01->83.56(-0.45pt) P 94.06->93.71(-0.35pt) 均<0.5pt gate内 ✅
- 新瓶颈: image_load_resize 35%(fd6 67.3s,每ref重复加载所有图像) -> Cycle 3 目标
- Current best: wall=565s R83.56%/P93.71% (config sam3_max_batch_size=32)

## 优化候选优先级（profiling 证后重排）
| 优先级 | 方向 | 收益 | 证据 | 复杂度 | 等价性风险 |
|---|---|---|---|---|---|
| **P0** | parallel_refs 跨 ref 并行 SAM3 | 大（75%→75%/N） | sam3 75%, CUDA 释放 GIL | 极低（CLI flag） | 低-中（OOM+浮点） |
| P1 | 跨 ref 共享图像 tensor | 中（省 7%） | image_load 6.9% 重复 N 次 | 中 | 低 |
| P2 | 投影向量化 | 极小 | projection 0.2% 非瓶颈 | 中-高 | 高 |

## Cycle Log

### Cycle 1 [discard] parallel_refs + SAM3 加锁
- 假设: --parallel_refs 4 跨 ref 并行 SAM3(75%瓶颈)，给 _BATCH_QUERY_COUNTER 加 Lock 原子化
- 改动: utils/sam3_utils.py 加 _BATCH_QUERY_LOCK（保留，正确线程安全修复）
- 结果(fd6 smoke): wall-clock 156s->191s(+22%), R87.88->90.80(+2.92pt), P96.09->96.89(+0.80pt)
- 失败双重: ①R/P 超 gate 根因=采样 RNG 全局状态非确定(np.random.choice/torch.randperm 多线程交错)非forward不安全 ②wall-clock 反升 根因=SAM3 GPU-bound CUDA kernel 跨线程串行 4线程争抢每call变慢4.1x
- **决定性结论: parallel_refs 对 SAM3 无效(GPU-bound CUDA串行)；跨 ref batching 不可行(SAM3 单图 API 限制 datapoint.images 单元素)**
- Executor 发现: batch循环内每批重复 set_image(line 1274-1275)；maybe_run_sam3 调 self_exemplar 未传 max_batch_size 用默认5

### Cycle 2 [KEEP ✅] max_batch_size 5->32
- 假设: max_batch_size 5->32 使每 ref 1次forward替代⌈N/5⌉次,省batch overhead
- 结果(可靠baseline5 720s): wall 720->565s(-21.5%), sam3 fd6 91.8->56.8s(-38%), R-0.45pt/P-0.35pt gate内
- 新瓶颈: image_load_resize 35% -> Cycle 3

### Cycle 3 [KEEP ✅] 跨 ref 缓存图像 tensor
- 假设: 模块级缓存图像tensor(_DA3_IMAGE_CACHE, key=sorted paths+TW+TH+device), 跨ref复用, 省N×image_load
- 改动: utils/sku_matching_system.py 加 _DA3_IMAGE_CACHE + da3 分支查cache命中复用
- 结果(Cycle2 565s口径): wall 565->402s(**-29%**), image_load fd6 67.3->6.9s(**-90%**)
- 等价性: R/P **0pt位级一致**(TP554/GT663,correct551/common588 完全相同) -> images只读缓存假设铁证 ✅
- Current best: wall=402s R83.56%/P93.71% (config sam3_max_batch_size=32 + _DA3_IMAGE_CACHE)

### 瓶颈结构(Cycle3后, fd6 profiling)
| stage | total | 占per_ref |
|---|---|---|
| **未计时开销** | ~63s | **56%** ⭐(待补计时定位) |
| sam3_mask | 50.2s | 44%(已Cycle2优化,绝对值降) |
| image_load | 6.9s | 6%(已Cycle3优化) |
| 三重循环 | ~3s | 3% |
- per_ref_total 113s 但 sam3+image_load+循环=60s -> **63s未计时**(build_transforms/find_object_correspondences非匹配部分/Python/torch/可视化)
- Cycle4方向: 补计时定位63s未计时开销

## Cycle 4 [待启动] 补计时定位未计时开销
- 事实: per_ref_total 113s vs 已计时stage 60s = 63s(56%)未覆盖
- 候选未计时区: build_transforms(sku_matching_system.py:152)、find_object_correspondences_3d_mapping整体(matching_algorithms.py:279-560非sam3非projection部分)、_post_process可视化、torch tensor操作
- 任务: 补细化计时(子stage)定位63s主因, 再设计优化

## Notes
- 与 program.md 精度优化正交: 不改算法语义，只改执行路径
- 等价性 gate: 每次改动跑 fd5/6/7 评估，R/P 须 ±0.5pt 内
- 搜索不可用（WebSearch 工具故障 + 搜索引擎反爬，15+ 次尝试证），用代码分析 + 已有 GIL/torch 知识
