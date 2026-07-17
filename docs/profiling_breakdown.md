# Profiling Breakdown - da3 SKU Matching 速度优化

## Phase 0 Profiling 结果（fd2，5 ref，76.3s，pairing_3d=next）

计时 instrumentation: `utils/profiling.py`（StageTimer，纯加法，零开销 no-op）。
验证: 带/不带 `--enable_profiling` 的 matching_summary.txt **字节级完全一致**（5 ref 全 IDENTICAL）-> 计时不破坏算法。

### Stage 耗时排序
| stage | total | calls | per-call | 占比 |
|---|---|---|---|---|
| **sam3_mask** | **57.585s** | 5 | 11.5s | **75.4%** ⭐ |
| image_load_resize | 5.232s | 5 | 1.05s | 6.9% |
| target_bbox_match | 1.659s | 398 | 4.2ms | 2.2% |
| ref_point_sampling | 0.687s | 436 | 1.6ms | 0.9% |
| projection_3d_to_2d | 0.145s | 398 | 0.36ms | 0.2% |
| post_process | 0.087s | 5 | 17ms | 0.1% |
| load_data | 0.011s | 5 | 2.2ms | 0.01% |
| cache_npz_load | 0.002s | 5 | 0.4ms | ~0% |
| uniqueness_constraint | 0.001s | 4 | 0.25ms | ~0% |
| **总计** | **76.3s** | | | |

计时自洽: batch_all_refs_total(76.35s) ≈ per_ref_total(76.0s) ≈ process_images(76.0s)。

## 瓶颈分析

### ⭐ 瓶颈 1: sam3_mask (75.4%)
- 每 ref 一次 SAM3 forward，~11.5s/ref
- 模型已模块级缓存（`_SAM3_PREDICT_INST_CACHE`），开销=推理本身非加载
- 已对 ref 所有 bbox batch（一次 forward 处理多 bbox）
- **加速方向**: SAM3 forward 在 CUDA 释放 GIL -> `parallel_refs` 跨 ref 线程并行（N ref 同时 forward）；或 reduce SAM3 调用频率

### 瓶颈 2: image_load_resize (6.9%)
- 每 ref 逐张 PIL open+resize（sku_matching_system.py:146-164），N ref 重复 N 次
- **加速方向**: 跨 ref 模块级缓存图像 tensor（图像只读）

### 非瓶颈（已优化或极小）
- cache_npz_load (0.002s): PI3_SCENE_CACHE 模块级缓存生效，仅首 ref load
- projection_3d_to_2d (0.145s): 逐点投影非瓶颈（曾怀疑，profiling 证否）
- 三重 Python 循环（target_bbox_match+ref_point_sampling+projection 合计 2.5s，3.3%）: CPU-bound 但占比小，向量化收益有限
- uniqueness_constraint (0.001s): 贪心 fallback 极快

## 数据集规模（ref 数 = 图片数）
| 数据集 | 图片数 | 预估 sam3 时间(@11.5s/ref) | 预估总时间 |
|---|---|---|---|
| fd2 (profiling) | 5 | 57.6s | 76.3s（实测） |
| fd5 | 8 | ~92s | ~120s |
| fd6 | 11 | ~127s | ~165s |
| fd7 | 16 | ~184s | ~240s |
| 三数据集合计 | 35 | ~403s | ~525s |

## 优化候选优先级（基于 profiling 事实，已重排）

| 优先级 | 方向 | 预期收益 | 证据 | 复杂度 | 等价性风险 |
|---|---|---|---|---|---|
| **P0** | **parallel_refs 跨 ref 并行 SAM3** | 大（75%→75%/workers） | sam3_mask 75%，CUDA 释放 GIL | 极低（CLI flag，不改代码） | 低-中（GPU 浮点非确定+OOM） |
| P1 | 跨 ref 共享图像 tensor | 中（省 7%） | image_load_resize 6.9% 重复 N 次 | 中（模块级缓存） | 低（图像只读） |
| P2 | 投影向量化 batch | 极小 | projection 0.2%，非瓶颈 | 中-高 | 高（浮点非确定） |
| - | 关闭可视化 | 极小 | post_process 0.1% | 低 | 低 |
| - | npz mmap | 极小 | cache_npz_load 0.002s | 低 | 低 |

**结论**: profiling 证伪了"三重循环/投影"是瓶颈的假设，证实 **SAM3 推理占 75%** 是唯一显著瓶颈。优化焦点应完全集中在此：并行化 SAM3（P0）。

## Cycle 1 候选假设（待 Planner 正式提出）
- **假设**: `--parallel_refs N`（N=4-8）跨 ref 并行，将 sam3_mask 75% 并行化
- **机制**: ThreadPoolExecutor 跨 ref 并行；SAM3 forward 在 CUDA 后端释放 GIL；cache/SAM3 模块级单例只读线程安全
- **改动**: 仅 CLI flag，不改代码
- **等价性**: 每 ref 独立 SKUMatchingSystem + 独立输出目录，无共享写入 -> 应等价；风险=Gpu 浮点非确定 + OOM
- **待验证**: GPU 内存（24G，N worker SAM3 forward 争用）-> smoke fd6 测
