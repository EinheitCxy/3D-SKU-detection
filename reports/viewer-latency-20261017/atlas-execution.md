# Atlas 1/2/4 线程执行结果 — 2026-10-17

两个完整固定视频导出序列均满足预先固定标准：2 线程相对各自串行 baseline 的两对节省为 42.155 / 40.903 秒；4 线程为 43.009 / 39.650 秒。每对都为正，runner 均为 `pass`。这只证明当前固定视频、固定 CPU 与相同 capture 条件下的完整 GLB 导出改善；不是 GPU、服务、上传或浏览器端到端耗时。生产默认仍为 1，等待主线程独立评估。

## 固定条件与相等性

输入为 `runtime/video3-full-profile-20261016/viewer/runs/c9c3be6802144fcf97a942085b118422`：29 frames、2,402,589 points、grid `[280,504]`。每个正式 arm 产生完整 31 页，capture 数组共 646,398,888 bytes，最终 GLB 为 121,513,344 bytes。

每个序列单 process 按 B1/C1/C2/B2 执行，唯一变项为 `encoder_workers` 与对应 FIFO 容量（1 vs 2 或 1 vs 4）。固定 CPU `{3,22,27,29}`，数值库 thread 设置为 4，GPU 隐藏、NumPy seed 0。radius 2、tile 8、LOD、128 palette、MEDIANCUT、dither NONE、PNG optimize 与既有长边规则未改。未安装依赖、访问网络、运行 GPU/model/service 或修改输入 sidecar。

所有正式 arm 都逐页比较完整 prequant RGBA（含透明 RGB）、positions、UV、indices 的 dtype/shape/元素；并比较完整 GLB 全部 bytes、全部 PNG decoded RGBA、全部 compact accessors 和完整 JSON/材质关联。全部通过；B1 包括自检。相同 page copies 在每个 raw export timer 内，全部比较在 timer 外；未从主计时扣除 capture、组件或重叠时间。

## 每臂结果

| 独立序列 | Arm | workers / queue | raw export wall (s) | export CPU (s) | sampled export RSS (GiB) | 累计 ru_maxrss (GiB) | equality |
|---|---|---:|---:|---:|---:|---:|---|
| 1 vs 2 | B1 | 1 / 1 | 112.818 | 251.481 | 1.459 | 1.459 | pass, 31 pages |
| 1 vs 2 | C1 | 2 / 2 | 70.663 | 253.928 | 2.235 | 2.235 | pass, 31 pages |
| 1 vs 2 | C2 | 2 / 2 | 70.555 | 254.091 | 2.207 | 2.275 | pass, 31 pages |
| 1 vs 2 | B2 | 1 / 1 | 111.459 | 249.352 | 2.241 | 2.321 | pass, 31 pages |
| 1 vs 4 | B1 | 1 / 1 | 112.707 | 251.647 | 1.462 | 1.462 | pass, 31 pages |
| 1 vs 4 | C1 | 4 / 4 | 69.699 | 250.863 | 2.277 | 2.277 | pass, 31 pages |
| 1 vs 4 | C2 | 4 / 4 | 69.908 | 251.711 | 2.265 | 2.277 | pass, 31 pages |
| 1 vs 4 | B2 | 1 / 1 | 109.559 | 245.316 | 2.183 | 2.277 | pass, 31 pages |

CPU 是同一进程 user+system 的导出区间差值，包括 capture 和测量线程开销。RSS 每 0.1 秒采样，可能漏过更短峰值；后续 arm 保留 B1 的约 616.45 MiB page captures 与约 115.88 MiB GLB bytes，且 allocator 缓存与进程高水位不会重置。因此 B1 与 C/B2 的 RSS 不能解释为 encoder 额外分配；`ru_maxrss` 是整个同一进程的累计高水位，不能当作 arm 独立峰值。raw 页约 20 MiB/页；2/4 页队列约 40/80 MiB，仅是原始页数据，不是总 RSS。

## 完整命令、退出码与资源

### atlas-official-2w-01

- UTC 开始：`2026-10-17T11:10:45.833369+00:00`；结束：`2026-10-17T11:17:09.716271+00:00`。Derived retrospectively from command receipt write mtime and measured outer wall duration; not direct clock captures。
- 两对节省：`42.154631` / `40.903232` s（`37.37%` / `36.70%`）；minimum paired saving `40.903232` s。
- exit `0`；outer wall `383.883` s；子命令 CPU `1033.343` s（含导入/比较）；runner RSS 峰 `2.321` GiB；无 observed child、GPU 0。
- [完整原始记录](atlas-official-2w-01/partial.json)、[command/environment/exit receipt](atlas-official-2w-01/command.json)、[stdout](atlas-official-2w-01/stdout.log)。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/viewer-atlas-uv-cache timeout --signal=TERM --kill-after=1s 889s uv run --no-project --python /home/xingyu/3D_Recognization/runtime/viewer-latency-20261017/test-env/bin/python python /home/xingyu/3D_Recognization/reports/viewer-latency-20261017/run-atlas-compare.py --phase official --candidate-workers 2 --run-dir /home/xingyu/3D_Recognization/reports/viewer-latency-20261017/atlas-official-2w-01
```

### atlas-official-4w-01

- UTC 开始：`2026-10-17T11:18:03.408987+00:00`；结束：`2026-10-17T11:24:23.719010+00:00`。Direct UTC captures immediately before launch and after wait。
- 两对节省：`43.008757` / `39.650235` s（`38.16%` / `36.19%`）；minimum paired saving `39.650235` s。
- exit `0`；outer wall `380.310` s；子命令 CPU `1024.461` s（含导入/比较）；runner RSS 峰 `2.296` GiB；无 observed child、GPU 0。
- [完整原始记录](atlas-official-4w-01/partial.json)、[command/environment/exit receipt](atlas-official-4w-01/command.json)、[stdout](atlas-official-4w-01/stdout.log)。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/viewer-atlas-uv-cache timeout --signal=TERM --kill-after=1s 889s uv run --no-project --python /home/xingyu/3D_Recognization/runtime/viewer-latency-20261017/test-env/bin/python python /home/xingyu/3D_Recognization/reports/viewer-latency-20261017/run-atlas-compare.py --phase official --candidate-workers 4 --run-dir /home/xingyu/3D_Recognization/reports/viewer-latency-20261017/atlas-official-4w-01
```

实际正式子命令使用 `timeout 889s --kill-after=1s`，外层收集器使用 899s deadline；均在 900s/command 上限内结束。runner RLIMIT_CPU soft/hard 为 1999/2000 s，RLIMIT_AS 为 12 GiB，单文件 256 MiB。

## Smoke、保留失败与唯一修复

初始 2w smoke 在 B1 已完成导出后的比较阶段失败：NumPy 1.26.4 首次加载 `numpy.testing` 时启动 SVE CPU-feature 检查子进程，触发 runner 的禁止子进程约束并发出 SIGUSR1。它不是导出异常，也没有得到 equality/性能通过结论。原始失败、B1 GLB、日志和耗时完整保留，未覆盖。向 coordinator 报告并获准唯一诊断修复后，以直接 `np.array_equal` + 显式 raise 替换 testing helper；保留严格 dtype/shape、全部元素和完整 GLB bytes 检查，不更改 timer/输入/判定。没有放宽禁止子进程约束。

| Attempt | status / exit | outer wall (s) | 子命令 CPU (s) | 记录 |
|---|---|---:|---:|---|
| atlas-smoke-2w-01 | failed / 1 | 7.109 | 13.764 | [partial](atlas-smoke-2w-01/partial.json), [log](atlas-smoke-2w-01/stdout.log), [command](atlas-smoke-2w-01/command.json) |
| atlas-smoke-2w-01/retry | smoke_pass / 0 | 7.059 | 15.004 | [partial](atlas-smoke-2w-01/retry/partial.json), [log](atlas-smoke-2w-01/retry/stdout.log), [command](atlas-smoke-2w-01/retry/command.json) |
| atlas-smoke-4w-01 | smoke_pass / 0 | 7.041 | 14.912 | [partial](atlas-smoke-4w-01/partial.json), [log](atlas-smoke-4w-01/stdout.log), [command](atlas-smoke-4w-01/command.json) |

2w 重试与修复后首次 4w smoke 都完整四臂通过。Smoke 使用同一 generation 前 2,048 点，仅产生 1 页；其计时不是性能证据，也不单独证明多页并行。正式各 31 页及先前五页四线程功能测试负责覆盖多页路径。之后没有再 retry、补测或扩大候选。

## 资源总量、建议与限制

- 所有五次尝试的子命令 receipt 合计 CPU `2101.483` s、outer wall `785.402` s（不含 resource detector、报告生成及交互间隔）；低于本 worker 5,000 CPU s / 2,400 elapsed s 分配，余量充足。所有执行均顺序、最多一个 heavy command。
- 所有新 run 目录当前 linked files 合计 `301,078` bytes；只留原失败 B1 小 GLB 与 logs/JSON。成功当前 GLB 在比较后删除，临时 BIN 正常关闭。峰值磁盘按单 BIN + 单当前 GLB、各不超过 256 MiB 的结构约束限制，远低于 1 GiB；未存 raw atlas 到磁盘。
- [资源就绪记录](atlas-resource-readiness.json)：可用 RAM 433.49 GiB、disk 22.55 GiB，四个固定 CPU 可用；测试解释器 Python 3.11.11、NumPy 1.26.4、Pillow 12.3.0。没有使用共享主机独占性假设。
- 两个候选均满足对各自串行 baseline 的固定验收标准。建议优先让主线程独立评估 2 线程作为采用候选：其队列更小，而现有独立序列并未建立 4 相对 2 的额外收益。4 可保留为显式选项；没有 8 线程证据。
- 这是两个独立的 1vs2 和 1vs4 sequence，不能仅比较 C 绝对值或相近 wall 值就认定 2/4 全局最优、等效或非劣。共享主机负载、page capture、baseline retained RAM 和顺序效应仍是限制；没有误差条或跨运行统计显著性主张。
- 生产默认、源码、tests、输入 sidecars 和 controller state 本执行未改。唯一修改是 runner 在任何正式计时前的 CPU/RSS诊断和已披露的比较 helper 修复。未推断 E2E Viewer/service/browser/COS 收益，未部署或发布。
