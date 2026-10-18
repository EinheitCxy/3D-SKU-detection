# 2026-10-18 GPU2 profiling 记录

当前入口：[docs/profiling_breakdown.md](../../docs/profiling_breakdown.md)，已扩展为[多视频同次细分](../profiling-multivideo-20261018/README.md)。本目录保留前一轮29帧video3实测248.899597秒；Viewer128.749960秒，ZIP137.148453秒。它是历史单次主跑，未混入新多视频均值，也不代表热服务延迟。

本目录保存可移植的机器汇总、独立组件数字、逐reference表、时间线、诊断源码和算法解释。本机大产物、原始logs/200ms采样、模型和环境在 `runtime/profiling-20261018/`，不作为报告正文重复保存。没有提交或上传权重/数据/GLB。

## 文件

- [summary.json](summary.json)：主跑互斥阶段、可用里程碑、准备重叠、GPU/RSS口径、每进程CPU/allocator峰、产物及29refs。
- [references.csv](references.csv)：ref/target slot、wrapper wall、PVS wall/encoder、boxes；reference按当前串行0..28对应。
- [timeline.svg](timeline.svg)：主跑时间线，准备线程与matching重叠。
- [results/da3.json](results/da3.json)、[PVS](results/pvs.json)、[Viewer/dedup](results/viewer.json)、[ZIP/GLB](results/delivery.json)、[初始化重叠](results/overlap.json)：独立细分，禁止和主跑相加。
- [GPU/matching详细算法](algorithms_gpu_matching.md)、[几何/采样详细算法](algorithms_geometry.md)、[交付详细算法](algorithms_delivery.md)：source pointers、loops、shape、阈值和边界。
- [review.md](review.md)：独立复核；[PLAN.md](PLAN.md)：本轮范围；[工作记录](../../docs/work-journal/2026-10-18-profiling/JOURNAL.md)。

## 实际执行顺序

1. 只读资源、输入、代码与权重前置核对；原HF权重链接失效，从已有本地项目镜像提取同snapshot，未启动服务。
2. `run_profile.py` 启动粗粒度 `profile_core.py`（通过DA3_VENV_PYTHON接入`profile_da3.py`），之后 `profile_delivery.py`。全链路主值只来自这次。
3. 主跑结束后，依次执行 `diagnose_gpu.py --mode da3`、`--mode pvs`、`diagnose_viewer.py --dedup`、`diagnose_delivery.py`。
4. 因PVS主跑load20.118s与独立8.053s有差异，追加一次`diagnose_overlap.py`，30s timeout内成功，定位重叠时模型构建17.362s。
5. `build_summary.py` 读取主receipt/events/CSV，过滤本任务PID并对齐显存时点，生成主summary；结果文档与旧入口替换随后完成。

原始证据：[receipt](../../runtime/profiling-20261018/receipt.json)、[core log](../../runtime/profiling-20261018/core.log)、[delivery log](../../runtime/profiling-20261018/delivery.log)、[host samples](../../runtime/profiling-20261018/host-processes.jsonl)、[GPU samples](../../runtime/profiling-20261018/gpu-processes.csv)、[线程采样](../../runtime/profiling-20261018/diagnostics/overlap/thread-samples.jsonl)。这些runtime链接仅在保留本机产物的checkout可读。

## 重现

项目根运行，显式使用独立环境，避免根失效`.venv`。主脚本会拒绝已有receipt/outputs；再次测量必须设置新的绝对`PROFILE_OUT`，在该目录准备`da3-model`指向已恢复的同snapshot根。它默认绑定物理GPU2 UUID、OMP/BLAS/MKL=4、全29帧、新结果缓存；资源预检要求至少17GiB GPU空闲、4GiB磁盘空闲。每stage上限30min，先核对共享GPU状态。

```bash
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/profiling-20261018/run_profile.py
```

以上是本轮实际入口；该默认输出已完成，直接再次执行会拒绝覆盖。主脚本中`prepare()`只复用已有JPEG/canonical JSON，不重跑检测/分类。原始29帧数据目录须仍在。

单独重放组件时，`--out`必须是新目录，GPU诊断只在主跑结束后串行执行：

```bash
CUDA_VISIBLE_DEVICES=GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/profiling-20261018/diagnose_gpu.py --mode da3 \
  --base runtime/profiling-20261018 --out runtime/profiling-20261018/diagnostics/da3-new \
  --model-path runtime/profiling-20261018/da3-model/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7
```

PVS改`--mode pvs`及新out。CPU Viewer/delivery以`CUDA_VISIBLE_DEVICES=''`运行，命令和算法说明在各自md。初始化重叠诊断用同GPU2环境及`timeout --signal=TERM --kill-after=5s 30s`包住uv入口；只prepare+load，不predict。单次重放不是受控速度比较。

## 结果解释

主CPU/GPU计时均有明确边界；没有把nested inclusive、异步wall、CUDA events和独立窗口相加。`self`只减去已插桩同线程直接子区间，剩余不等于纯Python或纯CPU。

新诊断复现PVS构建膨胀，线程采样/CPU和官方Open3D未释放GIL的绑定支持GIL阻塞解释；没有精确测出全部等待归属或进程隔离的可回收时间。独立Viewer的ground结果与主跑不同，最终总点数相同并不能证明来源/画面等价。本轮没有算法promotion或新画质结论。
