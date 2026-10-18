# GPU2 多视频完整 profiling 与内部 breakdown

本目录对应2026-10-18用户要求“不同fd-videos视频、平均时长、原六项指标的每项内部breakdown”。当前汇总入口为[docs/profiling_breakdown.md](../../docs/profiling_breakdown.md)。测量进行中时，JSON的pending记录明确保留；最终结果以complete receipt和最终aggregate为准。

最终完成：video1/3/4/5/6/7，161帧/6508检测，6/6成功。完整平均229.250920秒，Viewer119.265629秒、ZIP127.241765秒、GLB自身101.178546秒；任务显存峰均值14.729818GiB、全样本最大17.226562GiB。每video一次、显式输入recipe、细分仪器开销未量化；不与上一轮单video混平均。[验证](validation.json) · [复核](review.md)。

逐视频完整JSON：[video1](video1-summary.json)、[video3](video3-summary.json)、[video4](video4-summary.json)、[video5](video5-summary.json)、[video6](video6-summary.json)、[video7](video7-summary.json)；[video2](video2-summary.json)仅待确认状态。

## 结果与阅读入口

- [aggregate.json](aggregate.json)：每视频等权的mean/median/min/max/n，以及逐指标缺失与来源；所有内部指标各自独立，不能逐行相加。
- [INTERNALS.md](INTERNALS.md)：完整host wall标签表、self、calls、范围和n，另列CUDA与点数；对应[内部均值CSV](detail-averages.csv)。
- [per-video-metrics.csv](per-video-metrics.csv)：逐视频原始标量及源JSON pointer。[references.csv](references.csv)：所有ref的框数、wrapper/predict/load/native计时。
- 各`videoN-summary.json`：同次运行的互斥时间线、Viewer/ZIP前缀、过程/allocator快照、采样资源和点数流向。未测case也有状态JSON，不能把文件存在等同成功。
- [输入证据](inputs.md)、[显式实验recipe](sampling-records)、[计划](PLAN.md)、[前一轮单video3正文快照](previous-single-video.md)。
- [独立point range CPU probe](point-range-probe.json)：在全部视频结束后运行，typed查询与Python int标量对照；未计入均值或实施生产改动。

默认case目录为`runtime/profiling-multivideo-20261018/videoN/`，保存input/settings/receipt、生产pipeline summary、matching stages、detail-PID与process-PID、events、200ms GPU/RSS样本、原始logs，以及dataset/outputs/viewer/ZIP/GLB大产物。模型只复用上一轮恢复的同snapshot，未另复制。大数据仅保留本机runtime，没有发布/上传。

## 协议与边界

GPU2固定UUID `GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7`，已有resident保留；OMP/BLAS/MKL=4，新process/新DA3-PVS结果cache，OS/模型文件页cache不清空。所有视频重计算串行。使用当前默认NN/SOR准备、DBSCAN/ground/LOD、商品2M/背景500K、atlas worker1、128色和最终Surfel长边限制；没有改变生产算法。

video3沿用既有0起1.5s recipe；1/4/5/6/7使用帧数吻合的显式1.5s实验recipe，未声称找回原时间戳或验证检测对齐，性能仅适用于这些声明的输入。video2采样有歧义待确认，8无JSON。未评估匹配准确率/人工画质。全部JSON帧保留，最长48帧也完整执行。

所有细分在同次core+delivery完整运行中收集。AST host spans、scalar观察、thread CPU读取与CUDA events有未量化开销；这不是无仪器生产延迟。CUDA只在算法原有同步边界后取event elapsed，不新增逐层sync；allocator不重置峰。host与CUDA、父子inclusive、并发prepare与matching、allocator/driver峰不可相加。只有总时间线互斥阶段及各milestone prefix可相加并闭合。

不包含detector/classifier推理、HTTP/BSON、COS/网络/浏览器、自动footprint计量。视频抽帧/JSON适配单独计时，不算进原六项的主流程定义。每视频一次，只报跨场景均值，不报热服务或重复实验的P95。

## 重现和更新汇总

根目录使用现成uv环境（根`.venv`失效未修）；原实验目录拒绝覆盖，重跑必须使用新的case目录。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/profiling-multivideo-20261018/prepare_inputs.py \
  --video 3 --out runtime/profiling-multivideo-20261018/new-video3

PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/profiling-multivideo-20261018/run_case.py \
  --case runtime/profiling-multivideo-20261018/new-video3
```

其他video显式传`--sampling-record`，格式见inputs.md；勿将未确认的例子视作历史事实。`run_batch.py`仅用于本次1/4/5/6/7实验recipe，新batch receipt不可覆盖。每stage上限30min、GPU空闲至少17GiB、磁盘空闲至少4GiB；错误保留，不为通过而缩帧/换模型。只终止自己创建的进程组。

汇总与文档生成是纯CPU/JSON，不启动模型或导出：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/profiling-multivideo-20261018/build_summary.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/profiling-multivideo-20261018/render_report.py
```

脚本分工：`detail_common`统一span/event/快照；`core_detail`接入DA3/PVS/Viewer AST；`profile_core/profile_da3/profile_delivery`运行实际生产入口；`delivery_detail`记录同次ZIP/GLB；`summarize_core/summarize_delivery/detail_metrics`只读原始JSON；`build_summary`按每video聚合；`render_report`更新唯一主文档与完整内部表。

详细算法继续引用[DA3/PVS/matching](../profiling-20261018/algorithms_gpu_matching.md)、[几何/过滤/采样](../profiling-20261018/algorithms_geometry.md)、[LOD/烘焙/atlas/ZIP/GLB](../profiling-20261018/algorithms_delivery.md)。这些文件中的旧独立诊断时间属于旧窗口；最新平均计时只取本目录。
