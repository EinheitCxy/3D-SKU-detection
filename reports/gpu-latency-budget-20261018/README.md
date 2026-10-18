# 14GB预算下的GPU延迟优化

Rick允许其他流程使用GPU，要求峰值不增加且不超过14GB。本轮先按更严格的十进制14,000,000,000bytes、**本任务所有GPU PID同一时点合计**验证；外部resident单列。14GiB与14GB的最终单位澄清仍待回复，本次结果低于两种上限。

## 已完成：video3受限DA3候选

GPU2、同29帧／504／seed42／原模型与精度，新进程一次。只改变native allocator策略：进程启动前`PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True`，模型搬GPU前把Torch allocator限制为12GiB。没有修改生产默认、输入、数学算术或checkpoint。

| 指标 | 本次受限候选 |
|---|---:|
| 任务driver采样峰 | **12,918,456,320bytes = 12.918GB = 12.03125GiB** |
| Torch allocated生命周期峰 | 12,169,374,720bytes |
| Torch reserved生命周期峰 | 12,475,957,248bytes（11.619GiB） |
| allocator限制 | 12,884,901,888bytes（12GiB） |
| 预警停止线／预算 | 13.5GB／14GB，未触发 |
| 采样 | nominal50ms，302timestamp组，最大实际间隔约68ms |
| 数值输出 | 15项完整数组dtype／shape／值精确相等 |
| worker退出码 | 0 |

实际backend为native；snapshot有2→10个expandable segments，无不支持警告。未出现monitor解析失败。原GPU2 PID6221／8550分别1580／3450MiB，单列且保持原状；worker3511594在完成后退出。两位medium参与设计，GPU reviewer独立核对结果PASS。[完整标量结果](allocator-result.json)、[执行探针](probe_da3.py)、[审查依据](da3_memory_audit.md)。

worker生命周期13.760725s、其中runner main10.011166s；本轮含提前CUDA初始化与snapshot，旧完整流水线内部窗口不同，**不计算严格配对速度收益**。旧15132MiB／14.777GiB属于另一轮200ms采样，只作历史参照，也不能从两次独立最大值之差推导精确context或碎片大小。

PyTorch区分活跃tensor与allocator保留，保留的闲置缓存也显示在nvidia-smi；expandable_segments用于调查不同大小分配留下的碎片，但本次没有逐块分解其收益。[官方2.7内存说明](https://docs.pytorch.org/docs/2.7/notes/cuda.html#memory-management)。fraction只约束Torch allocator，context、库和其他PID仍须独立观察；50ms监测与预警不能证明亚采样瞬时绝不越界。因此当前结论严格为**这个video3候选的本次任务合计采样峰符合预算**，不是所有输入／全流水线的硬上限证明。

## 仍未推广的部分

- video6历史活跃tensor峰13,688,829,440bytes，高于本候选12GiB allocator cap。未在这个cap下再运行video6，不通过OOM或降低质量来假装完成；仅修改allocator不足以保证所有视频都符合十进制14GB。
- 生产DA3默认未写入该配置，避免让原本较大输入未经验证就因固定cap失败。下一步应根据确认的14GB／14GiB单位验证大输入，必要时研究模型分支驻留生命周期。
- 已验证的Torch bake完整GLB66.589→44.671s仍是独立实验。CPU/GPU共享投影与frame/page循环、显式设备传递、模型进程退出后独立delivery的方案见[正式接入设计](delivery_integration_design.md)。根视频入口目前只有Viewer，新增GLB入口或改变公共参数属于后续实施范围；没有修改独立Docker checkout或部署服务。
- NN／SOR／DBSCAN等流程可继续调查GPU，但需保留质量与邻域语义；没有把全量N²距离矩阵作为默认替代，也不声称所有CPU工作迁GPU都会更快。

## 执行方式与边界

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/gpu-latency-budget-20261018/probe_da3.py \
  --out runtime/gpu-latency-budget-20261018/allocator-01
```

输出目录必须不存在；上例已执行，重跑需新目录。parent不导入Torch，uv启动新session worker，显式隔离GPU2并复用本机模型／数据；模型搬GPU前限allocator，50ms合计owned PID，观测超过13.5GB或120s超时仅终止该自有进程组。worker成功退出后再比较既有reference的15数组；不重跑已知超预算control，不hash、不降低精度／分辨率／帧数、不自动重试／放宽预算。

实际执行AST parse、CLI help、一次完整受限DA3及15数组比较；没有生产行为改动，因此未额外跑全量pytest。原始采样CSV、日志、NPZ仅runtime保留。状态与事件：[STATUS](STATUS.md)、[JOURNAL](JOURNAL.md)。
