# 14GB预算下的GPU延迟优化

Rick允许其他流程使用GPU，要求峰值不增加且不超过14GB。本轮先按更严格的十进制14,000,000,000bytes、**本任务所有GPU PID同一时点合计**验证；外部resident单列。14GiB与14GB的最终单位澄清仍待回复。2026-10-19补测结果：**video6的GPU GLB导出快28.8%且交付内容精确一致；12GiB allocator配置未能完成video6 DA3。** 不能声称整条流水线已经符合14GB。

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

## video6补测：GPU bake有效，DA3固定cap失败

GPU2／RTX4090D，既有48帧generation、250万Viewer点；CPU线程4，encoder2、source_world、tile8、batch≤1024、LOD／palette／长边限制均保持原样。原始输入、帧数、模型精度没有降低。CPU完整控制和GPU完整候选各一个成功新进程；另一个GPU候选因监控误判中断并保留。OS缓存、共享CPU负载没有隔离，新进程不等于物理冷缓存。这里只测已有Viewer之后的GLB组件。

| 同次video6 GLB指标 | Native CPU | Torch GPU |
|---|---:|---:|
| 完整export wall | 100.497878s | **71.574736s** |
| 脚本至GLB就绪，含import | 103.826038s | 75.098048s |
| producer窗口，含LOD等、扣主线程compact/write/join | 95.979967s | 55.011276s |
| 主线程compact/write/join | 4.516137s | 16.561592s |
| ↳ encoding join | 0.984595s | 12.629608s |
| LOD后候选／baked | 2027824／2007183 | 相同 |
| 三角／atlas页／GLB bytes | 4014366／49／198805712 | 相同 |

完整导出减少**28.923142s／28.779853%**。GPU计时含首次CUDA context、frame上传、CPU投影、GPU采样／mask／延展、D2H、编码和写盘。就绪后17.903855s的逐项比较单列，不算交付耗时；whole_process93.001903s含验证，不能代替71.574736s。producer与compact为原报告的互斥窗口，encode worker会与producer重叠，不能重复相加。

先完成48×1024＝49152输入的raw sample：48560个配对quad／4856000个raw RGBA texels，corners顺序、透明RGB、alpha全部exact。完整导出的document关联、全部accessor字节和49页**200736000个decoded RGBA texels**精确一致，缺页／shape／alpha／RGB变化均0。完整处理2006 batches，RGB／depth各上传48次。没有逐项比较完整导出的全部raw tiles或整个GLB文件字节，也没有浏览器人工画质或匹配准确率验收。

GPU任务driver采样峰**476053504bytes＝0.476054GB（454MiB）**，allocated峰38639616、reserved峰54525952bytes；这些口径不混用。nominal50ms，1852组、最大timestamp gap56ms、解析失败0，任务结束后只剩原6221／8550 resident。但supervisor消费采样最大延迟**11.099499s**，停止判断也受此延迟影响：原始CSV支持事后采样峰，不支持实时停止或瞬时硬cap保证。独立reviewer复核性能／输出／该限制PASS。

DA3另一次受限测试保持video6完整48帧／504／seed42。native expandable实际生效，但depth head `refinenet1/out_conv`的conv2d申请360MiB时触及12GiB allocatorcap而OOM；当时整卡仍约6.46GiB空闲。worker退出1，allocated峰12498550272／reserved峰12641632256bytes，任务driver采样峰13021216768bytes（13.021GB）、298组／最大gap55ms，无监测线触发。**没有完整输出，也未执行15数组比较。** 这证明当前固定cap不足，不证明所有保持质量的方案都不可行。未自动放宽cap或降低输入／精度。

首个GPU完整尝试因新鲜度检查先于dequeue而中断：CSV300组／最大gap65ms，但父线程仅flush276组。代码已改为reader记录最新arrival、队列空才判断新鲜度，同时记录消费lag；SIGTERM清理、reader异常传播及原阈值保留。idle pipe／stdbuf对照未发现缓冲问题，未引入stdbuf。中断运行不参与速度结论，CPU控制没有重跑。[中断记录](video6-bake-interrupted-result.json) · [监控诊断](monitor-diagnostic.json)。

[video6汇总](video6-summary.json) · [CPU完整结果](video6-native-result.json) · [GPU完整结果](video6-bake-result.json) · [raw样本](video6-bake-sample-result.json) · [DA3失败](video6-allocator-result.json)。原始CSV、worker日志和GLB仅保留`runtime/gpu-latency-budget-20261018/`。

## 仍未推广的部分

- video6已在这个cap下运行并OOM；video3通过不能推广所有视频。完整流水线在14GB下完成仍未证明。
- 生产DA3默认未写入该配置，避免让原本较大输入未经验证就因固定cap失败。下一步应根据确认的14GB／14GiB单位验证大输入，必要时研究模型分支驻留生命周期。
- 已验证的Torch bake完整GLB66.589→44.671s仍是独立实验。CPU/GPU共享投影与frame/page循环、显式设备传递、模型进程退出后独立delivery的方案见[正式接入设计](delivery_integration_design.md)。根视频入口目前只有Viewer，新增GLB入口或改变公共参数属于后续实施范围；没有修改独立Docker checkout或部署服务。
- NN／SOR／DBSCAN等流程可继续调查GPU，但需保留质量与邻域语义；没有把全量N²距离矩阵作为默认替代，也不声称所有CPU工作迁GPU都会更快。
- GPU bake变快后，video6编码join增至12.630s；下一项值得测试的是GPU bake配合显式encoder4，而不是把该等待时间直接承诺为可省时间。本轮未执行这个新对照。

## 执行方式与边界

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/gpu-latency-budget-20261018/probe_da3.py \
  --out runtime/gpu-latency-budget-20261018/allocator-01
```

输出目录必须不存在；上例已执行，重跑需新目录。parent不导入Torch，uv启动新session worker，显式隔离GPU2并复用本机模型／数据；模型搬GPU前限allocator，50ms合计owned PID，观测超过13.5GB或120s超时仅终止该自有进程组。worker成功退出后再比较既有reference的15数组；不重跑已知超预算control，不hash、不降低精度／分辨率／帧数、不自动重试／放宽预算。

video3阶段实际执行AST parse、CLI help、一次完整受限DA3及15数组比较；没有生产行为改动，因此未额外跑全量pytest。原始采样CSV、日志、NPZ仅runtime保留。状态与事件：[STATUS](STATUS.md)、[JOURNAL](JOURNAL.md)。

补测入口：DA3探针加`--video 6`并指定新out；CPU控制用`reports/torch-bake-fusion-20261018/native_export.py`；GPU样本／完整导出用本目录[probe_bake.py](probe_bake.py)的`--mode sample/glb`。完整GPU模式要求`--reference-glb`，均固定GPU2、encoder2和1024batch。准确执行命令在各结果JSON的`command`字段；输出路径已存在，复现必须换新目录。新增脚本AST／CLI检查通过；实测之外没有重跑无关pytest或生产全流水线。实验阶段未commit／push／部署，后续本地Git提交见[JOURNAL](JOURNAL.md)。
