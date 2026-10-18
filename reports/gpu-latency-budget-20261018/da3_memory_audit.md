# DA3 14GB任务显存预算：只读审查与有界验证计划

2026-10-18。仅审查源码与既有PID/计时标量；本次没有初始化CUDA、运行模型或基准，也没有修改生产源码。
预算严格使用十进制 **14,000,000,000 bytes**（约13.039GiB），按本任务所有GPU PID在同一时刻的driver用量合计；不是14GiB，也不是整卡总用量。其他任务resident不计入本任务，但会影响可用容量。

## 已定位：旧峰属于DA3

证据目录：`runtime/latency-optimization-20261018/{baseline,candidate}-video3/`，文件为`gpu-processes.csv`、`process-PID.json`、`events-PID.jsonl`及`receipt.json`；阶段总量另见`reports/latency-optimization-20261018/pair-summary.json`。

| 同一次完整运行 | baseline | candidate |
|---|---:|---:|
| DA3 PID | 2857934 | 3106308 |
| DA3 driver采样峰 | 15132MiB | 15132MiB |
| DA3 allocated生命周期峰bytes | 12251022848 | 12251022848 |
| DA3 reserved生命周期峰bytes | 15422455808 | 15422455808 |
| core/PVS PID | 2857733 | 3105898 |
| core/PVS driver采样峰 | 3788MiB | 3788MiB |
| core/PVS allocated峰bytes | 3341922304 | 3341922304 |
| core/PVS reserved峰bytes | 3527409664 | 3527409664 |

两臂CSV按timestamp分组、仅取对应任务PID求和后，任务同时峰仍为15132MiB，分别在21:03:56.991与21:28:17.723；该采样时刻只有DA3 PID计入。15132MiB=15,867,052,032bytes，超过预算1,867,052,032bytes。
events显示DA3重建结束后才进入matching：candidate reconstruction `[108119.369429,108133.075752]`，matching `[108133.076777,108162.784321]`；baseline同样顺序。不能把两个PID各自生命周期峰相加。
因此本问题首先是DA3，而非PVS。现有200ms采样不能排除更短暂的driver尖峰，也没有足够数据定位到某个具体forward层。

reserved峰比allocated峰大3,171,432,960bytes，但两个独立最大值不一定同时发生；这不是已经测出的“可释放碎片”。driver峰减reserved峰为444,596,224bytes，同样不是精确同步测得的context大小，更不是未来开销上界。

## 源码确定的精度与生命周期

- `src/da3_runner.py:224–242`：独立runner将模型搬到GPU后，调用同一inference；没有在这里转换模型精度或分批改变帧数。
- `src/da3_model_loading.py:141–159`：CPU构造、覆盖审核、原生strict safetensors加载；选择性跳过随机初始化没有改变GPU参数精度，因此旧优化未改变上述两臂显存峰。
- `Depth-Anything-3/src/depth_anything_3/api.py:99,125–130`：forward被`inference_mode`装饰；内部根据设备支持选择bf16/fp16 autocast，没有显式传`cache_enabled`。
- 同文件`:200–214,310–325`：输入上GPU后跑forward，再转换prediction。`:379–386`已有forward前后同步；保留这些边界，不增加逐层同步。
- `model/da3.py:132–146`：backbone在外层autocast中；depth head/camera等在局部`enabled=False`区间，不能把整个模型强制half替代。
- 同文件`:362–370`：anyview随后metric，再做alignment；两个分支的模型参数同时驻留。anyview的输出仍被metric后的alignment使用。不能直接删除这些输出，也不能假设缩短局部变量作用域就能回收主要峰。
- `api.py:209–222`：raw_output与GPU输入局部变量可存活到inference结束；在forward之后删除它们或`empty_cache()`不能回溯降低已经发生的forward峰。

## 首选候选：保留数学运算，只改allocator策略

建议由协调者批准后，在新的独立DA3进程验证一个候选：进程启动前设`PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True`；同一进程在任何模型/输入GPU分配前设置allocator预算 **12GiB=12,884,901,888bytes**。
调用形式为`torch.cuda.set_per_process_memory_fraction(float(12 * 2**30 / total_device_memory_bytes), device=0)`；分母必须来自该实际设备总容量，不能用当前free或24GB字符串推算。调用会初始化CUDA，必须放进候选初始化计时与driver监测范围。
12GiB比旧allocated峰多633,879,040bytes，给总预算留下1,115,098,112bytes供context、库和其他任务内GPU PID使用；这些只是预算分配，不能证明一定够用。

本机Torch2.7.1安装源码入口（均位于`runtime/viewer-latency-20261017/test-env/lib/python3.11/site-packages/torch/`）：
- `cuda/memory.py:165–191`：fraction限制caching allocator，超过会OOM；分母是设备总容量，不是free。它不是整个任务/driver的硬限额，且每个进程独立。
- `include/c10/cuda/CUDAAllocatorConfig.h:25–33`：expandable_segments取决于编译期`PYTORCH_C10_DRIVER_API_SUPPORTED`；不支持时警告并返回false。
- `include/c10/cuda/CUDACachingAllocator.h:70–80,105`：存在expandable segment状态和SEGMENT_MAP记录。
头文件证明该功能接口存在，不足以证明当前wheel+driver实际启用。正式候选必须记录backend、配置、警告和snapshot中的segment状态；若不支持，应明确失败，不能把退回普通allocator的结果标成expandable成功。
它可能减少不同大小分配造成的保留/碎片，也可能无益或OOM；本次未检查wheel对应的完整C++实现，不能承诺它在本模型上的收益或准确的保留量变化。

## 次选可证伪候选：autocast cache_enabled=False

只在上述候选不足且仍允许继续时，单独比较cacheoff；不同时更换dtype、分辨率、帧数、seed、attention实现或质量参数。
本机`amp/autocast_mode.py:260–273,365–371`显示内层autocast未传cache参数时继承当前cache flag；`:395–400`在最外层退出时清cache并恢复状态。因此可在DA3 inference调用的同一线程临时设置cache flag=false，并在finally恢复，无需改vendor；内层原来的bf16/fp16选择与局部FP32区间仍保留。
本机wheel没有附带`aten/src/ATen/autocast_mode.cpp`，已读Python/header没有给出`cached_cast`的完整资格判断。**不能仅凭inference_mode断言当前没有cache，也不能断言关闭一定释放约一份低精度权重。** 该假设须通过allocated/reserved实测变化证伪；没有变化即不支持其收益。
预期只改变权重转换复用/临时生命周期，数学精度策略保持相同；但exact输出仍须验证，不能由“只改cache”直接推出。它也可能增加重复转换与latency。

## 有界验收与失败停止

1. 不重跑已知超14GB的原配置；复用既有baseline NPZ。只运行固定video3同29帧、504、seed42的单个候选，独立新输出，保持现有权重与strict加载。
2. 首先记录任务PID树、逻辑设备/物理UUID、模型CPU加载结束与CUDA初始化前状态；监测仅汇总本任务所有GPU PID，同一timestamp求和。后续PVS/Torch bake不得与DA3并行驻留占用任务预算；若流程改成persistent worker，需要重新核算驻留和重叠，不能沿用旧峰。
3. allocator fraction在模型搬运前生效；所有任务内Torch GPU进程都必须分配明确子预算，不能每个进程各给12GiB。其他库/直接CUDA分配不受此fraction覆盖。
4. 保留正常forward同步/D2H；记录初始化、to-device、forward后、prediction完成的allocated/reserved和生命周期峰，以及可用的inactive-split/segment统计；不重置峰来隐藏初始化。高频driver采样记录任务合计与缺失样本。
5. 14e9是driver总约束。可设置更低的监测停止阈值（例如13.5e9）留响应余量，但外部采样/终止只能限制风险，不能保证阻止亚采样瞬时越界。若要求可证明的严格硬上限，现有Torch fraction+采样工具不足，不能宣称已经满足；需要协调者决定可接受的执行隔离/保证方式后才运行。
6. allocator OOM、配置不支持、任务driver合计接近停止线、输出不一致均停止，保留失败结果；不自动放宽预算或切CPU/half/降帧。
7. 成功要求既有15项DA3完整数值数组精确相同（含frame/camera绑定），且任务driver采样峰低于14e9；同时明确这仍为采样观测，不是瞬时硬上限证明。报告完整reconstruction wall，不能把初始化/额外转换排除后宣称净加速。

当前结论：已证明旧峰来自DA3；优先验证12GiB allocator限制与expandable_segments，cacheoff为独立后续候选。两者均未运行，未证明能在总14GB内完成，也未证明输出或latency收益。
