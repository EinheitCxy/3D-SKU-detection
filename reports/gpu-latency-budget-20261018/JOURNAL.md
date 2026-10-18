# Budget investigation journal

## G01 — 2026-10-18 23:48 CST

Rick表示其他流程可自由使用GPU但任务峰不得超过14GB。先按十进制14e9bytes保守执行，并保留之前任务PID合计／外部resident分开的口径。两个medium agents只读调查DA3显存与实际GLB接入，不启动CUDA、不写生产。

当前资源历史证明峰来自DA3：candidate PID3106308，任务15132MiB；allocated12.251GB而reserved15.422GB，提示需区分活跃tensor和allocator保留。尚未实测碎片治理候选，不把差额全归因为可回收内存。

查询本机PyTorch源码与官方2.7 AMP／CUDA文档。cache_enabled=False只是待查方向，不能假设inference_mode下仍有可省的权重缓存。下一候选优先考虑expandable_segments且限制allocator12GiB，保留数学和输入不变；total driver仍需独立采样，非allocator分配不受该fraction覆盖。

Git核对main@c0edc06，索引为空，之前本人优化全部提交；当前剩余原有用户改动不打包成新提交。

## G02 — 2026-10-18 23:57 CST，固定video3受限候选通过

运行probe_da3.py parent，经uv启动隔离GPU2 worker3511594。allocator设native／expandable_segments且限12GiB，保留29帧／504／seed42／原精度。输出在runtime/gpu-latency-budget-20261018/allocator-01，没有重跑已知超预算control。

worker exit0，lifecycle13.760725s／runner main10.011166s；采样峰12918456320bytes（12.91845632GB／12.03125GiB），未触发13.5GB停止线或14GB预算。nominal50ms，302组、最大实际间隔约68ms，无解析失败；原6221／8550resident单列。allocated峰12169374720／reserved峰12475957248bytes，snapshot实际有expandable segments，无不支持警告。15完整数值数组dtype／shape／值全部exact。GPU2收尾只保留原resident，任务worker退出。

GPU reviewer独立审查PASS：旧15132MiB属于另一次200ms窗口，不计算严格配对降幅；fraction与采样都不证明所有时刻的driver硬cap。本次可说video3合计采样峰低于14GB，不能推广所有场景。

另查video6历史allocated13688829440bytes，高于此12GiB allocatorcap，故不把它设成生产默认。已异步询问14GiB或十进制14GB单位，等待期间保持严格口径；本次样本满足两者。正式GPU bake调用链方案写完，根视频流程目前没有GLB，不把研究runner称正式接入；未扩展到Docker／部署。
