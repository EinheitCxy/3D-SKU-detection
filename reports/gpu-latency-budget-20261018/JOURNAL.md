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

## G03 — 2026-10-19 00:03 CST，扩大到video6验证

Rick要求“帮我测试效果”。继续使用autoresearch、get-available-resources和work-journal，两个medium agents分别泛化既有受限探针至video6／只读核对GLB输入和比较范围。资源GPU2空闲19038MiB、原6221／8550resident，约419GiB RAM可用、磁盘10.84GiB；GPU0／1不用于本轮。新结果均新runtime目录，不覆盖旧证据。

方案：先对video6完整48帧／504／原精度运行同12GiB allocator候选，budget14e9／stop13.5e9不放宽。若OOM或超停止线，保留失败并如实报告，不删帧或降精度；不重跑已知超预算control。随后对既有同generation做48帧raw tile样本，正确性通过才各一次CPU2／GPU2完整GLB对照；计时含GPU启动和传输，质量比较置于timer外。video6帧来自既有显式1.5s研究recipe，未验证原检测对齐，不作准确率推广。

## G04 — video6 DA3失败；GPU bake样本通过

video6-allocator-01 worker3571597退出1：depth head refinenet1/out_conv的conv2d申请360MiB时触及12GiB allocator cap，整卡仍约6.46GiB空闲。allocated生命周期峰12498550272、reserved峰12641632256bytes；native expandable实际启用。任务driver采样峰13021216768bytes、298组、最大实际间隔约55ms、无无效行，未触发13.5GB停止线。没有生成完整结果，也没有执行15数组比较。GPU reviewer独立复核：这是cap下OOM，不能说video6完整推理满足预算，也不能据此排除所有其他保持质量的显存方案。没有自动放宽cap或降帧／精度。

新增probe_bake.py仅外部监督实验runner：GPU2隔离，50ms采样owned PID时点和，保留resident，13.5GB停止线／300s deadline，只清理本次session。video6-bake-sample-01覆盖48个frame，raw corners／完整RGBA均exact，任务driver采样峰473956352bytes，320组、最大间隔约54ms，无无效行。两臂同批划分的bake累计CPU2.611821／GPU2.022958s，GPU含第一次context与传输，图片decode／准备和比较不在这两个timer内；不把sample差额当完整导出收益。样本通过后启动全量CPU encoder2控制video6-native-01，完成后才能启动GPU完整候选。

## G05 — CPU完整控制完成；监控中断与修复

video6-native-01完整导出100.497878s，脚本至ready103.826038s，2007183 baked／4014366三角／49页／198805712bytes。CPU2控制一次，不重跑以挑选更好时间。

delivery reviewer发现新增监督脚本缺SIGTERM清理及采样新鲜度gate，先修复并获静态PASS。首个完整GPU候选full-01随后在约15.34s被新鲜度检查停止（任务采样峰473956352bytes），没有完整GLB／输出比较，不作成功或加速证据。进一步复核发现新检查使用“上次已消费”采样时间、且位于dequeue之前：父线程落后时，即使队列有新数据也能误中止。原CSV共300组／15.183s／最大gap65ms，而result仅flush276组，支持消费落后解释；没有arrival日志，不能完全还原判定瞬间。另做两个各5s的只读GPU2 idle监控对照，plain pipe与stdbuf均正常约50ms逐行到达，buffering假设不获支持，未引入stdbuf。

修复：reader记录最新arrival，仅队列为空才检查1s新鲜度，并新增最大消费lag。SIGTERM／reader异常传播／首尾覆盖／原1s source gap gate／13.5GB停止线保留。独立新目录full-02恢复一次，只重跑被中断GPU候选，不重跑CPU或改变算法／样本／encoder数。full-01失败结果、监控诊断保留在本目录，原日志保留runtime。

## G06 — 2026-10-19 00:15 CST，完整video6 GPU bake通过

full-02正常退出，export71.574736s／脚本ready75.098048s；相对本轮CPU100.497878s节省28.923142s／28.779853%。whole_process93.001903s含就绪后17.903855s比较，不能当导出耗时。2006 batches／2027824LOD候选／2007183baked／4014366三角／49页，RGB／depth各48次上传。document关联、全部accessor、200736000decoded RGBA texels全部exact；不是整个GLB字节或全raw tile验证。

driver任务采样峰476053504bytes，allocated38639616／reserved54525952bytes；1852组、最大timestamp gap56ms、无无效行。supervisor最大消费lag11.099499s，意味着停止响应延迟；本轮只能使用完整CSV作事后采样峰结论，不宣称实时停止或瞬时14GB硬cap。GPU reviewer独立核对全部指标／输出／预算限制PASS。producer95.979967→55.011276s，但encoding join0.984595→12.629608s；后续值得比较GPU bake+encoder4，不在本轮额外启动。

收尾GPU2只剩原6221=1580MiB／8550=3450MiB，无本次作业。结果与失败标量拷贝至本report目录，原GLB／CSV／日志留runtime；更新STATUS／README及canonical profiling。实验源由main@1edf6ff加本轮probe_da3／probe_bake改动构成，生产代码保持原样。未重新测全流水线、浏览器、准确率或P95，未commit／push／部署；原有cleanup／camera／accuracy工作保留。

## G07 — 2026-10-19 00:18 CST，本地Git提交

Rick引用CPU100.498／GPU71.575s对照并要求提交Git。按commit-helper在main@1edf6ff、空暂存区起步，将本轮video6探针／监督脚本、精简成功及失败结果和相关文档合为一个功能提交，共14个文件、约200KB。生产Torch bake实现已在此前c0edc06提交，本次不改变生产默认。原始GLB／NPZ／CSV、模型／环境及既有cleanup／camera／accuracy工作不纳入；只暂存明确路径，不push，不重复GPU实验。既有AST／JSON／链接与实际对照验证沿用，提交前检查暂存diff，提交后核对未纳入的工作区改动保留。
