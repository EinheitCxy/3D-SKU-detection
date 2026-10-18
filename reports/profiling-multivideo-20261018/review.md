# 多视频 profiling 复核结果

2026-10-18；3个medium subagents协作实现/审查，root执行并综合。最终结果：PASS。

- 输入与geometry reviewer：1/3/4/5/6/7完整161帧、6508检测；requested7/attempted6/success6/failed0、2待采样确认；全部milestone前缀误差0。SAM商品域与cluster域区分，ground apply n=5不补0；video6商品预算求和/1330组/末组337字段已修正。GPU只有有效compute-query且无任务PID才记观测0，否则缺失。
- Delivery reviewer：六视频ZIP/GLB均值重算一致；LOD输入−移除=选入、选入−跳过=baked、两三角/Surfel、页数/bytes一致；七块不超过bake父段，compact=join+write，未把inclusive/独立组件加到总量。ZIP表outer wrapper和INTERNALS inner span口径已明确。
- GPU/core reviewer：process detail唯一读取、按role稳定键提取，未重复平均投影；同名snapshot只称边界采样均值，allocator生命周期峰/driver阶段峰分开；typed searchsorted值得下一步真实数据验证，未修改生产代码。

复核中修复的是**本轮诊断汇总/呈现问题**：epoch/monotonic锚点、采样窗口、running/pending分母、误将末组quota称总预算、条件性n、features说明及ZIP成员范围。没有借此改变默认算法或重跑已有成功视频。

root最终一次必要验证见[validation.json](validation.json)：6/6全阶段成功、输入/refs/PVS/输出数量、所有前缀和时间线闭合、GLB小header和ZIP成员/压缩方法、15脚本AST及35本地文档链接通过。没有CRC/hash、全量pytest、浏览器或准确率/画质验收。GPU2已回到原6221/8550两resident，总5030MiB。

[Point range独立CPU probe](point-range-probe.json)在所有视频完成后执行：250万int32、200queries，Python int1.152541秒、typed scalar0.001243秒、typed vector0.000141秒，端点一致；这是合成诊断，未混入均值，未证明生产全流程收益。
