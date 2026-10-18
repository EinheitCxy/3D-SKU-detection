# GPU latency under a 14 GB task budget

更新：2026-10-19 00:18 CST。按本任务GPU PID同时合计≤14,000,000,000bytes执行；外部resident单列，14GB／14GiB单位确认仍待回复。此处均为采样证据，不是瞬时硬cap。

**本轮效果测试结束，无运行作业。** video6完整GPU GLB导出100.497878→71.574736s，减少28.923142s／28.779853%；document、全部accessor、49页／200736000个decoded RGBA texels精确一致。原48帧／250万点／LOD／长边／encoder2不变。任务driver采样峰476053504bytes，无缺页或解析失败。[汇总](video6-summary.json) · [实测说明](README.md)。

完整GPU过程1852组／最大gap56ms，但supervisor消费延迟峰11.099499s；不能把记录峰值称为实时预算保障。首次尝试因dequeue前新鲜度检查中断，修复后新目录重测成功；中断证据保留，不用于速度结论。CPU控制仅跑一次。

**DA3显存方案尚未全通过。** 之前video3／29帧在12GiB allocator下通过，driver峰12.918GB、15数组exact。本次video6／48帧触及allocatorcap OOM，driver记录峰13.021GB；没有完整输出或数组比较。未放宽cap、减少帧数／分辨率或降精度。[失败结果](video6-allocator-result.json)。

生产默认、完整流水线和Docker部署均未改；本轮只修改实验脚本、结果和文档。原GPU2 PID6221／8550仍1580／3450MiB，测试进程已退出。两个medium agents完成输入／脚本／失败及最终结果的独立只读复核。

下一步候选：DA3降低活跃／同时驻留内存；GPU bake配合encoder4实测；监督端避免消费积压。尚未执行，不承诺收益。Rick追加要求将本轮实验与报告提交本地Git，本次不push；提交范围见[JOURNAL](JOURNAL.md)，当前性能入口为[profiling breakdown](../../docs/profiling_breakdown.md)。
