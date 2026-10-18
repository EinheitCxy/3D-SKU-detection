# GPU latency under a 14 GB task budget

更新：2026-10-18 23:57 CST。Rick允许其他流程使用GPU，条件是峰值不增加且不超过14GB。按此前任务进程口径，保守采用**本任务所有GPU PID同时合计≤14,000,000,000bytes**；其他人的resident单独记录，不修改。14GB／14GiB单位澄清待回复，本次成功结果低于两种上限。

当前已确认：旧video3峰15132MiB＝14.777GiB来自DA3子进程。其Torch allocated峰12,251,022,848bytes，reserved峰15,422,455,808bytes；driver峰超预算，不能继续把它当符合14GB的基线。PVS不是此峰的来源。

工作范围：先只读定位DA3保留显存与GPU bake正式接入点；候选只写本目录，不改生产默认，不部署其他checkout。优先试不改变算术的allocator碎片治理，显式限制Torch allocator为12GiB，为CUDA context／非allocator分配留余量；不降低帧数、分辨率、精度或画质。allocator限制不等于总driver硬上限，必须另采样任务PID合计，检测超预算即停止，不能以allocated小于14GB冒充合规。

- Coordinator：预算、实验调度和判断，必要的有界DA3探针。
- Medium GPU reviewer：DA3/autocast/allocator只读审计，da3_memory_audit.md。
- Medium delivery reviewer：真实导出调用链和最小API/生命周期方案，delivery_integration_design.md。

本轮video3受限候选已完成：native expandable_segments、allocator限12GiB，任务driver采样峰**12.918GB／12.03125GiB**，reserved峰11.619GiB，15项完整输出数组精确相同。nominal50ms／实际最大间隔68ms，302组，无解析失败、不支持警告或预警触发；worker exit0，GPU reviewer独立复核PASS。它是单输入n1采样结果，不证明瞬时硬上限或所有输入可运行。[报告](README.md) · [结果](allocator-result.json)。

上一轮GPU bake已验证GLB66.589→44.671s、最终交付一致，仍为实验。正式接入设计已完成：当前根视频入口只生成Viewer，新增显式GPU GLB交付须独立delivery及模型退出屏障；未改Docker／生产默认。video6历史活跃tensor13.689GB高于本候选12GiB allocator，尚未按此cap运行，不推广统一固定cap。

Git main起点c0edc06；此前源码已commit/push，剩余原有cleanup／accuracy／camera工作保留。本轮事件见[JOURNAL.md](JOURNAL.md)。
