# GPU2 多视频 profiling

状态：多视频profiling及后续video3优化实现/组件实验/最终GPU2完整对照均已完成，三位medium最终复核PASS。更新：2026-10-18 21:39 CST。性能入口：[profiling_breakdown.md](../../profiling_breakdown.md)，实施见[task_plan](../../../reports/latency-optimization-20261018/task_plan.md)、[progress](../../../reports/latency-optimization-20261018/progress.md)。

完成video1/3/4/5/6/7各一次完整运行，161帧/6508检测，6/6成功。完整平均229.251秒，Viewer119.266秒，ZIP127.242秒，GLB自身101.179秒；任务driver采样峰均值14.730GiB、最大17.227GiB，Viewer点均值1764058.5。完整范围147.990–380.629秒。

所有细分来自同一次完整运行：266个host函数/block标签、2922项计时/计数/显存指标；逐video/ref表和source pointer完整。前缀/阶段闭合、数量及必要格式、AST/链接检查PASS，两位medium最终审查PASS。另在视频全部结束后做一次typed searchsorted CPU合成probe，端点一致，仅作下一步候选证据，未混入均值。

限制：1/4/5/6/7为显式1.5s实验recipe，未验证原检测对齐；video2采样歧义仍待确认，8无JSON。每video一次、仪器开销未量化；不代表热服务/HTTP/COS/浏览器或准确率/画质。旧单video3结果保留历史，未混入平均。

本轮video3原始完整baseline已完成253.139s。真实CPU检查证实1377个物理参数覆盖完整，6个state缺项是共享别名；构造11.827→.652s、含审核加载12.994→2.045s。Point ranges同数组711端点一致，3.490→.000964s。完整GLB1/2/4worker为108.396/67.627/66.638s，全部geometry与128295600decodedRGBA texels相等，选默认2。

最终同次video3 pair：完整253.139→186.894s（缩短26.2%），Viewer129.854→109.163s，ZIP138.097→117.362s，GLB114.041→68.770s。同次DA3构造12.624→.914s、重建25.133→13.706s，point ranges3.939481→.001608s。各臂一次、相同function instrumentation和diagnostic捕获；不与前述六视频AST均值混算。

15项DA3数值数组、global mapping、8项Viewer数组及objects/manifest、完整最终GLB payload精确一致；两臂2402589Viewer点/1282759baked/31页/121197580GLB bytes。任务driver采样峰均14.777GiB。编码累计wall43.640→45.379s与bake重叠，主线程join43.649→.900s不代表编码计算仅.900s。

Ground50K虽10.879→.980s，但平面近垂直于摆正地面，保留默认0；camera_basis出现6alpha变化，保留source_world默认，只共享frame constants。两项显式实验已实现，未计入安全默认收益。原GPU2 resident保留，任务GPU进程均退出；未commit/push/部署。未测优化后多视频均值/P95、浏览器或外部准确率。[最终报告](../../../reports/latency-optimization-20261018/README.md) · [pair验证](../../../reports/latency-optimization-20261018/pair-validation.json) · [定向测试与复核](../../../reports/latency-optimization-20261018/review.md)。

[多视频报告](../../../reports/profiling-multivideo-20261018/README.md) · [验证](../../../reports/profiling-multivideo-20261018/validation.json) · [复核](../../../reports/profiling-multivideo-20261018/review.md) · [日志](JOURNAL.md)
