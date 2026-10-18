# 多视频完整 profiling 与内部 breakdown

2026-10-18，用户要求在fd-videos不同视频实测平均耗时，并细分原六项指标的内部组成。本轮沿用autoresearch、get-available-resources、work-journal，复用3个medium subagents协作；root统一执行GPU2。

- 输入候选：1..7各有mp4+JSON，173帧/6639检测；8缺JSON，排除同口径集合并明确列出。video3沿用已建立0起1.5s recipe。对齐澄清等待期间，1/4/5/6/7按JSON数量恰好吻合的0起1.5s建立显式实验recipe，标注未验证原检测对齐；性能结论仅适用于该recipe，不作匹配准确率结论。video2可能1.5s前段或2s全程，仍待确认、不启动、不混入均值。这是研究者选择的输入协议，不声称找回原采样metadata。
- 实验：每视频一次新进程/新DA3和PVS结果cache；全部标注帧保留，禁止为显存截断或切换模型。GPU2 UUID固定，已有进程保留。CPU OMP/BLAS/MKL=4。Atlas仍默认1，DBSCAN/ground及所有近似开关不变；quiet模式，检测分类由JSON提供。
- 与前轮区别：在同一完整运行内统计细分，得到同场景同窗口父子关系；不是另跑组件后拼到总量。仪器开销不假定为0，均值明确称统一细分profiling模式。旧video3单次不混入新均值。
- 细分：完整互斥阶段、Viewer/ZIP累计就绪的前缀组成、准备与matching重叠；DA3导入/构建/权重/to/preprocess/forward/D2H/backproject/NPZ；PVS模型/encoder/batch/后处理与matching；方向/标签/NN/SOR/DBSCAN/ground/voxel/adaptive/序列化/ranges/发布；ZIP各类成员；GLB LOD/7个bake连续block/长边约束/bilinear/RGB/页/alpha/palette/PNG/量化/写入。
- 显存与点数：driver按任务PID及同采样时点筛选，分阶段峰；allocator snapshot与生命周期峰分别报告，不相加。点数按dense/valid/filter/product/background/voxel/budget/LOD/bake拆分，不把峰值当时间分解。
- 聚合：逐视频表、算术平均/中位数/min/max及样本数；不同视频一次的横截面均值，不是同视频重复或热服务分布。失败完整保留，若有失败不从分母中静默删除；只在明确标识的成功子集给mean。
- 执行顺序：先已确认video3验证新增仪器，再执行可对齐的其他视频，最大48帧场景尽早测；所有重计算串行，不争用彼此。每stage30min、GPU空闲17GiB预检、磁盘空闲4GiB下限；复用现有6.3GiB模型不再复制，新增产物预计<8GiB。
- 交付：更新唯一docs/profiling_breakdown.md为多视频及内部均值，保留逐视频JSON/CSV、原始本机日志、各指标定义和未测范围。无生产算法改动、无发布或COS。
