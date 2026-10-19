# 下一轮加速方向：保持原峰值、保持效果、缩短实际交付

执行更新：八项已按本提案实施和实测，当前采用/未采用决定及video3新同协议183.533→137.643秒结果见[实现与验证](gpu-acceleration-20261019/README.md)。本文以下保留2026-10-19执行前的调查上下文，不再作为待启动清单；旧计时与候选描述不是最新默认状态。

2026-10-19。三个 medium subagents 分别只读审查几何、GLB 和模型/匹配链；协调者核对现有计时、关键调用及官方源码。本轮没有运行新 GPU 实验、修改生产算法、安装依赖或部署。以下新增方案均为待验证候选。

## 当前约束与已知事实

Rick 已澄清：其他模块可以使用 GPU，只要不抬高原有峰值；不要求把 DA3 压到 14GB。之前的十进制 14GB 总门槛和 12GiB allocator 配额是误解下的实验条件，**不再作为后续加速的前提**。已有成功、失败及原始数字保留为历史证据。

video3 的原任务 driver 采样峰为 **15132MiB = 14.77734375GiB**，来自 DA3 子进程；Matching 为 3788MiB，最终 Viewer 阶段为 2914MiB。后者包含主进程模型/cache 驻留，并不表示几何过滤已经使用 GPU。[同次阶段峰](latency-optimization-20261018/pair-validation.json)。video6 原始 DA3 峰为 17640MiB，说明“约15GB”是 video3 的实测值，不能宣称所有输入都已有相同峰值；新增模块以用户指明的15132MiB作为本轮预算参照，不自动放宽到更大值。

顺序阶段比较最大占用；并行阶段统计同一时点本任务所有 GPU PID 的和。外部任务单列且保持原状。模型进程退出是简单的隔离方式，**不是所有后续 GPU 工作的强制前置条件**：有余量时可以保留已有驻留，但必须计入同时占用。采样峰仍是观测值，不冒充瞬时硬上限。

## 剩余时间主要在哪里

以下 video3 数字来自同一次最新安全默认候选；嵌套/重叠区间不可相加。[pair JSON](latency-optimization-20261018/pair-summary.json)。

| 区间 | 秒 | 对交付的含义 |
|---|---:|---|
| 完整流水线 | 186.894 | 既有 CPU GLB 路径的完整结果，尚未改用正式 GPU baker |
| DA3 重建 | 13.706 | 网络构造已降至0.914秒，不再拿原12.6秒当剩余机会 |
| Matching | 29.708 | PVS 首次构造、推理、匹配；该轮没有PVS独立细分 |
| geometry 准备 | 14.628 | 与Matching重叠；不能把全段当作可直接节省的时间 |
| 最终 Viewer 导出 | 60.615 | 其中DBSCAN/ground仍在交付主路径上 |
| ↳ DBSCAN `_cluster_labels` | **37.707** | 最大未解决的单项几何成本 |
| ↳ ground `_ground_filter_mask` | **10.654** | 多次全点域平面评价，不只是最后一次mask |
| ↳ 显示方向 | 3.164 | 不依赖标签，有前移机会，需保持随机及缓存语义 |
| GLB 自身 | 68.770 | 此完整候选仍使用CPU baker、两线程编码 |

另一个独立 video6 GPU GLB 实验为 **71.575秒**，producer55.011秒，主线程compact/write/join16.562秒，其中等待编码12.630秒、其余3.932秒。producer含LOD、CPU投影、图片准备和搬运，不是纯GPU kernel时间。[video6 JSON](gpu-latency-budget-20261018/video6-bake-result.json)。不能把这轮或video3独立GPU组件的差额直接扣除旧完整流水线。

## 建议优先顺序

| 顺序 | 工作 | 已有证据 | 下一步的性质 |
|---|---|---|---|
| 1 | 正式复用已验证的GPU bake | video3完整GLB66.589→44.671秒；video6 100.498→71.575秒，geometry及decoded RGBA exact | 集成已有成果，再测真实入口完整交付 |
| 2 | GPU bake配合encoder4 | video6编码等待12.630秒；已有有界4线程实现 | 单变量小实验，不能预设能省完等待 |
| 3 | canonical PVS选择性skip-init | 当前loader未使用；旧六视频CPU构造16.979秒，但受准备线程干扰 | 先审权重覆盖，再隔离测构造，不承诺旧成本全可回收 |
| 4 | Matching复用/批量计算框命中数 | TopK已有counts，验证函数又逐框`.sum().item()` | 小范围消除重复计算和同步，保持采样/SVD |
| 5 | DBSCAN内部诊断与等价优化 | 最新37.707秒；官方源码关闭progress仍逐点进critical | 先分清邻域/簇扩展/锁成本，再决定native改动；GPU索引另立实验 |
| 6 | NN/SOR与PVS隔离；再研究共享KNN | NN3.326/SOR5.736秒为旧六视频均值；当前准备与Matching重叠 | 首先测GIL隔离是否改善截止点；融合helper另算维护成本 |
| 7 | ground全点域评价分块GPU | 最新10.654秒，旧video6 RANSAC18.154秒 | 保留全点域、假设与判定的算法移植，数值等价待证 |
| 8 | GPU bake减少动态压缩/批间同步 | video6共2006批，现有GPU显存工作集很小 | 固定shape及搬运组织实验，先确认同步占比 |

## 1. 将已有GPU baker变成可复用实现

`src/surfel_glb.py:372` 当前compact入口仍为CPU。实验在 `reports/torch-bake-fusion-20261018/probe.py:137` 替换page producer，不是正式库接入。原实验通过不等于集成后自动通过。

应共用CPU `source_world`投影准备及唯一frame/batch/page循环，设备端只承担已经验证的mask、深度采样、float64双线性与方向延展。显式传递设备、lazy import Torch，保持原JPEG/f16边界、LOD、palette、alpha和最终Surfel长边限制；不复制两套完整算法，不引入自动fallback。

video6 GPU bake任务driver采样峰454MiB，完整geometry accessor及49页200736000个decoded RGBA texels exact。模型已经退出的交付进程最易隔离；若接在仍有PVS驻留的主进程中，也不因“未退出”自动拒绝，而应观测合计峰与实际耗时。

最小验收：共用准备重构后的CPU/GPU真实样本与完整产物比较，然后用实际入口测Viewer/ZIP/GLB截止点及同时任务峰。当前实验范围不包含Docker服务集成或部署。

## 2. GPU bake + encoder4

`src/surfel_glb.py:177–205` 已有有界FIFO，最多workers页在途，单writer保持顺序。先只把encoder2改为4，不动palette、PNG设置或页大小。CPU baker旧1/2/4对照不能预测GPU producer变快后的最优值。

验收同一video6 generation的完整export wall、encoding join、RSS与最终decoded RGBA。CPU并发、主机RAM和共享负载是主要新增成本；GPU工作集算法不变，但仍记录任务峰。12.630秒是等待窗口，不是可直接承诺的净收益。

## 3. PVS初始化与准备线程：分开验证两个原因

`utils/sam31_pvs.py:15–49` 构造canonical tracker后strict加载权重，没有使用skip-init。`utils/sam3_utils.py:421–426` 的旧API已有跳初始化，不能把旧API能力当作canonical PVS已经优化。

候选仅跳被checkpoint覆盖的大Linear/Conv随机weight填充，保留token、buffer、位置cache及必要初始化。位置cache在 `sam31/sam3/model/position_encoding.py:37–58` 的普通dict内，构造时会用当前CUDA设备；strict state加载不能证明所有构造代码都能跳过。

先在无geometry准备干扰的新进程比较构造/加载，审计被跳过Parameter的覆盖并比较同输入mask。当前六视频旧构造16.979秒混有GIL等待，不能照搬为收益。随后另测 `main.py:1091–1098` 的准备线程改为CPU进程：返回需要的数组/metadata，不能直接跨进程复用Open3D对象；IPC、PCD重建、RAM均计入。

只有Matching或Viewer累计截止点实际改善才算隔离有效，不能拿已重叠的14.628秒再扣一次。NN/SOR公式不必同时重写；这样能区分调度与算子两种收益来源。

## 4. Matching消除已知重复命中统计

`utils/matching_algorithms.py:40–60` 已批量计算inclusive bbox命中数；`:617–653` 的TopK分支只带着候选bbox进入验证，没有传递counts。`utils/geometry_3d.py:453–465` 又为每框执行相同比较及GPU `.sum().item()`。

已核对当前调用使用同一个 `projected_points`，中间无点域修改，边界均为inclusive。因此可以按top_indices同步传counts；没有预先计算的分支才批量重算。限定为同一次调用、同一投影点域和bbox，不能跨对象缓存。

保持候选顺序、稳定并列排序、每目标采样RNG消费、CPU float64 SVD及0.4m商品进深规则。先比较counts/分数/对应关系，再测target_bbox_match与完整Matching；旧六视频target bbox匹配均值3.110秒仅标记工作量，不等于这项同步可省3.110秒。

## 5. DBSCAN：最大热点，先优化原算法的成本

`utils/pointcloud_filter.py:267–276` 的DBSCAN输入是应用商品保护后的SOR点域，不能无条件提前到matching。簇大小门槛和无达标簇时的最大簇选择也必须保留。

[Open3D0.19官方实现](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloudCluster.cpp#L29) 在每点radius查询后无条件进入`omp critical`递增progress，即便print_progress=false。可研究只有启用progress才进入锁；这不改邻域或簇扩展，但**尚未测锁占比**。本地wheel为静态构建，修它需要native构建/扩展及维护成本，不是Python加一个开关。

先拆分KD-tree/半径邻域、串行簇扩展与progress成本。若锁很小，就不为此维护整套Open3D patch。GPU版本需要真正的空间索引、受限邻域存储及稳定核心/边界点归属；直接换库不自动保持簇计数、最大簇并列选择和eps边界结果。

[Open3D tensor DBSCAN官方文档](https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.cluster_dbscan) 明确其仍包装CPU实现并复制数据；把点云搬到CUDA不会自动让DBSCAN使用GPU。[cuML C++接口](https://docs.rapids.ai/api/libcuml/stable/group__dbscancpp) 默认邻域方法为BRUTE_FORCE，batch参数限制的是pairwise批次，不应据此承诺百万点索引加速或总峰上限。本轮未安装或验证cuML。

最小验证先比较完整labels及最终保留mask，再测同域耗时。不能只比较簇数或最终点数。400万点全距离float32矩阵约64TB，分块也不会消除二次工作量。

## 6. NN/SOR复用一次邻域查询

当前 `utils/pointcloud_filter.py:95–103` 分别调用NN和SOR，两者各建树。Open3D NN查询KNN2取第二距离，SOR查询KNN20且包含自身；同一float64点域的一次KNN20可以同时提供两者所需距离。[官方实现](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloud.cpp#L570)。

原生窄helper可返回NN数组及SOR mask，避免第二建树/查询，也避免生成被调用方丢弃的SOR filtered cloud。现有Python返回值不暴露邻域，不能零改动复用。需要保留零距离处理、样本标准差、归约顺序与严格阈值；SciPy或GPU新实现不自动等价。

先测进程隔离，再决定是否值得引入native helper。重复点、等距、零距离和阈值邻近点需覆盖；真实完整域比较medianNN/raw SOR及商品保护后的mask。该准备大部分已被Matching隐藏，若没有减轻互阻，其组件加速对交付的收益可能很小。

## 7. ground保持完整点域，GPU分块评价平面

`utils/pointcloud_filter.py:137–199,281–296` 的主要工作是RANSAC为多个候选反复扫全点域，不是最后一次point-plane mask；后者旧probe仅约0.05秒。只将最后几行改Torch不会解决主要成本。

可以研究固定原三点假设/float64、点仅上传一次，逐块计算全点距离并归约inlier count与RMSE。342万点float64 XYZ约82MB，但临时量、context和驻留模型另计；禁止生成1000×N距离矩阵。匹配后运行较易避免争用，也允许有预算时并行。

难点是Open3D并行提前停止、fitness/RMSE并列、阈值附近浮点、先定内点再refine；批量评完所有假设不自动等价。先比较固定假设逐个统计、胜出plane和refine前内点；改变mask则不推广。旧50K采样方案额外删除51286背景点，仍不作为保效果优化。

## 8. GPU bake固定shape，减少同步边界

`reports/torch-bake-fusion-20261018/torch_bake.py:136–148` 每批动态压缩valid/tx/ty，再下载keep/tiles；video6重复2006次。每frame RGB/depth已经仅上传一次，不能重复把这项当作新收益。

候选先对全部visible batch进行固定shape RGB采样/延展，回CPU后按keep稳定压缩，消除部分设备动态shape边界。会多处理最终无tile的候选，需要实测额外工作是否抵消收益。之后才研究合并小上传、双缓冲及更大batch；当前1024为代码硬限，不能当成现成参数随意调大。

动态shape及D2H是同步调查方向，[PyTorch CUDA语义](https://docs.pytorch.org/docs/2.7/notes/cuda.html#asynchronous-execution)及[官方同步说明](https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/)支持这一机制，但没有本项目trace量化其占比。保留source_world CPU算术顺序，先比较真实样本corners和全部RGBA（含透明RGB），再测完整导出和同时任务峰。

## 暂不优先的工作与本轮结论

- 不再压低DA3显存；其selective skip-init与CPU standby已实现，不把已实现能力重复列为新收益。
- 不把GPU常驻DA3作为首选：需要改变跨请求生命周期，可能抬高与PVS/其他模块共驻留时的峰值。
- 不直接GPU SVD或camera_basis投影。前者改变小矩阵数值及阈值，后者已有真实alpha差异。
- 不把关闭DBSCAN、代理点过滤、背景提前缩减当作等价优化；原本保持效果的证据还不足。
- 自适应预算、ZIP和最终写盘相对较小；先处理关键路径大项，不引入复杂异步writer。
- 显示方向可以提前约3.164秒窗口，但必须使用原get_scene_orientation点域并维持随机调用关系；作为调度补充，不优先于大项。

最先执行的低成本新判别应为GPU bake + encoder4，以及独立canonical PVS选择性skip-init；DBSCAN内部成本诊断与准备进程隔离可作为独立工作。正式GPU baker集成是把已验证组件收益带到真实入口的必要工作。以上都没有新的实测收益，不能将候选节省值叠加成承诺的全流程时间。
