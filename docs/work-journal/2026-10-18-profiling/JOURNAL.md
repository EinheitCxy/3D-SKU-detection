# GPU2 profiling 事件记录

## P01 — 2026-10-18 CST：启动与前置核对

用户要求在 GPU2 对最新代码做非常细致 profiling，替换旧主文档，并记录小步骤实现。使用 autoresearch、get-available-resources、work-journal；三个 medium 只读 subagents 分别核对 GPU/matching、几何/采样、交付源码。

观测：HEAD bf9bde8，本地 main ahead 4（只表示本地 tracking ref）；当前 12 份已有文档改动与未跟踪文件保留。GPU2 UUID GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7，总量 24,564 MiB，原有两进程 1,580/3,450 MiB。根 .venv 与 DA3 HF cache 都指向已不存在的旧 build 路径；独立 test-env 在场，PVS 权重在场。不会重置 symlink 或清理其他工作。

决定：相同 29 帧输入、新结果 cache、最新默认配置。主跑与细分诊断区分；先恢复已有项目镜像内的同一 DA3 snapshot，准备时间不计入流水线。旧 267.565s 保留至新主跑通过。

## P02 — 2026-10-18 19:32 CST：模型与前置计数修正

已从本地 global-id-mapping:cos-background-20261013 镜像取出相同 b2359b... snapshot（约6.3GiB）；临时容器只处于 created、从未启动，提取后删除。根旧 symlink 保留。环境导入与粗粒度插桩 targets smoke 通过，CPU线程4；资源脚本实测3块GPU、40逻辑CPU、约399.65GiB可用RAM、15.77GiB磁盘。

第一次 launcher 在计时/启动模型前失败：输入是 canonical {skus:[{objects:...}]}，preflight 误从顶层取 objects。已仅修正runner计数、允许复用经目标校验的输入symlink；失败日志 preflight-failed-01.log 保留。此失败未启动 DA3/PVS，未产生正式结果，不计为流水线样本。

## P03 — 2026-10-18 19:37 CST：完整主跑通过

运行 reports/profiling-20261018/run_profile.py，独立 test-env、GPU2 UUID、4线程、离线、29帧/711 observations、新输出cache。core/delivery退出码均0；总墙钟248.899597s。重建27.100441、matching29.837225、dedup0.134608、最终Viewer66.483175、ZIP2.681316、GLB111.118460、导入/进程生命周期/未单列间隙11.544374s，互斥主表闭合。

Viewer publish128.747393s、export返回128.749960s、ZIP137.148453s、GLB248.267107s；最终输出2402589点、711缩略图、385global IDs。prepare15.106235s，其中15.106026s与matching重叠；约0.000209s为提交与matching入口先后差，并不表示future最终阻塞。

GPU任务采样峰15132MiB（14.777GiB），同时间按任务PID过滤；既有进程恒5030MiB；整卡峰20181MiB。200ms采样可能漏短峰。进程树RSS同时采样和峰14,860,701,696 bytes，包含共享页重复可能，不是独占物理RAM。

248.900s已是本次条件下的新结果，但旧GPU0与新GPU2、OS页缓存/共享任务/代码均不完全相同；新Viewer就绪比旧记录更晚，不能只报告GLB改善。新主文档将在细分诊断及独立复核完成后替换。已备份旧入口到runtime/previous-profiling_breakdown.md。

## P04 — 2026-10-18 19:38 CST：独立GPU细分

DA3诊断成功（22.398237s进程内窗口；不替换27.100441s主跑重建）。新脚本先正常导入父package再AST重编并更新引用，CUDA隐藏环境已验证不建模；events在runner原同步和D2H之后读取，无新逐算子同步。模型构造11.408798s、搬GPU1.879405s、原生同步forward2.220267s；CUDA events分别标注anyview/metric及backbone、alignment，不与wall相加。当前串行执行PVS细分，Viewer/delivery CPU诊断随后执行。

## P05 — 2026-10-18 19:58:51 CST：细分、重叠复现与来源差异

PVS、Viewer/dedup、delivery均成功。PVS独立load8.052901s、predict4.278822s；Viewer独立82.655465s，raw NN3.322962、SOR5.739382、DBSCAN38.727894、ground RANSAC10.517558s。独立filter输出3367575，主跑3425675；最终都2402589，不将同点数当来源或画面等价。独立ground保护额外救回246927商品点，仅删除58100背景点，不能补写成主跑逐阶段记录。

delivery独立117.484454s窗口，其中ZIP2.878987、GLB110.955937；1280 bake batches，31atlas页。RGB延展22.716723s、bilinear15.543698、MEDIANCUT18.621591、PNG21.540498；全部属于各自父区间。

追加一次30s timeout的prepare+PVS load复现，24.110781s内成功。load19.306925s，其中construct17.362315s、该线程CPU6.607440s；NN3.791112/SOR6.120773s。Python采样空档3.903487/6.154753s，main CPU仅0.102612/0.049146s、runqueue等待约61/69微秒。官方Open3D0.19绑定直接def NN/SOR无GIL release；pybind11官方GIL规则支持机制解释。保留wheel机器码未核验/无GIL专用探针、不同进程缓存状态和仪器影响等限制，不把全部差额说成可回收收益。

## P06 — 2026-10-18 19:58:51 CST：文档替换、独立复核与结束

替换docs/profiling_breakdown.md与docs/viewer_export_algorithm.md，更新3d_core/surfel_glb入口；旧两个入口原文保存在本任务runtime历史快照。详细3份算法、10份诊断/重现脚本、5份组件JSON、summary/逐ref CSV/时间线/validation均在reports/profiling-20261018。报告原始日志/模型/大二进制留runtime，不提交或上传。

三个medium subagents独立复核：主表闭合0、PID对齐的15132MiB任务显存峰、整卡采样覆盖、RSS与包含关系、阶段/点数与GIL措辞均通过。已修正两处算法映射（序列化与point_ranges次序、bool valid与integer valid_indices区别）及head wall/event术语；delivery旧“未测量”文字已替换。

最终最小验证通过：1次完整主跑，两stage退出0；4类独立细分及1次重叠复现；29refs/29完整PVS manifests/711 observations/385 gids/2402589点；751ZIP成员压缩方法/GLB header与大小；10脚本AST、46本地Markdown链接、所改文档diff whitespace检查。未跑pytest/浏览器、hash/CRC或画质验收，无生产源码/默认配置改动。GPU2最终仅保留原6221/8550两进程，任务GPU进程已退出；磁盘约15GiB可用。

## M01 — 2026-10-18 20:03 CST：扩展多视频与同轮细分

用户要求在fd-videos不同视频测平均时长，并对完整流水线、Viewer/ZIP就绪、GLB、显存和最终点数做内部breakdown。新子任务目录 reports/profiling-multivideo-20261018；前轮结果保持完整。3个medium subagents分别负责输入对齐、core/DA3/PVS/geometry细分、delivery细分；root负责运行和聚合。

前置元数据：1..7有JSON，173标注帧/6639objects；8缺JSON。video2为12帧、23.37秒，其1.5s前段或2s全程不能凭数量断定；其余也需要采样依据。已发异步文本澄清并继续只读查找，不能将经过时间当回答。GPU2原有5030MiB进程保留，GPU0/1不触碰；约414GiB RAM、15GiB磁盘可用。新均值将用统一同轮细分profile，不拼接旧独立组件诊断。

## M02 — 2026-10-18 20:27 CST：输入实验协议、前三场景及聚合复核

未找到其他原始时间表。为完成性能任务，1/4/5/6/7以JSON数量恰好对应的0起1.5s建立明确的研究者实验recipe；sampling-records标注非原timestamp证据、未验证检测对齐，仅报告该输入协议的性能，不作准确率结论。video2因采样有歧义继续待答，不启动；8缺JSON。没有把未答当确认。video3沿用已建立recipe。

video3完整236.866210s，Viewer120.553064s，ZIP129.384797s，GLB106.576903s，最终2402589点；不同输入还原/随机拟合/共享负载/细分仪器，未将比上一轮少的秒数归因为生产优化。video6完整48帧、2780检测成功：380.629313s，Viewer197.780715s，ZIP207.433270s，GLB172.135072s，最终2500000点；未截断帧、未换模型。原商品3778160点，预算选择2000000；ground内点352892/5933195≈5.947%，小于8%条件，未进入apply。video1也完整成功，video4当前串行运行。

两个只读medium reviews发现并修复诊断汇总的三类问题：GPU时间戳改用相邻epoch/monotonic独立锚点，并裁剪主窗口；固定1..7 requested集合，区分running/pending/failed而非成功目录决定分母；商品预算指标由误读末gid quota改为分组quota总和/末组/组数，明确不代表配置cap。video6 1330groups总quota2000000，末组337；video3一组1902589。pointflow缺失不补0，条件性均值带n；修正features文案为depth/valid/语义region的edge/halo/weights/caps。

已实现逐视频summary、所有原始wall/context/CUDA/snapshot/pointflow指标、mean/median/min/max/n、逐ref CSV和主文档生成。只从唯一raw process detail提取，避免将投影重复平均。主前缀可相加闭合；内层父子、CUDA/host、重叠prepare和峰值均不相加。旧单video3正文保留在新reports目录previous-single-video.md。

## M03 — 2026-10-18 CST：六视频完成、最终聚合和验证

全部1/3/4/5/6/7完整成功，161frames/6508objects；总均值229.250920383s、Viewer119.265629263s、ZIP127.241764743s、GLB自身101.178545928s。完整min/max147.990334445/380.629312573s。任务driver峰均值15083.333333MiB=14.729818GiB，所有样本最大17640MiB=17.226562GiB；Viewer点均值1764058.5，范围837531..2500000。requested7/attempted6/success6/failed0，2仍采样待答、8无JSON；不把未测混入分母。

同次完整链路保留266个host函数/block label、2922项wall/context/CUDA/显存/计数指标，逐video和逐ref CSV齐全。主文档与算法入口更新为多视频/动态F，旧单次保留历史；所有5种milestone×6video prefix closure_error=0。平均互斥阶段DA324.199s、matching31.871s、dedup0.248s、finalViewer57.844s、ZIP2.214s、GLB101.179s，导入/仪器/生命周期间隙另列。

视频全部结束后追加一个30s硬上限CPU合成probe，250万int32/200queries：Python int标量1.152541s、typed scalar0.001243s、typed vector0.000141s，range endpoints完全相同。固定NumPy1.26.4官方共同dtype/转换源码支持重复临时转换机制解释；未直接测生产转换次数、未改生产代码，也不从主表扣时间。这是下一步精确优化候选，非全链路收益证明。

两位medium最终只读复核均PASS，GPU/core reviewer另核对source/metrics及NumPy机制。root一次最终验证PASS：6case输入/refs/PVS/完整输出、前缀和主时间线闭合、必要ZIP成员方法/GLB header、15脚本AST及35本地链接。未跑全量pytest/CRC/hash/browser/画质；所有原任务变更保留，未commit/push/部署。GPU2最终仅6221/8550 resident，1580+3450MiB；没有本任务GPU进程残留。
# O20261018 — 2026-10-18 21:27 CST，批准后的优化实现与组件实验

Rick明确要求实现DA3初始化、point ranges、ground代理候选、GLB投影复用与atlas流水，并使用文件记录。三个medium workers分工，主线程保留源码快照、统一调度GPU2与串行CPU对照。完整任务记录：[progress](../../../reports/latency-optimization-20261018/progress.md)。

修改前video3同输入/seed42完整baseline253.139s，函数仪器模式独立于此前六视频AST均值。真实DA3初审误把6个共享LayerNorm别名当漏载，拒绝记录保留；修正identity别名后1377physical参数全部覆盖。CPU构造11.827→.652s、含审核加载12.994→2.045s；尚待GPUforward验证。Point ranges2402589点/711观察区间端点全相等，3.490→.000964s。

Ground50K组件10.879→.980s但增加51286背景删除，平面与orientationfloor夹角87.77度，未推广；camera_basis真实样本6个alpha变化，source_world几何/RGBA完全一致，默认source_world。完整GLB同generation1/2/4workers108.396/67.627/66.638s，全部geometry和128295600decodedRGBA texels一致，选默认2而保留显式4。最终GPU2组合实验待执行；不将独立组件收益相加成全链路收益。

## O20261018-FINAL — 2026-10-18 21:39 CST，完整优化验收

GPU2最终组合两stage均exit0，253.138706→186.894435s（减少66.244271s/26.169%）；Viewer129.854374→109.162660s、ZIP138.096691→117.362010s、GLB114.041383→68.770372s。同次DA3构造12.624388→.914212s、CPUload13.562327→2.424568s、重建25.132625→13.706324s；同次point ranges3.939481→.001608s。各臂一次、固定29帧/711检测/seed42/4线程、新模型与结果进程；页缓存不清空，function spans模式独立于旧六视频AST均值。

完整输出直接对比PASS：15DA3数值数组、mapping、8Viewer二进制、objects/manifest和最终GLB整个payload相等。两臂386gids/2402589Viewer点/1282759baked/2565518三角/31pages/121197580bytes；任务driver200ms采样峰均15132MiB=14.777GiB。累计encode43.640→45.379s与bake重叠，main join43.649→.900s，未消除全部编码运算。

三位medium最终只读复核PASS（DA3/geometry/delivery）。定向测试明细见[review](../../../reports/latency-optimization-20261018/review.md)，未相加重叠测试数、未重跑全量回归/hash/CRC。所有任务GPU/runtime进程退出，21:32CST快照仅原6221/8550resident；未操作GPU0/1。默认source_world/ground0/encoder2，不叠加不满足输出或质量推广条件的camera_basis/ground50K实验。PyTorch调查完成，建议下一项是有界RGB+mask+方向延展融合，含搬运/context计时；本轮未新增GPU算法。主文档§12与状态已更新；未commit/push/部署，未测优化后多视频均值/P95或人工准确率/画质。
