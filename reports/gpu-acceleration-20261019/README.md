# 八项延迟方案：实现与验证

2026-10-19，已完成八项本地实现与对应实验。三个medium workers并行实现、互相审查，coordinator整合并串行实测；六项形成可用优化，DBSCAN原生helper和GPU ground scorer保留实验状态。保持原峰值、效果并降低实际交付；没有新14GB/12GiB DA3门槛。

本次main发布仅包含通过的PVS初始化、hit-count复用、共享KNN及GPU bake/有界编码；较慢process接线、GPU ground scorer和native DBSCAN helper留在本地实验，不作为main接口。下文保留这些实验的正反结果，标量汇总同时保留失败记录。[发布及视觉影响审查](RELEASE_REVIEW.md)明确了历史opt-in选项与当前默认的区别。

## 完整video3交付结果

同29帧、GPU2/RTX4090D、seed42、原精度/LOD/palette/长边，CPU线程环境4。各臂一个新进程，固定顺序；共享主机与OS缓存未隔离，n=1。输入已有检测/分类；范围是**帧就绪后的DA3、Matching、Viewer、ZIP、GLB**，不含检测、分类、COS或浏览器加载。

| 指标，秒 | Control：原PVS＋CPU2 | PVS skip＋GPU fixed4 | 再加共享KNN，最终采用 |
|---|---:|---:|---:|
| GLB累计就绪 | 183.533016 | 148.058645 | **137.642563** |
| Viewer累计就绪 | 110.842337 | 104.833556 | **93.935496** |
| ZIP累计就绪 | 113.674038 | 107.686906 | **96.634042** |
| GLB自身 | 69.858972 | 40.371731 | **41.008508** |
| DA3父阶段 | 14.110161 | 14.068722 | 13.831491 |
| Matching父阶段 | 27.806182 | 22.489850 | **11.986647** |
| ↳ PVS load | 18.532415 | 2.169143 | 2.099707 |
| Viewer几何准备，并发支线 | 13.874693 | 14.490779 | 10.570518 |
| 最终Viewer export | 62.250569 | 61.740246 | 61.770907 |
| 任务driver采样峰，MiB | 15132 | 15132 | 15132 |

最终累计减少45.890453秒（25.004%），Viewer提前16.906841秒。共享KNN单变量增量又省10.416082秒；它同时减轻与Matching的互阻，不能只按孤立KNN组件3.169秒推算。父子/并发区间不相加，各臂均有Matching hit-count复用，不单独给它归因收益。GLB最后一组略慢于中间组，报告保留真实值。

三组15项DA3数组、8份Viewer二进制、manifest、objects、global mapping精确一致；GLB关联、所有accessors、31页128295600个decoded RGBA texels一致，alpha/RGB差异0。核对在就绪后另花11.438/11.949/11.613秒，不计入交付。均为2402589 Viewer点、1297686 LOD候选、1282759 baked surfels、121197580字节GLB。峰值15132MiB＝14.777GiB来自DA3子进程。全部内部数值、输入与raw receipt路径见[汇总JSON](summary.json)。

## 已完成的组件对照

GPU2/RTX4090D，CPU线程环境4；同一既有video6 48帧generation、250万Viewer点，source_world/tile8/原LOD/palette/长边。各臂一个新进程，固定顺序；共享主机/页缓存未隔离。

| 正式GLB实现 | export秒 | 含import就绪秒 | driver采样峰MiB |
|---|---:|---:|---:|
| GPU动态shape、encoder2 | 71.513331 | 75.100404 | 456 |
| GPU动态shape、encoder4 | 62.282952 | 65.813080 | 456 |
| GPU固定shape、encoder4 | 59.734091 | 63.210315 | 456 |

三个完整导出均与既有CPU参考的document、全部accessor和49页/200736000个decoded RGBA texels精确一致。比较在交付计时外。这里没有重跑CPU控制，不用旧CPU100.498秒计算新的严格配对提速率。新2→4及动态4→固定4才是本轮单变量对照；n=1不代表普遍收益/P95。

PVS独立真实帧：原生/选择性skip各新进程、无准备线程；加载到GPU就绪8.420716→2.492964秒，构造6.803251→0.681401秒。131个大weight/1804599296bytes覆盖审核通过；该帧全部mask、scores、presence、areas、boxes精确相等，无nonpersistent buffer，普通cache结构一致。任务driver采样峰3690MiB。全matching/mapping与完整流水线验证已通过，默认开启skip；不能将有准备线程时load18.532秒全部当作关键路径节省。

NN/SOR隔离的反例：同candidate PVS、完整Viewer准备，两边就绪截止点thread14.983108秒、process20.370611秒，全部10项输出精确一致。进程启动、临时数组IPC和父进程PCD重建已计入。**process慢5.387504秒，不推广默认**；完整候选继续使用thread。这是组件重叠实验，不是全Matching成本。

## 八项当前状态

| 项 | 实现 | 验证/采用情况 |
|---|---|---|
| 1 | 正式bake_device API、共享CPU准备与分页、薄CLI和可选视频GLB | video6完整GLB、video3完整core→delivery通过；GPU显式选择 |
| 2 | 有界encoder4 | 本轮GPU场景比2快9.230秒；保留显式选择，CPU默认2 |
| 3 | PVS选择性skip及严格覆盖审核 | 完整video3通过；默认开启，保留token/bias/norm/buffer/cache和strict load |
| 4 | 同调用复用/批量hit counts | 生产采用；真实mapping通过，RNG、CPU SVD及0.4m进深规则保留 |
| 5 | 原nanoflann DBSCAN分段/去无效锁独立native候选 | 3889146点labels/keep exact，但去锁仅省0.347秒；不替换Open3D默认 |
| 6 | CPU NN/SOR进程隔离；单索引KNN统计 | 默认Viewer线程采用共享KNN；两真实raw域及完整video3通过；process更慢，不推广 |
| 7 | 有界全点域float64固定plane评分 | GPU完整域counts/mask/refine通过；RMSE不exact、原RANSAC RNG/早停未复现，不替换默认 |
| 8 | 固定shape GPU颜色/延展、CPU稳定压缩 | 全video6输出exact，比动态4快2.549秒；显式选项 |

## 实现与最小检查

- delivery最初136项CPU检查：共享准备、唯一页循环、显式CUDA、CLI与视频接线；旧/新CPU两projection八例exact。正式实现见`src/surfel_mesh_torch.py`、`src/surfel_glb.py`，使用说明见[GLB文档](../../docs/surfel_glb.md)。
- model最初17项CPU检查：PVS选择性初始化、别名/身份/shape/dtype审核、hit counts与RNG；真实GPU比较由coordinator执行。对应正式测试为`test/test_sam31_initialization.py`和`test/test_matching_hit_counts.py`。
- geometry最初7项CPU检查涵盖本地KNN、native DBSCAN和固定plane scorer。main只保留共享KNN正式实现及其边界测试，原RANSAC RNG/early-stop尚未移植。
- coordinator新增CPU进程worker及pipeline接线8项检查通过；独立agent复核PASS。进程临时文件失败/成功均清理，错误直接传播。
- PVS默认提升后仅复测8项初始化测试通过；共享raw准备6项、Viewer默认接线4项通过；serial encoder异常清理扩展现有用例10项通过。分组存在重叠，不相加为唯一测试数。

## 几何真实域及未推广原因

| NN＋SOR＋median独立组件 | Open3D秒 | 共享KNN秒 | 等价 |
|---|---:|---:|---|
| video3，4092480点 | 9.572414 | 6.403446 | 全部NN距离、median、raw SOR mask exact |
| video6，6773760点 | 14.961355 | 10.804380 | 全部NN距离、median、raw SOR mask exact |

默认Viewer准备线程在完整float64 base域建一个cKDTree，按50000点分块查询包含自身的KNN20：第二邻居距离提供medianNN，20邻居均距提供SOR，保留顺序归约、样本标准差、零距离和严格阈值。低层prepare_scene_filter默认保留原Open3D控制，由Viewer显式传shared_neighbors=True；实验process仍用原生统计。SciPy exact query不证明任意点域bitwise Open3D等价；当前两真实域及重复点、短域等测试通过。

商品标签出现后仍由原filter合并保护、应用20% guard，再执行原DBSCAN和ground。video3 raw4092480→保护后SOR3889146→DBSCAN3425675点；ground仍因support约7.9554%不足8%不删点，没有直接跳过RANSAC。最终全部1902589商品点＋500000背景点，未预缩背景。

DBSCAN正确post-SOR域/原eps0.01966750138532755：Open3D39.187108秒；独立native带锁31.915607、去锁31.568111秒，两臂完整labels及keep exact。去锁版本建树2.170921、邻域12.827846、串行扩展16.186785秒；408126570条邻居，存储下界1725845784字节。native与wheel的7.619秒差还混有编译/实现差异，不能都归因去锁；锁本身仅差0.347496秒，不为此替换全局Open3D或维护整套发行包。

Ground3425675点/1000固定NumPy三点候选：GPU含context、传输、评分和胜出mask2.216965秒，driver504MiB/allocator91212800bytes；counts、winner542、mask及同CPU公式refine exact，RMSE最大差2.3418766925686896e-16。CPU85.947310秒是Python/NumPy参考评分，**不是原Open3D约10.654秒RANSAC**，不能按该比值承诺交付收益。原候选RNG、并行早停与tie还未保持，因此不推广。

进程隔离反例使用共享KNN接线之前的原Open3D统计；不冒充process与新线程默认的配对结果。共享KNN正式实现为`utils/pointcloud_neighbors.py`，真实域、计时范围和实验限制保存在[标量汇总](summary.json)。未推广的geometry探针及process实现不在本次main发布范围。

## 使用和失败记录

已有generation可运行`uv run python -m src.surfel_glb_delivery --generation GENERATION --output NEW.glb --bake-device cuda:0 --encoder-workers 4 --bake-fixed-shape`；设备按CUDA_VISIBLE_DEVICES解释。视频脚本可加`--glb-output PATH --glb-bake-device cuda:0 --glb-encoder-workers 4 --glb-fixed-shape`，未给输出不生成额外GLB。CPU仍默认可用；CUDA错误直接报错，无auto或fallback。完整视频脚本的检测/分类本轮未实跑，实际执行的是core→delivery。

首个完整control研究probe只seed父进程，遗漏DA3子进程--seed42，depth/scale和Viewer出现差异。失败目录保留并排除收益结论；修正probe后在新目录重跑，比较标准未放宽。delivery迁移残留变量、PVS模板身份审核、监督器生命周期问题在采用前修复；各owner报告和根progress保留记录。

GPU监督采用独立nvidia-smi文件采样、owned PID及进程组身份；100ms记录为采样峰，不是瞬时硬保证。独立审查发现的包装进程提前退出/短生命周期PID归属问题已修正；前两个导出的CSV已确认只含其完整known task PID及原6221/8550，无未知GPU PID，未重复有效导出。原GPU2 residents保留，GPU0/1未用于本任务。

原始GLB、NPZ、点域、采样与日志留在runtime/gpu-acceleration-20261019；此目录只放源码、约30KB标量汇总和解释。新增runtime限3GiB、磁盘4GiB底线。原14份无关tracked patch、HEAD、index及已有untracked路径保留。实验测量没有安装/替换全局依赖；后续main仅发布上述通过项，未部署服务。未运行浏览器人工画质、检测/分类准确率或新多视频全流程均值/P95。没有采集本轮host峰RSS，不声称编码4线程主机内存不变；有界FIFO最多4页在途的行为测试通过。
