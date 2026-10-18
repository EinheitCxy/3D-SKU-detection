# 最新 profiling：GPU2 / 多视频 / 2026-10-18

本文件是项目唯一当前 profiling 入口。**最新优化前后对照见第12节：GPU2/video3完整253.139→186.894秒，最终GLB内容完全一致。** 第1–11节保留修改前`bf9bde8`的六视频测量：video1/3/4/5/6/7，共161帧/6508检测，完整平均229.251秒、Viewer119.266秒、ZIP127.242秒；该批内部数字均来自各视频同一次完整运行。后续优化已修改本地源码与默认atlas worker数，但没有修改服务或发布。新单视频对照不替换旧六视频均值。

## 1. 六项指标的平均值与逐视频结果

| 指标 | 平均（n=6） | 中位数 | 最小–最大 | 单位 |
| --- | ---: | ---: | ---: | ---: |
| 完整流水线 | 229.251 | 211.890 | 147.990–380.629 | 秒 |
| Viewer 可用（export 返回） | 119.266 | 108.599 | 83.169–197.781 | 秒 |
| ZIP 就绪（累计） | 127.242 | 116.629 | 89.735–207.433 | 秒 |
| GLB 导出自身 | 101.179 | 94.080 | 57.584–172.135 | 秒 |
| 任务 driver 显存采样峰 | 14.730 | 14.370 | 13.664–17.227 | GiB |
| 最终 Viewer 点数 | 1,764,058.5 | 1,730,111.5 | 837,531–2,500,000 | 点 |

| video | 帧 / 检测 | 完整秒 | Viewer累计秒 | ZIP累计秒 | GLB自身秒 | 任务峰GiB | Viewer点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 21 / 342 | 186.913 | 96.550 | 104.418 | 81.582 | 14.125 | 1,497,310 |
| 3 | 29 / 711 | 236.866 | 120.553 | 129.385 | 106.577 | 14.777 | 2,402,589 |
| 4 | 28 / 1381 | 242.746 | 120.896 | 128.840 | 113.138 | 14.615 | 1,962,913 |
| 5 | 19 / 899 | 180.361 | 96.645 | 103.640 | 76.055 | 13.971 | 1,384,008 |
| 6 | 48 / 2780 | 380.629 | 197.781 | 207.433 | 172.135 | 17.227 | 2,500,000 |
| 7 | 16 / 395 | 147.990 | 83.169 | 89.735 | 57.584 | 13.664 | 837,531 |

均值为每个成功视频等权算术平均，非按帧或检测数加权。每视频只跑一次，是不同场景的横截面，不是同视频重复或热服务延迟分布；不提供P95。任务峰均值与最大video任务峰不同，显存峰不能按时间相加。平均点数只是规模统计，不能实际生成一份“平均点云”。

本轮 requested=7（有JSON的1–7），attempted=6，success=6；失败：`[]`；未运行：`[{"video_id": 2, "status": "awaiting_sampling"}]`。video8缺同名检测JSON，排除同协议集合。video2有12帧，可能1.5s前段或2s全程，原始时间戳缺失、尚未确认，不混入均值。

采样协议：video3沿用既有0起1.5s recipe；1/4/5/6/7按JSON帧数吻合的0起1.5s建立显式实验recipe，**没有验证原检测与画面完全对齐**。因此这些是该recipe下的性能结果，不作匹配准确率或画质结论。全部JSON帧保留，没有为通过测试截断长视频；输入依据、源frame index和预处理时间见每case input.json及[输入说明](../reports/profiling-multivideo-20261018/inputs.md)。

| 环境与范围 | 本次条件 |
|---|---|
| 代码 | 本地main `bf9bde8`，生产源码/默认参数未改；原有dirty工作保留 |
| GPU | 物理GPU2，RTX4090D，24564MiB，UUID `GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7`；进程内逻辑cuda:0 |
| 共享负载 | 既有两compute进程约5030MiB保留；主机CPU共享，未设置affinity |
| 模型 | DA3NESTED-GIANT-LARGE-1.1，同snapshot `b2359bdf726fb44ef62acca04d629dcf158053e7`；SAM3.1 PVS |
| 软件 | Python3.11.11，Torch2.7.1+cu126，NumPy1.26.4，Open3D0.19.0，Pillow12.3.0；复用同test-env，未安装/升级 |
| CPU | 40逻辑/20物理核；OMP/OpenBLAS/MKL环境=4，Torch interop默认未强制修改 |
| 输入 | 1080×1920 JPEG quality95、原JSON canonical分类；504×280 DA3 grid；全部F帧一次推理 |
| 主计时起止 | 启动core进程之前的单调时钟 → delivery退出被观察到；含导入/进程生命周期/必要文件写入/轮询间隙 |
| 输出截止 | 本地Viewer generation/CURRENT、ZIP和GLB；写入返回，无fsync承诺，无浏览器首屏承诺 |

根失效`.venv`和旧HF DA3链接未改。使用现成独立uv环境、复用上一轮从本地项目镜像恢复的同模型snapshot；本轮没有下载/另复制权重或启动模型服务。

[均值JSON](../reports/profiling-multivideo-20261018/aggregate.json) · [所有内部计时表](../reports/profiling-multivideo-20261018/INTERNALS.md) · [全部内部均值CSV](../reports/profiling-multivideo-20261018/detail-averages.csv) · [逐视频全部指标CSV](../reports/profiling-multivideo-20261018/per-video-metrics.csv) · [逐reference表](../reports/profiling-multivideo-20261018/references.csv) · [复现与逐视频JSON](../reports/profiling-multivideo-20261018/README.md)

## 2. 完整流水线、Viewer和ZIP：同一时间线的三种前缀

| 互斥阶段 | 完整平均秒 | Viewer前缀贡献秒 | ZIP前缀贡献秒 |
| --- | ---: | ---: | ---: |
| core imports + 仪器安装 | 5.016 | 5.016 | 5.016 |
| DA3重建 | 24.199 | 24.199 | 24.199 |
| PVS + matching | 31.871 | 31.871 | 31.871 |
| Dedup | 0.248 | 0.248 | 0.248 |
| 最终Viewer导出 | 57.844 | 57.844 | 57.844 |
| delivery imports | 3.552 | 0.000 | 3.552 |
| delivery仪器安装 | 0.042 | 0.000 | 0.042 |
| ZIP打包 | 2.214 | 0.000 | 2.214 |
| GLB导出 | 101.179 | 0.000 | 0.000 |
| 进程启动/退出及其余间隙 | 3.087 | 0.088 | 2.256 |

每列可相加，各列是同一工作量的不同截止点，**不能将三列再相加**。Viewer/ZIP是从主起点起的累计时间；GLB是自身耗时。Viewer就绪以后仍有core退出、delivery导入/启动和ZIP打包，因此ZIP累计值不是Viewer加GLB。精确单调时钟与各milestone前缀闭合误差保留在逐视频JSON。

准备线程平均 14.131 秒，其中与matching交集 14.130 秒，matching之外 0.001 秒。它是并发支线，未额外加到上表。重叠区间不能等价为净节省；NN/SOR绑定的GIL机制与PVS构造膨胀见第11节。

## 3. DA3内部：构造、权重、推理、反投影、保存

DA3父阶段平均 24.199 秒，含子进程导入、runner、父进程必要检查和退出。下表子项有明确包含关系，不能把父项和子项相加。

| 同次运行的内部区间 | 平均 wall 秒 | 平均调用次数 | n | 做什么 / 包含关系 |
| --- | ---: | ---: | ---: | ---: |
| 导入与AST仪器安装 | 5.082 | 1.0 | 6 | 在重建子进程内；含依赖导入和插桩准备 |
| JPEG读取 / RGB | 0.436 | 1.0 | 6 | 逐全部F帧；PIL decode/convert |
| from_pretrained CPU总段 | 12.203 | 1.0 | 6 | 父区间，含构造与权重加载 |
| ↳ 网络构造 | 11.320 | 1.0 | 6 | 创建Nested两分支和head，含默认参数初始化 |
| ↳ safetensors load_model | 0.881 | 1.0 | 6 | 含下面load_file和assignment |
| 　↳ load_file | 0.110 | 1.0 | 6 | 打开/mmap；不等于全部物理磁盘读取 |
| 　↳ state assignment | 0.638 | 1.0 | 6 | 参数赋值及可能的页触发 |
| 模型to(cuda) | 1.878 | 1.0 | 6 | 原生调用wall；没有额外同步 |
| inference API | 2.452 | 1.0 | 6 | 父区间，含预处理、forward、输出整理 |
| ↳ preprocess | 0.182 | 1.0 | 6 | resize/normalize/stack |
| ↳ 输入整理/H2D提交 | 0.010 | 1.0 | 6 | 主机调用，不称completed-copy CUDA时间 |
| ↳ forward原生同步区间 | 2.078 | 1.0 | 6 | 保留算法原有前后synchronize |
| ↳ Prediction / D2H | 0.013 | 1.0 | 6 | 转CPU Prediction数组 |
| ↳ processed RGB | 0.150 | 1.0 | 6 | NHWC、反归一化、clip、uint8 |
| CPU反投影 | 0.085 | 1.0 | 6 | 逐F帧K^-1 p d，再C2W |
| NPZ整条保存with语句 | 0.218 | 1.0 | 6 | 包含参数astype等求值及函数内保存 |
| ↳ np.savez函数内 | 0.183 | 1.0 | 6 | 未压缩序列化/写入；无fsync承诺 |

一次输入全部F帧；processed grid为 `[F,504,280]`，不是逐帧独立重建。Nested顺序跑anyview和metric分支，再做metric scaling/depth alignment/sky处理。构造时间含参数默认初始化，不能叫作全部“读权重耗时”。source图像读取、NPZ参数astype、父子进程间隙均有各自范围，不用np.load打开archive的时间代表完整解包。

## 4. Matching与PVS：逐ref、逐框batch

`pairing_3d=next` 环形，共F个reference；每ref只配下一帧，最后回第0帧。全局RNG关键区串行，首次PVS模型构造发生在本次请求内。box batch=4，不同视频帧数/框数不同。

| Matching计时区间 | 平均秒 | 平均次数 | 含义 |
| --- | ---: | ---: | ---: |
| SAM masks | 24.103 | 26.8 | 父区间，含首次加载、predict及cache包装 |
| 数据读取 | 0.518 | 26.8 | 每ref输入 |
| 场景Tensor构造 | 0.152 | 1.0 | 首ref场景准备 |
| transform构造 | 0.004 | 26.8 | 每ref camera transforms |
| 参考mask点采样 | 1.492 | 1,084.7 | 每检测最多70点 |
| world→camera | 0.184 | 1,070.8 | 含设备等待 |
| 3D→2D投影 | 0.361 | 1,070.8 | 投影到next帧 |
| 投影后处理 | 0.332 | 1,070.8 | 有效域等 |
| 目标bbox匹配 | 3.110 | 1,070.8 | 候选采样、SVD/进深和评分 |
| 唯一性 | 0.001 | 25.2 | 每对约束 |
| 结果后处理 | 0.014 | 26.8 | 匹配结果结构 |
| process_images父段 | 30.572 | 26.8 | 嵌套父区间，不能再加到以上子项 |

| 同次运行的内部区间 | 平均 wall 秒 | 平均调用次数 | n | 做什么 / 包含关系 |
| --- | ---: | ---: | ---: | ---: |
| 首次模型加载 | 18.680 | 1.0 | 6 | 一次/视频；与NN/SOR准备线程重叠 |
| ↳ model_builder导入 | 0.663 | 1.0 | 6 | 动态导入 |
| ↳ CPU模型构造 | 16.979 | 1.0 | 6 | 位于cuda.device作用域内；不等于读取权重 |
| checkpoint mmap | 0.155 | 1.0 | 6 | weights_only / mmap |
| 严格state assignment | 0.252 | 1.0 | 6 | 只保留tracker和共享backbone权重 |
| eval/to(cuda) | 0.581 | 1.0 | 6 | 实际设备搬运 |
| 全部predict API | 4.850 | 26.8 | 6 | 每图一次encoder，box batch=4 |
| 图像transform/H2D语句 | 0.444 | 26.8 | 6 | 1008×1008，归一化后提交GPU |
| box Tensor构造 | 0.003 | 26.8 | 6 | 缩放语句L79另有AST记录 |
| encoder主机调用 | 1.258 | 26.8 | 6 | enqueue wall，不能视作GPU完成时间 |
| decoder主机调用 | 1.946 | 281.2 | 6 | ceil(boxes/4)批，不同视频批数不同 |
| logits上采样 | 0.010 | 281.2 | 6 | bilinear到源输入图大小 |
| finite检测 | 0.063 | 281.2 | 6 | 可能引发GPU→CPU依赖 |
| mask阈值/D2H | 0.064 | 281.2 | 6 | CPU numpy |
| 按框裁剪mask | 0.069 | 1,084.7 | 6 | 逐box |
| scores D2H | 0.024 | 281.2 | 6 | CPU list |
| presence D2H | 0.011 | 281.2 | 6 | CPU list |
| 最终同步 | 0.001 | 26.8 | 6 | 保留原算法边界 |
| stack/返回 | 0.040 | 26.8 | 6 | 含np.stack；native总量包含encoder |

PVS native encoder平均 1.941 秒；native inference_and_postprocess平均 4.384 秒，后者包含encoder。`mask_postprocess`原有空span不作为有效工作；`cache_npz_load`只覆盖archive打开，不代表完整NPZ读取。所有ref的具体wall、boxes、首次模型加载与native区间在CSV中，不能把某一个后续ref的耗时当作整次热请求。

## 5. 几何准备、最终Viewer、Dedup

| 同次运行的内部区间 | 平均 wall 秒 | 平均调用次数 | n | 做什么 / 包含关系 |
| --- | ---: | ---: | ---: | ---: |
| 几何准备总段 | 14.131 | 1.0 | 6 | 后台线程；不能加到matching主表 |
| cache读取/校验 | 0.185 | 1.0 | 6 | 全native grid |
| U/V与depth | 1.790 | 1.0 | 6 | 相邻grid差分与连续性 |
| raw filter准备 | 11.034 | 1.0 | 6 | snapshot/cloud/NN/median/SOR raw mask |
| ↳ NN原生 | 3.326 | 1.0 | 6 | 建索引、每点最近邻；包含在prepare中 |
| ↳ SOR原生 | 5.736 | 1.0 | 6 | k=20/std=2；尚未套用商品保护 |
| 源照片/相机纹理 | 0.903 | 1.0 | 6 | JPEG与affine/camera metadata |

| 同次运行的内部区间 | 平均 wall 秒 | 平均调用次数 | n | 做什么 / 包含关系 |
| --- | ---: | ---: | ---: | ---: |
| 最终Viewer总段 | 57.844 | 1.0 | 6 | 当前可用里程碑的最后主阶段 |
| thumbnails | 1.650 | 1.0 | 6 | 逐observation，Viewer必要缩略图 |
| 显示方向 | 1.980 | 1.0 | 6 | 与ground删除不同的RANSAC拟合 |
| ↳ 方向NN | 0.138 | 1.0 | 6 | 最多200K候选域 |
| ↳ 方向RANSAC | 1.431 | 2.8 | 6 | 候选方向平面 |
| 选点总段 | 49.264 | 1.0 | 6 | 包含以下标签/过滤/预算等 |
| ↳ mask→商品label | 0.328 | 1.0 | 6 | canonical processed PVS mask |
| ↳ 最终几何过滤 | 44.101 | 1.0 | 6 | raw NN/SOR复用，再保护/DBSCAN/ground |
| 　↳ SOR应用整段 | 0.698 | 1.0 | 6 | 商品保护、20% guard、select；不含提前SOR原生 |
| 　↳ DBSCAN整段 | 32.943 | 1.0 | 6 | native+簇阈值+保护+select |
| 　　↳ DBSCAN原生 | 32.259 | 1.0 | 6 | eps=5×medianNN，min_points=10 |
| 　↳ ground整段 | 10.316 | 1.0 | 6 | RANSAC+双侧判断+保护+select |
| 　　↳ ground RANSAC | 9.868 | 1.0 | 6 | 当前点域，最多1000假设，可能提前停止 |
| ↳ full-grid特征 | 0.436 | 1.0 | 6 | depth/valid/语义region生成edge、halo、weights、caps |
| ↳ 背景voxel | 1.027 | 1.0 | 6 | 5mm，置信度优先 |
| ↳ 商品/背景预算 | 2.520 | 1.0 | 6 | 分ID预算与adaptive scale/selection |
| 　↳ adaptive.select | 2.332 | 223.5 | 6 | 包含于预算分组；勿重复相加 |
| ↳ 选点数组序列化 | 0.404 | 1.0 | 6 | 稳定索引及紧凑数组 |
| Surfel打包 | 0.153 | 1.0 | 6 | U/V/scale/depth与纹理metadata |
| point ranges | 4.333 | 1.0 | 6 | 每observation关联选点半开区间 |
| generation发布 | 0.140 | 1.0 | 6 | 写binary/JSON/JPEG及CURRENT |

| 同次运行的内部区间 | 平均 wall 秒 | 平均调用次数 | n | 做什么 / 包含关系 |
| --- | ---: | ---: | ---: | ---: |
| Dedup总段 | 0.248 | 1.0 | 6 | 全部F帧 |
| parse summaries | 0.004 | 1.0 | 6 | 匹配边 |
| union-find/GID | 0.033 | 1.0 | 6 | 同图冲突不合并 |
| 逐帧global ID | 0.092 | 1.0 | 6 | 分类聚合 |
| global pair发布 | 0.043 | 1.0 | 6 | 原子发布相关JSON |

NN用全有效点的最近邻距离中位数设置随尺度变化的阈值；SOR检查每点20个近邻的平均距离，std_ratio=2，raw布尔结果提前计算。最终label到达后用商品点保护修复raw拒绝，并执行20%最小保留guard。DBSCAN在SOR应用后的点域做半径邻域和连通扩展；ground在当前点域做RANSAC、双侧判断、保护与guard。显示方向另用最多200K候选拟合，不能与ground计时混用。

Python计时只能分开native NN/SOR/DBSCAN/RANSAC整体。其内部KD-tree建树、邻域查询、DBSCAN扩展、RANSAC实际迭代数**本轮没有逐段测量**；没有用复杂度推导冒充实测秒数。[原生循环说明](viewer_export_algorithm.md#61-open3d-0190-原生循环)保留算法细节。

## 6. 最终点数的完整流向

| 点数域 | 跨视频平均点数 | video1 | video3 | video4 | video5 | video6 | video7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense native grid | 3,786,720 (n=6) | 2,963,520 | 4,092,480 | 3,951,360 | 2,681,280 | 6,773,760 | 2,257,920 |
| export-valid | 3,786,720 (n=6) | 2,963,520 | 4,092,480 | 3,951,360 | 2,681,280 | 6,773,760 | 2,257,920 |
| 原SAM商品 | 1,560,418 (n=6) | 997,310 | 1,902,589 | 1,462,913 | 884,008 | 3,778,160 | 337,531 |
| 原背景 | 2,226,302 (n=6) | 1,966,210 | 2,189,891 | 2,488,447 | 1,797,272 | 2,995,600 | 1,920,389 |
| SOR raw保留 | 3,624,269 (n=6) | 2,835,143 | 3,888,295 | 3,803,234 | 2,588,164 | 6,453,776 | 2,177,002 |
| SOR商品救回 | 1,032 (n=6) | 196 | 825 | 579 | 506 | 3,298 | 788 |
| SOR应用后 | 3,625,301 (n=6) | 2,835,339 | 3,889,120 | 3,803,813 | 2,588,670 | 6,457,074 | 2,177,790 |
| DBSCAN raw保留 | 3,136,689 (n=6) | 2,357,677 | 3,402,853 | 3,244,481 | 2,070,858 | 5,894,636 | 1,849,627 |
| DBSCAN商品救回 | 34,408 (n=6) | 2,108 | 22,822 | 87,114 | 23,102 | 38,559 | 32,740 |
| DBSCAN应用后 | 3,171,096 (n=6) | 2,359,785 | 3,425,675 | 3,331,595 | 2,093,960 | 5,933,195 | 1,882,367 |
| ground RANSAC输入 | 3,171,096 (n=6) | 2,359,785 | 3,425,675 | 3,331,595 | 2,093,960 | 5,933,195 | 1,882,367 |
| ground RANSAC内点 | 324,564 (n=6) | 364,414 | 313,004 | 333,295 | 398,691 | 352,892 | 185,086 |
| ground raw保留 | 2,299,778 (n=5) | 1,995,371 | 3,112,671 | 2,998,300 | 1,695,269 | — | 1,697,281 |
| ground商品救回 | 91,223 (n=5) | 6 | 230,825 | 172,123 | 0 | — | 53,161 |
| ground应用后（仅调用apply时） | 2,391,001 (n=5) | 1,995,377 | 3,343,496 | 3,170,423 | 1,695,269 | — | 1,750,442 |
| 最终filter后 | 2,981,367 (n=6) | 1,995,377 | 3,343,496 | 3,170,423 | 1,695,269 | 5,933,195 | 1,750,442 |
| filter后商品 | 1,560,418 (n=6) | 997,310 | 1,902,589 | 1,462,913 | 884,008 | 3,778,160 | 337,531 |
| filter后背景 | 1,420,948 (n=6) | 998,067 | 1,440,907 | 1,707,510 | 811,261 | 2,155,035 | 1,412,911 |
| 背景voxel候选 | 1,081,097 (n=6) | 803,686 | 1,084,992 | 1,273,600 | 705,892 | 1,715,050 | 903,363 |
| 商品选入 | 1,264,058 (n=6) | 997,310 | 1,902,589 | 1,462,913 | 884,008 | 2,000,000 | 337,531 |
| 背景选入 | 500,000 (n=6) | 500,000 | 500,000 | 500,000 | 500,000 | 500,000 | 500,000 |
| Viewer选入总量 | 1,764,058 (n=6) | 1,497,310 | 2,402,589 | 1,962,913 | 1,384,008 | 2,500,000 | 837,531 |

商品保护在几何过滤时生效；最终商品预算仍为2M、背景预算500K，商品超过预算时也会采样。raw保留+商品救回不一定等于应用后：20% guard若触发会跳过该过滤，guard记录在完整CSV里。没有以cluster的非负标签冒充SAM商品标签。

ground没有进入apply时，该行记为缺失，均值按实际n；这不等于RANSAC未计算，也不等于输出0点。具体例子：video6的RANSAC为352892/5933195≈5.947%内点，未达到8%条件，所以没有应用删除；最终filter仍保留5,933,195点。

| 输出元素 | 平均 | 最小–最大 |
| --- | ---: | ---: |
| GLB LOD选入点 | 1,195,338 | 689,394–2,027,824 |
| LOD移除点 | 568,721 | 148,137–1,104,762 |
| 实际baked Surfel | 1,180,904 | 684,791–2,007,183 |
| 无有效tile跳过 | 14,433 | 4,603–20,641 |
| 三角形 | 2,361,809 | 1,369,582–4,014,366 |
| atlas页 | 29 | 17–49 |

Viewer选点与GLB LOD/bake是不同表示，GLB不回写Viewer点集。完整depth grid仍用于连续性、遮挡和来源判定。选点相同或总数相同都不能证明来源/背景/方向或画面相同。Open3D RANSAC流程没有显式固定随机seed，本轮逐阶段计数显示实际发生了什么，不以旧独立诊断补写。

## 7. ZIP内部：成员读入、压缩、写入

ZIP自身平均 2.214 秒；累计就绪平均 127.242 秒，前缀组成见第2节。

| 成员类别 | 平均成员数 | 平均原始MiB | 平均归档payload MiB | 平均成员write秒 |
| --- | ---: | ---: | ---: | ---: |
| binary / DEFLATE level1 | 8.0 | 66.104 | 39.805 | 2.060 |
| JPEG / STORED | 1,111.5 | 23.257 | 23.257 | 0.131 |
| JSON / STORED | 3.0 | 0.329 | 0.329 | 0.001 |

成员write wall包含文件读入、压缩与归档写入，不是纯zlib。binary为8个文件、DEFLATE level1；JPEG/JSON为STORED。上表采用成员wrapper范围，含每成员metadata/统计小开销；INTERNALS的zip.member.*是更内层的write span，两者不可当同一口径。类别加总不覆盖成员外的枚举、归档metadata和关闭。最终ZIP包含必要Viewer数据、源纹理与缩略图，不含GLB/模型或本轮没有提供的masterdata。

## 8. GLB内部：LOD、七块bake、RGB、atlas、最终文件

| video | LOD选入 | 实际baked | atlas页 | bake批次 | GLB MiB | GLB秒 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 963,232 | 946,155 | 23 | 953 | 83.828 | 81.582 |
| 3 | 1,297,827 | 1,282,942 | 31 | 1282 | 115.399 | 106.577 |
| 4 | 1,280,314 | 1,265,250 | 31 | 1265 | 113.383 | 113.138 |
| 5 | 913,434 | 899,106 | 22 | 901 | 79.292 | 76.055 |
| 6 | 2,027,824 | 2,007,183 | 49 | 2006 | 189.596 | 172.135 |
| 7 | 689,394 | 684,791 | 17 | 682 | 64.176 | 57.584 |

| 同次运行的内部区间 | 平均 wall 秒 | 平均调用次数 | n | 做什么 / 包含关系 |
| --- | ---: | ---: | ---: | ---: |
| GLB总段 | 101.179 | 1.0 | 6 | 同一次实际导出，默认encoder_workers=1 |
| 输入_load | 0.144 | 1.0 | 6 | generation文件读取与校验 |
| 保守2×2 LOD | 3.303 | 1.0 | 6 | 商品block满足法向/depth/coverage才合并 |
| ↳ coverage | 0.921 | 45.8 | 6 | 每批最多8192组，已含于LOD |
| 全部bake batch | 54.401 | 1,181.5 | 6 | 按frame，每批最多1024候选 |
| ↳ 中心/相机可见性 | 0.440 | 1,181.5 | 6 | 七个连续AST块之一 |
| ↳ 初始footprint/连续性 | 1.010 | 1,181.5 | 6 | 含_bound_footprint |
| ↳ radius/最终直径 | 0.505 | 1,181.5 | 6 | 保持长边≤6×native最大间距 |
| 　↳ 最终长边限制 | 0.432 | 1,181.5 | 6 | 内层函数，不再加到七块 |
| ↳ 8×8 texel投影 | 10.704 | 1,181.5 | 6 | world→camera、圆盘/边界mask |
| ↳ depth/source mask | 4.816 | 1,181.5 | 6 | 遮挡、来源/面积与有效tile |
| ↳ source/RGB采样 | 14.674 | 1,181.5 | 6 | source→texture坐标及双线性采样 |
| 　↳ bilinear函数 | 14.333 | 1,181.5 | 6 | 包含于上行 |
| ↳ tile/corners/延展 | 22.070 | 1,181.5 | 6 | 包含RGB延展 |
| 　↳ RGB延展 | 20.619 | 1,181.5 | 6 | 固定四方向、旧轮known，alpha不变 |
| atlas组页 | 0.587 | 28.8 | 6 | 8×8内部+一圈gutter；最多41616 tiles/页 |
| 几何量化/写BIN | 1.570 | 28.8 | 6 | u16 position/UV/index，含bufferView |
| atlas编码总段 | 39.802 | 28.8 | 6 | 包含alpha、palette、PNG |
| ↳ alpha检查 | 0.933 | 28.8 | 6 | 二值alpha |
| ↳ compact RGBA | 38.864 | 28.8 | 6 | 含MEDIANCUT和PNG子项 |
| 　↳ palette量化 | 17.018 | 28.8 | 6 | 128色，无dither |
| 　↳ PNG save | 20.842 | 28.8 | 6 | optimize=True；PIL累计按contexts可再分 |
| 图片写BIN | 0.032 | 28.8 | 6 | 编码字节，不重解码 |
| 最终GLB写入 | 0.077 | 1.0 | 6 | JSON/BIN chunk与header，文件写入返回 |

生产report另外提供两个调度区间：bake/load/LOD侧平均 59.693 秒，compact/encode/write侧平均 41.484 秒；后者包含串行encode调用（encoding_join）平均 39.802 秒和main_thread_write平均 1.682 秒。生产迭代器的计时归属与函数span不同，不能与上表再合并求和。

实际循环：frame → 每1024候选bake → 有效10×10 RGBA tile → 每41616个tile组atlas → 编码 → geometry/image/material顺序写BIN → 最终GLB。7个bake块连续覆盖函数主体，内层bilinear/RGB/长边函数已在所属块内。MEDIANCUT与PNG同样是encode子项。`self_s`仅减已测直接子区间，仍含其余操作和仪器成本，不能叫“纯Python时间”。

六视频测量时`encoder_workers=1`，已有2/4worker功能没有自动开启。后续源码默认改为2，依据见第12节。原始128色、PNG optimize、LOD和清理仍保留，尤其是最终长直径≤6×native最大间距；没有靠放宽长边、删DBSCAN或降纹理获得该批数字。

## 9. 显存与RAM：阶段峰、模型参数、当前占用、allocator峰分开

| 阶段 | 任务driver阶段峰均值GiB | 最大一条video阶段峰GiB |
| --- | ---: | ---: |
| DA3重建 | 14.730 | 17.227 |
| PVS模型加载 | 2.627 | 2.830 |
| 全部PVS predict中的最高采样 | 3.695 | 3.758 |
| matching | 3.696 | 3.758 |
| 几何准备（与matching重叠） | 1.023 | 1.086 |
| 最终Viewer | 2.838 | 2.904 |
| GLB | 0.000 | 0.000 |

| 进程 / 边界 | 参数GiB | buffer GiB | 当时allocated GiB | 当时reserved GiB | 当时未用reservation GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| DA3 model ready | 6.295 | 0.000 | 6.358 | 6.377 | 0.019 |
| DA3 forward之前 | 6.295 | 0.000 | 6.401 | 6.420 | 0.019 |
| DA3 forward之后 | 6.295 | 0.000 | 6.437 | 14.316 | 7.878 |
| PVS model ready | 1.813 | 0.009 | 2.292 | 2.399 | 0.107 |
| PVS predict结束边界的均值 | 1.813 | 0.009 | 2.885 | 3.281 | 0.396 |

| 进程 | allocated峰均值GiB | reserved峰均值GiB | RSS HWM均值GiB | user CPU秒均值 | system CPU秒均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| da3 | 11.259 | 14.316 | 13.903 | 22.564 | 4.323 |
| core | 3.106 | 3.282 | 5.567 | 169.458 | 11.707 |
| delivery | — | — | 0.874 | 128.702 | 106.392 |

表中每video先取同一时点任务PID的driver和，再取该阶段最大；跨video平均的是这些峰。predict行先取每video全部predict里的最大，不按ref数加权。GPU已有resident约5030MiB排除在任务峰外；整卡峰平均 19.660 GiB。200ms采样会漏短峰，极短阶段可能无采样，按缺失处理。

阶段内有完整compute-app查询、且无本任务GPU进程的时点记为观测0（例如CPU的GLB阶段）；没有查询覆盖的短段才记缺失。进程生命周期allocator未初始化则保留缺失，不用driver的0替代。

allocated是PyTorch活跃分配，reserved是allocator保留；同一snapshot的reserved−allocated是未用reservation，不含driver/context。参数/buffer按实际device/numel/element_size计算。snapshot剩余allocated还含场景Tensor、激活和其他数据，不称KV cache；forward后也不是最大激活的时点。生命周期峰、阶段driver峰、当时快照和各进程峰互不相加。

同时进程树RSS采样和峰的跨video均值为 13.580 GiB；共享页可能重复计。单进程HWM和CPU user/system分别列出，CPU秒可超过wall。时钟用紧邻采集的epoch/monotonic锚点映射GPU时间戳，并裁剪到每case主窗口；video3锚点在本次运行中补采、未改变其定时或进程。

## 10. CUDA events与视频输入准备（均不再加到主表）

| 进程 | CUDA event范围（保留父上下文） | 平均秒 | n |
| --- | ---: | ---: | ---: |
| core | `pvs.decoder` | 1.962 | 6 |
| core | `pvs.encoder` | 1.936 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L362 > da3.branch_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L132 > da3.backbone_forward` | 1.248 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L362 > da3.branch_forward` | 1.669 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L365 > da3.branch_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:DepthAnything3Net.forward:L132 > da3.backbone_forward` | 0.145 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L365 > da3.branch_forward` | 0.403 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L368 > da3.metric_scaling` | 0.000 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L369 > da3.depth_alignment` | 0.004 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward.host > Depth-Anything-3/src/depth_anything_3/model/da3.py:NestedDepthAnything3Net.forward:L370 > da3.sky_handling` | 0.001 | 6 |
| da3 | `da3.runner_main > src/da3_runner.py:main:L237 > da3.inference_api > da3.forward_native_sync > da3.nested_forward` | 2.077 | 6 |

这些是同一运行内CUDA stream elapsed，保留嵌套父上下文；包含CPU enqueue间隙的可能性，不等于exclusive kernel和。不增加逐层synchronize，只在算法原有同步后读取events。host wall与CUDA event不能相减得到“纯CPU”。

| 视频输入准备区间 | 平均秒 |
| --- | ---: |
| canonical_adapter_load | 3.907 |
| canonical_frame_adaptation | 0.058 |
| canonical_json_encode_and_write | 0.021 |
| jpeg_encode_and_write | 0.326 |
| json_video_metadata_and_sampling_plan | 0.051 |
| video_grab_retrieve | 1.684 |
| 完整输入准备wall（父区间） | 6.056 |

输入准备使用视频顺序grab、选帧retrieve、JPEG quality95及现有canonical分类适配。视频抽帧/检测JSON适配在主计时外，单独如上；detector和classifier均没有推理，沿用JSON。若要原视频至全部产物的本地耗时，可按每video加其preprocessing wall，但这仍不包含原检测/分类/HTTP/COS/下载/浏览器。

## 11. 解释边界、为什么耗时、下一步

大点域上的DBSCAN/ground、PVS CPU构造及Surfel逐texel颜色/透明延展、palette/PNG是不同工作。F决定全grid规模，检测数还决定decoder批次、缩略图和point ranges；Viewer预算封顶不会让此前全grid过滤消失。GLB LOD减少候选，但每个候选仍可能处理64texels，编码还要遍历整atlas透明区，所以不能按最终GLB字节数推断导出应很快。

具体例子：video6的ground RANSAC计算了18.154秒，因内点比例不足最终未应用删除；这不是0成本的跳过。point ranges关联在video6为13.950秒、video3为3.429秒，属于观察数和选点规模共同影响的CPU工作。这里给出成本与分支事实，不据单次跨场景差异推导严格复杂度或可回收时间。

完整视频结束后另做一次有界CPU合成probe：250万int32有序labels、200个query值，Python int逐个left/right共1.152541秒，同dtype逐标量0.001243秒，typed vector两次整批查询0.000141秒，全部range端点精确一致。它没有使用实际生产labels，也没有修改生产代码；不计入六video均值，不支持直接从13.950秒扣除差额。[probe原始结果](../reports/profiling-multivideo-20261018/point-range-probe.json)。

生产labels为int32，而enumerate产生Python int。[NumPy1.26.4搜索源码](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/item_selection.c#L1964)先求共同dtype再准备整个haystack；[Python整数dtype发现](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/abstractdtypes.c)与[dtype提升](https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/multiarray/array_coercion.c#L643)支持本机64位long导致重复int32→int64临时转换的解释。这是源码机制与合成现象一致的推断，未直接测生产转换次数。下一步值得先在真实labels上验证typed vector精确替换、空range/缺label边界，再跑完整交付对照。

上一轮独立初始化复现与固定版Open3D绑定/pybind规则支持调查NN/SOR持GIL造成的PVS构造膨胀；该证据保留于[历史单视频记录](../reports/profiling-20261018/README.md)及其overlap结果。它不是本轮的受控加速实验，不能把线程重叠秒数直接承诺为可回收秒数。下一步仍宜验证GIL隔离、现有atlas worker的完整交付收益，再做DBSCAN-off/proxy/early-background的视觉对照，本轮没有实施或promotion。

这是统一**细分profiling模式**：AST span、scalar点数观察、CPU时间读取与CUDA events增加了未量化开销。跨video同模式适合描述这些输入的规模差异；不宣称无仪器生产延迟。GPU2共享resident、主机CPU共享、OS页缓存/文件缓存未清空；新进程/新DA3-PVS结果cache不等于物理冷盘。

前一轮video3的248.900/128.750/137.148秒仍是那次实测，保留[历史summary](../reports/profiling-20261018/summary.json)，**未混进本轮均值**。不同抽帧还原、共享负载、随机拟合和仪器条件不支持把前后差额叫作受控提速率。当前文档取代旧单样本作为性能入口。

验证只覆盖诊断脚本接线、完整阶段/帧数/输出、计时闭合和必要格式边界；没有修改生产算法，因此未重复跑全量pytest或额外hash/CRC。未测热服务、HTTP/BSON/COS、网络、浏览器首屏、人工画质、匹配/面积准确率、native内部细分、精确GIL等待归因和仪器开销。逐视频JSON与所有内部均值/范围/n完整保留，失败和未测不静默删除。

## 12. 后续已实现的优化：固定video3前后完整对照

2026-10-18晚，按Rick批准实现并测试。相同既有29帧/711检测、显式seed42、GPU2/RTX4090D、OMP/OpenBLAS/MKL=4；各臂独立模型/结果进程，源码在修改前后分别隔离保存。**每臂一次**，OS/模型页缓存不清空，主机共享负载未隔离。本轮改用function-level spans和两臂相同的诊断点域捕获，不与旧AST仪器的236.866/248.900秒或六视频均值混算提速率。仍不包含抽帧/检测分类/HTTP/COS/浏览器。

| 同次完整运行指标 | 修改前 | 安全默认候选 | 单位 |
|---|---:|---:|---|
| 完整流水线 | 253.139 | 186.894 | 秒，缩短26.2% |
| Viewer可用累计 | 129.854 | 109.163 | 秒，缩短15.9% |
| ZIP就绪累计 | 138.097 | 117.362 | 秒，缩短15.0% |
| DA3重建自身 | 25.133 | 13.706 | 秒 |
| ↳ 网络构造 | 12.624 | 0.914 | 秒，包含于重建 |
| ↳ CPU构造+加载 | 13.562 | 2.425 | 秒，包含网络构造 |
| Matching | 30.663 | 29.708 | 秒，算法未改，不归因这点波动 |
| Dedup | 0.143 | 0.148 | 秒 |
| 最终Viewer导出 | 68.859 | 60.615 | 秒 |
| ↳ point ranges | 3.939481 | 0.001608 | 秒，包含于Viewer |
| ↳ 几何filter | 53.673 | 49.839 | 秒，仍为原完整点域算法 |
| ZIP打包自身 | 2.745 | 2.774 | 秒 |
| GLB导出自身 | 114.041 | 68.770 | 秒，缩短39.7% |
| ↳ LOD | 4.964 | 4.896 | 秒 |
| ↳ 所有bake batch | 61.205 | 58.505 | 秒 |
| ↳ atlas encode累计调用wall | 43.640 | 45.379 | 秒，候选跨worker且与bake重叠 |
| ↳ 主线程encoding join | 43.649 | 0.900 | 秒，候选主要等待已被重叠隐藏 |
| 任务driver显存采样峰 | 14.777 | 14.777 | GiB，200ms采样 |
| Viewer点数 | 2,402,589 | 2,402,589 | 点 |
| GLB大小 | 121,197,580 | 121,197,580 | bytes |

累计前缀、阶段和嵌套子项不能再相加。候选atlas累计encode并没有变成0.9秒：0.9是主线程等待，45.379是worker调用wall之和，它与58.505秒bake重叠。未修改filter的时间波动也不归因为某项优化；控制组ground在完整计时里没有单独native子span，不能拿独立组件秒数填入。

**实际采用：** DA3只跳过已覆盖的大Linear/Conv weight随机填充；point ranges用同dtype批量查询；GLB按frame共享常量并默认两页有界FIFO编码流水，保持`source_world`投影顺序。源f16几何、规范JPEG、alpha/depth边界、LOD、128色、最终6倍native直径上限不变。

真实checkpoint审计：1383个state名字对应1377个独立参数，checkpoint1377键完整覆盖；6个省略名字是同一个LayerNorm的别名，并非漏载。保留1个nonpersistent buffer原生初始化。跳过406个大weight的随机填充，共6737780736bytes；不使用meta/降精度。独立新CPU进程对照构造11.826516→0.651774秒、构造+审核+加载12.993895→2.045339秒，单独标记组件证据，不替换上表GPU完整运行内部数值。

**输出验证：** 15项DA3数值数组（含全部depth/worldpoints/confidence/相机/RGB及帧绑定）、global mapping、8项Viewer二进制、objects、manifest以及最终GLB完整payload均精确相等。两臂各386globalIDs/711observations、2402589Viewer点、1282759baked surfels/2565518三角/31atlas页。最终GPU进程全部退出，GPU2只保留原6221/8550resident。

线程数选择另用同一个完成generation做完整source_world导出，各一次且固定顺序：1/2/4workers为108.396069/67.627083/66.638055秒，采样selfRSS峰957362176/1041502208/1084465152bytes。原GLB→新1、新1→2/4的全部accessor/document和128295600decodedRGBA texel精确相等。默认选2；4仅再快0.989秒(1.46%)，未证明稳定额外收益，保留显式4。不把该CPU组件表替代完整流水线表。

**已实现但不推广：** ground50K代理候选在同一3425675点域上10.879015→0.980081秒，但额外删除51286背景点；虽然商品0变化，平面法向与用于摆正的floor夹角87.77度，且81.7%内点靠商品保护救回，因此保留`ground_proposal_max_points=0`默认。`camera_basis`增量投影在29帧/29696样本中几何相同，但6个alpha texel变化，共同不透明RGB最大差1；保持显式实验选项。**这两项的速度未计入安全默认候选收益。**

[实现、复现与定向测试](../reports/latency-optimization-20261018/README.md) · [完整pair JSON](../reports/latency-optimization-20261018/pair-summary.json) · [输出与资源验证](../reports/latency-optimization-20261018/pair-validation.json) · [状态/失败与更正](../reports/latency-optimization-20261018/progress.md) · [PyTorch后续方向](../reports/latency-optimization-20261018/pytorch_candidates.md)。未重新测优化后六视频平均、P95、热服务或外部准确率；本轮没有部署/上传/commit/push。

## 13. 有界Torch采样、mask和方向延展：完整GLB组件实验

2026-10-18 23时，Rick批准测试。复用§12同一video3 generation，native CPU与Torch候选各新进程、按顺序各一次，均source_world／encoder2／tile8／batch≤1024；GPU仅物理2，CPU线程4，页缓存和共享负载未隔离。这里只计已有Viewer后的GLB组件，**没有重跑DA3／matching／Viewer完整流水线**，不直接替换186.894秒。

| 同次GLB组件指标 | Native CPU | Torch候选 |
|---|---:|---:|
| 完整export wall | 66.589322s | 44.671011s（减少32.916%） |
| 脚本计时至GLB就绪，含import | 69.928963s | 48.084271s |
| imports | 3.335594s | 3.413232s |
| producer窗口，含LOD等／扣主线程写入与join | 63.420602s | 36.657004s |
| 主线程compact／write／join | 3.166538s | 8.012695s |
| ↳ encoding join | .877399s | 5.669511s |
| baked／triangles／atlas页 | 1282759／2565518／31 | 相同 |
| GLB bytes | 121197580 | 相同 |

export计时包含候选首次CUDA context、frame／batch上传、CPU投影、GPU mask／RGB／延展、D2H和最终编码／写入；正确性比较在timer外。脚本ready不含外部uv／解释器启动，两脚本起点略有差异；candidate whole59.170519秒还含就绪后约11.086秒验证，不是导出耗时。producer更快后FIFO等待增加，不将join与重叠编码重复相加。

候选仅在[实验实现](../reports/torch-bake-fusion-20261018/torch_bake.py)：CPU保留原source_world、affine／half-pixel浮点运算顺序；Torch显式float64四tap bilinear保持NumPy提升语义，合并mask／nearest depth／round与固定old-known四方向延展，用static-shape where避免每方向动态布尔索引。它是GPU阶段整合，不是单一融合kernel。每frame RGB／depth各一次上传，共29／29；1280batch处理1297686LOD候选，每batch最多1024，只持有当前frame GPU纹理和depth。生产默认未改。

数值验证：真实29批sample共29696input／29316幸存quad，全部corners及2931600raw RGBA texel精确相等（含透明RGB／alpha）；完整GLB的document、所有accessor bufferView字节、31页／128295600decoded RGBA精确相等。未逐项比较全导出的全部raw tiles，也未检查整个GLB文件字节。CPU30项通过，where修改后受影响10项补测通过，12tiny smoke相等，两位medium独立最终结果复核PASS。

Torch allocator生命周期allocated峰37.119MiB、reserved54MiB，不含完整CUDA context／driver；candidate RSS包含后续比较，native不含，不能推导export内存升降。每臂一次、单generation；未测跨视频／重复分布／P95、完整pipeline新耗时或人工准确率。原GPU2 resident保留。[完整报告与命令](../reports/torch-bake-fusion-20261018/README.md) · [汇总](../reports/torch-bake-fusion-20261018/summary.json) · [状态](../reports/torch-bake-fusion-20261018/STATUS.md)。

## 14. 后续14GB预算验证：DA3 allocator候选

Rick随后允许其他流程用GPU，但限制任务峰值。本轮先按十进制14e9bytes、所有owned GPU PID同时合计验证。GPU2固定video3同29帧／504／seed42／原精度，各一次新进程：native allocator启用expandable_segments，并在模型搬GPU前限制allocator12GiB。实际task driver采样峰**12918456320bytes＝12.918GB＝12.03125GiB**，allocated峰12169374720、reserved峰12475957248bytes；15项完整DA3数组dtype／shape／值精确相等。nominal50ms、302timestamp组、实际最大间隔约68ms，未达到13.5GB预警线，无解析失败或不支持警告，worker正常退出；原GPU2 resident保留。

此前15132MiB是另一轮200ms采样，只作历史参照，不计算严格配对降幅或速度收益。本次worker生命周期13.760725s、runner main10.011166s，包含提前初始化／snapshot，与§12窗口不同。fraction不约束所有context／外部CUDA分配，采样也不能证明亚采样瞬时绝不超限；当前结论仅是此video3候选采样达标。

没有修改生产默认，也没有验证完整pipeline新耗时。video6历史活跃tensor峰13688829440bytes已高于该12GiB allocatorcap，不能把此配置推广所有输入；后续video6补测在该cap下OOM，见§15。14GB／14GiB单位确认仍待回复。GPU bake正式接入方案已形成，当前根视频入口只有Viewer，未更改Docker或部署。[预算报告／结果与命令](../reports/gpu-latency-budget-20261018/README.md)。

## 15. 2026-10-19补测：video6完整GPU GLB有效，DA3固定cap失败

Rick要求测试效果。固定既有video6／48帧／250万Viewer点generation，GPU2／RTX4090D，CPU线程4，source_world／encoder2／tile8／batch≤1024，LOD、palette与Surfel长边限制不变。CPU与GPU完整导出各一次成功新进程，另有一次监控中止的GPU尝试保留而不计入速度。OS缓存与共享负载未隔离；这是已有Viewer之后的GLB组件，不是完整流水线新耗时。

| 同次video6 GLB指标 | Native CPU | Torch GPU |
|---|---:|---:|
| 完整export wall | 100.497878s | **71.574736s（减少28.780%）** |
| 脚本至GLB就绪，含import | 103.826038s | 75.098048s |
| producer窗口，扣主线程compact/write/join | 95.979967s | 55.011276s |
| 主线程compact/write/join | 4.516137s | 16.561592s |
| ↳ encoding join | 0.984595s | 12.629608s |
| LOD候选／baked | 2027824／2007183 | 相同 |
| 三角／atlas页／GLB bytes | 4014366／49／198805712 | 相同 |

完整导出节省28.923142s。GPU计时包含第一次CUDA context、H2D、CPU投影、GPU采样/mask/延展、D2H、PNG和写盘；GPU就绪后的17.903855s正确性比较单列，whole_process93.001903s含此比较。producer与compact为互斥窗口，worker编码会重叠，不能重复相加。GPU producer加速后编码join变大，下一项值得测GPU bake配合encoder4，本轮未执行。

48帧raw样本49152输入／48560配对quad／4856000raw RGBA texels，corners与完整RGBA均exact。完整2006batches，RGB／depth各上传48次；全部document关联、accessor字节及49页**200736000decoded RGBA texels**精确相同，未配对页／shape／alpha／RGB差异均0。没有比较整个GLB字节、完整导出的所有raw tiles，也未做浏览器人工画质或匹配准确率评估。

GPU任务driver采样峰**476053504bytes＝0.476054GB（454MiB）**，Torch allocated38639616／reserved54525952bytes。1852组、nominal50ms／最大timestamp gap56ms、解析失败0；原resident单列并保留。但supervisor消费采样最大延迟11.099499s，停止判断受此延迟影响，不能称实时停止保障或瞬时硬上限。首个完整尝试在约15s因dequeue前新鲜度检查中断；修复队列处理后新目录重测，失败与监控诊断完整保留。

同轮DA3独立测试保留完整48帧／504／seed42／原精度，native expandable实际启用，在depth head conv2d申请360MiB时触及12GiB allocatorcap而OOM，worker退出1，未产生完整结果／未比较15数组。allocated峰12498550272／reserved12641632256bytes、driver采样峰13021216768bytes＝13.021GB；298组／最大gap55ms，无监测停止线触发。整卡当时仍约6.46GiB空闲，故失败来自本候选cap，不能说video6完整推理符合14GB，也不能据此排除所有其他保质量方案。

两个medium agents独立只读复核输入、监控、DA3失败及最终GLB结果。测试进程全部退出，GPU2只剩原6221／8550resident。生产默认／Docker未改，无全流水线新均值或P95；实验阶段未commit/push，后续本地提交见报告JOURNAL。[实测报告](../reports/gpu-latency-budget-20261018/README.md) · [video6汇总](../reports/gpu-latency-budget-20261018/video6-summary.json) · [GPU完整结果](../reports/gpu-latency-budget-20261018/video6-bake-result.json) · [DA3失败](../reports/gpu-latency-budget-20261018/video6-allocator-result.json)。
