# 统一 profiling：GPU0 / video3 / 29 帧

本文件是项目唯一 profiling 入口。正式基线为 **267.565 秒**，Viewer 在 **121.214 秒**可用，ZIP 在 **129.371 秒**就绪。原始运行和本轮独立组件诊断都位于 [同一记录目录](../runtime/video3-full-profile-20261016/)，不再引用旧 fd2–4、旧 Docker 或旧组件计时作为性能基线。

## 固定口径

- 输入：`fd-videos/3.json` 的 29 帧、711 个检测/分类；对应视频时刻 0、1.5…42.0 秒。旧独立 JPEG 已清理，本轮从视频 frame 0、45…1260 还原 JPEG quality=95，1080×1920。输入准备 2.345 秒单列，不计入主链路。
- 计算：DA3NESTED-GIANT-LARGE-1.1，本地 snapshot；自动内参，processed grid H=504/W=280；SAM3.1 PVS、batch=4、29 个 reference、pairing=next。不重跑 detector/classifier，分类由输入提供。
- 输出截止点：本地完整 Surfel generation、Viewer ZIP 和 scene GLB 写入返回。无 fsync，表示进程可见文件完成，不承诺物理介质同步完成；不含 COS、HTTP/BSON、浏览器下载/渲染、人工准确率评估和辅助 bbox 图。
- 设备/环境：GPU0 RTX 4090 D 49,140 MiB；OMP/OpenBLAS/MKL=4；Torch 2.7.1、NumPy 1.26.4、Open3D 0.19.0。新进程、新 DA3/PVS 输出缓存；本机模型文件和 OS 页缓存保留。GPU 与 CPU 为共享主机，未控制其他任务。
- 实现：`c1af83a` 加本次运行时的 Surfel 提前准备改动，实际差异见 [source-at-run.patch](../runtime/video3-full-profile-20261016/source-at-run.patch)，运行前后这三份核心源文件差异未变。没有把未部署代码称作在线服务实测。
- 样本数：正式全链路一次；细分诊断各一次。独立诊断用于归因，保留自身总量，**不按比例塞回正式基线，也不由它们拼出另一套总耗时**。

## 正式基线：可相加主表

| 组成 | 秒 | 说明 |
|---|---:|---|
| DA3 重建 | 24.561 | 含子进程启动、导入、加载、推理、NPZ、退出与父进程验证 |
| PVS + matching | 17.187 | 完整 29 refs；含首次 PVS 模型准备 |
| Dedup | 0.134 | 分类输入来自 3.json |
| 最终 Surfel 导出 | 74.376 | 方向、缩略图、标签、过滤、采样、point ranges、发布 |
| ZIP | 2.752 | 8 个二进制 DEFLATE level 1；JSON/JPEG STORED |
| GLB | 137.389 | LOD、烘焙、量化和最终写入 |
| 导入、启动退出与未单列间隙 | 11.164 | 含 core/delivery 包导入和外层采样等待 |
| 合计 | 267.565 | 唯一全链路基线 |

提前准备 2.999 秒，与 matching 重叠 2.999 秒，不另加。以下时间线保留原始起止时点；当前实现并未在 matching 前发布完整 Viewer。

![正式运行时间线](../runtime/video3-full-profile-20261016/timeline.svg)

## 1. DA3：冷进程主要花在建网络

正式 reconstruction 为 24.561 秒；其中原运行测得 CPU 构造/加载 12.905 秒、inference API 2.453 秒、反投影 0.103 秒、NPZ 写入 0.218 秒。

下表是补做的独立 DA3 组件诊断（进程内 24.565 秒），边界包含嵌套，未把子项再加到父项：

| 组成 | 秒 | 说明 |
|---|---:|---|
| Python/Torch + DA3 依赖导入 | 4.802 | 诊断脚本窗口 |
| JPEG decode/RGB | 0.559 | 29 帧 |
| 网络构造 | 13.102 | 含默认参数初始化 |
| safetensors 打开/映射 | 0.131 | 不等于完整物理磁盘读取 |
| checkpoint 赋值 | 0.574 | 可包含 mmap 触页 |
| 模型搬 GPU | 1.983 | 原生 model.to，未新增同步 |
| 输入预处理 | 0.223 | resize/normalize/stack |
| 输入 H2D enqueue | 0.013 | 异步提交和 CPU 工作，不是 CUDA copy events |
| 网络 forward 原生同步边界 | 2.240 | 含现有同步/队列排空边界 |
| 输出 D2H/Prediction | 0.012 | CPU 数组转换 |
| processed RGB 整理 | 0.173 | denormalize/clip/uint8 |
| 深度反投影 | 0.092 | world_points |
| NPZ 数组准备/open/close | 0.045 | 参数求值与外层保存差额 |
| NPZ 序列化/写入 | 0.230 | np.savez；无 fsync |

网络构造 13.102 秒与权重打开/赋值约 0.705 秒应分开解释。不能把整个模型准备叫作磁盘读权重，也不能把完整 reconstruction 叫 GPU forward。常驻/延迟初始化的收益仍需单独实现和验证。

## 2. Matching / PVS：正式运行内部组成

| 组成 | 秒 | 说明 |
|---|---:|---|
| sam3_mask 总区间 | 12.327 | 含下面模型加载、predict、cache 边界；本行不与它们相加 |
| ↳ PVS 模型准备 | 7.949 | 首次进程加载 |
| ↳ 29 帧 predict | 4.057 | 711 框；encoder 与 decoder/postprocess |
| ↳ 其余 mask/cache 包装 | 0.321 | 差额：cache 校验/发布与调用包装；未逐项拆开 |
| 参考物体点采样 | 1.019 | 711 次 |
| 目标框命中与选择 | 1.603 | 697 次 |
| world→camera | 0.118 | 697 次 |
| 3D→2D 投影 | 0.227 | 697 次 |
| 投影后处理 | 0.191 | 697 次 |
| 数据加载 | 0.197 | 29 次 |
| 场景数据构造 | 0.185 | 首次 cache 数据转 Tensor |
| transforms 构造 | 0.004 | 29 次 |
| 结果后处理 | 0.014 | 29 次 |

`batch_all_refs_total`、`per_ref_total`、`process_images` 属于嵌套父区间，不能与上表再次求和。未列出的 parser、几何判断、文件写入和调度保留在 matching 总量中，不能全部归为 GPU forward。

补充 PVS 组件诊断：模型准备 8.772 秒；native 模型构造约 7.026 秒，checkpoint mmap 0.184 秒、严格赋值 0.247 秒、eval/to GPU 0.471 秒。

29 帧/711 框 predict 共 4.826 秒：图像 transform/image H2D 0.664 秒、encoder 2.169 秒、decoder/掩码后处理/最终 stack 1.968 秒。原生 `inference_and_postprocess_s` 包含 encoder 和 stack；187 个 decoder batch 的 CUDA events 合计 1.620 秒，也是其中子区间，不能相加或用 CPU wall 相减推导“纯 CPU 后处理”。

诊断复用了 baseline processed bbox 和图像预处理，未重放整条 matching 的 prefetch、cache、重排与 RNG 边界；711 masks 的数量完整，但这不是新增的 matching 准确率或 mask 逐像素等价结论。完整模型准备表和同步说明见 [GPU 细分记录](../runtime/video3-full-profile-20261016/diagnostics/gpu/REPORT.md)。

## 3. Dedup：当前占比很小

正式运行 0.134 秒。独立 CPU 诊断 0.155 秒中：解析 matches 0.004 秒，构造 global mapping 0.025 秒，生成带 global ID 的检测数据 0.036 秒，原子发布 global mapping/SKU pair 0.031 秒，其余为输入解析与逐帧输出。输出仍为 385 个 global IDs、711 observations；数量完整不等于准确率。

## 4. Surfel 提前准备与最终导出

提前准备的正式组成：

| 组成 | 秒 | 说明 |
|---|---:|---|
| 读取并校验 DA3 cache | 0.189 | 准备线程 |
| 生成原始 U/V/depth | 1.845 | 完整 native grid |
| JPEG/相机纹理准备 | 0.964 | 不依赖最终 IDs |
| 其余准备 | 0.001 | 输入关联和返回值整理 |

以上三项在一个 CPU worker 内进行，与 matching 并行；本次全部落在 matching 区间内。它们已实现并有重叠证据，不能再当成“待实现提速”。NN/SOR 及最终标签保护不属于当前提前准备。

最终导出的正式组成：

| 组成 | 秒 | 说明 |
|---|---:|---|
| 缩略图 | 1.570 | 711 个 observations |
| 场景方向 | 3.749 | 首次拟合和 sidecar 发布 |
| 采样总区间 | 65.013 | 含标签、几何过滤、voxel、自适应选择和排序 |
| ↳ mask→实例标签 | 0.424 | 读取 matching 的 processed masks |
| ↳ 几何过滤 | 60.096 | 内部诊断见下表 |
| ↳ 背景 voxel | 0.662 | NumPy 分组，最高 confidence/首个并列输入 |
| ↳ 采样其余工作 | 3.831 | 自适应 grid、预算、选择、数组整理与稳定排序 |
| 最终 Surfel 数组准备 | 0.186 | 复用提前准备的 U/V/depth/JPEG |
| generation 发布 | 0.114 | 二进制/JSON/JPEG 和 CURRENT |
| 其他导出工作 | 3.744 | 含 point ranges、world_to_view、对象/manifest 整理 |

独立 Viewer 诊断再次生成 2,402,589 点、711 张缩略图。它的最终导出总量为 83.213 秒，包含独立方向拟合和共享主机波动；不能直接替换正式 74.376 秒。

### 几何过滤：DBSCAN 最大，SOR 不是整段耗时

| 组成 | 秒 | 说明 |
|---|---:|---|
| 初始 NN 原生调用 | 3.811 | 4,092,480 点的最近邻距离；不是 k=20 SOR |
| SOR 整段 | 7.418 | 含 native 调用、商品保护 OR、门槛和 select |
| ↳ SOR 原生调用 | 6.420 | k=20、std_ratio=2 |
| DBSCAN 整段 | 43.396 | 聚类、主簇判断、商品保护、select |
| ↳ DBSCAN 原生调用 | 42.713 | eps=median_NN×5，min_points=10 |
| RANSAC 整段 | 11.708 | 拟合、双侧质量检查、商品保护、select |
| ↳ RANSAC 原生调用 | 11.045 | num_iterations=1000，保留现有随机行为 |
| 有效域、pcd 构造与其他 | 0.512 | 互斥差额 |
| 诊断过滤合计 | 66.845 | 独立组件测量；不替换基线 60.096 秒 |

DBSCAN 原生调用约占这次诊断过滤总量的 63.9%。SOR 和初始 NN 两个原生调用合计 10.231 秒，明显小于 DBSCAN。`k=20` 仅用于 SOR，不能用改 k 推断初始 NN 或 DBSCAN 会等比例变快。

所有保护 OR、保留比例门槛、簇判断及 ground 判定仍照原实现执行。三个 `pcd.select_by_index` 合计 1.928 秒已计在各整段内。方向拟合中的 RANSAC 另为 2.329 秒，与过滤里的 ground RANSAC 不同。

独立诊断还定位到：自适应 grid 特征 0.448 秒、两次自适应选择合计 2.726 秒，point ranges 构造 3.708 秒。它们已分别包含于采样或最终导出，不另计入基线。

## 5. ZIP：时间集中在 8 个二进制的压缩

正式 2.752 秒。独立 ZIP 诊断 3.128 秒：8 个二进制成员 2.992 秒，740 张 JPEG 0.113 秒，3 个 JSON 0.001 秒，余量为成员表、目录与 ZIP 收尾。二进制为 DEFLATE level 1；JPEG/JSON 为 STORED。此测量含读文件/压缩/ZIP 写入，没有把纯 zlib 与 I/O 分离。

## 6. GLB：RGB 延展、调色板量化和 PNG 编码主导

正式 GLB 137.389 秒，生产报告分为 bake/LOD 94.215 秒、compact/encode/write 43.164 秒，两项互斥。

下表使用原始 baseline Viewer generation 做一次独立 CPU 诊断，GLB 151.089 秒。保留 31 页 atlas、2,564,994 个三角面和 121,513,344 bytes；没有做额外 hash 或全量语义复查。

| 组成 | 秒 | 说明 |
|---|---:|---|
| 输入读取与校验 | 0.198 | 原始 baseline generation |
| 保守 2×2 LOD | 5.473 | 2,402,589 → 1,297,863 个候选 |
| ↳ LOD 覆盖检查 | 1.713 | 72 批；其余为归组、排序、数组整理 |
| Surfel tile 烘焙 | 95.137 | 1,280 批；下列四项互斥地构成此区间 |
| ↳ footprint + 最终直径约束 | 0.836 | 保留原几何裁切规则 |
| ↳ 双线性照片采样 | 16.374 | 每 Surfel 的 8×8 tile |
| ↳ 透明边界 RGB 延展 | 56.040 | 只填 RGB，保持 alpha；避免线性采样黑边 |
| ↳ 其他烘焙工作 | 21.887 | 投影、深度连续性、可见域、tile/corners 数组 |
| atlas 页面组装 | 0.645 | 31 页 |
| 几何量化与写 BIN | 1.853 | 16-bit geometry/UV/index |
| atlas 编码 | 46.248 | 下列 quantize/save/alpha 是其子集 |
| ↳ 二值 alpha 检查 | 1.083 | 保留透明/不透明两组颜色 |
| ↳ MEDIANCUT 调色板量化 | 21.652 | 128 色 RGB |
| ↳ PNG 编码 | 22.399 | optimize=True |
| 图片字节写 BIN | 0.041 | 31 张 PNG |
| 最终 GLB 拼装/写文件 | 0.096 | 约 115.88 MiB |
| 其他读取/转换与调度包装 | 1.397 | 含来源 JPEG 转换和未单列间隙 |
| 诊断 GLB 合计 | 151.089 | 独立诊断，不替换基线 137.389 秒 |

RGB 延展 56.040 秒、调色板量化 21.652 秒、PNG 编码 22.399 秒合计占独立 GLB 诊断的 66.2%。单纯优化最终文件写入或几何量化，无法解释大部分 GLB 时间。

`_dilate_rgb` 在每个 tile 内向透明边界传播 RGB，alpha 不变；直接删掉可能造成边缘黑线。调色板/PNG 参数也涉及体积和画质，以上是定位证据，不是已经验证的新优化。

## 显存、RAM、产物与验证

- 正式任务 driver 显存采样峰值 **14.777 GiB**；GPU0 原有进程约 7.145 GiB，已排除。采样间隔 200 ms，可能漏掉短峰值。
- DA3 allocator 生命周期 allocated/reserved：11.410/14.363 GiB；PVS 所在 Core 进程：3.112/3.285 GiB。两种指标及不同进程/时段的峰值不能相加。
- 正式 Core/Delivery 阶段 uv 进程树 RSS 采样峰值 **14.166 GiB**，共享页可能重复计入；不包含 launcher 和 nvidia-smi。
- 29 refs、29 份 mask manifests、385 global IDs、711 observations、2,402,589 Viewer 点；ZIP 75.94 MiB，GLB 115.88 MiB。基线 core/delivery、GPU 两个诊断和 CPU 两个诊断均成功；没有运行服务部署或 COS 上传。
- 临时细分脚本不改生产源码，不保存 tensor 主体；CPU 诊断仅生成独立 Viewer/cache sidecar 和报告，额外 ZIP/GLB 已删除。原始基线产物与记录保留。

## 旧记录退役与证据入口

按 Rick 授权删除 223 个过时 profiling/对照文件，其中包含旧 fd2–4、旧 Docker、旧 DA3 三份专用报告，以及旧 per-dataset timing 和组件计时输入。混合研究保留 F1、mask/数组等价、几何、视觉及发布证据，仅去掉性能部分；模型、输入数据、当前 baseline 和其他 checkout 源码未删除。

- [精确删除清单](../runtime/video3-full-profile-20261016/retirement-result.json)
- [正式墙钟和进程命令](../runtime/video3-full-profile-20261016/receipt.json)
- [正式机器可读汇总](../runtime/video3-full-profile-20261016/summary.json)
- [全部表格 JSON](../runtime/video3-full-profile-20261016/detailed-tables.json)
- [CPU Viewer 细分](../runtime/video3-full-profile-20261016/diagnostics/cpu/viewer/result.json)
- [CPU ZIP/GLB 细分](../runtime/video3-full-profile-20261016/diagnostics/cpu/delivery/result.json)
- [GPU 细分报告](../runtime/video3-full-profile-20261016/diagnostics/gpu/REPORT.md)

后续优化优先围绕 DBSCAN、tile RGB 延展、PNG 量化/编码及冷进程网络构造设计单变量验证；不从本轮诊断直接承诺提速比例或省略质量验收。

## 2026-10-16 CPU 优化实验收尾：无候选通过验收

本轮仅在 runtime 实现 RGB 切片延展与 bounded 两 worker atlas 编码候选，未集成生产；独立评估对 dilation/encoding 为 inconclusive，DBSCAN-off 为 debuggable（smoke 失败，未进入正式比较）。既有 Surfel 提前准备不是本轮新增。唯一全链路基线仍为 **267.565 s**，Viewer **121.214 s**，GLB **137.389 s**。

Dilation 原始 B1/C1/C2 为 169.464647/133.610064/126.081133 s，原 B2 被 CPU 限额中断；独立 B2 continuation 为 156.874763 s。直接算术差为 35.854583 与 30.793630 s，仅供诊断：同核干扰、timer 内不同验证开销、中断后的新进程、累计 1,200 CPU s bound 冲突和缺失 B2 baked reference 阻断正式接受。C1/C2 完整 baked RGBA/geometry 与 final typed arrays、associations、31 张 decoded RGBA atlas 比较未发现差异，已完成文件均为 121,513,344 bytes；continuation 对历史 GLB 的 final 比较通过，不能补证缺失的量化前比较。

Encoding 只有首张真实页两副本的 smoke/pilot，正式运行因同核重叠取消；DBSCAN runner 调用可写 orientation API 导致 sidecar 重算，内容已恢复但原 filesystem metadata 未保留。这些协议失败不证明算法回归，也不证明 GLB 或 Viewer 已正式提速。controller failed-stage transition 仍 blocked，未 promotion 或修复状态。完整证据、资源实测/保守预留的区分、限制与未来最小验证建议见 [Git 内精简报告与原始数值](../reports/cpu-latency-20261016/README.md)（详细本机记录为 `runtime/latency-optimization-20261016/FINAL_REPORT.md`）。
