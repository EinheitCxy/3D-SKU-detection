# Viewer / GLB 当前算法与采样入口

场景入口现通过显式 backend 接受 DA3 和已适配的 PromptDA 逐帧米制网格；提前准备、过滤、采样及 GLB bake 共用下述循环，输出保留真实 backend。PromptDA 的原始字段、缓存转换、无 confidence 的有效性语义和 PVS 隔离见 [接入说明](promptda_integration.md)。这里的历史 DA3 计时不作为 PromptDA 实测。

核对：2026-10-19。当前默认使用 proxy DBSCAN scale2；背景平面只从未保护点拟合，已取消全局 8% 支持率门槛。具体循环见第4节。此前完整流水线实测见[GPU2 profiling](profiling_breakdown.md)；历史测量使用当时的过滤策略，不能将其点数或耗时直接当作新策略结果。GPU/matching 与交付的详细实现仍见[GPU/matching](../reports/profiling-20261018/algorithms_gpu_matching.md)、[交付](../reports/profiling-20261018/algorithms_delivery.md)。

## 1. 当前到底用了多少点

最新不同视频的dense/valid/SOR/DBSCAN/ground/voxel/预算/Viewer/LOD/bake逐阶段数量，统一见[多视频profiling第6节](profiling_breakdown.md)。每视频同一次完整运行都记录了中间计数，不用另一轮诊断补写。

| 点数域 | 当前算法关系 |
|---|---|
| DA3 native grid | F×504×280，F取该video全部输入帧 |
| export-valid / 几何过滤后 | 有效域及保护/guard条件决定，不是固定比例 |
| 商品选入 | 不超过2M；小于预算时全保留，大于预算时按gid分配 |
| 背景选入 | 5mm voxel后按adaptive预算，不超过500K |
| Viewer实例 | 商品选入＋背景选入 |
| GLB LOD后 / baked | 保守2×2合并后，再去掉无有效tile候选 |
| 三角形 | 每个baked Surfel两个三角形 |

商品保护在几何过滤阶段保点，不能豁免最终商品预算。总点数相同不证明来源、方向或画面相同；未新增浏览器加载/画质结论。

## 2. 完整算法主循环与依赖

```text
DA3一次处理全部F帧 -> schema-v3 metric NPZ
    |
    +-- 单个CPU准备线程：
    |     load cache
    |     full-grid depth、U/V
    |     export-valid indices
    |     float64 full-domain point snapshot + Open3D cloud
    |     共享KNN查询 -> median NN与raw SOR bool mask
    |     source JPEG + 相机/affine metadata
    |     返回PreparedViewerGeometry，不读商品mask、不发布
    |
    +-- 主线程：
          for ref in 0..F-1:              # 全局RNG锁保护，默认串行
              target=(ref+1)%F
              PVS：整图encoder一次；box按4分批decoder
              ref商品mask采样最多70点
              world->target camera->pixel
              Top-5 target bbox候选，几何/SVD/命中率评分
              已验证候选唯一性分配，写matching summary
          分类join（本次分类由输入提供）
          union-find dedup -> global_mapping
    |
    准备future就绪 + 完整PVS cache + global IDs
    |
    全部observation缩略图；方向拟合/sidecar
    masks -> observation labels；同gid region
    full-grid断裂/边界特征、权重、caps
    复用raw SOR，合回商品保护和guard
    proxy DBSCAN -> 标签映射回原始点 -> 保簇 + 商品保护 + guard
    背景RANSAC -> 双侧安全检查 + 商品保护 + guard，不设8%支持率门槛
    背景5mm voxel -> 最多500K adaptive选择
    商品按gid公平预算 -> 最多2M adaptive选择
    同步整理source indices、labels、positions、U/V scale
    按label稳定排序
    序列化Surfel U/V/frame/scale、完整depth及照片
    observation point_ranges
    临时generation写完 -> rename -> atomic CURRENT
    |
    本次交付调用方：ZIP -> compact streamed GLB
```

几何准备与matching有重叠，但交集不等于净节省。额外诊断复现了NN/SOR与PVS构建的GIL限制，边界详见[当前profiling第11节](profiling_breakdown.md)。增加后台线程数不能绕过持有GIL的native调用。

`main.py`负责Viewer generation；ZIP/GLB由后续交付调用方执行，本次profiling明确包含二者，不表示每个CLI默认都生成二者。

## 3. 对齐的数据结构

记F为该video输入帧数，H=504、W=280、N=FHW，M为有效域，K为最终选择数。

| 数组 | shape / dtype | 作用 |
|---|---|---|
| XYZ / confidence / processed RGB | `[F,H,W,3] f32` / `[F,H,W] f32` / `[F,H,W,3] u8` | 原始grid与置信度/颜色 |
| E / K / affine | `[F,3,4]` / `[F,3,3]` / `[F,2,3]` | world→camera、相机投影、source→processed |
| U/V/depth | `[F,H,W,3] f32` / `[F,H,W] f32` | 真实相邻像素切向量和相机Z |
| valid / valid_indices / source_indices | `[N] bool` / `[M]` / `[K]`整数索引 | 有效mask和筛选索引保持对齐，仍回到完整grid |
| labels / region | `[M] int32` / `[F,H,W]` | observation label与跨帧同gid的语义region；背景-1 |
| adaptive weights / caps | `[N] u8` / `[N,2] f32` | full-grid边界重要性与U/V方向最大scale |
| 输出positions/colors/normals | `[K,3] f32/u8/i8` | normals是固定占位[0,0,1]，不是额外native估计 |
| Surfel U/V/scale/frame | `[K,3] f16` / `[K,2] f16` / `[K] u8` | 最终实例面片与来源帧 |
| 输出完整depth | `[F,H,W] f16` | 遮挡/断裂支持，不随选点变成K长度 |

先计算相邻像素U/V，再选择展示点；不能把抽样空洞当作真实深度断裂。source index始终是连接所有数组的键，不能分别随机采样positions和U/V/labels。

## 4. 过滤如何保护商品

默认过滤不是生成Surfel的必要数学条件，而是当前噪声/背景清理策略。原始NN/SOR不看商品；最终每阶段删除前执行：

```text
stage_keep = raw_keep OR protected[current_indices]
if count(stage_keep) < 0.2 * current_point_count:
    整阶段跳过
else:
    合并到全域keep，更新current_indices与pcd
```

保护仍受finite/nonzero base validity限制。20%相对每阶段当前域；不是最终至少保留原始20%的保证。初始median NN全程复用，不在每阶段重新估计。

### 4.1 默认代理 DBSCAN

`cluster_proxy_voxel_scale=2.0`，体素边长为 `2×medianNN`，邻域半径仍为 `5×medianNN`。每个体素选择最先出现的原始点作为代理，按原始出现顺序排列；在代理上运行 Open3D DBSCAN，再将簇标签映射回每个原始成员。聚类本身不会把导出几何替换成体素中心。

核心点门槛10统计代理邻居，属于近似算法。保簇时统计映射后的原始成员数，保留≥当前输入1%的簇，无达标簇且存在簇时取最大簇；商品保护和20% guard随后照常执行。它与footprint中的“相对最大分量20%”是不同规则。显式`cluster_proxy_voxel_scale=0`仍可运行原始全点DBSCAN；没有自动切换。

### 4.2 背景平面清理：删除什么

`remove_ground`沿用既有参数名，实际操作是清理非商品的平面背景。候选可能是地板、墙面或其他大平面；该步骤不使用语义地面分类，也不以摆正方向约束法向。因此不能把所有被删点称为真实地面。未被SAM标注的商品也可能落在背景域。

记当前经过SOR/聚类/保护的点集为`P`，保护mask为`protected`：

```text
background_indices = where(~protected)
B = P[background_indices]
若len(B) < max(3, min_points)，跳过拟合
dist = clip(3 × medianNN, 0.02, 0.15) 米
proposal = B                          # 默认ground_proposal_max_points=0
plane, local_inliers = RANSAC(proposal, 三点假设, 最多1000轮)
检查平面系数有限且法向非退化
将background局部内点索引映射回P
检查当前P中平面两侧的非内点分布
raw_keep = ~background_inliers
应用商品保护与20%最少保留guard
将工作集索引映射回全域keep
```

商品点不参与平面拟合，始终不会被该步骤列为删除候选。默认完整背景域沿用Open3D返回的pre-refinement内点索引；归一化后的refined plane用于双侧检查和诊断。`support_ratio=内点数/背景点数`仅记录诊断，不参与接受/拒绝；`ground_min_inlier_ratio`参数已移除，没有换成另一个支持率门槛。

保留的双侧规则是：去掉背景候选内点后，在当前完整P（包括受保护商品）上统计signed distance；若正、负两侧超过dist的点数均超过其余点数的15%，拒绝这个可能穿过主体的平面。随后保留原商品保护及每阶段20% guard。这两项是既有安全规则，本次只取消8%支持率依赖。

显式`ground_proposal_max_points=50_000`只从B均匀无放回抽样提案，使用本地seed42且保留原顺序；返回的refined plane会在完整B上按严格`abs(distance)<dist`重新求内点，不使用样本支持率，也不设完整域支持率门槛。该路径与完整背景域的native内点规则不同，仍不是等价加速，默认不启用。旧全P采样实验及其失败结论保留为[历史记录](../reports/latency-optimization-20261018/ground_report.md)，不能用来推算新背景拟合效果。

最终删除通过同一keep/source_indices选择同步作用于位置、颜色、法线、标签及Surfel来源；它发生在Viewer预算采样之前，不改原始DA3缓存。显示摆正和footprint支撑平面估计继续使用独立的共享方向/面积路径。修改与缓存实景验证见[本轮记录](../reports/viewer-background-cleanup-20261019/README.md)。

## 5. 预算采样为什么仍要支持完整grid

背景先5mm voxel：每格优先最高confidence，同分保留先出现者；再按16×16原图tiles的edge/halo/interior权重分配最多500K。商品超2M时按gid公平分配余量，不超预算也计算adaptive scale。

adaptive选择先算每tile capacity/mass，再做饱和配额分配；tile内Morton排序，累计概率中点确定取样。scale与保留概率、candidate/support质量比有关，clip到.5..8，再受full-grid caps限制。support和可选candidate不是同一概念。

最终点仍按observation label稳定排序。`_attach_point_index_ranges`生成与labels相同dtype的全部query，分别一次批量`searchsorted(left/right)`，避免Python整数逐个查找时重复转换整个int32点标签数组。缺失label得到相同插入位置的空半开区间，写JSON前转Python整数；不重新排序或改变点数组。真实2402589点/711观察的端点与原实现完全一致，组件3.489543→0.000964秒；不把这一组件比例当整流水线比例。

标签、真实边界和完整depth决定scale及遮挡；这不要求全部原始点最后都渲染。显式`early_background_candidates`会先保留全部商品和有限背景候选，但会改变NN/SOR/DBSCAN/ground输入，不能当等价加速。

## 6. 源码与计时的层次

详细伪代码/行号在三份实现文档；本轮[多视频可重现脚本](../reports/profiling-multivideo-20261018/README.md)在同一完整运行里使用nested inclusive/self和独立CUDA events。每函数总时间、连续block与内部native/PIL调用是父子关系，不能一起求和；独立组件旧诊断也不能填进新主跑总量。

### 6.1 Open3D 0.19.0 原生循环

以下是固定版本源码的算法解释，**不是**内部各子循环的实测时间。当前runner只测native API整次调用。

NN与SOR分别建KD-tree；NN对每点KNN2，取第二个距离平方根；SOR对每点KNN20，平均距离再用全域均值和样本标准差作严格阈值，k包含自身。SOR原生还构造一个过滤cloud，项目丢弃它再于保护/guard之后select。[NN/SOR源码](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloud.cpp#L570)

```text
NN:  建树 -> 并行for每点KNN2 -> sqrt(second distance) -> NumPy median
SOR: 再建树 -> 并行for每点KNN20 -> mean(sqrt(distances))
     -> 全域mu/sigma -> 0<avg<mu+2*sigma -> indices和selected cloud
```

Open3D DBSCAN先为其输入点查半径邻居并保留列表，再串行按输入顺序扩展簇。当前默认传入代理点；显式scale0传入全部当前点。frontier使用unordered_set；曾是noise的点仍可成为边界点。邻居总量取决于所有半径邻居数之和，不仅是点数。[DBSCAN源码](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloudCluster.cpp#L18)

```text
建树；并行for i: neighbors[i]=SearchRadius(P[i],eps)
labels=-2
串行for未访问i:
    邻居不足10 -> noise
    否则建立cluster、frontier和visited
    while frontier非空:
        取一点；将noise/未访问点归簇
        若核心点，加入未访问邻居
```

RANSAC预生成最多1000组三点假设；并行测试有效平面，每个假设扫描当前点集；fitness/RMSE与成功概率更新提前停止上限。最后用最佳平面重新查内点并重拟合平面；不是一定执行1000次，也不建KD-tree。[RANSAC源码](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloudSegmentation.cpp#L133)

原生可能有OpenMP并行，但其[Python绑定](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/pybind/geometry/pointcloud.cpp#L126)没有自动释放GIL；这与Python准备线程能否同PVS构建有效并行不同。

## 7. GLB算法与长边限制

GLB只对满足四点/法向/深度/coverage条件的商品2×2 block合并，背景保留；批次8192组。LOD后按frame、1024候选bake，8×8内部texel加一圈透明gutter，投影/深度遮挡/圆盘mask后做双线性颜色采样。

U/V初始footprint限制与乘radius/scale后的最终限制都保留，最长直径≤6×native最大间距。RGB延展固定方向、固定旧轮来源，只改RGB不改alpha；每满41,616个tile组一页。默认128色MEDIANCUT与PNG optimize=True，几何/UV/index量化u16，顺序写BIN后拼GLB。

已有2/4编码worker是有界FIFO：worker只编码，主线程保留页序与BIN写序。前述六视频profiling使用当时默认1；后续video3完整对照已将`export_scene_glb`默认改为2，结果见[主文档§12](profiling_breakdown.md#12-后续已实现的优化固定video3前后完整对照)。没有用降低纹理/取消清理/放宽长边换取这些数字。[完整LOD、bake、编码循环](../reports/profiling-20261018/algorithms_delivery.md)

## 8. 当前证据能支持什么

新多视频完整运行支持声明recipe下本机各场景的耗时、内部breakdown、点数和资源峰；每video一次，不代表重复/热服务分布，也不验证原检测对齐。历史PVS初始化重叠复现支持调查GIL机制，不能直接承诺节省全部差额。代理DBSCAN scale2已在后续video1/3/6对照中由Rick进行视觉验收并设为默认，属于近似过滤，不能称逐点等价。关闭DBSCAN、背景提前采样及NN/SOR进程隔离不在本次默认变更范围；新的背景平面策略也不复用历史全链路耗时结论。

10月17日原算法/历史时间说明已由本文替换，原文保留为[本机历史快照](../runtime/profiling-20261018/previous-viewer_export_algorithm.md)，该runtime文件不随Git发布。当前时间统一引用主profiling，避免旧数字继续作为最新事实。
