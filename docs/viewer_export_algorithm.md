# Viewer / GLB 当前算法与采样入口

核对：2026-10-18，本地main `bf9bde8`。完整实测已更新为[GPU2 profiling](profiling_breakdown.md)，本文不再复述旧GPU0/旧generation的耗时。完整实现细节见[GPU/matching](../reports/profiling-20261018/algorithms_gpu_matching.md)、[几何与采样](../reports/profiling-20261018/algorithms_geometry.md)、[交付](../reports/profiling-20261018/algorithms_delivery.md)三份源码说明。

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
    |     初始NN -> median；raw SOR -> bool mask
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
    DBSCAN -> 保簇 + 商品保护 + guard
    RANSAC ground -> 双侧/比例判定 + 商品保护 + guard
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

DBSCAN保留≥当前输入1%的簇，无达标簇时有簇则取最大簇。它与footprint中的“相对最大分量20%”是不同规则。ground的dist=clip(3×medianNN,.02,.15)，内点至少8%，非内点两侧质量检查后才删除；商品仍合回。

`PointCloudFilterConfig(ground_proposal_max_points=50_000)`是显式代理候选实验：均匀抽样生成平面后，在完整当前点域按原threshold严格`<dist`重计内点，再沿用8%、双侧、商品保护和20%guard。默认`0`保留原全量RANSAC及其返回inliers；没有自动重跑或切换。video3组件虽然10.879→0.980秒，但候选近乎垂直于摆正地面，并额外删除51286背景点，因此不默认推广。计数与局限见[ground报告](../reports/latency-optimization-20261018/ground_report.md)。

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

DBSCAN先为全部当前点查半径邻居并保留列表，再串行按输入顺序扩展簇。frontier使用unordered_set；曾是noise的点仍可成为边界点。邻居总量取决于所有半径邻居数之和，不仅是点数；本次未采完整邻居列表或拆开内部建树/查询/扩展计时。[DBSCAN源码](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloudCluster.cpp#L18)

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

新多视频完整运行支持声明recipe下本机各场景的耗时、内部breakdown、点数和资源峰；每video一次，不代表重复/热服务分布，也不验证原检测对齐。历史PVS初始化重叠复现支持调查GIL机制，不能直接承诺节省全部差额。近似过滤、关闭DBSCAN、背景提前采样及NN/SOR进程隔离均没有本轮画质/全链路收益验收。

10月17日原算法/历史时间说明已由本文替换，原文保留为[本机历史快照](../runtime/profiling-20261018/previous-viewer_export_algorithm.md)，该runtime文件不随Git发布。当前时间统一引用主profiling，避免旧数字继续作为最新事实。
