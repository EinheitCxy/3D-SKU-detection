# Viewer 几何、过滤与采样算法：2026-10-18 源码映射

本文核对当前 checkout 的源码，说明独立 CPU 诊断的边界与算法。不包含新的性能结果；正式主跑、独立组件诊断和旧 video3 测量应各自保留口径。`diagnose_viewer.py` 的结果只有运行成功后才可用于该次诊断。

## 1. 调度和数据依赖

入口 `main.py:1086–1099` 在 DA3 新缓存或复用缓存就绪后提交一个 `surfel-prepare` CPU worker；主线程 `1106–1125` 运行 matching、classification join、dedup；`1130–1139` 的最终 Viewer 导出取 preparation future。matching 发布 canonical PVS masks，dedup 发布 global mapping；标签只能在两者就绪后生成。

`src/web_viewer_export.py:97–136` 的 preparation 顺序为 NPZ load → full-grid U/V/depth → valid indices → 全有效域 NN/SOR raw → source JPEG/相机参数 → 可选 frame sharpness。它不读商品 masks、不拟合方向、不运行 DBSCAN/ground，也不发布 generation。`main.py:1160–1164` 在异常时也等待 worker 收尾。

`PreparedViewerGeometry`（`web_viewer_export.py:80–94`）保存缓存、U/V/depth、JPEG、valid_indices、原始过滤准备结果。finalize `158–169` 核对路径、文件 stat identity、texture edge、tuning/config；`180` 复用 cache，`570–584` 复用几何并确认 valid domain/order，`641–644` 复用 NN/SOR raw，`surfel_export.py:90–92` 复用 JPEG bytes。此调度改变计算时机；NN/SOR 结果仍在取得 labels 后应用商品保护与 guard。

`ViewerExportTuning`（`src/viewer_export_tuning.py:9–24`）默认 quality_filter=False、product_spill_trim=False、background_edge_weights=(4,2,1)、single_owner=False、owner_patch_cells=4、early_background_candidates=0。`main.py:1094–1099` 没有覆盖这些默认。显式启用 early background 候选时，`web_viewer_export.py:116–119` 强制推迟 NN/SOR，因为过滤输入域依赖 labels。

## 2. shape、有效域、一像素切向量和纹理

`_load_da3_cache`（`web_viewer_export.py:359–454`）检查 schema=3；world_points=float32[F,H,W,3]、confidence=float32[F,H,W]、images=uint8[F,H,W,3]。返回 E=float64[F,3,4]、affine=float64[F,2,3]，image_ids=int32[F]，source sizes=[F,2]。图像帧槽位 f 与 image_id 不同。

`grid_tangents`（`src/surfel_export.py:11–39`）：

1. einsum 计算相机深度 `(R@P+t).z`；valid要求 finite/nonzero XYZ、finite positive depth。
2. axis=(2,1)，两次全 grid 向量化：roll得到前向/后向差分，去除图像边界环绕和无效邻点。
3. 世界坐标差分长度必须 `<max(.03*depth,.001)`；两侧均合法时选较短者；均不合法时切向量为0。
4. 输出 U/V=float32[F,H,W,3]、depth=float32[F,H,W]。零切向量表示无面片面积，不能随意用邻域补齐。

`_viewer_valid_mask`（`web_viewer_export.py:529–539`）要求 finite/nonzero XYZ、finite confidence；Surfel 时再要求 depth>0。记录 full grid N=FHW、export-valid M，不能假定 N=M。

`prepare_surfel_textures`（`surfel_export.py:48–85`）另从 NPZ 读取 intrinsic[F,3,3]；帧数必须1..256。逐帧打开源图转RGB，核对 source dimensions，按朝向缩至 longest_edge=1920、short_edge≤1080，LANCZOS thumbnail，JPEG quality=95、subsampling=0。保存 K/E/texture size/source size/inverse affine；这些工作与选点无关。

## 3. 几何过滤的原生调用与保护

默认配置 `utils/pointcloud_filter.py:18–40`：enabled=True，SOR k=20/std_ratio=2；DBSCAN enabled，eps=5×initial median NN，min_points=10；保留≥本阶段输入1%的簇；ground enabled，dist=clip(3×median NN,.02,.15)，inlier_ratio≥.08；各阶段保留 guard=.2；几何过滤最小输入1000；proxy voxel scale=0。

`prepare_scene_filter:71–100`：

- 输入复制为 float64[M,3]，base mask=finite/nonzero XYZ；保留不可写 points/base/rawSOR arrays和原始 Open3D cloud。
- enabled 且 base≥1000才运行几何；`_median_nn_distance:49` 调 `compute_nearest_neighbor_distance()`，`52` 在 Python NumPy median。
- median NN>0且base>20，`92–94` 调 `remove_statistical_outlier(20,2)`；返回 cloud丢弃，indices转base域 rawSOR mask。
- 此处没有商品保护、20% guard、DBSCAN或地面删除。

`filter_scene_points:130–234`：

1. prepared缺失时同步准备；`158–161` 要求 prepared config、float64输入值及顺序一致。
2. protected=`protect_mask & base`；disabled、base<1000或median NN≤0时返回base。
3. `apply()`（`174–188`）先 `stage_keep=raw_keep|protected[current_indices]`；若保留数 `<.2*current_count`，跳过整个阶段；否则更新总keep、current_indices，并 `pcd.select_by_index`。
4. SOR apply：`191–192` 将提前计算的rawSOR经过上面的保护和guard。
5. DBSCAN block：`195–204`；默认native在`_cluster_labels:107`，输入是SOR apply后的cloud。对非noise labels计数，保留≥当前1%的簇；有簇但均未达门槛时取最大簇；全noise也执行apply保护/guard。
6. ground block：`209–226`，当前≥1000才调用`segment_plane(dist, ransac_n=3, num_iterations=1000)`（`211`）。inliers≥当前8%，才评估非内点signed distance；若两侧中较少侧仍占rest>15%，判为穿过主体，跳过；否则apply非内点。

商品参与NN/SOR/DBSCAN/ground估计，在应用删除结果时获保护。median NN只在初始域算一次；20% guard相对每阶段当前点集，不能当作最终占原始点20%的保证。

显式 proxy 分支 `_cluster_labels:110–127`：voxel edge=initial median NN×scale，np.unique取每格最早原始点为representative，proxy native DBSCAN在`126`，标签映回原成员。簇1%/guard仍按原始成员，min_points=10按proxy邻居数；这改变聚类语义，不是默认等价加速。

原生内部算法依据 [现有算法说明 §6.1](../../docs/viewer_export_algorithm.md#61-open3d-0190-原生循环) 与已核对的 Open3D v0.19.0 官方源码：NN新建KDTree逐点KNN2；SOR再建KDTree逐点KNN20，k含自身；DBSCAN预计算全部半径邻居并串行扩展簇；RANSAC预生成假设、并行测试、按fitness/probability提前停止。1000是迭代上限。此runner只测native API调用的整段，不冒充测到了内部建树、查询、簇扩展各段。

## 4. 场景方向与居中

DA3 finalize `web_viewer_export.py:194–199` 调 `get_scene_orientation`；方向不属于prepared worker。`utils/scene_orientation.py:235–274` 可命中独立stat绑定sidecar；miss/stale时重读world_points/conf/E，valid要求finite/nonzero XYZ、finite confidence，没有Surfel positive-depth条件。

`fit_scene_orientation:36–114`：最多均匀取200,000点（`51–56`），native NN在`60`，distance=clip(3×median NN,.02,.15)；最多8候选，native RANSAC在`68–70`，n=3/iterations=1000。每个候选必须内点≥原样本5%、tilt≤60°、原样本在平面下超过.1m者≤15%；不接受则移除该候选内点继续。返回fitted旋转或not_found固定相机轴翻转。

方向NN与过滤NN使用不同域；方向RANSAC与删除地面RANSAC也不能合并。诊断hardlink NPZ到全新out/cache，sidecar隔离，首次方向阶段为miss，不代表主跑必然miss。

`web_viewer_export.py:665` 冻结几何/quality/spill过滤后的filtered_points；发生在ownership和预算前。`_compute_world_to_view:1066–1075` 将其旋转后以各轴median居中。最终预算点数不改变此居中输入。

## 5. 标签与full-grid特征

`_instance_labels_v2`（`web_viewer_export.py:764–839`）按数值global ID、observation构建label_keys；逐image读取完整canonical processed PVS masks，逐observation填尚未归属像素。重叠像素先到者保留；输出labels[M]，背景=-1。

`595–612` 把同gid的跨帧observations合并为region；背景=-1、无效=-2。每16个valid source取U/V长度的有限正数median为spacing_grid；与过滤median NN及SOR k20均不同。

`src/surfel_sampling.py:99–145` 的 `EdgeAdaptiveSampler.__init__` 逐frame构建weights=uint8[N]、caps=float32[N,2]。`_depth_breaks:18–34` 的断裂阈值为max(2×spacing,.005×min_pair_depth)+3×neighbor_slope；还阻断不同region。edge包括断裂两端与图像边，halo膨胀2轮，默认edge/halo/interior权重4/2/1；背景可显式调权。`_axis_caps:37–48` 双向累计最近边界距离，乘.95/1.05并clip .5..8。

完整grid用于识别真实深度/语义边界；候选抽稀不应变成grid空洞。support与candidate是不同集合。

## 6. 候选、voxel、预算和自适应scale

默认 `web_viewer_export.py:641–644` 对全valid域过滤。默认关闭的quality、spill、ownership入口在`649–686`；开启时必须单独记录各阶段点数。ownership依赖完整候选geometry/frames/regions/confidence/sharpness，不能提前抽稀后声称等价。

early-background实验（`613–639`）在labels后先对完整背景做voxel，再adaptive选最多early预算，和全部商品合并为过滤候选；不refill。full背景support仍供scale估计。该模式使NN/SOR域改变且不能提前准备。

背景 `_select_background_voxels:497–526`：默认voxel_size=.005m，对floor(XYZ/size)排序，每格取最高confidence，同分取最早；结果按voxel首次出现顺序。它发生在几何过滤后、500K预算前。

`MAX_PRODUCT_POINTS=2,000,000`、`MAX_BACKGROUND_POINTS=500,000`（`62–63`）。商品 `_sample_product_points:457–494` 在超预算时按gid公平分配，小组不足则余量再分配；Surfel在配额内调用adaptive，纯points用seed42无放回。商品不足2M也调用adaptive，候选不会删减但计算scale。

`EdgeAdaptiveSampler.select:157–203`：

1. 候选原grid indices，分入16×16 tiles；四次bit循环生成tile内Morton顺序。
2. bincount得到tile capacity/mass；有support时另算support mass。
3. `_apportion:51–72` 逐轮处理饱和容量，余量按mass分配；预算足够时先给每个occupied tile一个名额。
4. 对有配额tile循环；`_probabilities:75–96` 逐轮饱和概率，累计概率中点确定取样位置。
5. density=selection_probability×candidate_mass/support_mass；scale=clip(1/sqrt(density),.5,8)，再取full-grid caps较小者。返回candidate内位置与aligned U/V scale。

`web_viewer_export.py:700–724` 背景最多500K、商品最多2M；source_indices通过所有过滤/采样保持对齐。

## 7. point_ranges与发布

`web_viewer_export.py:726–754` 对最终labels稳定argsort，所有positions/colors/normals/source_indices/scales同步排序。`_estimate_scene_normals:759–761` 是固定[0,0,1]数组，没有native法向估计。positions=float32、colors=uint8、normals=int8。

`_attach_point_index_ranges:857–866` 对每个observation label两次searchsorted，写半开[start,end)；空observation有空range。`_minimal_objects:1028–1056` 发布非空point_ranges并检查边界/重叠；同gid允许多个observation ranges。

`prepare_surfels`（`surfel_export.py:88–122`）验证U/V/scale aligned shape及scale∈[.5,8]；source_index//HW作为Uint8 frame；U/V/scales与完整depth grid序列化Float16，校验有限范围和正depth下溢。完整depth不随选点裁掉。准备过的JPEG直接复用。

`_publish_bundle:1094–1126` 依次写Surfel payload、manifest、positions/colors/normals、objects/masterdata/thumbs到临时generation，rename后atomic CURRENT。runner保留全部产物及失败遗留，清理由协调者决定。

## 8. 独立诊断用法、统计与限制

只在主benchmark结束后运行，例如：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
reports/profiling-20261018/diagnose_viewer.py \
--base runtime/profiling-20261018 --out runtime/profiling-20261018/diagnostics/viewer-cpu-01 --dedup
```

out必须是不存在的新目录。默认只诊断Viewer；`--dedup`额外在out/outputs下重做dedup，原matching结果只通过只读用途symlink alias读取。Viewer始终使用base主跑global_mapping/PVS masks和同一dataset/images。NPZ通过hardlink放入out/cache，runner只读该NPZ，方向sidecar另写。hardlink要求同文件系统，错误直接记录，不静默复制。

`result.json`保存status/异常、线程配置、CPU user/system时间、RSS HWM、数值counts及source_map。thread-local嵌套spans聚合calls/inclusive/self/max；并发线程不会串stack。本诊断本身串行prepare/finalize，不能测matching重叠收益。

AST覆盖`prepare_scene_filter`、`_median_nn_distance`、`_cluster_labels`、`filter_scene_points`、`fit_scene_orientation`；不插入随机调用、不seed、不更改原生参数/调用顺序。native API调用与外层block分别计时；SOR raw准备和SOR apply分开。`_sample_points`连续粗阶段按当前源码行分组：560–587 setup、588–590 labels、591–612 full-grid features、613–644 geometry filter、645–686 optional selection filters、687–693 postfilter/voxel、694–725 budget、726–755 array serialization。source_map记录实际函数范围/native call sites及上述映射；源码改行后应重新核对分组。

数字观察包括N/valid/base、rawSOR保留量、prepared medianNN；apply前raw/protected/rescued/guard/后点数；DBSCAN输入/labels/noise和proxy输入；ground输入/inliers/rest/side counts；过滤前后product/background、voxel前后、预算与最终数量；方向candidate remaining/inliers/tilt/below；early分支计数。记录只保留数字或小标量，不保存array或locals。数值观察存在诊断开销，native计时把前后观察排除，外层block包含这部分开销。

不测Open3D内部子阶段、不记录全部邻居列表、不测GPU、accuracy、浏览器、COS/HTTP或GLB。dedup本次使用base输入检测JSON重新跑，独立结果不会替换主跑mapping。此结果不能与主跑墙钟或旧组件窗口直接拼加。

旧`runtime/video3-full-profile-20261016/diagnostics/cpu/detail.py:79–80`仅改写medianNN和filter函数；当前SOR迁至prepare_scene_filter、DBSCAN迁至_cluster_labels，旧runner会漏native SOR/DBSCAN。其SOR_block字符串匹配旧if也失效。旧全局STACK不支持线程嵌套；本runner修为thread-local。旧文档prepare=2.999s的窗口尚无当前NN/SOR提前工作，不能当作此次prepare耗时。旧点数和时间仍属旧记录。
