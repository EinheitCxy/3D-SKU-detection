# Ground proposal：实现、固定域实测与未推广建议

2026-10-18。此次是Rick批准的**算法变更**，不是保持原输出的等价优化。初始实现以50K供component实验；固定域实测显示速度改善伴随背景mask变化，且未做视觉验收。协调者已决定并落地**`ground_proposal_max_points=0`为安全默认**，保持原全域RANSAC；50K保留为显式可选实验，未推广。组合正式candidate也使用默认0，不能将下面50K component差额计作组合已获得收益。

## 实现与保留规则

`utils/pointcloud_filter.py`新增明确config字段，追加在现有字段之后，现有调用不必提供新参数；允许0或整数≥3。准备阶段config快照包含该字段。NN、SOR、DBSCAN、方向拟合和可选quality/ownership均未修改。

`_ground_filter_mask()`读取SOR/DBSCAN后当前cloud：

1. 沿用原initial medianNN和`clip(ground_dist_scale*medianNN,.02,.15)`阈值。
2. cap>0且当前点数超过cap，用局部`np.random.default_rng(42)`均匀无放回抽样，随后按原索引排序；不改变NumPy全局RNG状态，也不改变生产Open3D RNG的既有行为。
3. 只对proposal cloud调用一次`segment_plane(dist,ransac_n=3,num_iterations=1000)`，不增加全量RANSAC重跑路径。
4. 将返回的有限非退化plane按法向长度规范化，在**完整当前点域**重算`abs(P@n+d)<dist`，严格小于，不用`<=`。
5. 用完整当前点数作8%支持率分母；达到后才检查非内点双侧比例15%；通过后仍调用原`apply()`合回商品保护并执行20%保留guard。
6. 不足支持、双侧拒绝或退化plane时不应用ground删除。低proposal支持率不作为提前拒绝条件，也不被解释成不存在其他可删平面。

cap>0但当前域≤cap时，使用整个cloud提案，仍按refined plane重计全量内点；因此这个分支也不是原生返回indices的逐bit等价替代。cap=0 control使用原生返回的indices，保留原控制的正常有效plane行为；退化plane统一按无删除处理。

## 核对Open3D最后refine行为

本次重新读取了固定版本[Open3D v0.19.0 PointCloudSegmentation.cpp](https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/geometry/PointCloudSegmentation.cpp#L211)：先按best_plane_model找final_inliers，随后用这些内点refine plane，最后返回**refined plane与refine前的final_inliers**。candidate明确使用返回的refined plane重算完整域内点；控制臂保留原生indices。这是除了采样提案外另一项明确的边界语义变化。

法向为零、过小、非有限或plane shape不合法时返回无删除；没有隐藏全量重试。Open3D原生异常仍传播，不转成成功或静默降级。

## Scalar计时与分支记录

ground每次只产生一条scalar日志，包括input_points、proposal_points、threshold_m、proposal_inliers、sampling_s、proposal_s、evaluation_s、full inliers/support_ratio及reason；进入apply时还记录protected_rescued、apply_s、output_points、removed_points。reason区分退化、完整支持不足、切过主体、20%guard与实际应用。

proposal native API现在位于独立helper；旧只改写`filter_scene_points`的AST native计时会漏此调用。新harness可直接包helper并读取上述native/evaluation scalar，不能把缺旧span当成没有RANSAC成本。日志没有保留点数组，也没有逐点Python插桩。

## 最小验证结果

指定现成uv Python、`CUDA_VISIBLE_DEVICES=''`：

- `test/test_pointcloud_filter.py`全文件39项通过（6.29s），覆盖原过滤行为和新增单元。
- 随后低成本scalar日志调整后，对完整支持/保护与拒绝分支再跑最小focused集合：6项通过（33 deselected，4.11s）。
- probe最终captured-domain-only版本语法通过。实现者未自行启动重型比较；其后由协调者串行执行真实固定域CPU对照，结果见后文。
- 默认回到0后仅运行新增的默认/显式opt-in断言focused test：1 passed（4.04s）；未重复全套过滤测试，guard和保护代码未变。

新增单元覆盖：cap校验；局部固定seed不扰动全局NumPy RNG；提案只来自post-SOR/DBSCAN当前域；proposal原生返回0内点但全量支持充分仍可接受；非单位法向规范化与strict threshold；商品保护；20%guard；低完整支持/双侧/退化拒绝；只有一次native调用无全量重试；cap=0保留原生pre-refinement indices。

## 独立CPU probe入口与实际执行

`ground_probe.py`只接受协调者已捕获的同一baseline domain：

- `CASE/ground_points.npy`：float64[N,3]；
- `CASE/ground_protected.npy`：aligned bool[N]；
- `CASE/ground_meta.json`：`median_nn`与`config`。

它不会再跑NN/SOR/DBSCAN、不重做Viewer/GLB、不保存新点云。两臂通过PreparedSceneFilter复用完整当前domain，禁用已执行过的DBSCAN且不提供rawSOR mask；baseline从只读snapshot导入原函数，candidate使用工作区新函数。每臂一次，probe局部在各臂前设相同Open3D seed=42用于受控比较，生产函数不新增Open3D全局seed。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
UV_CACHE_DIR=/tmp/3d-profile-20261018-uv \
uv run --no-project --python runtime/viewer-latency-20261017/test-env/bin/python python \
reports/latency-optimization-20261018/ground_probe.py \
--captured-domain runtime/latency-optimization-20261018/baseline-video3 \
--out runtime/latency-optimization-20261018/ground-probe-new --cap 50000
```

上例CASE名称应替换为协调者实际baseline case。out必须不存在。result.json保留失败异常、每臂wall、原生proposal耗时、candidate全量评价/应用耗时、产品/背景保留数、candidate增删数、完整keep Jaccard、产品是否全部保护及CPU/RSS；不保存大型新点云。load/cloud准备与两臂计时分开，wall含PreparedSceneFilter输入关联检查和mask应用，不冒称纯native时间。

首次执行在创建Open3D cloud时因read-only mmap被拒绝，尚未调用native RANSAC；失败记录保留于`runtime/latency-optimization-20261018/ground-probe/result.json`，不计性能样本。probe现以`np.array(domain,dtype=np.float64,copy=True)`向Open3D提供可写buffer，原domain仍是只读mmap。copy和cloud构造都计入`captured_domain_load_s`，在两臂timer之外。

协调者重跑的[原始结果](../../runtime/latency-optimization-20261018/ground-probe-writable/result.json)为complete，OMP/OpenBLAS/MKL均4线程、CPU-only、每臂一次、同一captured baseline domain：3,425,675点，商品1,902,589，背景1,523,086，原medianNN=.0039335003m，ground阈值=.02m。输入加载/可写copy/cloud构造为.096513s，独立列出。

| 同次固定域指标 | 原snapshot全量control | 50K candidate |
|---|---:|---:|
| ground wrapper wall | 10.879015s | .980081s |
| 原生proposal调用 | 10.703740s | .157808s |
| candidate采样/构造 | — | .026135s |
| candidate完整域评价 | — | .048175s |
| candidate保护/guard/select应用 | — | .591180s |
| 原生proposal返回内点 | 238,076（全域） | 3,496（50K域） |
| 完整域接受内点 | 原生indices为238,076，未达8% | refined plane重计280,535（8.18919%） |
| 最终保留商品 | 1,902,589 | 1,902,589 |
| 最终保留背景 | 1,523,086 | 1,471,800 |

此次ground wrapper差额为9.898933s，仅属于固定域component。没有用它直接扣减完整流水线，也不把sampling/proposal/evaluation/apply子项再加到wrapper wall。

candidate完整域内点280,535中，229,249受商品保护恢复，实际多删51,286个背景点；相对原背景删除3.36724%，完整keep Jaccard=.98502894，商品变化0。最终mask不等价，商品全保留不能替代背景/画质验收。sample原生indices比例6.992%低于8%并未提前否定候选；candidate按返回refined plane重计完整域后通过8%。这同时涉及proposal选择与refine后重计语义，不能把差异全归因于普通抽样误差。

## Plane方向核对：不能把删除点称为已确认地面

只读核对同case[scene_orientation.json](../../runtime/latency-optimization-20261018/baseline-video3/outputs/dataset/da3_cache/scene_orientation.json)：status=fitted，world normal=`[.0754208214,-.9390331411,-.3354526190]`，offset=1.4077135591m，方向拟合阈值=.0382593113m。此方向是算法拟合，不是人工标注的地面真值。

- control返回normal=`[-.5940707347,-.3040042489,.7447559190]`，与方向normal的无向夹角为89.474845°。
- candidate返回normal=`[.8084554271,-.1008339149,.5798554509]`，与方向normal的无向夹角为87.773311°。
- 计算为`acos(abs(dot(n,n_orientation)/(norm(n)*norm(n_orientation))))`，消除plane normal正负号歧义；plane与方向都处于同一原世界坐标，无须Viewer rotation或translation。

candidate平面与用于摆正的拟合floor平面几乎垂直；其完整内点81.7185%又是被保护的商品点。这些是**不能自动认定为真实floor、需要审查背景删除是否合理**的风险信号，不能在无可视化/人工标签的情况下声称已证明误删了哪种结构，也不能把51,286点记成已正确清理的地面。

现有ground只做支持率与双侧检验，没有方向一致性约束，所以“通过guard”本身不提供地面语义证明。这里不顺带新增tilt过滤或orientation复用规则。

## 推广建议

**已按协调者决定维持原默认0，50K显式可选；当前不做默认promotion。**组合正式candidate使用安全默认0，不叠加该50K component收益。理由是只有一个video、一次fixed-seed对照，输出背景已变化、candidate法向与摆正floor几乎垂直，且无视觉acceptance。速度证据成立于此component，地面语义和整体视觉等价尚未成立；这不是确认误删，但不足以直接推广。

## 精确风险和接受限制

均匀50K仍可能改变胜出的plane、漏掉小/稀疏地面或误选其他大平面；完整评价可防止仅靠sample支持率误删，但不能恢复没提出的plane。固定NumPy采样不等于Open3D多线程RANSAC输出完全确定。单video固定域计数和Jaccard不能代替多场景视觉/背景质量验收；原始完整地面策略本身也无绝对地面语义保证。当前默认已保持0；显式实验若质量不接受则不推广，而不是自动全域重试。

## 可复用PyTorch算子的候选（未实现）

固定plane全量评价可表达为`torch.matmul/abs/count_nonzero`，point ranges对应`torch.searchsorted`；若数据已在GPU，才值得独立核算搬运/同步与等价边界。当前ground cloud驻CPU且使用float64，NumPy已调用向量化算子，不能承诺换成PyTorch有收益。SOR/DBSCAN不应以全量`torch.cdist`替代，避免M²邻接存储；palette/PNG也没有可直接等价替换的现有PyTorch算子。
