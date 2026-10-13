# DA3 地堆 Footprint 并集面积算法

## 目标与物理定义

本算法估计一堆**已检测并去重的纸箱**在其承载平面上的投影占地面积，单位为 `m²`。正式结果是成功纳入的 OBB 并集；被排除 ID 的可见高架投影覆盖检查是独立的发布条件。

对每个跨帧唯一物体 `global_id`，算法融合其所有合格观测得到 3D 点云，将点云沿承载平面法向投影，恢复一个二维定向包围矩形（oriented bounding box, OBB）。成功生成可信 OBB 的 ID 构成集合 $I$，最终面积为该集合的二维多边形并集：

\[
A_{\mathrm{stack}} = \operatorname{area}\left(\bigcup_{i\in I}\mathrm{OBB}_i\right)
\]

因此该面积：

- 重叠、上下堆叠部分只计算一次；
- 纸箱在桌面外的悬挑投影会计入；
- 不等于包装表面积、正面面积、SAM mask 像素面积、纸箱接触面积或 bbox 面积算术和；
- 不会推断未检测或完全遮挡的纸箱。

## 输入契约

公开入口为：

```bash
# 在仓库根目录运行；先完成 reconstruction、完整 matching 和 dedup。
uv run python main.py --mode ground-stack-area \
  --dataset <dataset> --save_root <save_root>
```

matching 使用自定义 mask 缓存目录时，追加相同的 `--sam3_mask_cache_root <path-to-v2>`。面积阶段只读取指定目录，不会搜索替代缓存或运行 SAM3。

测量阶段读取以下输入；方向结果另存于同一 DA3 cache 目录：

| 输入 | 用途 |
| --- | --- |
| `<dataset>/images/<frame>.*` | 原始图像，用于验证 cache 与图像尺寸、内容一致 |
| `<dataset>/detections_results/<frame>.json` | 每帧检测框 |
| `<save_root>/<dataset>/da3_cache/predictions.npz` | DA3 metric `world_points`、置信度、world-to-camera `extrinsic` 和像素映射信息 |
| `predictions.npz` 旁的 `scene_orientation.json` | 与 Viewer 共用的方向；缺失时计算一次并保存 |
| `<save_root>/<dataset>/dedup_detections/global_mapping.json` | 检测框到物理纸箱 `global_id` 的一一映射 |
| `<save_root>/<dataset>/sam3_mask_cache_sam31/v2` 或显式指定目录 | matching 发布的完整 processed-space self-exemplar masks |

DA3 cache 必须为 schema-v3、`is_metric == 1`，且包含原图 SHA256、原图尺寸、pixel-centre affine、预处理分辨率与方法。程序会验证 cache 与当前原图、检测 JSON、global mapping 的帧和对象集合完全一致。旧 cache 或任一不一致输入会被拒绝，而不会继续测量。报告中的 metric 为 `da3_self_exemplar_ground_footprint_union`，不能与旧的 source-mask 指标直接比较。

## 算法流程

### 1. 输入完整性与坐标契约校验

1. 枚举图片、检测 JSON 和 DA3 cache 的数字帧 ID，要求三个集合完全相等。
2. 验证每个 cache frame 的原图尺寸与 SHA256。
3. 依据 DA3 的 `upper_bound_resize`、patch 对齐和 batch centre crop 规则，重新计算原图到 DA3 网格的 pixel-centre affine；cache 中 affine 必须逐元素匹配。
4. 展开 `global_mapping.json`，要求每个 `(image_id, object_id)` 恰好出现一次，且 bbox 与检测 JSON 完全相同。

这一层保证后续 SAM3 mask、DA3 世界点和 `global_id` 指向同一个实际检测对象。

### 2. 从 bbox 得到每个物体的多视角 3D 点

对每一帧：

1. 读取 matching 已发布的完整 v2 mask cache；每个 mask 已位于 DA3 processed grid。
2. 用 DA3 cache 的完整 affine 映射检测框，验证 mask 的对象、尺寸、bbox 和坐标契约。
3. 保留同时满足以下条件的 mask 内点：3D 坐标有限、坐标非零、`world_points_conf >= 1.0`。
4. 将该观测点云归入对应的 `global_id`。

同一 `global_id` 不会选取“最佳单帧”，而是在后续融合它的所有合格观测（不足 32 点的观测按第 4 节排除）。这样可由不同视角互补纸箱的顶面和侧面。

### 3. 优先复用场景方向，再确定水平支撑高度

面积阶段与 DA3 Viewer 共用 `utils/scene_orientation.py`。`scene_orientation.json` 保存 `status`、`rotation`、`normal_world`、`plane_offset_m` 和拟合诊断；参考平面方程是 `normal_world · X + plane_offset_m = 0`。它不包含 Viewer 的居中平移。

方向读取规则：

1. 同一 `predictions.npz` 的方向缓存有效时直接复用，报告 `cache_event: hit`，不重新运行方向拟合。
2. 缓存缺失或 NPZ 文件状态变化时，调用与 Viewer 共享的 numpy 方向拟合算法并原子保存结果，分别报告 `computed` 或 `stale_recomputed`。关联使用真实路径及文件 stat，不增加文件 hash 扫描。
3. 两个消费者统一使用有限、非零世界点及有限 confidence 的整场景点集。方向估计仍沿用 Viewer 的 5% sampled-point 支持、相对原始 −Y 倾角最多 60°、平面以下 0.1 m 点不超过 15% 等启发式条件；它不是 IMU 重力测量。
4. 没有拟合方向时，面积阶段拒绝；Viewer 保留已有轴翻转显示行为，但不会把该状态标成方向拟合成功。损坏的方向文件明确报错。

方向拟合（`viewer-plane-v2`）使用固定种子、固定 1,000 次试验的 numpy RANSAC，并对内点做 SVD 复拟合。固定输入和运行环境中的重复测试通过；不承诺跨平台逐位相等。v1 使用 Open3D `segment_plane`，设置种子仍不足以保证其并行采样可复现；v1 sidecar 会被判为 stale 并重算。

所有 detection masks 膨胀 5×5 像素后从有效世界点中排除，得到背景。随后固定共享的 up 方向，仅估计平面高度：

1. 复用 `src.surfel_export.grid_tangents`，在原始 DA3 网格上计算局部表面法向；保留与 up 平行或反平行、夹角不超过 15° 的背景点。局部法向在高度切片**之前**计算，防止把墙上的水平条带当作地面。15° 为初始工程参数，尚未通过独立实测标定。
2. 按帧均衡采样最多 50,000 个水平支持点；沿 up 的高度排序，以 24 mm 滑窗进行最多 8 次密集高度候选搜索，并抑制重复候选。通过质量条件后仍落入已有合格候选 24 mm 邻域的项标记 `duplicate_of`，保留诊断，但不作为另一个合格平面参与选择。
3. 在 ±12 mm 邻域中，先求每帧高度中位数，再取帧间中位数，最多更新 10 次。最终使用同一 offset 的 ±10 mm 支持点计算全部质量诊断；此候选选择阶段不重新拟合法向。p95 残差保留作诊断，不作为质量门。
4. 每个候选需至少 10,000 个水平内点，跨至少 3 帧及 30% 背景帧；支撑 hull 的 minimum rotated rectangle 两边均至少 0.30 m，hull 面积至少 0.25 m²。商品位于平面上方（容忍 −12 mm）的点比例按 observation 等权平均，需达到 95%。
5. 不再要求支撑点占全部背景 10%、物体 p01 高度低于 80 mm，或可见地面 hull 覆盖商品投影。背景占比和 hull 覆盖率仍记录为诊断。
6. 在通过质量条件的候选中选最低平面。报告 `method: direction_constrained_height_modes`、`semantics: lowest_observed_horizontal_support`，以及全部候选和选中索引。

7. **法向精炼。** 在选中平面 ±100 mm 内取水平支持点（按帧均衡，最多 50,000），用固定种子 RANSAC（12 mm）求内点，对内点做 SVD 得到支撑面自身法向，并以此作为投影方向。内点少于 10,000，或与共享 up 夹角超过 3° 时拒绝。报告字段为 `plane.normal_refinement`。带宽与 3° 均为初始工程参数。原 p95 残差门先筛到 ≤10 mm 再比较 p95 与 10 mm，无法发挥质量筛选作用，已删除。`plane.selected` 中保留的内点/比例/残差统计标为 `statistics_basis: direction_constrained_selection`，不可当作精炼后的统计。
8. **逐帧支撑高度。** 对每帧取精炼平面 ±80 mm 内的水平支持点，用 24 mm 滑窗找最密模式，再以 ±12 mm 中位数精炼，得到该帧 offset。少于 1,000 点的帧使用选中平面（`source: selected_plane`）。报告字段为 `plane.frame_support_offsets`，记录初始工程参数与逐帧来源。最低可见水平支撑面的选择语义不变，但 15 mm 高度过滤改为相对各观测所属帧的支撑面。

这里的最低平面是**最低可见水平支撑面**，不能仅凭几何断定其就是地板。如果地板完全不可见，最低棚板也可能成为候选。对固定商品点集，平面沿法向平移不改变正交投影面积；但当前 15 mm 高度过滤依赖所选平面位置，仍需保留高度语义与诊断。

### 4. 每个 `global_id` 的二维 footprint 恢复

选定承载平面后，对每个物理纸箱：

1. 有效点少于 32 的观测不参与融合，记录在 `excluded_observations`；只有没有任何合格观测时，该 `global_id` 才因此失败。合格观测通过正式阶段与 shadow evidence 共用的融合函数处理。
2. 只保留高于所属帧支撑面 15 mm 的点，避免支撑背景参与纸箱形状。
3. 沿平面法向正交投影到局部米制坐标 `(u, v)`。
4. 以 5 mm 二维 voxel 保留每格一个点，均衡多视角、不同高度和不同可见面积带来的点密度。
5. 在二维平面中运行 DBSCAN：`eps=20 mm`、`min_samples=4`。
6. 若次大分量点数不低于最大分量的 20%（整数门限为 `ceil(0.20 × largest)`，含边界），视为混入其他物体或严重断裂，该 `global_id` 失败。数值比例记录在诊断中。
7. （D4 = B）先对组件拟合 minimum rotated rectangle，在该矩形自身两条轴上各裁掉 0.5% 的极端点，再对剩余点重新拟合。报告 `extent_estimator: obb_frame_trim_0.005` 和 `extent_trim_fraction: 0.005`。0.40×0.30 m、0°/15°/30°/45°、有/无稀疏贴边点链的合成回归通过面积 spread <1.5%、误差 <2.5%、点链增幅 ≤0.5% 的检查；这仅证明所测 fixture 的朝向/点链偏差减小，不证明精确旋转不变或任意点分布下的密度独立性。

每个 OBB 必须是非退化、有效且面积为正的多边形。

### 5. 多边形并集与面积稳定性检查

对已纳入集合 $I$ 的纸箱 OBB（排除 ID 的覆盖判定另见下节）：

1. 先将几何量化到 0.1 mm 网格，计算主结果 `area_0_1mm_m2`。
2. 再量化到 1 mm 网格，计算敏感性对照 `area_1mm_m2`。
3. 若两个结果差异超过 `max(主面积×0.5%, 1e-4 m²)`，说明 polygon union 对数值精度过于敏感，拒绝结果。
4. 否则发布 OBB 并集面积为 `value_m2`。

## Fail-closed 规则

未能生成可信 OBB 的 `global_id` 被排除。只有每个被排除 ID 的可见高架投影点至少 95% 落在已纳入 OBB 并集外扩 20 mm 的范围内（`covered`）时才发布面积；任何 `uncovered` 或没有投影点的 `unverifiable` ID 都会使整次测量被拒绝。覆盖检查使用该 ID 全部观测的有效高架点，包括不足 32 点的观测。报告字段为 `coverage` 以及逐 ID 的 `status` 和 `coverage`。95% 与 20 mm 为初始工程参数。

支撑平面、至少一个可信 OBB、union 稳定性和输入完整性仍是发布条件。缓存契约不一致、mask 缺失/不匹配、支撑选择或法向精炼失败、所有 ID 均被排除、覆盖失败或 union 精度敏感性超标均拒绝测量。单个 ID 的几何失败记录原因；程序错误（如 `KeyError`）不再伪装成正常测量拒绝。

拒绝输出固定为：

```json
{
  "status": "rejected",
  "value_m2": null
}
```

`null` 表示“没有通过可信度门”，不是 `0 m²`，也不是部分纸箱面积。

## 输出与复核

输出目录：`<save_root>/<dataset>/ground_stack_footprint/`。`CURRENT` 是记录 `run_id` 的 JSON，以下文件位于它指向的 `runs/<run_id>/` 中。

| 文件 | 内容 |
| --- | --- |
| `measurement_report.json` | 输入 provenance、平面候选及 gates、逐 global ID 观测/voxel/分量诊断、union 精度敏感性、最终 accepted/rejected 状态 |
| `footprints.geojson` | accepted 时包含每个纳入 ID 的 OBB 和 `union`；rejected 时为空 feature collection 且标记 `measurement_complete: false` |
| `top_down_footprint.png` | 俯视复核图；rejected 时带有 `REJECTED` 水印 |

测量报告 schema 为 `3.0.0`，区别于输入 DA3 cache 的 schema-v3 和 SAM mask cache 的 v2。报告还包含 mask 腐蚀/膨胀、固定平面的 leave-one-observation-out 和跨视图重投影诊断。mask robustness 的 union 扰动只使用正式纳入的 ID，并保持法向/逐帧 offsets 固定。leave-one-observation-out 的面积比较要求该 ID 有正式纳入的 polygon；被排除 ID 仍可记录 unavailable 或其他观测诊断。跨视图重投影是独立的可见性诊断。每个来源观测在全部合格像素的排序中等距选择最多 512 点，避免只检查图像顶部；小于预算的观测全部检查。这些均是 shadow evidence，不改变正式结果，扰动面积范围也不是统计置信区间。

## 已知适用边界

- 方向来自现有 Viewer 的平面启发式，而不是独立重力传感器。水平面仍需至少 10,000 个内点、3 帧及 30% 背景帧、足够二维范围；低分辨率、很少可见地面或方向误差仍可能导致拒绝。已有 Viewer manifest 缺少原始地面 offset 和成功状态，不能单独作为可信方向缓存；旧输入首次运行时计算共享缓存。
- 多层货架共用一个投影方向，重叠位置只计一次。仅看到棚板时不能证明找到了真实地板；新的方向策略也不解决空 mask、缺失商品或 OBB 的形状近似。
- 不足 32 点的弱观测被排除；没有合格观测的 ID 无法生成 OBB。OBB 每条边至少 50 mm，次大分量达到最大分量的 20% 时拒绝该 ID；这些被排除 ID 仍需通过覆盖条件。
- 5 mm voxel 仍保留首个观测点，结果可能随点顺序有亚体素变化。OBB 范围估计的已知偏差见第 4 节第 7 条。
- 法向精炼只纠正 ≤3° 的方向偏差。逐帧 offset 只补偿帧间高度不一致，不修复帧间水平错位；多视角融合的 OBB 仍可能被帧间水平错位放大。
- 覆盖只说明被排除 ID 的可见高架投影点与已纳入并集一致。隐藏的纸箱深度和未检测商品没有测量；`covered` 与 `accepted` 均不能证明实测面积准确。
- 与主体相连的错误分割区域可能通过 DBSCAN 并扩大 OBB；矩形近似不能表达瓶罐或袋装商品的真实投影轮廓。
- 真实准确率应通过独立实测面积评估，同时报告成功输出比例和 accepted 样本的误差；合成测试、内部重投影一致性和 `accepted` 状态都不能替代该评估。

## 尺度与 reference 的关系

DA3 提供的是 metric world points，因此算法运行本身不要求额外放置已知尺寸 reference，也不使用它来把 bbox 像素面积换算为 m²。

但 DA3 的 metric scale 仍应在部署场景中通过已知尺寸纸箱或标尺进行 QA 校验。reference 在此流程中用于验证或校正 DA3 尺度质量，而不是替代支撑平面、mask、3D 融合或 polygon union。

## 验证记录（2026-10-13 修复）

fd4–fd8，CPU，SAM3.1 mask cache；baseline 为 `main` 快照 `2f2b9f8` 的原代码，fixed 为 Tasks 1–9（至 `4e0f1df`）加本次 schema 更新。没有独立实测面积，下表只说明发布与诊断行为，不代表准确率。

| dataset | baseline status / value_m2 | fixed status / value_m2 | fixed id_outcomes | normal_angle_deg | frame_offset_range_m |
| --- | --- | --- | --- | --- | --- |
| floor_display4 | rejected / null | rejected / null | `{"ok": 333, "fused global id has fewer than 64 projected voxel points": 149, "carton OBB side length must be at least 0.05 m": 96, "global id has no observation with at least 32 valid masked points": 71, "carton footprint has multiple substantial components": 47, "fused global id has fewer than 64 valid points": 6}` | 0.15014155184787045 | `[-0.01129202101342859, 0.01191043194547362]` |
| floor_display5 | rejected / null | rejected / null | `{"ok": 153, "carton OBB side length must be at least 0.05 m": 33, "fused global id has fewer than 64 projected voxel points": 21, "global id has no observation with at least 32 valid masked points": 18, "carton footprint has multiple substantial components": 4}` | 0.15532236669000676 | `[-0.019607632636345153, 0.010207073242589466]` |
| floor_display6 | rejected / null | rejected / null | `{"ok": 232, "carton footprint has multiple substantial components": 23, "carton OBB side length must be at least 0.05 m": 20, "global id has no observation with at least 32 valid masked points": 14, "fused global id has fewer than 64 projected voxel points": 9, "fused global id has fewer than 64 valid points": 1}` | 0.5442954367947541 | `[-0.01439819636236581, 0.018914237226094664]` |
| floor_display7 | rejected / null | rejected / null | `{"ok": 290, "carton OBB side length must be at least 0.05 m": 161, "global id has no observation with at least 32 valid masked points": 42, "fused global id has fewer than 64 projected voxel points": 13, "carton footprint has multiple substantial components": 12}` | 0.060220644154202516 | `[-0.044407316758864473, 0.0178747521945282]` |
| floor_display8 | rejected / null | rejected / null | `{"ok": 110, "carton OBB side length must be at least 0.05 m": 109, "global id has no observation with at least 32 valid masked points": 21, "carton footprint has multiple substantial components": 5, "fused global id has fewer than 64 projected voxel points": 3}` | 0.02452611293719203 | `[-0.03424448210474293, 0.008061109400849607]` |

数值与完整逐 ID 原因来自以下本地验证记录；`null` 表示该项不可用，不能当作零。法向精炼或 offsets 之前失败的测量不补造角度/高度范围，已载入但未进入逐 ID 拟合的项为 `not_evaluated`。`ok` 只表示该 ID 成功生成 OBB，不代表整次测量 accepted。

- [baseline 摘要](/home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/baseline-summary.jsonl)、[fixed 摘要](/home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/fixed-summary.jsonl)。
- [baseline 真实 CLI exits](/home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/baseline/exit_codes.tsv)、[fixed 真实 CLI exits](/home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/fixed/exit_codes.tsv)；各目录保留逐数据集 `.command.txt`、`.log`、`ground_stack_footprint/CURRENT` 与其指向的 `runs/<run_id>/measurement_report.json`。replay 仅运行一次，复用已有 DA3/SAM cache，未运行模型推理。
- [合并六文件测试](/home/xingyu/3D_Recognization/runtime/worktrees/footprint-area-fix/.superpowers/sdd/2026-10-13-footprint-area-fixes/task10-logs/suite.log)：196 passed in 86.05s，exit 0；schema/source/test 稳定后一次执行，文档修改后复用。

D4 决策在法向/逐帧 offset/覆盖修复前进行，候选共用同一 formal ID 的同一组件点，不能把该实验的 per-ID 比例当作最终面积变化。五数据集 B 的配对比例 <0.8 数量为 0，C 为 7；B 新拒绝 18 个原可生成 OBB 的 ID，并恢复 20 个原拒绝 ID；C 对应 22/26。新拒绝均来自保留的 50 mm 最小边门。B 按预先决策规则采用，未放宽门限。[逐数据集比较摘要](/home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/obb-capture/summaries.json)与[逐 ID outliers](/home/xingyu/3D_Recognization/runtime/footprint-fix-validation-20261013/obb-outliers.json)保留配对比例、缺失配对与全部拒绝。

复核中修正了三个易混淆项：弱观测虽然被正式融合排除，原 evidence 数值预检仍曾让它触发拒绝，Task3 已将预检限定为合格观测；Task8 的 included-ID mask robustness 曾临时缺少局部 polygons 初始化，失败记录保留且已恢复；Task10 摘要曾将没有拒绝字段的观测条目误标 `ok`，现要求真实面积或 included 状态，否则标 `not_evaluated`。上述修复均有保留的报告/测试记录，最终六文件测试通过。

没有场景独立实测面积；方向/offset、覆盖率与成功生成 OBB 数量只是诊断。可见覆盖不能还原隐藏深度或未检测商品，逐帧 offset 不处理水平错位，B 的合成稳定性也不建立任意密度下的准确性。

五个 fixed 均因覆盖条件拒绝：

| dataset | included | excluded covered / total | uncovered | unverifiable | CLI exit |
| --- | --- | --- | --- | --- | --- |
| floor_display4 | 333 | 82 / 369 | 212 | 75 | 2 |
| floor_display5 | 153 | 36 / 76 | 24 | 16 | 2 |
| floor_display6 | 232 | 28 / 67 | 25 | 14 | 2 |
| floor_display7 | 290 | 141 / 228 | 47 | 40 | 2 |
| floor_display8 | 110 | 99 / 138 | 21 | 18 | 2 |

汇总：成功生成 OBB 的 ID 为 1030 → 1118（总计 1996），排除 ID 中 386 个 covered、329 个 uncovered、163 个 unverifiable；fixed 发布成功率为 0/5。ID 数增加只描述拟合/发布诊断，不代表面积准确率提升。
