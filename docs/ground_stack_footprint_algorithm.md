# DA3 地堆 Footprint 并集面积算法

## 目标与物理定义

本算法估计一堆**已检测并去重的纸箱**在其承载平面上的实际占地面积，单位为 `m²`。

对每个跨帧唯一物体 `global_id`，算法融合其所有有效观测得到 3D 点云，将点云沿承载平面法向投影，恢复一个二维定向包围矩形（oriented bounding box, OBB）。最终面积为所有 OBB 的二维多边形并集：

\[
A_{\mathrm{stack}} = \operatorname{area}\left(\bigcup_{i=1}^{N}\mathrm{OBB}_i\right)
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

同一 `global_id` 不会选取“最佳单帧”，而是在后续融合它的所有有效观测。这样可由不同视角互补纸箱的顶面和侧面。

### 3. 优先复用场景方向，再确定水平支撑高度

面积阶段与 DA3 Viewer 共用 `utils/scene_orientation.py`。`scene_orientation.json` 保存 `status`、`rotation`、`normal_world`、`plane_offset_m` 和拟合诊断；参考平面方程是 `normal_world · X + plane_offset_m = 0`。它不包含 Viewer 的居中平移。

方向读取规则：

1. 同一 `predictions.npz` 的方向缓存有效时直接复用，报告 `cache_event: hit`，不重新运行方向拟合。
2. 缓存缺失或 NPZ 文件状态变化时，调用原 Viewer 的 Open3D 平面摆正算法并原子保存结果，分别报告 `computed` 或 `stale_recomputed`。关联使用真实路径及文件 stat，不增加文件 hash 扫描。
3. 两个消费者统一使用有限、非零世界点及有限 confidence 的整场景点集。方向估计仍沿用 Viewer 的 5% sampled-point 支持、相对原始 −Y 倾角最多 60°、平面以下 0.1 m 点不超过 15% 等启发式条件；它不是 IMU 重力测量。
4. 没有拟合方向时，面积阶段拒绝；Viewer 保留已有轴翻转显示行为，但不会把该状态标成方向拟合成功。损坏的方向文件明确报错。

所有 detection masks 膨胀 5×5 像素后从有效世界点中排除，得到背景。随后固定共享的 up 方向，仅估计平面高度：

1. 复用 `src.surfel_export.grid_tangents`，在原始 DA3 网格上计算局部表面法向；保留与 up 平行或反平行、夹角不超过 15° 的背景点。局部法向在高度切片**之前**计算，防止把墙上的水平条带当作地面。15° 为初始工程参数，尚未通过独立实测标定。
2. 按帧均衡采样最多 50,000 个水平支持点；沿 up 的高度排序，以 24 mm 滑窗进行最多 8 次密集高度候选搜索，并抑制重复候选。通过质量条件后仍落入已有合格候选 24 mm 邻域的项标记 `duplicate_of`，保留诊断，但不作为另一个合格平面参与选择。
3. 在 ±12 mm 邻域中，先求每帧高度中位数，再取帧间中位数，最多更新 10 次。最终使用同一 offset 的 ±10 mm 支持点计算全部质量诊断；法向不重新拟合。
4. 每个候选需至少 10,000 个水平内点，跨至少 3 帧及 30% 背景帧；支撑 hull 的 minimum rotated rectangle 两边均至少 0.30 m，hull 面积至少 0.25 m²。商品位于平面上方（容忍 −12 mm）的点比例按 observation 等权平均，需达到 95%。
5. 不再要求支撑点占全部背景 10%、物体 p01 高度低于 80 mm，或可见地面 hull 覆盖商品投影。背景占比和 hull 覆盖率仍记录为诊断。
6. 在通过质量条件的候选中选最低平面。报告 `method: direction_constrained_height_modes`、`semantics: lowest_observed_horizontal_support`，以及全部候选和选中索引。

这里的最低平面是**最低可见水平支撑面**，不能仅凭几何断定其就是地板。如果地板完全不可见，最低棚板也可能成为候选。对固定商品点集，平面沿法向平移不改变正交投影面积；但当前 15 mm 高度过滤依赖所选平面位置，仍需保留高度语义与诊断。

### 4. 每个 `global_id` 的二维 footprint 恢复

选定承载平面后，对每个物理纸箱：

1. 合并所有帧的有效 mask 内世界点。
2. 仅保留高于承载平面 15 mm 的点，避免桌面背景参与纸箱形状。
3. 沿平面法向正交投影到局部米制坐标 `(u, v)`。
4. 以 5 mm 二维 voxel 保留每格一个点，均衡多视角、不同高度和不同可见面积带来的点密度。
5. 在二维平面中运行 DBSCAN：`eps=20 mm`、`min_samples=4`。
6. 若存在两个足够大的分离组件，则认为该纸箱点云混入了其他物体或严重断裂，拒绝该 `global_id`。
7. 删除二维坐标的 1%–99% 极端点，并对保留点的 convex hull 拟合 minimum rotated rectangle，得到 OBB。

每个 OBB 必须是非退化、有效且面积为正的多边形。

### 5. 多边形并集与面积稳定性检查

对所有纸箱 OBB：

1. 先将几何量化到 0.1 mm 网格，计算主结果 `area_0_1mm_m2`。
2. 再量化到 1 mm 网格，计算敏感性对照 `area_1mm_m2`。
3. 若两个结果差异超过 `max(主面积×0.5%, 1e-4 m²)`，说明 polygon union 对数值精度过于敏感，拒绝结果。
4. 否则发布 OBB 并集面积为 `value_m2`。

## Fail-closed 规则

总面积只有在**每一个** `global_id` 都成功生成可信 OBB、支撑平面通过全部质量门、且 union 稳定时才会发布。

以下任一情况会使整次测量被拒绝：

- 缓存 schema、hash、affine、帧集合或 mapping 不完整；
- canonical mask cache 缺失或不匹配、mask 空、有效 DA3 点不足；
- 支撑平面不存在、歧义或质量门失败；
- 任一 `global_id` 点云不足、DBSCAN 多组件或 OBB 退化；
- polygon union 精度敏感性超标。

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
| `footprints.geojson` | accepted 时包含每个纸箱 OBB 和 `union`；rejected 时为空 feature collection 且标记 `measurement_complete: false` |
| `top_down_footprint.png` | 俯视复核图；rejected 时带有 `REJECTED` 水印 |

报告还包含 mask 腐蚀/膨胀、固定平面的 leave-one-observation-out 和跨视图重投影诊断。每个来源观测在全部合格像素的排序中等距选择最多 512 点，避免只检查图像顶部；小于预算的观测全部检查。这些均是 shadow evidence，不改变正式结果，扰动面积范围也不是统计置信区间。

## 已知适用边界

- 方向来自现有 Viewer 的平面启发式，而不是独立重力传感器。水平面仍需至少 10,000 个内点、3 帧及 30% 背景帧、足够二维范围；低分辨率、很少可见地面或方向误差仍可能导致拒绝。已有 Viewer manifest 缺少原始地面 offset 和成功状态，不能单独作为可信方向缓存；旧输入首次运行时计算共享缓存。
- 多层货架共用一个投影方向，重叠位置只计一次。仅看到棚板时不能证明找到了真实地板；新的方向策略也不解决空 mask、缺失商品或 OBB 的形状近似。
- 每个 global ID 的任一观测不足 32 个有效点，仍会导致该 ID 及整次结果被拒绝。OBB 每条边至少 50 mm，多个显著分离组件也会被拒绝。
- 5 mm voxel 当前保留首个观测点，结果仍可能随点顺序变化；1%/99% 裁剪也可能随坐标轴方向变化。简单改为最外侧点、最近网格中心点或均值点会在部分噪声或干净样本上增加误差，不能视为已经解决。
- 与主体相连的错误分割区域可能通过 DBSCAN 并扩大 OBB；矩形近似不能表达瓶罐或袋装商品的真实投影轮廓。
- 真实准确率应通过独立实测面积评估，同时报告成功输出比例和 accepted 样本的误差；合成测试、内部重投影一致性和 `accepted` 状态都不能替代该评估。

## 尺度与 reference 的关系

DA3 提供的是 metric world points，因此算法运行本身不要求额外放置已知尺寸 reference，也不使用它来把 bbox 像素面积换算为 m²。

但 DA3 的 metric scale 仍应在部署场景中通过已知尺寸纸箱或标尺进行 QA 校验。reference 在此流程中用于验证或校正 DA3 尺度质量，而不是替代支撑平面、mask、3D 融合或 polygon union。
