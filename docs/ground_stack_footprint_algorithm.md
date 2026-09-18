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
cd code
uv run python main.py --mode ground-stack-area \
  --dataset <dataset> --save_root <save_root>
```

测量阶段只读以下输入：

| 输入 | 用途 |
| --- | --- |
| `<dataset>/images/<frame>.*` | 原始图像，供 SAM3 生成实例 mask，并验证 cache hash/尺寸 |
| `<dataset>/detections_results/<frame>.json` | 每帧检测框 |
| `<save_root>/<dataset>/da3_cache/predictions.npz` | DA3 metric `world_points`、置信度和像素映射信息 |
| `<save_root>/<dataset>/dedup_detections/global_mapping.json` | 检测框到物理纸箱 `global_id` 的一一映射 |
| 本地 SAM3 checkpoint | 按 bbox 分割每个纸箱 |

DA3 cache 必须为 schema-v2，且包含原图 SHA256、原图尺寸、pixel-centre affine、预处理分辨率与方法。程序会验证 cache 与当前原图、检测 JSON、global mapping 的帧和对象集合完全一致。旧 cache 或任一不一致输入会被拒绝，而不会继续测量。

## 算法流程

### 1. 输入完整性与坐标契约校验

1. 枚举图片、检测 JSON 和 DA3 cache 的数字帧 ID，要求三个集合完全相等。
2. 验证每个 cache frame 的原图尺寸与 SHA256。
3. 依据 DA3 的 `upper_bound_resize`、patch 对齐和 batch centre crop 规则，重新计算原图到 DA3 网格的 pixel-centre affine；cache 中 affine 必须逐元素匹配。
4. 展开 `global_mapping.json`，要求每个 `(image_id, object_id)` 恰好出现一次，且 bbox 与检测 JSON 完全相同。

这一层保证后续 SAM3 mask、DA3 世界点和 `global_id` 指向同一个实际检测对象。

### 2. 从 bbox 得到每个物体的多视角 3D 点

对每一帧：

1. 用 SAM3 对全部检测 bbox 生成一个 mask。
2. 用已验证的 affine 将原图 mask 映射到 DA3 `world_points` 网格。
3. 保留同时满足以下条件的 mask 内点：3D 坐标有限、坐标非零、`world_points_conf >= 1.0`。
4. 将该观测点云归入对应的 `global_id`。

同一 `global_id` 不会选取“最佳单帧”，而是在后续融合它的所有有效观测。这样可由不同视角互补纸箱的顶面和侧面。

### 3. 从 mask 外背景点拟合承载平面

所有检测 mask 会先做 5×5 像素膨胀，再从背景中排除，减少纸箱边界混入桌面点。剩余的有效 DA3 世界点构成平面候选背景。

平面选择过程：

1. 按帧均衡采样，最多保留 50,000 个背景点，避免某一帧主导 RANSAC。
2. 以固定随机种子执行最多五个自适应 RANSAC 候选；每轮会移除前一候选的局部内点，以保留桌面、墙面等不同平面。
3. 对每个候选的全部背景内点执行 SVD refinement。
4. 记录每个候选的原始法向、原始点、内点数、内点比例和 refinement 结果；即使 refinement 被拒绝也写入报告。

通过 refinement 的候选还必须满足 table compatibility gates：

- 至少 10,000 个内点，且背景内点比例至少 10%；
- 95% 残差满足平面质量门；
- 内点跨足够多帧，且在平面内具有足够的长宽跨度和 hull 面积；
- 至少 80% 的物体观测中心位于桌面 hull（带 150 mm buffer）内；
- 绝大多数物体点位于所选平面上方，且最低高度分位数接近该平面。

如果没有候选通过 refinement，报告明确标记为 `no support-plane candidate passed refinement gates`；如果候选已通过 refinement 但不符合桌面语义，则标记为 `support plane candidates failed table compatibility gates`。若两个方向明显不同的候选得分相近，则以平面歧义拒绝。

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
- SAM3 失败、mask 空、有效 DA3 点不足；
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

输出目录：`<save_root>/<dataset>/ground_stack_footprint/`

| 文件 | 内容 |
| --- | --- |
| `measurement_report.json` | 输入 provenance、平面候选及 gates、逐 global ID 观测/voxel/分量诊断、union 精度敏感性、最终 accepted/rejected 状态 |
| `footprints.geojson` | accepted 时包含每个纸箱 OBB 和 `union`；rejected 时为空 feature collection 且标记 `measurement_complete: false` |
| `top_down_footprint.png` | 俯视复核图；rejected 时带有 `REJECTED` 水印 |

## 尺度与 reference 的关系

DA3 提供的是 metric world points，因此算法运行本身不要求额外放置已知尺寸 reference，也不使用它来把 bbox 像素面积换算为 m²。

但 DA3 的 metric scale 仍应在部署场景中通过已知尺寸纸箱或标尺进行 QA 校验。reference 在此流程中用于验证或校正 DA3 尺度质量，而不是替代支撑平面、mask、3D 融合或 polygon union。
