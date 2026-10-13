# 自动货堆外轮廓与人工圈选对比

目标是每一堆货物的水平外轮廓面积：内部小缝计入，外部 L/U 凹口保留，分离货堆分别计算后相加。新 metric 为 `da3_manual_equivalent_stack_outline_candidate`。

当前实现是可运行的候选路线。它读取已有 DA3 与 SAM mask 缓存，以 Viewer 共享水平方向投影，提取米制栅格外轮廓；不要求每个 global ID 先成功拟合 OBB。精炼支撑面只参与高度过滤。显示用 Surfel 面片大小不参与面积计算。

## 生成候选

从仓库根目录运行。数据、缓存与 Viewer manifest 必须对应同一场景。`--save-root` 使用已有管线输出根目录；`--sam3-mask-cache-root` 指向该场景已有 mask 缓存。

```bash
uv run python -m src.stack_outline_stage \
  --dataset imdata/floor_display4 \
  --save-root Output \
  --sam3-mask-cache-root Output/floor_display4/sam3_mask_cache_sam31/v2 \
  --viewer-manifest /path/to/viewer/runs/RUN_ID/manifest.json \
  --output /path/to/floor_display4-outline.json \
  --grid-resolution-m 0.005 \
  --gap-radius-m 0.02 \
  --max-hole-area-m2 0.04
```

这些数值只是一组演示设置，尚未根据人工参考标定。补缝半径和最大可填孔面积必须显式提供。半径决定局部 closing 的尺度，并不等于所有几何情况下的最大可连接缝宽。5 mm 栅格不是 5 mm 实测精度承诺。

输出包含逐堆 world-space 顶点、面积、参数、输入与分量诊断，以及同名 PNG 俯视图。已有输出默认不能覆盖。输入无效时命令非零退出，不生成有效候选。此入口不运行重建或 SAM 模型，不写正式面积 `CURRENT`。

默认最多分配 4,000,000 个栅格；包含离群投影点的大范围场景可能超限。确认内存足够后，可显式设置 `--max-grid-cells 16000000`。这只提高资源上限，不会改变几何阈值，也不会自动裁掉离群点。

## Viewer 使用

使用与候选 `--viewer-manifest` 相同的 Viewer 数据包。纯点云包通过 `?data=/path/to/bundle/&render=points` 打开。进入“圈选面积”，开始俯视圈选，闭合后保存当前货堆；新一轮草稿与已保存的货堆分开计数。

圈选时可收起面板，避免面板遮挡顶点。静态服务器须对缺失的数据文件返回真正的 404；把所有未知路径重写到 HTML 首页会导致可选 JSON 的加载报错。

多个已保存的货堆组成手工参考，可导出或导入 JSON。再导入自动候选 JSON，叠加查看外部凹口、堆间通道和缺边。手工与自动轮廓使用不同颜色。错误场景、非法 polygon、重叠区域或面积字段与顶点不符的文件会被拒绝，原有状态保留。

手工保存的参考应由人确认。程序生成的矩形 fixture 只验证计算与坐标转换，不能作为人工口径已经通过的证据。

## 状态与比较

| 状态 | 含义 |
| --- | --- |
| `reference` | 人工参考文件，`complete: true`；允许尚未保存任何货堆 |
| `candidate` | 本轮输入的轮廓生成完整，但参数与边界尚未人工验证 |
| `needs_review` | 存在无法表示的分量、过大孔洞等；已生成的面积仅是部分小计 |

候选的正式 `value_m2` 始终为 `null`。没有有效区域的候选不能解释为场景面积为零；输入被拒绝时也没有有效面积。`total_area_m2` 仅表示文件中不重叠区域的面积之和。

保存人工参考后运行：

```bash
uv run python -m src.stack_outline_compare \
  --candidate /path/to/floor_display4-outline.json \
  --reference /path/to/floor_display4-manual.json \
  --output /path/to/floor_display4-comparison.json
```

比较包括堆数、面积相对差、polygon 并集 IoU，以及沿完整边界按 0.01 m 弧长采样的双向距离 p95。面积相近仍可能边界错位；这些量衡量圈选一致性，不是现场尺寸精度。

JSON 使用 `viewer-horizontal-outlines` version 1，米制坐标。`world_to_view` 为 row-major 刚体矩阵；顶点列表不重复首点。场景名称、backend、帧数、变换与显示边界用于检查目标包一致性，不是唯一来源身份凭据。

## 当前限制

- 补缝可能误连两堆或堵住窄入口。超过显式孔面积限制的分量会留待复核，而不是强行填平。
- 单点、共线正面投影不能提供进深，不能据此制造有宽度的货堆。稀疏边界、遮挡和跨帧几何错位仍会影响结果。
- 不按最大堆的点数或面积比例丢弃小堆；它必须被表示，或明确列为未解决分量。
- 栅格尺寸有上限；超大稀疏范围会明确失败，避免无界内存分配。
- 目前没有真实人工参考轮廓或现场面积真值。旧生产 OBB 入口保持原行为，尚未由候选替换。

## 2026-10-13 缓存实验

fd4/fd5 使用全部有效 mapped mask 点，固定 5 mm 栅格、20 mm closing 半径和 0.04 m² 最大填孔面积；未按结果调参。默认 400 万格上限先拒绝了两场景，随后只将显式分配上限提高到 1,600 万格，保留原失败记录。

| 场景 | 有效 mapped 点 / 高度过滤后点 | 候选区域数 | 已表示部分小计 | 完整性 |
| --- | --- | --- | --- | --- |
| fd4 | 444,094 / 439,936 | 32 | 0.349625 m² | `needs_review` |
| fd5 | 318,336 / 318,295 | 106 | 0.586225 m² | `needs_review` |

这些数值不能用作整个货堆面积。fd4 最大分量有 0.24665、0.132225、0.050575 m² 的大孔，另一主要分量无法形成简单无孔外轮廓；fd5 最大分量有约 2.290225 m² 的孔。许多孤立或共线观测也被明确列为未解决。当前参数的真实场景结果尚未达到整堆外轮廓目标；下一步需要人工确认哪些空白是内部缺测、哪些是应保留的通道，不能仅增大填孔阈值来得到总面积。

合成矩形验证中，两堆分别为 6 和 1 m²。Python JSON 与 Viewer 导入后的面积、屏幕边界一致；浏览器实际点击保存得到约 6.99999986 m²，导出重载、草稿独立计数和错误导入保留状态均通过。fd4/fd5 的实际候选也已导入构建后的 Viewer，部分状态与 32/106 条轮廓显示正常。以上验证证明功能链路可用，不代表真实候选准确；fd6–fd8 尚未评分。
