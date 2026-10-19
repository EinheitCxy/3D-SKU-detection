# Viewer 代理聚类与背景平面清理

2026-10-19，Rick查看video1/3/6后接受proxy2的画面，并要求保留video1的背景清理收益、取消全局8%支持率依赖、提交并推送main。本轮源码、测试和算法说明作为同一功能发布；不部署服务。

## 实现

- 默认`cluster_proxy_voxel_scale=2.0`：每个`2×medianNN`体素选最先出现的原始点，在代理上做DBSCAN，再把标签传回全部原始成员。核心点邻居数按代理统计，簇规模仍按原始成员统计。显式0保留已有原始DBSCAN控制路径。
- 平面拟合只使用当前未保护背景点；少于`max(3,min_points)`时不拟合。受保护商品既不投票，也不会进入删除候选。
- 删除`ground_min_inlier_ratio`及8%门槛。`support_ratio`只记录背景内点比例，不参与接受/拒绝；没有替换成新的支持率阈值。
- 保留现有双侧15%检查、商品保护和每阶段最少保留20%的guard。双侧检查仍使用完整当前场景，属于防止平面切穿主体的既有安全规则。
- 默认`ground_proposal_max_points=0`在完整背景上运行Open3D，并映射其native内点索引；显式正数只抽背景提案，再用refined plane重查完整背景。未启用50K默认，也未替换Open3D的随机候选/早停。

`remove_ground`是既有参数名，实际是平面背景清理。地面、墙面等都可能被识别为候选；它没有语义地面分类，未被SAM标注的商品仍可能被视为背景。keep和source_indices在预算采样前同步选择位置、颜色、标签、法线和Surfel来源，不修改原始DA3缓存。共享摆正/footprint、2M商品/500K背景预算、LOD和长边限制未改。

完整循环及边界见[算法说明第4节](../../docs/viewer_export_algorithm.md#4-过滤如何保护商品)。

## 最小验证

`test/test_pointcloud_filter.py`：51通过，覆盖默认代理成员回填、低于8%仍可清理、商品多数时不进入拟合、局部/全域索引映射、背景不足、native/refined内点边界、双侧拒绝、过量删除保护及准备复用。

`test/test_web_viewer_export.py::test_prepared_surfel_matches_serial_and_prepares_once`：2通过，覆盖准备/串行输出及过滤启用的接口连接。未运行无关全量回归。

## 缓存实景验证

复用三视频既有DA3、PVS、mapping和原图，只运行一次新的完整Viewer准备/导出。CPU执行，不重跑模型、ZIP、GLB、GPU测试或浏览器。结果保存在本机`runtime/viewer-background-cleanup-20261019/cached-viewers/result.json`，不随Git发布。

| 视频 | DBSCAN秒 | 背景平面拟合/检查秒 | 过滤总秒 | 实际背景删除 | 最终Viewer点 | 最终商品点 |
|---|---:|---:|---:|---:|---:|---:|
| video1 | 9.371 | 4.554 | 15.538 | 360,834 | 1,497,310 | 997,310 |
| video3 | 12.104 | 5.208 | 19.408 | 286,624 | 2,402,589 | 1,902,589 |
| video6 | 20.683 | 7.482 | 32.025 | 356,069 | 2,500,000 | 2,000,000 |

平面列包含背景选择、Open3D拟合和双侧评估，不包含随后应用mask的时间；过滤总时间包含应用。串行准备至Viewer返回分别28.274/37.086/64.744秒。每视频一次、共享主机，这些不是帧就绪至完整GLB的端到端耗时，也不能与旧“Viewer不含准备”数字直接相减。

三视频所有有效受保护点均保留，最终商品的positions、colors、normals、U/V、scale、frame七类逐点payload与各自原缓存Viewer完全一致。video6的3778160个商品候选仍按既有预算选入2000000个；不能把“过滤保点”说成最终全商品导出。

video1/3候选法向与已有摆正参考相差0.288°/1.392°；video6共享方向为not_found，没有这个对照。参考方向不是人工真值，不能据此断言删除的全是真实地板。新增背景删除结果未做新一轮GLB像素或人工视觉验收，proxy2此前的人工验收也不能代替它。

复现缓存导出（模型和原视频不随仓库提供）：

```bash
uv run python reports/viewer-background-cleanup-20261019/verify_cached.py \
  --case runtime/profiling-multivideo-20261018/video1 \
  --case runtime/latency-optimization-20261018/baseline-video3 \
  --case runtime/profiling-multivideo-20261018/video6 \
  --out runtime/viewer-background-cleanup-local
```

`--out`必须是新目录；失败会保留记录，不自动重试或切换过滤模式。
