# Surfel 最终长度限制（2026-10-12）

本轮按新增要求，在现有 Viewer clarity 功能分支上限制最终面片长度，并做独立可视化探索。原 D1 因固定 crop 的 frame 24 没有可见点而未能判定；本轮不改变它的帧、阈值或验收结论。

## 修复与取值

旧约束先限制 native U/V，之后的自适应 scale 和点大小仍可把面片拉长。现在本地 Viewer、英文 Viewer 和 CPU GLB 烘焙都在放大后计算 `D = 2 * sigma_max([U, V])`，将最终椭圆最长直径限制为 `6 * max(nativeStepU, nativeStepV)`。分别压缩超限奇异值，保留未超限短轴；斜交轴也受限。此处使用显示变换前的场景坐标，随来源深度及相机内参变化，并非固定厘米或屏幕像素。

不改变中心点、点索引、商品归属、来源连续性判断或深度容差。已有 ZIP 更新 Viewer 即可使用；已有 GLB 则需重新烘焙才带有新几何。源码修改不等于更新已运行的服务。

同一批现存 X0 crop，在默认半径 1.05 下，超过不同直径上限的点占比：

| 场景 | 点数 | cap 4 | cap 6 | cap 8 | cap 6 命中商品点 | cap 6 命中背景点 |
|---|---:|---:|---:|---:|---:|---:|
| video3 | 254,552 | 15.31% | 2.52% | 1.86% | 0% | 21.15% |
| fd6 | 246,197 | 18.31% | 2.26% | 0.01% | 0% | 7.36% |

选择 6 作为探索值：4 会缩小约 11% 的商品面片，8 对 fd6 背景长面片作用很少。商品按 `objects.json` 的 `point_ranges` 并集划分。这只是两份裁剪数据的几何分布，不是全量场景、其他半径或新数据的保证。原始记录在本机 `runtime/viewer-extent-20261012/distribution/footprint_cap_distribution.json`。

## 验证与可视化

- `uv run --no-sync pytest -q test/test_surfel_mesh.py`：14 passed；从实际烘焙四角验证 scale/radius 放大、斜交轴、双轴超限、小面片不变和深度变化。
- 英文 Viewer 定向测试：6 passed；英文及本地 Viewer 的 TypeScript/Vite 构建通过。
- 真实 WebGL probe 和相同相机 A/B 工具：[运行说明](../perf/surfel_extent/README.md)。测试复用生产 renderer，截图与读回结果保留在本机 `runtime/viewer-extent-20261012/visual/`。
- 最终 WebGL 合成盘：`scale=(8,.5), radius=2` 的画布内外接框从 `512×52`（长边被画布截断）变为 `154×52`，短边保留；斜交盘从 `200×74` 变为 `150×68`，普通盘保持 `54×54`。中心点击仍命中，缩小后边缘以外不再命中；英文和本地 renderer 均通过，WebGL/browser errors 为空。
- 可交互入口：`http://192.168.2.6:53110/?scene=video3`，以及 `?scene=fd6`；Before/After 使用同一 ZIP 和相机，可切换 Overview/Close/Side。

先比较了两轴等比缩小的 Uniform 方案：两场景各有默认半径的三个视角，以及半径 2 的近景，共 16 张图。video3 的稀疏背景出现更多孔洞，因此继续比较只压超限主轴的 Axis cap，补充两场景、两种半径的近景四张图。有效截图统一为 960×543；新增按钮曾改变画布高度，该轮 Axis cap 截图已作废并重拍，未用于结论。

人工查看最终近景，Axis cap 相比 Uniform 保留了更多未超限方向的覆盖，video3 大半径下的背景条带较连贯；fd6 的差异较小。默认半径下商品外观整体变化不大。最终采用 Axis cap，Uniform 留在比较工具中。`candidate-*.png` 是先前 Uniform 方案，`major-close.png` 才是最终 Axis cap；不得按文件名把二者混称。

可直接查看本机近景：

- video3 大半径：[Before](../runtime/viewer-extent-20261012/visual/video3/radius2/baseline-close.png)、[Uniform](../runtime/viewer-extent-20261012/visual/video3/radius2/candidate-close.png)、[Axis cap](../runtime/viewer-extent-20261012/visual/video3/radius2/major-close.png)。
- fd6 大半径：[Before](../runtime/viewer-extent-20261012/visual/fd6/radius2/baseline-close.png)、[Uniform](../runtime/viewer-extent-20261012/visual/fd6/radius2/candidate-close.png)、[Axis cap](../runtime/viewer-extent-20261012/visual/fd6/radius2/major-close.png)。

这些是裁剪场景的定性观察，并没有得到可推广的清晰度或精度提升结论。图中的错层、孔洞和边缘破碎仍然存在；长度上限不能校正位姿或补齐缺失几何。软件 WebGL 的截图不作为硬件 GPU 性能证据。
