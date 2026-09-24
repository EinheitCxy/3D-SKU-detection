# Surfel 集成说明

Surfel 接入已有 DA3 缓存导出与 `modules/viewer_web`，不重新推理 DA3/SAM3，不改变 SKU 匹配、去重或计数。当前导出 Surfel v3，使用固定预算的边缘自适应采样与逐点 U/V 尺度；中英文 COS Viewer 均须支持 v3 后才能部署新后端。Viewer 继续明确支持原有 v2 单位尺度包，未知版本或缺失的 v3 尺度文件直接失败。此前各版的场景测试记录见 [历史实施报告](surfel_implementation_report.md)，不作为本次验证结果。

## 固定预算的边缘采样

`src/surfel_sampling.py` 从深度突变、有效域边界和商品 global-ID mask 边界计算权重，边界/附近两像素/内部为 4/2/1。深度突变阈值包含邻域斜率，避免把连续斜面当断层。16×16 图块按权重获得整数名额，预算允许时每个非空块至少一个点；块内按 Morton 顺序确定性分层选择原始观测，不生成深度或移动点。

商品仍跨帧按 global ID 均分最多 200 万点，小组保留全部、剩余额度回流。背景仍经现有过滤和最高置信度体素代表选择，独立最多 50 万点。只替换额度内的选择方法。用于边缘检测的密集几何有效域与候选池分离；背景圆盘密度同时计入体素抽稀，避免将被淘汰的候选误当深度空洞。普通 points 导出保留原选择方式。

每点 U/V 尺度按目标密度的反平方根计算，分别受源图边界距离限制，范围 [0.5, 8]；global-ID 边界也限制扩张。原始一像素 U/V 保持不变，所有位置、颜色、来源帧、尺度与点击区间共用最终 slot 排序。浏览器先约束 native U/V，再仅在圆盘展开时乘尺度；邻点深度预测、纹理深度门和混合深度容忍不随尺度放宽。

这保留了隔离实验方案 2 的设计，但正式商品预算、背景体素筛选和渲染器均不同，不能直接套用实验的 +5.11 个百分点覆盖或 +4.1% 单次总耗时。新尺度文件未压缩大小为每点 4 bytes，达到 250 万点上限时为 10 MB。旧任务 ZIP 需要重新导出才获得新采样结果。

## 入口和数据流

```text
DA3 predictions.npz + 原图 + global_mapping.json + 同网格 SAM3 v2 masks
  -> scripts/export_surfel_viewer.py
  -> src/web_viewer_export.py:export_web_viewer_bundle
     -> SAM 商品按 global ID 均分预算；背景过滤、体素选点、独立预算；额度内边缘采样
     -> source_indices 保留每个输出点的原始网格位置
     -> src/surfel_export.py:prepare_surfels
     -> 原子发布 CURRENT -> runs/<run_id>/
  -> ?data=/data-surfel/&render=surfel
  -> loadViewerBundle（跳过 RGB）-> loadSurfels
  -> createViewerScene -> createSurfelRenderer
     -> 最近深度 -> 加权纹理累加 -> 归一化显示
     -> 点击时另绘 ID 通道 -> point_ranges -> 全局id
```

常规 `main.py --mode viewer-web` 没有传入 `surfel_texture_edge`，仍走普通 points 导出。独立脚本传入该参数，启用同一导出函数的 Surfel 扩展。命令示例见 [Viewer README](../modules/viewer_web/README.md#深度约束-surfel)。

## 输入及点序

用 F 表示帧数、H/W 表示 DA3 processed grid 的高/宽、P 表示最终保留点数：

| 输入 | 形状 | 用途 |
|---|---|---|
| world_points | `[F,H,W,3]` | 原始世界坐标网格 |
| world_points_conf | `[F,H,W]` | 普通体素选点时选择最高置信度代表 |
| images | `[F,H,W,3]` | 既有导出流程的网格 RGB，与最终点序同步发布 |
| extrinsic | `[F,3,4]` | 世界到来源相机的变换 R、t |
| intrinsic | `[F,3,3]` | processed grid 对应的内参 K |
| source_to_processed_affine | `[F,2,3]` | 原图像素到 processed 像素的预处理变换 |
| source_image_sizes / image_ids | `[F,2]` / `[F]` | 原图宽高、帧对应关系 |

`_sample_points` 先去除非有限/全零点（Surfel 另排除非正来源深度），在商品保护下执行现有场景过滤，再为背景按体素选择置信度最高的点；Surfel 在各自预算内进行上述边缘采样，最后按商品标签稳定排序。
`source_indices = valid_indices[keep_filter][keep][order]` 与 positions 共序。由该索引可计算来源帧 `f = index // (H*W)`，以及网格行列。

SAM3 mask 和 global mapping 在既有 `_instance_labels_v2` 阶段产生对象标签。Surfel 不重新分配商品 ID；U、V、来源帧逐项跟随最终点序，所以原有半开区间 `point_ranges=[start,end)` 可直接用于批量选择、隐藏和 Focus。

## 从点变成局部表面

`grid_tangents` 计算来源相机深度 `d=(R*p+t).z`。非有限、全零世界点和非正深度无效，其深度写 0。

沿图像宽度计算 U、沿高度计算 V。每个方向分别考虑前向差分 `p_next-p` 和后向差分 `p-p_prev`：邻点必须有效，边界不允许 wrap-around，三维边长必须小于 `max(0.03*d, 0.001)`；两边都有效时取较短的一边，均无效则该切向量为零。阈值约束的是三维邻边长度，不只是 z 差。

U/V 的形状起初均为 `[F,H,W,3]`，随后按 source_indices 选为 `[P,3]`。它们代表一个原始网格像素的世界空间位移，隐式确定局部表面方向。当前没有融合成新的网格，也没有估计/优化高斯协方差。

浏览器每点实例化一个由两个三角形组成的 quad。其世界坐标为：

```text
p_fragment = center + radius * (x * scale_u * U + y * scale_v * V)
x*x + y*y <= 1
```

像素着色器丢弃单位圆外部分，得到沿局部表面展开的圆盘。默认 radius=1.05 个网格像素；Point size 映射为 `clamp(1.05*size/0.004, 0.6, 2)`。扩大半径只扩大覆盖，不增加采样点、不恢复新几何；切向量为零的退化面可能不可见。

`normals.i8.bin` 仍是普通 schema3 的固定 `[0,0,1]` 占位数组。Surfel renderer 不读取它来确定表面方向，也不使用它做光照。

## Sidecar 与纹理

| 文件 | 编码/形状 |
|---|---|
| positions.f32.bin | little-endian Float32 `[P,3]` |
| normals.i8.bin | Int8 `[P,3]`，共用元数据格式所需的占位数组 |
| surfel-u.f16.bin / surfel-v.f16.bin | 各为 little-endian Float16 `[P,3]` |
| surfel-scale.f16.bin | v3 必需；little-endian Float16 `[P,2]`，正有限尺度 [0.5,8] |
| surfel-frame.u8.bin | Uint8 `[P]`，从 0 开始的来源帧编号 |
| surfel-depth.f16.bin | little-endian Float16 `[F,H,W]`，保留完整来源网格，0 为无效 |
| surfel.json | version=3、P、grid_size=[W,H]、每帧相机/映射及纹理信息 |
| surfel-texture-<f>.jpg | 每帧原图缩小后的纹理 |
| manifest.json / objects.json / sku_masterdata.json / thumbs | 复用发布元数据、对象区间、主数据和商品缩略图 |

当前导出器也写入 `colors.u8.bin`，但 Surfel 渲染使用照片纹理。二者共用 manifest schema 3.0.0；显式 render 模式决定是否要求 Surfel sidecar。

Float16 输出拒绝非有限值、超过 65504 的数值和正深度下溢为零。浏览器以 Uint16Array 保存 half 位模式，U/V 用标记为 `isFloat16BufferAttribute` 的实例属性直接上传 GL_HALF_FLOAT，深度用 HalfFloatType 的 DataArrayTexture，不展开为 Float32。来源帧始终使用 Uint8。

原图必须与缓存记录的 source_size 一致。导出以 LANCZOS 等比缩小，默认最长边 1920、短边最多 1080，不放大；JPEG quality=95、subsampling=0。metadata 记录实际原图及纹理尺寸，以及把 2×3 affine 扩为 3×3 后取逆的 processed_to_source。

导出器把点文件、sidecar、纹理、元数据和缩略图写入同一个临时 generation；完整写入后重命名，并原子更新 CURRENT。失败清理临时目录，不发布半个 generation。

## 浏览器加载与投影

`bootstrap` 默认 render=points，仅接受 points/surfel。Surfel 模式由 `loadViewerBundle` 加载通用数据并跳过 RGB，随后 `loadSurfels` 加载 sidecar、半精度数组和 JPEG。loader 检查 v2/v3、点数、数组长度、有限值、帧编号与图片尺寸；v2 单位尺度，v3 必须提供尺度文件。JPEG 顺序解码，以限制临时 bitmap/canvas 开销。

所有图片放入一个 RGBA DataArrayTexture；不同宽高按最大宽和高补齐。来源深度使用单通道半精度 DataArrayTexture。纹理内存约为 `F*max_width*max_height*4` bytes，CPU/GPU 各保留一份；不能用压缩 JPEG 大小推算显存。

渲染时 model/world_to_view 把几何放到 Viewer 坐标系，但传给来源投影的 vWorld 仍是原始世界坐标。对圆盘上的每个片元：

1. 用该实例的来源帧 E 把世界点变为 sourceCamera，再用 K 得到 processed 像素坐标。
2. 检查 processed 有效域；在完整来源深度网格上采样 d，要求 `abs(d-sourceCamera.z) <= 0.015*d + 0.001`。
3. 用 inverse affine 还原原图坐标并检查原图有效域。
4. 结合 source_size、texture_size 和纹理阵列宽高，把源像素中心映射到缩小的 JPEG 纹理。

因此每个圆盘内部可以显示连续图案。一个 Surfel 使用一个来源相机；多个 Surfel 的颜色在当前视角屏幕上融合，不是对一个 Surfel 进行多相机最佳纹理选择。

## 渲染通道

- 深度通道：打开 depth test/write，由真实 quad 光栅化得到倾斜表面的片元深度，记录当前屏幕最近可见线性深度。
- 颜色通道：关闭 depth test/write，以第一通道限制颜色贡献；只接受 `viewDepth <= frontDepth + 0.001 + 0.5*min(length(U),length(V))`。把纹理 sRGB 转为线性 RGB，按 `exp(-2*(x*x+y*y))` 加权并加法累积 RGB 和权重。
- 输出通道：累计 RGB 除以权重，转换回 sRGB；无贡献像素显示白色。

深度通道使用 Float32 render target，颜色累加用 Float16 target。该分支不调用普通 points 的 EDL/Lambert/fog pipeline，保留照片自身光照。当前是深度约束加权 oriented surfel splatting，未实现完整 EWA 屏幕空间滤波、纹理接缝优化或训练式重建。

## 交互与性能

选择使用独立 Uint8 aSelected 属性：选中时纹理颜色改为紫色，取消只清标记，无需恢复逐点 RGB。可见性 aVisible 与原有对象区间操作共享底层数组；更新版本后同步到实例属性。

点击额外绘制 ID 通道，共用圆盘和来源投影/深度有效性判断，按深度缓冲返回最近表面。slot+1 编码进 RGB，0 留给背景；读取一个像素后转回 slot，再通过 point_ranges 得到全局id。ID 编码为 24 位，当前默认 150 万点低于该范围；代码没有为任意大点数实现更宽 ID。Focus 仍根据所选对象原始点位置计算包围盒。

Surfel 使用合并请求的按需 RAF 循环。相机 change、阻尼、Focus/视角动画、选择、隐藏、半径及 resize 会请求绘制；静止时不继续排帧。普通 points 保持连续循环。dispose 取消待执行帧并释放事件、纹理、实例几何和 render targets。

加载进度包装 fetch 响应流，累计实际读取 bytes、已完成文件数和待完成名称；它不增加重试或自动切换模式。

## 边界与验证范围

Sidecar 接受 1..256 帧（Uint8 编号 0..255）。相机参数通过 16 列 RGBA Float32 纹理逐帧读取，不再使用随帧数增长的相机 uniform 数组。renderer 检查纹理层数、尺寸和浮点 render target 支持。合成 256 帧已验证加载、末帧纹理和点击；实际大图仍受显存与浏览器内存限制，未做多设备矩阵验证。

原图提高的是外观采样密度，几何仍受 DA3 processed grid 和筛选后点数限制。细杆、遮挡边界、跨视角深度误差和曝光差异可能产生空洞/接缝/重影。普通点云、Surfel 两个独立数据根用于显式比较，没有自动降级路径。历史浏览器结果和硬件 GPU 性能不能由本次源码审查或构建推导。
