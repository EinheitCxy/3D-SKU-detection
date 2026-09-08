# Surfel 实施报告

## 本次提交整理验证（2026-09-08）

- CPU 合成检查通过：倾斜面的解析 U/V、断层邻边、全零点在相机平移后仍为无效深度；完整 points/Surfel 导出保持位置及对象区间一致，Surfel 不生成 RGB 文件；v2 半精度长度/深度、inverse affine 和缩小纹理尺寸正确。使用临时目录，结束后清理。
- 在 HEAD 原有前端测试加待提交源码的临时快照中，6 个定向测试文件共 33 项通过：既有 bundle loader、入口、points geometry、选中颜色、拾取区间，加临时 v2 loader、按需重绘和进度流检查。临时测试已删除，不纳入提交。
- 同一待提交前端通过 `npm run build`（TypeScript + Vite）；仍有大于 500 kB chunk 提示。未重新导出真实场景，未重新做浏览器画质/硬件 GPU 性能验证。
- 提交不包含工作区已有测试迁移/删除、Pi3X、评估改动或生成的 public 数据。

## 2026-09-08 紧凑 v2 与按需重绘

本文件保留此前实施时的验证记录，未在本次提交整理中重跑这些真实场景及浏览器检查。当前代码已移除 RGB Surfel 模式，并将新导出纹理限制为 1080p；以下 2048/4096 参数、旧 generation 和 RGB 入口仅为历史记录，不能作为当前使用命令。当前机制与入口以 [集成说明](surfel_implementation.md) 和 [Viewer README](../modules/viewer_web/README.md#深度约束-surfel) 为准。

- U/V 与源深度改为 little-endian Float16，frame 改为 Uint8；loader 仅接受 v2，GPU 属性与源深度纹理保持紧凑存储。点位、slot 顺序、ID 和纹理分辨率不变。
- Surfel 改为按需重绘；相机 change、阻尼、Focus/视角动画、选择、隐藏、point size、resize 唤醒；dispose 取消待执行帧。普通 points 继续原连续绘制。
- 使用上文相同缓存参数重新导出；CURRENT 为 `a3b4a1152c3046589d627ff13a2d1e71`，1,367,432 点、493 thumbnails。旧 generation 未删除。
- 单 generation：89,461,691 → 64,758,951 bytes，减少 24,702,740 bytes（27.61%）。U/V 合计 16,409,184 bytes，frame 1,367,432 bytes，depth 4,191,264 bytes。
- 逐元素比较旧/新编码：frame 完全一致；U/V 最大单分量误差分别 0.0000305176 / 0.0000300407；depth 最大绝对误差 0.00780296 scene units、最大对应可见性容差占比 3.2346%，无有效深度变为 0。不是所有边缘可见性判定一致的证明。
- 合成 Python 导出检查：解析切向量、v2 文件宽度及深度值通过。前端定向 Vitest 3 项通过：v2 解码/半精度纹理与非法值拒绝、静止/动画/唤醒/释放、绘制期间阻尼继续请求帧。
- 合成浏览器检查通过：实际 GL_HALF_FLOAT 与 Uint8 属性绑定、静止停绘、选择/大小/隐藏/resize/Focus 唤醒、GPU ID 点选为 1、动画结束及 dispose 后不继续绘制；console/pageerror 为空。
- 独立只读代码审查未发现 critical/important 问题，核对了实际 Three.js 半精度属性和 instancing 绑定。
- 导出后重新 `npm run build` 通过（保留既有大于 500 kB chunk 提示）；dist 包含新 CURRENT 和 v2 数据。部署需一起发布新前端与新数据，未操作远端 nginx。
- 完整场景自动截图在 SwiftShader 环境超时，不视为截图/画质通过，也不作为硬件 GPU 帧率或提速数据。
- 完整 v2 场景再次仅检查加载/停绘：商品 UI 加载成功，实例 draw 调用计数停在 4（初始两帧、每帧深度与颜色两次），静止超过 1 秒不再提交；console/pageerror 为空。测试设备为 ANGLE SwiftShader。临时 Vitest 和浏览器脚本验证后删除。

## 已实现

新增独立 `data-surfel` bundle；现有 schema3 与默认 points 模式保持可用。新模式保留原采样点顺序及 `point_ranges`，每点新增深度网格切向量 U/V、来源 frame index。网格邻接边超过源深度尺度 3% 时禁止使用；缺失/全零 world point 的切向量和深度无效。

浏览器使用有方向的平面圆盘（三角形实例），由光栅化产生真实逐片元平面深度。先写最近可见线性深度，再对容差内片元累加高斯权重线性 RGB，归一化后输出 sRGB。当前是 **depth-weighted oriented surfel splatting**，不声称实现完整 EWA 屏幕空间协方差滤波或论文训练流程。

纹理模式逐片元执行源相机投影与 `source_to_processed_affine` 逆变换，采样原图 JPEG 纹理；processed/source 有效区域和源相机深度检查作用于深度、颜色和 GPU ID 通道。原图不额外打光，不加旧点云 EDL/fog。批量选中染色、隐藏、实际表面点选和 Focus 接入既有对象范围。

## 文件

- `src/surfel_export.py`：网格切向量、原图缩放、source projection metadata 与 sidecar 序列化。
- `src/web_viewer_export.py`：保留最终采样 source index；与既有 atomic generation 同时发布 Surfel sidecar。
- `scripts/export_surfel_viewer.py`：显式独立导出，支持 `--mask-cache-root`、`--texture-edge`。
- `modules/viewer_web/src/surfel-loader.ts`：sidecar shape/finite/frame/texture dimension 检查，原图纹理与深度阵列加载。
- `modules/viewer_web/src/surfel-renderer.ts`：深度、加权颜色、归一化与实际 Surfel ID picking，资源释放。
- `modules/viewer_web/src/{bundle-loader,main,scene}.ts`：显式 render 模式接入和既有交互集成。
- `modules/viewer_web/README.md`：入口、运行命令、尺度、纹理资源和局限。

## 已运行验证

- 合成倾斜平面：U/V 与解析单像素切向量一致。
- 深度断层：切向量不使用跨前后景长边。
- 全零 world point + 正相机平移：source depth 保持无效零值。
- 带缩放/平移 affine：逆投影得到预期源图坐标；记录真实缩放图像尺寸及 sidecar slot shape。
- `npm run build`：通过 TypeScript 和 Vite build；仅有既有类 bundle size advisory。最后构建入口 `index-_dvBr6vp.js`。
- 真实 floor_display6：导出成功，未重跑 DA3、matching 或 mask model。
- 最小合成验证以内联 `uv run --no-sync python` 运行；临时图片和 cache 使用 `TemporaryDirectory`，结束自动删除，没有新增临时测试文件。

浏览器近景/斜视与交互质量检查由主线程独立执行，后续结果补入本报告；上述构建和合成验证不替代实际浏览器画质验收。

### 主线程浏览器验收

Chromium 实际加载完整 1,367,432 点纹理版 bundle，通过 WebGL 编译与渲染；控制台和 pageerror 均为空。已查看全景和拉近截图，原图纹理可见，商品轮廓与部分表面仍有缺口，不能声称消除全部孔洞。截图保存在 `runtime/surfel-preview/`。

自动浏览器探测到 SwiftShader 软件 WebGL，尝试 EGL/GL 后仍为 SwiftShader。连续渲染截图曾超过 60 秒；随后测试 harness 限制动画帧才完成截图。这不是硬件 GPU FPS。测试 harness 的帧限制未写入产品代码。一次脚本时间探针没有确认待执行绘制回调，丢弃其结果，不作为单帧耗时。

RGB 对照截图已完成；实际 Surfel GPU ID 点击在场景中选中 Global ID 111（image 5 / object 46）。点选会切到 Global ID 面板，首次筛选脚本因此误点隐藏 SKU 按钮并超时；定点重跑先切回 SKU 后通过，无浏览器错误。截图确认 SKU 56642 的 65 个 Global ID 被选中并在实际 Surfel 上高亮（既有 SKU 交互是筛选列表＋高亮，不是隐藏所有其他商品）。相关截图见 `runtime/surfel-preview/surfel-filter.png`。

## 实际导出

```bash
UV_CACHE_DIR=/tmp/uv-viewer-cache uv run --no-sync python scripts/export_surfel_viewer.py \
  --dataset imdata/floor_display6 \
  --cache Output/floor_display6/da3_cache/predictions.npz \
  --mapping Output/floor_display6/dedup_detections/global_mapping.json \
  --mask-cache-root /tmp/viewer-da3-masks-5e595nvt/v2 \
  --output modules/viewer_web/public/data-surfel
```

发布 generation：`fe3a929d306a4423a2182892bcafbbb3`。共 1,367,432 surfel slots（旧基线同点数）、493 thumbnails。新 bundle 共 89,461,691 bytes；旧基线 26,850,123 bytes。原图最长边默认 2048；提高分辨率追加 `--texture-edge 4096`，内存约四倍。mask 临时归档路径只对本次环境有效。

11 帧实际纹理全部为 2048×1536；JPEG 文件合计 15,934,060 bytes，解码 RGBA 阵列为 138,412,032 bytes（132 MiB，CPU/GPU 各一份，另计几何与 render targets）。

| 入口 | 用途 |
|---|---|
| `/?data=/data/&render=points` | 原有 points 基线 |
| `/?data=/data-surfel/&render=surfel` | 高分辨率逐片元原图 |
| `/?data=/data-surfel/&render=surfel-rgb` | 同几何与融合，每点 RGB |

## 局限与取舍

没有提高 DA3 深度分辨率，原图纹理不能恢复未观测/错误几何。多视角深度误差、曝光差异、薄物体和遮挡轮廓仍可能表现为重影/接缝/缺口；未进行纹理接缝优化、几何融合或学习式优化。覆盖为源像素切向量圆盘，未实现完整屏幕空间 EWA 或 mipmapped 各向异性纹理过滤。

使用 WebGL2 + `EXT_color_buffer_float`；能力不足直接报错。三通道渲染和额外纹理增加显存/带宽；SwiftShader 检查用于浏览器正确性，不代表硬件 GPU FPS。取消 Viewer 时释放纹理、全部 render targets 和 Surfel geometry。默认 points 独立保留用于比较，没有自动切换算法。
