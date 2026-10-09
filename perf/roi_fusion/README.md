# Video3 单商品固定相机融合对照

隔离实验：使用 video3 的504分辨率 DA3 缓存、SAM masks 和 gid=3 的五帧观测（0、1、26、27、28）。不重新推理，不修改相机、匹配、计数或生产 Viewer。原始输入位于 `runtime/video3-resolution-review/504/outputs/dataset`，原图位于其同级 `dataset/images`。

比较原始多帧 Surfel、跨帧融合＋固定表面块主纹理、TSDF 网格＋固定表面块主纹理。两类 Surfel 都使用半径1.05的八边形圆盘和无光照 GLB 材质，**并非 production splatting 的逐像素复现**。基线保留每个原始 Surfel 的来源帧，候选按表面块选择固定来源。因此外观变化是几何与重新选纹理的共同结果。

## 重现

从仓库根目录执行，使用已有 Python 环境和已安装 Open3D/trimesh，无需下载模型：

```bash
export UV_CACHE_DIR=/tmp/uv-roi
uv run --no-sync python perf/roi_fusion/prepare.py
uv run --no-sync python -m perf.roi_fusion.fusion
uv run --no-sync python -m perf.roi_fusion.tsdf
uv run --no-sync python perf/roi_fusion/evaluate.py runtime/roi-fusion-video3-gid3
```

默认输出为 `runtime/roi-fusion-video3-gid3`，重复执行会更新该实验目录内的对应产物。不会更改旧 comparison、输入缓存或生产 bundle。`prepare.py --gid 19 --output <新目录>` 可显式选择另一商品；后续命令也须显式传入对应 input/output，不自动换样本。

两个终端分别启动仅本机可访问的服务：

```bash
UV_CACHE_DIR=/tmp/uv-roi uv run --no-sync python perf/roi_fusion/serve.py
```

```bash
cd modules/viewer_web
npm run dev -- --host 127.0.0.1 --port 5176 --strictPort
```

打开 `http://127.0.0.1:5176/roi-fusion.html`。远程使用时转发5176和8767两个端口；不必将图片目录开放到局域网。

截图与交互验证（使用已有本地 Chromium/Playwright，默认软件渲染）：

```bash
node perf/roi_fusion/capture.mjs 'http://127.0.0.1:5176/roi-fusion.html?assets=http://127.0.0.1:8767/'
```

共享观察位置包括来源0、前两来源相机位置中点、正面、30度与60度侧面。来源预设仅复用相机位置并朝向商品，不是原始内外参下的图像重建；不计算对原照片的 PSNR。支持三列同步旋转、缩放、来源帧着色和共同场景背景。

## 参数与限制

- 融合仅接受同商品的双向投影对应，通过法线、局部尺度、完整两两距离与每帧唯一性检查；保留无支持观测。真实 normals 来自 U×V，同步重建切向量。具体阈值写入 `fused-stats.json`。
- TSDF voxel 为有效网格邻点距离中位数（本样本3.465mm），截断距离为4倍（13.861mm），输入 camera-Z 为米，使用原始 w2c。ROI 外深度无效，不自动补洞。
- 三种方法使用相同的纹理有效性规则：三角形三个顶点及中心均需通过完整场景深度和商品 mask 检查。灰色面是没有有效主纹理来源的面，不等于没有几何。
- 候选每块选择一个固定来源，没有有效来源的面不自动切到其他来源；纹理上限1920长边/1080短边，无放大。UV 使用原始 affine 的逆与像素中心约定。
- `geometry-metrics.json` 对三组表面各按面积采样100000点，seed42；比较同一主法向方向上、共同参考格内的P90−P10分散。它是内部几何代理，不是真实物理厚度。参考格命中率也不是像素/表面积覆盖率；删掉几何可以降低分散，必须结合截图观察。
- source-color 显示最终纹理来源；融合成员关系另存 `fused.npz`，TSDF 没有单一原始观测归属。
- 本轮 CPU 进行几何处理，SwiftShader 验证浏览器。GPU2 已获授权但未使用；CUDA_VISIBLE_DEVICES 并不能保证 Vulkan 浏览器只使用 GPU2，因此没有把浏览器切到不确定的 GPU。

## 最小验证与产物

```bash
UV_CACHE_DIR=/tmp/uv-roi uv run --no-sync python -m pytest -q perf/roi_fusion/test_fusion.py perf/roi_fusion/test_texture.py
```

测试覆盖重复平面融合、真实双面/遮挡/跨商品边缘保护、非传递链式误合并、切向量、原来源保持、UV及GLB材质。运行结果、局限和截图见 `runtime/roi-fusion-video3-gid3/REPORT.md`；`captures/browser-report.json` 记录实际浏览器 renderer、视角与交互检查。

## 第二轮：TSDF 与贴图分开探索

保留首轮产物，新输出在 `runtime/roi-fusion-video3-gid3-v2/`。

```bash
export UV_CACHE_DIR=/tmp/uv-roi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
uv run --no-sync python -m perf.roi_fusion.probe_geometry
uv run --no-sync python -m perf.roi_fusion.probe_geometry --silhouette-only
uv run --no-sync python -m perf.roi_fusion.probe_texture
uv run --no-sync python -m perf.roi_fusion.probe_texture \
  --mesh runtime/roi-fusion-video3-gid3-v2/geometry/trunc8/tsdf.npz \
  --output runtime/roi-fusion-video3-gid3-v2/texture-trunc8 \
  --truncation 0.02772136963903904
uv run --no-sync python -m perf.roi_fusion.prepare_probe_views
uv run --no-sync python perf/roi_fusion/serve.py \
  --root runtime/roi-fusion-video3-gid3-v2 --port 8768
```

复用5176的Viewer，分别打开：

- 几何：`http://127.0.0.1:5176/roi-fusion.html?assets=http://127.0.0.1:8768/geometry-view/`
- 贴图：`http://127.0.0.1:5176/roi-fusion.html?assets=http://127.0.0.1:8768/texture-view/`

远程使用转发5176、8768。截图命令沿用 `capture.mjs '<URL>' '<输出目录>'`，`ROI_CAPTURE_PRESETS=front,side60` 只截取两种视角。

几何探针只改变 `sdf_trunc` 为4/8/12倍spacing；4倍直接复用原结果。相机、输入、体素、贴图不变，没有补洞或删除碎片。统计来自NPZ原始面序，旧报告从GLB按材质重排后的面序采样，不能把两轮seed42视为同一组采样点。

贴图探针固定几何，使用 mesh 邻接图和 Potts 接缝代价选择固定来源，以确定性 ICM 局部优化。这是小范围实现，不是 MVS-Texturing 论文复现或 graph-cut 全局优化。`strict` 保留原深度/mask/朝向条件；`mesh` 同时增加首交面检查并允许一倍TSDF截断距离内的深度偏差，两项是联合策略。未建模且mask未排除的遮挡物无法保证被检测。没有可用来源的面保持灰色，没有平均混合纹理或自动补洞。

射线使用朝中心内缩1%的三个近顶点和中心，要求首交 primitive ID 等于目标面，另保留距离检查；实际顶点仍用于 mask/depth 检查。较小的0.01%内缩在本样本中出现 float32 邻面误命中，诊断记录于 `ray-diagnostic.json`。少量采样不能保证三角形内部每个像素均无遮挡。

本轮改动的最小测试：

```bash
uv run --no-sync python -m pytest -q perf/roi_fusion/test_texture.py perf/roi_fusion/test_probe_texture.py
```

具体结果和限制见 `runtime/roi-fusion-video3-gid3-v2/REPORT.md`。

## 整个货架对比

`shelf.py` 读取 video3 全部29帧和全画面深度，不再限制 gid=3。
共同有效区域要求 confidence≥1.1、有限正深度和有效切向量；不套用生产
Viewer 的商品保护、背景体素过滤或去地面。Surfel 在原504网格每4像素
采样并扩大显示圆盘；TSDF 使用完整有效深度，voxel为原网格中位spacing，
截断8倍spacing。因此这是整场景视觉预览，不是两算法同密度质量基准。
TSDF 原始网格保存为 `tsdf-full.npz`；网页网格先以 quadric decimation
简化到最多100万面，再选主纹理，保留灰色未贴图面。Surfel 的显示尺寸
只在导出圆盘时放大，融合本身始终使用原始像素切向量。
融合不使用商品ID约束，仍保留双向投影、法线和距离检查。纹理仍使用原图。

```bash
export UV_CACHE_DIR=/tmp/uv-roi
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
uv run --no-sync python -m perf.roi_fusion.shelf --stage prepare
uv run --no-sync python -m perf.roi_fusion.shelf --stage baseline
uv run --no-sync python -m perf.roi_fusion.shelf --stage fused
uv run --no-sync python -m perf.roi_fusion.shelf --stage tsdf
uv run --no-sync python -m perf.roi_fusion.shelf --stage finalize
```

复用5176 Viewer与8768产物服务，打开
`http://127.0.0.1:5176/roi-fusion.html?assets=http://127.0.0.1:8768/shelf-view/`。
三列分别为原始全场景Surfel、融合Surfel、TSDF重新贴图，均为实际全场景资产。
`--stride` 和 `--confidence-min` 仅在prepare阶段生效；修改后需重新执行后续阶段。

### 英文双屏预览与资产 ZIP

启动上面的 5176 Viewer 与 8768 产物服务后，打开
`http://127.0.0.1:5176/shelf-split.html`。左侧是跨帧融合 Surfel，右侧是
TSDF 网格；两侧加载同一组 29 帧生成的全货架 GLB，并同步旋转、缩放、
平移与视角预设。页面 UI 为英文，不读取 `masterdata.json` 或厂商数据。
这仍是独立实验页：Surfel 是 GLB 圆盘渲染，未应用生产 Viewer 的商品保护、
背景过滤或 SKU/Global ID 交互。`capture.mjs` 也可用于双屏页：

```bash
ROI_CAPTURE_PRESETS=front,side60 node perf/roi_fusion/capture.mjs \
  http://127.0.0.1:5176/shelf-split.html \
  runtime/roi-fusion-video3-gid3-v2/shelf-view/split-captures
```

`shelf-view/shelf-split-assets.zip` 是仅供该实验页解压并静态托管的资产包，
包含 `fused.glb`、`tsdf.glb`、`meta.json` 和 `observations.jpg`；不包含
`baseline.glb`、`tsdf-full.npz`、推理缓存、源码或主数据。ZIP 使用 deflate
压缩；在线实验页直接请求两个 GLB，不会下载或解析此 ZIP。它不是生产
`viewer_bundle.zip`，当前 Docker Viewer 的 task ZIP loader
不接受 TSDF GLB。实测包成员、压缩后大小和一次打包耗时见同目录的
`shelf-split-package.json`。`fused-stats.json` 与 `tsdf-stats.json` 的阶段时间
只覆盖各自的导出调用；没有 prepare、DA3 推理或 ZIP 传输计时。
一次本机 SwiftShader 页面加载计时单独记录在
`split-captures/load-timing.json`，不能用作海外网络加载时间。
