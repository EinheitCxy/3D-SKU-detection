# 通用网页可加载的 Surfel GLB

将现有 **DA3 Surfel bundle** 导出为自包含的标准纹理 GLB。普通 glTF 2.0 加载器只需支持 Khronos `KHR_materials_unlit`，不需要本项目的 renderer、shader 或外部纹理文件。

```bash
uv run --no-sync python -m src.surfel_glb \
  path/to/viewer_bundle.zip Output/shelf.glb \
  --report Output/shelf-export.json
```

输入也可以是包含 `manifest.json`、`surfel.json` 和二进制/图片资源的 generation 目录。支持现有 Surfel v2/v3；不运行 DA3、SAM 或其他 GPU 推理。输出路径必须尚不存在。

只导出一个商品：

```bash
uv run --no-sync python -m src.surfel_glb \
  path/to/viewer_bundle.zip Output/product-3.glb --global-id 3
```

## 文件内容与效果

每个有效 Surfel 展开为两个三角面；原照片投影颜色、圆盘轮廓及来源深度裁剪被烘焙到内嵌 RGBA PNG atlas。材质使用 `alphaMode: MASK`、双面和 `KHR_materials_unlit`，保留照片自身颜色。场景方向来自原 manifest 的 `world_to_view`。

CPU 烘焙沿用当前 Viewer 的 footprint 限制、深度连续性和 source-depth 阈值。默认 `--radius 2.0` 对应当前 Viewer 初始圆盘大小；`--tile-size 8` 表示每个 Surfel 的有效纹理 tile 为 8×8 像素，另有透明 gutter。可显式增大 tile size 改善局部纹理和裁剪采样精度，但 atlas 显存随 tile 边长平方增长。`--atlas-size` 默认 2048。

默认遍历全部输入点，无静默下采样。完全透明、不可投影或零面积 Surfel 不产生三角面；报告分别记录输入点数、保留和跳过数量、三角数、atlas 页数、GLB 字节数及导出时间。跳过数量是本次有限分辨率烘焙的结果，不能当作原始场景无效点数量。

静态 GLB 用普通深度测试决定可见面，不能复现当前 Viewer 随观看视角变化的 Gaussian 累加与归一化。因此重叠处颜色、边缘和细小空隙可能有差异；本功能不承诺逐像素一致。纹理烘焙还会增加几何、纹理和加载内存。当前只支持 DA3 的像素坐标约定，其他 backend 明确报错。

此 GLB 交付三维外观，不包含当前 Viewer 的 SKU 列表、观测缩略图或 Global ID 点击逻辑。Docker 也可复用相同数据生成减量版 `scene.glb`，见下节；BSON 成功响应包含 `global_skus` 和二进制 `scene_glb`。

## Docker 输出

Docker 服务在现有 Viewer generation 目录生成后，调用 `src.surfel_glb.export_scene_glb(generation_dir, output_path)`，复用位置、Surfel U/V/scale、来源深度、相机、照片和商品点范围；不重新推理，不解压或复制另一套源数据。

GLB 在 Docker 请求临时目录内生成，以原始 `bytes` 放入响应的 `scene_glb` 字段，由现有 `bson.dumps` 编码为 BSON binary；不使用 Base64，不上传 COS，也不塞入 ZIP。成功响应为 `{"global_skus": [...], "scene_glb": <binary>}`；原 Viewer ZIP 的 COS 上传保持不变。普通加载器需支持标准 `KHR_materials_unlit` 和 `KHR_mesh_quantization`。

该入口固定使用上一轮减量方案：半径2.0、8×8纹理tile、2×2保守LOD、16-bit几何和128种RGB索引PNG。中间和最终GLB文件均随请求临时目录清理；最终内容在清理前读成bytes返回。导出、读取或原Viewer ZIP上传失败均抛出错误。BSON编码和客户端下载会完整持有GLB，响应大小与内存占用随文件大小增加。

这是额外的CPU步骤，会延长同步 `/api` 请求；当前约236万点样例的LOD烘焙曾耗时约155秒，之后还需压缩和传输响应，不能视为固定时限。实际文件大小随场景变化；117.16 MB仅为现有样例。运行容器须更新后，新任务才会生成该文件；旧任务不会自动补齐。

生成算法此前已在镜像 `global-id-mapping:surfel-glb-20261009` 中验证：在无网络、无GPU、4CPU/8GiB限制下用真实bundle完成导出：117,164,160 bytes，198.54秒（烘焙和压缩合计，不含上传）。产物与报告在 `runtime/surfel-glb-20261009/docker-output/artifacts/`。BSON改造未更改该生成算法；原tag是此前COS输出版本，当前BSON版本使用 `global-id-mapping:surfel-glb-bson-20261009`。BSON版本镜像已构建；镜像内复用该真实GLB，经过现有API的 `bson.dumps` 与 `bson.loads` 后内容一致，测试响应117,164,209 bytes（SKU结果为stub）。报告为 `docker-output/bson-report.json`。2026-10-09 17:19 CST已部署到8011，固定宿主机GPU2。宿主机HTTP检查200，容器单GPU可见且微型CUDA运算通过；未重复完整模型推理或COS上传。旧容器 `global-id-mapping-local-before-glb-bson-20261009` 停止保留。

客户端收到响应后直接保存：

```python
from pathlib import Path
import bson

result = bson.loads(response.content)
Path("scene.glb").write_bytes(result["scene_glb"])
# result["global_skus"] 仍是原来的逐帧 JSON 字符串数组。
```

## 缩小已有 GLB

对上面命令生成的文件执行：

```bash
uv run --no-sync python -m src.surfel_glb_compact \
  Output/shelf.glb Output/shelf.compact.glb \
  --report Output/shelf-compact.json
```

精简版保留所有三角面和纹理尺寸，使用16-bit位置/UV及分块16-bit索引，并把PNG改为128种RGB颜色、每种颜色保留透明/不透明两种状态的索引图片。透明像素仍保留颜色，避免双线性采样产生黑边。颜色量化是有损的；`--keep-textures` 可以只量化几何。报告记录实际文件大小以及位置和UV最大误差（位置为原始局部坐标各分量误差）。

精简版要求加载器支持标准 `KHR_mesh_quantization`，不需要额外压缩解码器。原始文件不覆盖。PNG体积下降不代表GPU纹理内存按同样比例下降，三角面数量也没有减少。

## 同时减少几何与渲染开销

```bash
uv run --no-sync python -m src.surfel_glb \
  path/to/viewer_bundle.zip Output/shelf.lod.glb --lod
uv run --no-sync python -m src.surfel_glb_compact \
  Output/shelf.lod.glb Output/shelf.fast.glb
```

`--lod` 显式启用保守的2×2来源像素抽稀，仅支持默认 `--radius 2.0`。在同来源帧、同商品、四点齐全且深度/法线连续的区域，检查有界footprint覆盖四个原始中心后，保留固定左上角的既有Surfel。其几何位置及footprint不变；不完整单元、未知商品归属、过小footprint及深度边缘保留。报告将 `lod_removed_point_count` 与烘焙无效的 `skipped_surfel_count` 分开。

该选项减少三角面、atlas像素及重叠绘制，有利于缩小文件与减少渲染工作量。但中心覆盖检查不等价于所有视角覆盖一致，细薄结构、近距离细节仍须检查；这是有损LOD选项，默认不启用。

本地完整场景 `scene.fast.glb` 为117.16 MB、约253万三角面、31张atlas；相对 `scene.compact.glb` 的205.75 MB、约470万三角面、57张atlas，文件减少43.1%，面数减少46.3%。商品 `gid3.fast.glb` 为0.80 MB，正侧视截图位于 `captures-gid3-fast/`。

同一来源相机、800×600、2帧预热和5帧测量的SwiftShader测试中，compact/fast含像素读回的帧耗时中位数为2904.10/1488.60ms（减少48.7%），draw calls为170/91。有效报告在 `runtime/surfel-glb-20261009/benchmark-fast-readback/benchmark.json`。这是单一视角的软件渲染结果，不能换算为用户设备的硬件GPU FPS；旧 `benchmark-fast/` 的finish计时不足以等待绘制完成，不用于性能结论。

可用 `scripts/3d/visualization/benchmark_glb_pair.mjs` 比较旧、新GLB：参数依次为Vite根URL、旧GLB URL、新GLB URL、输出目录、可选相机JSON。相机JSON包含 `position`、`target`、`up` 及可选 `fov`。工具固定同一相机和800×600画布，每文件2帧预热后测5帧 `renderer.render + readPixels`（读回1像素以等待绘制完成），记录含读回开销的帧耗时、draw calls、三角面及纹理像素。当前脚本使用SwiftShader，不能据此声称真实GPU帧率。

真实样例：`gid3.compact.glb` 为1.69 MB（减少51.6%），`scene.compact.glb` 为205.75 MB（减少54.6%）。商品精简版的Three.js正侧视截图在 `runtime/surfel-glb-20261009/captures-gid3-compact-fixed/`；完整精简场景已通过Babylon.js加载与截图检查，见同目录 `babylon-scene.compact-report.json` / `babylon-scene.compact.png`。完整场景仍含约470万三角面，文件变小不等于交互开销同等下降。

## 本地对照

`modules/viewer_web/surfel-glb-compare.html` 的左侧使用现有 Surfel renderer，右侧只使用原生 Three.js `GLTFLoader` 和其返回的标准材质，两侧相机同步：

```text
/surfel-glb-compare.html?bundle=<原bundle根URL>&glb=<导出GLB的URL>
```

`bundle` 使用原来的 `CURRENT` / `runs/<run_id>/` 布局；单商品对照增加 `&global_id=3`。跨端口资源服务需允许 CORS。浏览器打开该对照页时会额外加载左侧基线；交付给第三方加载器只需要右侧 GLB。

本机已有 Playwright/Chromium 时，可截图正视、30°侧视和来源相机 pose：

```bash
CHROMIUM_PATH=/path/to/chrome-headless-shell \
node scripts/3d/visualization/capture_surfel_glb.mjs \
  '<对照页URL>' runtime/surfel-glb-captures
```

该截图脚本使用 SwiftShader，仅用于外观和格式检查，不用于证明终端 GPU 性能。

## 已验证样例

本地 `runtime/surfel-glb-20261009/` 包含商品 `gid3.glb`（3.50 MB）和完整货架 `scene.glb`（453.28 MB）。商品在 Three.js 原生 GLTFLoader 与 Babylon.js 中加载显示成功，正视及30°侧视与原 renderer 的对照截图位于 `captures-gid3/`。完整货架在 Babylon.js 中加载显示成功，见 `babylon-scene.png` 和对应报告。

完整货架包含约470万三角面、57张 atlas。其 Three.js 双视图在 SwiftShader 下截图超时，尚未验证完整场景双视图外观一致性或终端交互性能；第三方网站也可能有文件大小或内存限制。相关 Python 单元测试共21个用例通过。
