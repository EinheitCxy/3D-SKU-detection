# 通用纹理 Surfel GLB

用户于 12:08 CST 明确：市面上通用 GLB 网页应直接显示纹理表面。此前 required vendor extension 设计取消，未实现。

输出使用 GLB 2.0 standard indexed TRIANGLES + embedded RGBA PNG + TEXCOORD_0 + KHR_materials_unlit。doubleSided、alphaMode MASK、alphaCutoff 0.5。不包含自定义扩展、专用 shader、外部资源 URI。不重新进行 DA3/SAM 推理、不融合改变点位置。

采用 CPU atlas baking：沿用 renderer 的 native footprint 奇异值 clamp、原始 U/V 深度连续性、逐轴 scale、圆盘半径；把 source-frame 投影颜色和 source-depth 裁剪烘焙到每个 Surfel 局部 RGBA tile。小矩形展开成两个三角面，alpha mask 去掉圆盘外和不可信区域。当前视角相关的 Gaussian 累加与归一化无法用静态 GLB 表达，不宣称像素等价。使用现有真实 bundle 做固定相机正侧视对照。

## 实现边界

- `src/surfel_mesh.py`：`bake_surfel_pages` generator，仅 numpy/Pillow/stdlib。keyword args `positions,u,v,scales,frame_indices,depths,frames,read_asset,radius=2.0,tile_size=8,atlas_size=2048,batch_size=1024`。positions/U/V 为 N×3，scales N×2，frame_indices N，depths F×H×W，frames 为 surfel.json.frames，read_asset(name)->bytes。调用者已执行子集选择。每页 yield dict：`positions` float32 [4M,3]，`uv` float32 [4M,2]，`indices` uint32 [2M,3]，`image_png` bytes，`surfel_count` int。UV 直接使用 glTF top-left convention；图片不得上下颠倒。内侧 tile_size×tile_size，外围 1 pixel gutter，atlas page 不超过 atlas_size。各页避免累积全部 RGBA 图。
- `src/surfel_glb.py`：验证并读取 generation dir 或既有 ZIP；解码原数组，调用上述 generator；流式临时 BIN 写入生成 GLB，输出不覆盖已存在文件。`export_surfel_glb(source, output, *, global_id=None, tile_size=8, atlas_size=2048, radius=2.0) -> dict`，CLI `uv run --no-sync python -m src.surfel_glb INPUT OUTPUT.glb [--global-id ID] [--tile-size N]`。全量默认保留所有 Surfel，无静默下采样。选 global-id 仅用于显式商品导出。report 包含点数、三角数、atlas页数、字节数、耗时。允许可选 `--report PATH` 保存报告。
- 每 atlas page 一个 mesh primitive/material/image，合并同一 scene/node；node.matrix 是 manifest.world_to_view 的列优先表示，固定默认 pose。geometry 只存原 world coordinates。
- 背景、颜色均取现有来源照片；本轮交付独立可移植导出命令、实物 GLB 和通用加载器对照，不提前更改 Docker 的生产输出或 Viewer 接口。
- 验证：小型合成 roundtrip 和深度边缘裁剪；真实商品 GLB；完整场景导出和通用 Three GLTFLoader 对照，必要时用第二引擎确认格式可移植性。仅运行直接相关检查。

## 12:40 起的精简版

`src/surfel_glb_compact.py` 对上述GLB后处理：POSITION/UV量化为uint16，POSITION使用8-byte stride；各块节点保存解码平移/缩放并挂在原变换之下。分块uint16索引严格避开保留值65535。声明标准 `KHR_mesh_quantization` 为 required，不增加专用解码器。全部三角面保留，报告记录几何误差。

`src/surfel_glb_texture.py` 把每页转为128色RGB，每种颜色各有透明及不透明索引的PNG。保留透明像素底色，防止标准双线性采样产生黑边；alpha和纹理尺寸不变，RGB有损。精简PNG仍在GLB内。源文件保留，输出必须不存在。

## 12:50 起的渲染减量版

导出器增加显式 `--lod`，经 `src/surfel_lod.py` 对同frame、同object的完整2×2像素单元选择固定左上代表点。仅通过深度、法线、scale、邻域源深度连续性及有界ellipse中心覆盖检查的单元允许减量；保留其余点及所选点原属性。当前仅支持radius=2，不改变默认全量导出。选择器输出索引及单元计数，导出报告分开记录LOD去点和无效烘焙。

LOD输出再运行compact命令得到 `.fast.glb`。减量会同时减少三角面和atlas数量，具有有损LOD的细节/视角限制。性能对照使用原生GLTFLoader、相同来源相机与800×600画布，SwiftShader render+单像素readPixels完成时间（含读回开销）；不承诺硬件GPU帧率。早期finish计时不足以等待实际绘制，不作性能证据。
