# 单文件 Surfel GLB

- 目标：通用 GLB 网页直接打开，尽量保持当前 Surfel 纹理表面外观。
- 阶段：标准纹理 mesh 导出、真实文件生成及最小验证已完成。
- 范围：CPU Surfel atlas baking、标准 GLB 导出命令、真实样例与通用加载器对照。
- 验收：标准自包含 GLB；纹理/深度裁剪；真实数据正侧视对照；相关最小测试。
- 部署：2026-10-09 17:19 CST，新BSON镜像已部署到8011，固定宿主机GPU2；旧容器停止并保留。
- 设计：[FORMAT.md](FORMAT.md)。

## 产物与验证

- `runtime/surfel-glb-20261009/gid3.glb`：3,497,404 bytes，36,920 三角面；Three.js 与 Babylon.js 均成功加载显示。Three.js 正视、30°侧视和来源相机 pose 对照截图位于 `captures-gid3/`。
- `runtime/surfel-glb-20261009/scene.glb`：453,281,476 bytes，4,700,538 三角面；Babylon.js 成功加载显示，截图及报告为 `babylon-scene.png` / `babylon-scene-report.json`。
- 21 个相关 Python 用例通过；两个新增 mjs 语法检查通过。未跑全量回归。
- 完整场景 Three.js 双视图在 SwiftShader 下截图超过120秒而失败；未验证完整场景双视图外观一致性或硬件交互性能。
- 静态 alpha-test mesh 近似原 Surfel 外观，不复现视角相关 Gaussian 混合。当前支持 DA3；完整场景文件较大。

## 精简版（后续请求）

- `scene.compact.glb`：205,753,312 bytes，减少54.6%；`gid3.compact.glb`：1,691,208 bytes，减少51.6%。原始文件保留。
- 全部三角面保留。16-bit几何和索引、带透明底色的索引PNG；需要标准 KHR_mesh_quantization 支持。
- 新增5个相关用例通过。商品精简版Three.js正侧视通过，有效截图为 `captures-gid3-compact-fixed/`；完整精简场景Babylon.js加载与截图通过，无页面错误，见 `babylon-scene.compact-report.json` / `babylon-scene.compact.png`。软件渲染检查，不作交互性能结论。

## LOD减量版（最新请求）

- `scene.fast.glb`：117,164,160 bytes、2,525,058三角面、31 atlas；比compact再小43.1%，面数减少46.3%。`gid3.fast.glb`：801,904 bytes。
- 显式 `--lod` 保守抽稀连续内部区域，再运行compact。原始文件和默认导出不变。
- LOD选择器13项、导出器13项检查通过；商品Three.js正侧视通过，见 `captures-gid3-fast/`。完整场景同相机800×600、5帧软件渲染+单像素读回：中位2904.10→1488.60ms（减少48.7%），draw calls 170→91，无页面错误。有效报告 `benchmark-fast-readback/benchmark.json`；旧 `benchmark-fast/` 的finish计时不作性能证据。
- 当前导出和性能检查均已结束。实测仅覆盖单一来源视角和SwiftShader，保留有损LOD细节差异及硬件性能未验证边界。


## Docker BSON 输出（2026-10-09 16:28 后）

- 用户更正：GLB只在Docker内生成，作为BSON binary返回，不上传COS。当前响应为 `global_skus` + `scene_glb`；原Viewer ZIP的COS流程保留。
- 已移除新增GLB上传函数；Docker `processor.py`读取最终GLB为bytes，在临时目录清理后由现有 `api.py` 执行 `bson.dumps`。
- 4项针对性检查通过：成功返回、GLB生成失败、原ZIP上传失败、HTTP BSON二进制编码。GLB生成算法未改，复用已有镜像内117,164,160 bytes / 198.54秒的真实导出证据。
- 新镜像 `global-id-mapping:surfel-glb-bson-20261009` 已构建；镜像内API编码117,164,160-byte真实GLB，BSON响应117,164,209 bytes，解码后bytes完全一致。使用stub pipeline复用既有文件，不声称完整推理端到端重测；报告 `runtime/surfel-glb-20261009/docker-output/bson-report.json`。原 `surfel-glb-20261009` 是先前COS输出版本，不是当前交付版本。
- 8011现运行 `global-id-mapping:surfel-glb-bson-20261009`，容器 `global-id-mapping-local`，DeviceIDs=[2]。宿主机HTTP检查200，容器可见1张RTX4090D，微型CUDA运算通过；未重新运行完整模型推理或COS上传。旧容器 `global-id-mapping-local-before-glb-bson-20261009` 已停止保留。部署记录见J12。
