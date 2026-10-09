# 工作记录

## J1 · 2026-10-09 · 初始设计

核对运行 Docker 打包器和三份 Viewer 源码。选择 GLB required vendor extension 保存原始 Surfel 资源；保留现有 renderer，避免 mesh 烘焙引入视觉变化。现有真实 105 MiB ZIP 可用于 CPU 转换与浏览器验证。尚未修改运行容器或上传 COS。

## J2 · 2026-10-09 · 用户修正目标

用户明确要求通用第三方 GLB 网页直接显示。停止专用扩展任务（子代理确认没有写入实现）。修订为标准 triangle mesh + RGBA atlas + unlit material；静态导出不能复现视角相关 Gaussian 混合。两位 medium 代理完成现有 mesh exporter 与 shader 源码审阅。机器可用 RAM 约395 GiB、磁盘约42 GiB，采用分页 CPU 烘焙及流式 GLB 写入。

## J3 · 2026-10-09 12:22 CST · 导出完成

实现 CPU atlas baker、目录/ZIP读取与标准 GLB writer、独立 CLI 和通用 Three.js 对照页。三位 medium 子代理分别执行实现与交叉审查。修复烘焙跳过点计数处理，限定 DA3，补充不可投影中心处理。相关分组测试共覆盖21个用例（未跑全量回归）。

- 商品3：18,771输入点，18,460保留，311跳过；独立检查311个输入本来即零面积。36,920三角面、1 atlas，3,497,404 bytes，CPU 2.40秒。
- 完整场景：2,363,298输入点，2,350,269保留，13,029跳过；4,700,538三角面、57 atlas，453,281,476 bytes，CPU 245.62秒。
- 报告：`runtime/surfel-glb-20261009/{gid3,scene}-report.json`。当前尚待浏览器实物外观验证，不能据格式导出宣称等效画面。

## J4 · 2026-10-09 12:29 CST · 浏览器验证

原 Chromium 缓存缺少二进制；官方 CDN 下载停滞后，从 npm 镜像取得同一版本临时浏览器。首轮商品对照因把整场景不可见点也送进 SwiftShader 而截图超时；对照页改为只提交所选商品的原始槽位后通过，生产 renderer 未改。

Three.js 原生 GLTFLoader：商品 GLB 正视/30°侧视/来源相机pose截图完成，无 pageerror、失败请求或外部 GLB 资源。对照可见包装纹理、轮廓相近，细小文字锐度与重叠边缘存在差异，未声称像素相等。

Babylon.js 9.30.0：商品和完整场景均完成加载与截图，无 pageerror。完整场景为4,700,538三角面、57个unlit alpha-test材质，全部纹理来自GLB内部。使用SwiftShader，不代表终端硬件GPU性能。完整场景Three.js对照仍运行中。

## J5 · 2026-10-09 · 收尾

完整场景 Three.js 双视图加载进入截图步骤，但截图超过120秒后失败，浏览器由 finally 关闭。保留这一限制，不重跑重型验证。商品正侧视对照与完整场景 Babylon.js 显示是当前外观与可加载性证据；未声称所有网页兼容、逐像素一致或完整场景交互性能达标。导出代码、使用文档、两个真实 GLB 及有效截图已交付；现有运行容器未替换。

## J6 · 2026-10-09 12:40 CST 起 · 缩小文件

用户要求立即减小文件。采用标准 KHR_mesh_quantization 的16-bit几何、分块16-bit索引及255色加透明索引PNG；保留所有三角面与纹理尺寸，无点下采样。新增独立压缩命令，对已有 GLB 后处理。纹理局部试算从1,575,966降至439,709 bytes，alpha完全保留，RGB平均绝对误差约1.88/255。原始文件保留。首轮几何审阅发现索引65535保留值问题，停止完整场景处理，修正后重新生成；不可交付首轮文件。

## J7 · 2026-10-09 · 透明底色修正与精简产物

J6的单一透明索引虽保留alpha，却丢失透明像素底色，Three.js截图出现细黑边。改为128色RGB，每色分别保留透明和不透明索引；增加非黑透明像素测试，重新生成全部精简文件。废弃 `captures-gid3-compact/`，有效截图为 `captures-gid3-compact-fixed/`，正视、侧视无此前黑边，无外部资源加载。几何3个、纹理2个相关用例通过。

- 商品：3,497,404 → 1,691,208 bytes，减少51.6%；36,920三角面不变。
- 整场景：453,281,476 → 205,753,312 bytes，减少54.6%；4,700,538三角面不变。几何141,016,140 bytes，PNG64,588,105 bytes。
- 整场景最大位置分量误差0.00008605原始坐标单位；UV误差0.000007618。使用所需标准扩展 KHR_mesh_quantization，无专有解码器。完整精简场景的Babylon.js加载检查进行中。

12:47 CST：完整精简场景Babylon.js加载及截图通过，报告 `babylon-scene.compact-report.json` 无页面错误，4,700,538三角面、57材质，图片已检查。保留完整场景软件渲染性能限制；本轮文件压缩完成。

## J8 · 2026-10-09 12:50 CST 起 · 减少渲染开销

用户进一步要求同时缩小和提速。只读代理统计发现完整同frame+object的2x2四像素单元有454,654个，可进行保守抽稀。实现显式 `--lod`：仅对完整四点、深度/法线连续、footprint足够的内部单元保留固定左上代表点；其位置、U/V、scale不变。稀疏、未知owner、深度边缘等保留。检查四个中心覆盖不代表所有视角像素覆盖等价，需真实外观验证。

商品候选18,771→8,208输入点，实际7,897 Surfel /15,794三角面，原零面积311个仍跳过。LOD选择器13个用例、导出器13个用例通过；完整场景CPU导出与商品浏览器对照进行中。新增标准Three.js比较工具将以相同800x600、来源帧相机、2帧warmup/5帧同步测量对比compact与fast版本；仅作为SwiftShader端到端帧耗时，不推断硬件GPU性能。

商品正侧视截图完成（`captures-gid3-fast/`），包装轮廓与主要图案接近，未宣称所有视角等价。完整场景选择1,275,558点，烘焙1,262,529 Surfel，13,029无效点跳过；2,525,058三角面、31 atlas。`scene.fast.glb` 为117,164,160 bytes，比compact的205,753,312 bytes减少43.1%。商品fast为801,904 bytes。固定相机渲染对比已启动。

## J9 · 2026-10-09 · 修正计时边界

首轮 `benchmark-fast/` 可证明加载、截图和静态计数，但render+finish中位数仅1.3/0.6ms，不能作为完整软件绘制耗时。Chromium有将WebGL finish改为flush的实现历史（https://codereview.chromium.org/16210006）。改用每帧读回1像素等待绘制完成；修正后仅重测同一对文件，结果另存 `benchmark-fast-readback/`。旧计时不用于提速结论。两版完整场景截图已检查，主体与轮廓相近，细小文字和纹理有所变化。

修正后的固定相机5帧测试完成：compact中位2904.10ms，fast中位1488.60ms，耗时减少48.7%（约1.95倍）；draw calls 170→91，纹理像素235,028,400→126,255,600。两版相机完全相同，无页面错误或失败请求。测量包含1像素读回开销，仅限800×600单一来源视角的SwiftShader结果，不是硬件GPU FPS。最终交付 `scene.fast.glb`、`gid3.fast.glb` 与原文件并存；无生产部署、无后台导出或benchmark遗留作业。

## J10 · 2026-10-09 15:37 CST 起 · Docker scene.glb 输出

用户确认直接输出可下载 scene.glb，复用现有 Viewer 数据。设计为 generation 目录 -> CPU LOD烘焙 -> compact -> 同taskID COS scene.glb，BSON仍仅global_skus。主线程负责共用export_scene_glb入口、临时文件清理及镜像验证；medium代理负责Docker processor/COS上传及文档和最小测试。运行服务当前镜像 global-id-mapping:surfel-adaptive-20260924；尚未替换容器。

15:50 CST：代理未产生代码后停止其写任务，由主线程接管并完成Docker接入。核心新增2项测试通过；Docker新增5项测试通过（先遇到pytest路径/package收集问题，使用importlib导入模式修正测试命令后通过，未更改既有package）。两次docker exec只读检查因自动审批超时未执行；离线docker build首次审批超时，允许的一次重试成功，镜像 `global-id-mapping:surfel-glb-20261009`。上下文为 `runtime/surfel-glb-20261009/docker-output/build`，只加入本任务5个src文件及processor/cos_upload。已提交无网络、无GPU、4CPU/8GiB限制的真实bundle导出验证，输出目录 `docker-output/artifacts`，尚未替换服务或上传COS。

15:55 CST：镜像内真实bundle转换退出0，输出117,164,160 bytes / 2,525,058三角面，export_scene_glb耗时198.54秒。`docker-output/artifacts`仅保留scene.glb和export-report.json，临时baked.glb已清理。处理器在镜像内成功导入；实际COS上传与运行服务切换未执行。

已只读核对运行容器配置：8011:80、GPU DeviceIDs=[2]、只读docker/.env -> /app/.env、无restart policy及CPU/内存限制、shm=64MiB。新镜像可复用配置，但切换会短暂停机；尚未执行，交由用户确认。构建/测试已完成，无导出任务仍在运行。

## J11 · 2026-10-09 16:28 CST 起 · 改为BSON内嵌GLB

用户明确GLB不上传COS，改为Docker内生成后随JSON结果一起bson.dumps导出。移除upload_scene_glb，processor返回global_skus和scene_glb原始bytes，API沿用现有bson.dumps。原Viewer ZIP上传保持不变。更新根README、Docker README和GLB文档及客户端保存示例。最小4项输出/异常/API二进制测试通过；生成算法未改，不重复烘焙。新tag为surfel-glb-bson-20261009，镜像构建和复用真实117MB GLB的BSON边界检查待完成。

16:34 CST：BSON版本镜像构建完成（首次审批超时、允许的一次重试成功）。镜像内调用现有API编码路径，复用117,164,160-byte真实GLB、stub pipeline，得到application/bson响应status=200，BSON共117,164,209 bytes，解码后scene_glb为bytes且与输入一致。报告docker-output/bson-report.json；无实际COS上传、模型推理或运行容器替换。导出与验证作业均已结束。

## J12 · 2026-10-09 17:19 CST · GPU2部署完成

用户授权更新Docker并部署GPU2。新镜像 `global-id-mapping:surfel-glb-bson-20261009` 已替换8011服务，容器名 `global-id-mapping-local`。沿用8011:80、只读docker/.env挂载、无restart policy及默认shm等配置，DeviceIDs=[2]。旧容器停止并改名 `global-id-mapping-local-before-glb-bson-20261009` 保留，未删除。

最小部署检查完成：Uvicorn启动成功；宿主机GET http://127.0.0.1:8011/openapi.json 返回200，schema包含POST /api；容器torch可见唯一RTX4090D，CUDA张量1+1结果2.0。GPU UUID为GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7，与切换前GPU一致。未重复完整模型推理、GLB烘焙或COS上传；复用J11的BSON边界验证。服务持续运行，无验证作业遗留。
