# 视频1：固定轨迹 surfel / 一致性 surfel / DA3 高斯对照

2026-10-09 收尾清理已删除 `runtime/surfel-comparison/` 下的 `gs-model/`、`gs/`、`deps/` 和 `torch-extensions/`。结果报告及网页回放保留；下文是历史运行记录。重跑高斯对比需重新准备权重、隔离依赖和扩展，并按 [perf 安装说明](../README.md#运行) 准备浏览器工具。

输入固定为 `runtime/reproduce-video1-1fps/frames/0.jpg..30.jpg`，即视频1每秒一帧，分类结果已用于既有149个global_id。复用原有DA3缓存及Viewer包，不重跑分类和匹配。

实验脚本不改变生产 `src/da3_runner.py`、默认推理配置或商品计数。

## 方法与相机

- 基线：既有150万点surfel、原分辨率1080×1920纹理、原shader及默认半径。
- 一致性：保持点位置、纹理、slot、shader完全相同，仅使用`consistency.py`的跨帧可见性过滤。深度误差≤1.5%为支持；后方为遮挡；前方为自由空间冲突。至少3冲突且冲突占比≥75%隐藏，深度边界/无效观测不参与。不试调阈值。
- 高斯：同一DA3 Nested checkpoint、31帧、504推理分辨率、完整SH2预测；没有逐场景训练、裁边、减点或降阶。高斯means和scales乘Nested scale_factor进入米制。相机用基线到新预测的Sim3变换，避免旋转SH造成额外误差。
- 轨迹：31个原相机、30个相邻相机中点插值，以及第15帧相机沿相机右方向−10到+10厘米的9个平移，共70视角。分辨率560×1008，白底。源图按相同affine及像素中心重采样。
- 实验页面：`http://127.0.0.1:5173/comparison/index.html`。这是离线渲染回放，不是已部署的原生高斯交互Viewer。

## 运行

当时从仓库根目录执行Python（uv），依赖隔离在`runtime/surfel-comparison/deps/`。以下命令依赖当时已准备好的基线缓存、权重和输入，不能直接用于清理后的环境。

```bash
UV_CACHE_DIR=/tmp/uv-viewer-cache uv run --no-sync python perf/surfel_compare/prepare.py
OPENBLAS_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-surfel-compare uv run --no-sync python perf/surfel_compare/consistency.py modules/viewer_web/public/data-video1-1fps/runs/112743ae4de5481bb07821432f19b58f runtime/surfel-comparison/consistency
cp runtime/surfel-comparison/consistency/visibility.bin modules/viewer_web/public/comparison/visibility.bin
CUDA_VISIBLE_DEVICES=1 UV_CACHE_DIR=/tmp/uv-gs uv run --no-sync python perf/surfel_compare/gaussians.py --count 31 --output runtime/surfel-comparison/gs
```

在`modules/viewer_web`启动`npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`。从仓库根目录执行：

```bash
node perf/surfel_compare/capture.mjs baseline
node perf/surfel_compare/capture.mjs consistent
CUDA_VISIBLE_DEVICES=0 TORCH_EXTENSIONS_DIR=runtime/surfel-comparison/torch-extensions MAX_JOBS=8 UV_CACHE_DIR=/tmp/uv-gs uv run --no-sync python perf/surfel_compare/render_gs.py
UV_CACHE_DIR=/tmp/uv-viewer-cache uv run --no-sync python perf/surfel_compare/evaluate.py
```

浏览器脚本使用本机缓存的Chromium1234和NVIDIA Vulkan，路径写在脚本内；不是通用部署安装器。当前脚本完整执行70帧，`CAPTURE_LIMIT=1`仅供最小渲染诊断。GS首次CUDA扩展编译与后续渲染分开计时。

## 评估口径

- 原拍摄视角PSNR/SSIM：图像拟合，不是独立测试集、几何真值或文字识别准确率。
- 点击图：surfel保持已有point_ranges；高斯用基线最近表面1.5厘米派生global_id，按每类透明度贡献取主要类别，置信不足记未知。两者来自同一已有pipeline，不是人工标注真值，不能用来声称高斯语义更准。
- 点击对照使用每个已有bbox向内缩20%、排除其他bbox覆盖的像素；报告有标签比例与编号吻合比例。
- 插值和横移没有实拍真值。轮廓、重影、文字可读性通过同步回放和原图人工复核；覆盖损失仅为代理指标，不能当作孔洞/漂移的真实物理误差。
- 浏览器耗时是本机页面导航后加载、解码、上传及首帧；文件页缓存未清空、单次测量。两组顺序运行，微小耗时差异不能当性能提升。
- 旧surfel捕获逐帧计时含可能的前一帧GPU工作，不用于性能排名。修正后额外执行`CAPTURE_LIMIT=0 node perf/surfel_compare/capture.mjs baseline`及`consistent`，加载记录写`*-loading.json`，`firstRenderedMs`不含PNG/编号回读。gsplat为服务器CUDA同步墙钟时间，不换算浏览器帧率。
- 高斯模型文件体积、服务器读取和渲染成本，与回放PNG的网页加载成本分别记录。没有测原生高斯浏览器加载/交互成本。

## 产物

- `runtime/surfel-comparison/`：资源、基线缓存、高斯权重/导出、日志、测量JSON。
- `modules/viewer_web/public/comparison/`：相机、三组图像及点击图、原图参考、中文比较页面。
- `runtime/surfel-comparison/REPORT.md`：本次结果与限制。

此次31帧高斯推理/导出完成后，报告阶段因基线字段名错误退出；已修脚本并从产物恢复报告，没有重跑推理。首次31帧的完整加载耗时与推理显存峰未留存，不能从两帧smoke外推。日志中的4.213秒仅模型前向。

页面验证：`node perf/surfel_compare/check_page.mjs`，检查三组点击、同步跳转与缩放，记录PNG回放加载成本。

页面加载诊断：初始状态显示正在读取的数据文件；JSON与图片请求设30秒上限，失败显示资源路径或脚本错误，避免永久停留在加载状态。
