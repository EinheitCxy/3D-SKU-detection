# GPU2 最新完整 profiling

2026-10-18 CST。用户授权：在 GPU2 对最新代码非常细致地 profiling，替换旧主 profiling，记录每个小步骤的实现。仅本地测量和文档，无推送、部署、COS 上传或其他 GPU 作业。

- 代码：当前 main，HEAD bf9bde8；保留现有 dirty 文档和未跟踪文件。
- 输入：video3 原视频对应 29 帧 / 711 observations，复用先前还原的 JPEG/规范检测 JSON，以避免输入差异；不重跑 detector/classifier。
- 主测量：新进程、新 DA3/PVS 输出 cache，现有模型文件/OS 页缓存保留；CPU OMP/OpenBLAS/MKL=4，GPU2 UUID GPU-d8012344-c2b6-f54b-7edc-9df3acda6fa7。NN/SOR 提前准备开启；DBSCAN/proxy/early-background 为当前默认；Atlas 默认 1。
- 截止点：本地 Viewer generation/CURRENT、ZIP、GLB 写完返回；分别记录可用时刻，不包含浏览器/HTTP/COS/fsync。
- 仪器：主跑仅低频函数 wall spans、既有 StageTimer、200 ms 进程树 RSS/driver GPU 采样、各进程 PyTorch allocator 生命周期峰值。重型逐语句/native/CUDA-event 诊断单独跑，不与主跑合计。
- 细分：DA3 构建/权重/to/预处理/forward/回投/NPZ；PVS 模型/图像/encoder/decoder与后处理；matching 每 ref 与投影步骤；dedup；NN/SOR/DBSCAN/RANSAC；标签/预算/采样/发布；ZIP成员；GLB LOD/烘焙/透明RGB/页面/量化/编码/写入。
- 资源：GPU2 上已有进程全部保留；新任务严格串行测量。准备阶段恢复同一离线 DA3 snapshot（原 host symlink 已失效）；新增模型/产物预计 < 12 GiB，保留 >=4 GiB 空闲。每次完整主跑/诊断 30 min 上限，出现异常保存失败记录，先定位再重试。
- 验收：29 refs、完整 PVS manifests、有效 dedup 和 Viewer/ZIP/GLB；阶段边界可解释、主表闭合、峰值口径清楚。单次结果不推导普遍性能，也不将 GPU0 旧测量与 GPU2 新测量的差异归因为单一优化。
- 交付：docs/profiling_breakdown.md 为唯一新主入口；reports/profiling-20261018 保存可移植汇总、算法细节与重现脚本；runtime/profiling-20261018 保存本机原始日志/产物。旧主文档保留带日期的历史副本并标记 superseded，不删除准确率/视觉证据。
