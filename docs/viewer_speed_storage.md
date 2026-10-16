# Viewer 速度与存储优化（2026-10-16）

本次保留现有画质参数与数据 schema，优化 ZIP、前端资源创建、GLB 导出和背景 voxel 聚合。源码发布不代表运行容器已经更新。

## 实现

- `.bin` 使用 DEFLATE level 1，JSON/JPEG 保持 STORED；三份 Viewer 支持两种方法。二进制直接传给 Surfel loader，避免 Blob/fetch 中转。
- 普通点云和 EDL 资源在首次使用时创建，保留此前选择和隐藏状态；重复切换复用资源。
- GLB 逐页完成既有量化和最终 PNG 编码，消除中间完整 GLB 与重复 PNG 编解码；最终 BIN 使用临时文件流。未生成的中间字节数报告为 `null`。
- 背景 voxel 用 NumPy 分组代替逐点 Python 字典，保留最高置信度、并列时首个输入及 voxel 首次出现顺序。

## 本地测量

输入为现存 video3-area 的 29 帧、2,349,242 点。MB 为十进制，MiB 为二进制。

| 项目 | 旧实现 | 新实现 |
|---|---:|---:|
| Viewer ZIP | 115.48 MB | 78.56 MB |
| ZIP 打包 | 0.347 s | 2.946 s |
| 完整 GLB 导出 | 201.30 s | 161.37 s |
| GLB 进程峰值 RSS | 1089.66 MiB | 877.17 MiB |
| 50 万背景点 voxel 内核 | 1.1848 s | 0.2631 s |
| 本机浏览器 fetch + parse + decode | 2.908 s | 2.656 s |

ZIP 减少 31.97%，751 个成员解码内容相等；压缩增加约 2.60 秒打包 CPU 时间。完整 GLB 的量化几何、PNG、材质和节点转换与旧版相同，两者均为 2,519,424 个三角面、31 页 atlas，最终文件仍约 118.70 MB，仅布局相差 44 bytes。

GLB 时间和 RSS 为相同项目初始化、CPU affinity 32–35、OMP/BLAS/MKL 单线程的独立进程对照，修正后的 baseline 在 candidate 后运行；未控制共享主机负载与文件 cache。首次 RSS 对照使用不同包初始化而失效，表中已使用补测结果。RSS 包含项目导入，不能解释为独立 exporter 的分配量。

voxel 数值为三次中位数，使用已导出坐标与合成置信度；100 个随机输入和 ordinary/adaptive 完整采样结果与旧逻辑相同。这不是完整过滤/采样阶段计时。

浏览器使用本机 Chromium 151，单次顺序对照、无渲染。DEFLATE parse 由 0.479 增至 0.679 秒；GC 后 backing store 为 315.672 / 315.688 MiB，没有证明常驻内存减少。这些组件数字不代表 Docker 请求端到端或终端硬件 GPU 性能。

## 验证与发布

发布 worktree 的核心 GLB/采样/clarity pack 定向测试 73 项、Docker pack 2 项通过；三份前端的相关测试与构建通过。此前浏览器小场景检查覆盖选择、隐藏、首次模式切换、测量相机、resize/dispose；真实包的 8 个数组共 90,408,430 bytes 与原包逐字节相同。

先更新三份前端，再部署使用压缩 ZIP 的 producer。前端要求原生 `DecompressionStream("deflate-raw")`；已验证 Chromium 151，其他浏览器版本未实测。各服务版本的上传时序保持原状；提前上传、分块 LOD 和 KTX2 未包含在此次发布中。

英文发布只移植速度/存储优化，保留 `visualization_en` 的现有渲染参数，没有合入其他 clarity 实验提交。相机先验、模型预热、异步 COS 实现和历史清理文档未纳入此次提交。

原始测量、对照脚本和浏览器记录保存在执行机器的 `runtime/viewer-speed-storage-20261016/`，不随源码发布。大文件和依赖未加入 Git；本次没有模型推理、COS 上传、镜像构建或服务部署。
