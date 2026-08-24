# DA3 到 Web Viewer 的端到端性能测量设计

## 目标

对 `floor_display2`、`floor_display3`、`floor_display4` 测量从 DA3 三维重建到
网页端可稳定交互的真实时间和显存开销。报告每个数据集的原始结果、三数据集均值、
阶段占比和瓶颈；所有运行时产物仅写入本目录的 `runs/`。

## 测量口径

模型权重已经安装，不把首次下载计入结果。每个数据集顺序运行两次：

1. **cold**：空的专用 `save_root`，因此 DA3 cache、matching 结果、SAM3 mask cache、
   footprint artifact 和 viewer bundle 都从头创建。
2. **warm**：保留同一 cold run 的 `save_root`，复用 DA3 和 SAM3 cache，但重新运行
   matching、dedup、footprint、bundle export 和浏览器的空 HTTP-cache 首次加载。

端到端总时间由下列用户可见必要阶段组成：

| 阶段 | 命令/边界 | 包含 | 不包含 |
| --- | --- | --- | --- |
| `reconstruction` | `main.py --mode reconstruct` | DA3 子进程启动、模型载入、图像读取、推理、世界点与 cache 写入 | 模型下载 |
| `matching` | `--mode concise --enable_profiling` | DA3 cache 消费、SKU 匹配、匹配 profiling | 精度评估时间单列 |
| `analysis_dedup` | analyzer + dedup | 全局 ID 映射生成 | 两套 bbox 调试图片 |
| `footprint` | `--mode ground-stack-area` | SAM3 mask cache、support plane、OBB union、正式 artifact | shadow-only evidence 不进入端到端临界路径 |
| `viewer_export` | `--mode viewer-web` | 点云过滤、法线、缩略图、bundle publication | Vite 编译 |
| `browser_first_interactive` | Playwright + 静态 HTTP server | HTML/JS、CURRENT/manifest/二进制资源、Three.js GPU upload、首帧、连续 10 帧稳定 | 浏览器安装、开发服务器启动 |

`accuracy_evaluation`、原始/去重检测框 PNG 不为 viewer 的输入，故记录为可选旁路，
不混入上述端到端临界路径。

## 资源与显存

每次阶段执行前记录机器、驱动、PyTorch 与数据集元数据。测试固定
`CUDA_VISIBLE_DEVICES=2`，以避免当前繁忙的 GPU 0/1。GPU 采样器在整个子进程期间以
100 ms 间隔记录 `nvidia-smi` 的显存、利用率和功耗；报告下列两个显存指标：

- `driver_peak_mib`：GPU 2 的采样峰值减去阶段前基线，包含 CUDA context 等驱动可见分配。
- `torch_peak_allocated_mib`：若阶段自身输出该字段，则记录 PyTorch allocator 峰值；它
  不替代 driver 指标。

当前 DA3 runner 不会输出 allocator 峰值，因此本轮以 driver 峰值为正式容量结论，并在
报告中明确这个限制。不会为了采样而改动 DA3 的数值路径。

## 浏览器测量

viewer 是原生 Three.js，不使用 React Three Fiber，故不接入 `r3f-perf`。专用
Playwright 脚本启动 Chromium，禁用 HTTP cache 后导航到对应 data root，并从页面内
Performance API 和 `requestAnimationFrame` 收集：导航开始、bundle 成功加载、第一帧、
第 10 帧、下载字节数、GPU renderer/vendor 和 WebGL context。浏览器时间采用三次导航的
中位数；每个数据集的原始三次结果也保留。若 Chromium 报告 SwiftShader、llvmpipe 或无法
读取 renderer，则标为 `software_or_unavailable`：资源加载和 JavaScript 时间仍记录，但不把
该样本解释成浏览器 GPU 渲染或浏览器显存结论。

## 结果契约

每次运行产生：

```text
perf/runs/<utc-run-id>/
  environment.json
  fd2/{cold,warm}/stages.json
  fd2/{cold,warm}/logs/*.log
  fd2/{cold,warm}/telemetry/*.csv
  fd2/{cold,warm}/browser.json
  summary.json
  report.md
```

任何阶段失败都会标记为失败并停止该数据集的后续依赖阶段，不以零填补；汇总会说明缺口，
不会把不完整数据算入均值。原始输入、既有 `Output/`、checkpoint 与 git 追踪文件不
被改写。

## 验收

1. 对 fd2–4 都有完整 cold/warm stage 记录，或明确的带日志失败原因。
2. 每个命令都有 wall-clock、退出码、GPU 采样基线和峰值。
3. 报告列出每阶段秒数、端到端总数、三数据集均值和百分比。
4. 浏览器报告至少包含三次 cache-disabled 导航、first frame 与 stable-interactive 时间。
5. 解析/汇总的单元测试、Python 格式检查、viewer build 与既有相关测试通过。
