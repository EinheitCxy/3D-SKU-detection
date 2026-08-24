# DA3 到 Web Viewer 性能基准

本目录包含从 DA3 重建到 Three.js viewer 首次可交互的可复现性能采集器。它测量
`floor_display2`、`floor_display3`、`floor_display4` 的 cold 与 warm 路径，生成逐阶段的
wall time、driver-visible GPU memory peak、原始日志、GPU telemetry、浏览器导航数据和
三数据集均值报告。

详细边界见 [DESIGN.md](DESIGN.md)，执行顺序见 [PLAN.md](PLAN.md)。

## 运行

先从 `perf/` 安装浏览器依赖并把 Chromium 留在本目录：

```bash
npm install
PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright" npm exec playwright install chromium
```

确认 GPU 2 空闲后，在仓库根目录运行：

```bash
uv run --offline python perf/benchmark.py --gpu-index 2
```

运行创建 `perf/runs/<utc-run-id>/`。每个 `fdN/cold` 从空的 `save_root` 开始；同一
`fdN/warm` 复用该 cold case 的 DA3/SAM3 cache，但重新运行 matching、dedup、footprint、
bundle export 与浏览器的 cache-disabled 导航。不会写入 `Output/`。

## 输出与解释

- `stages.json`：每个 stage 的 wall seconds、退出码、GPU baseline/peak、日志与 telemetry 路径。
- `browser.json`：三次导航的 bundle-loaded、first-frame、第 10 帧、传输字节和 WebGL renderer。
- `summary.json` / `report.md`：fd2–4 的 cold/warm 完整 case 平均值、时间占比和显存峰值。

`nvidia-smi` peak 是 driver 可见显存，包含 CUDA context 与非 PyTorch 分配；它是本报告的
正式显存值。DA3 runner 尚未输出 `torch.cuda.max_memory_allocated()`，所以不能将 driver peak
解释为纯 tensor allocation。浏览器若报告 `SwiftShader`/`llvmpipe`/无 renderer，会标记为
`software_or_unavailable`；其资源和 JS 加载时延仍可用，但不能作为硬件 WebGL 或浏览器 GPU
显存结论。

## 验证

```bash
uv run --offline pytest perf/tests/test_benchmark.py -q
(cd perf && npm test)
(cd modules/viewer_web && npm test -- --run && npm run build)
```
