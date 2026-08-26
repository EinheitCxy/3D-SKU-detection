# DA3 到 Web Viewer 性能基准

本目录包含从 DA3 重建到 Three.js viewer 首次可交互的可复现性能采集器。它对
`floor_display2`、`floor_display3`、`floor_display4` 各执行一次独立 cold run，生成逐阶段的
wall time、driver-visible GPU memory peak、原始日志、GPU telemetry、一次浏览器导航数据和
三数据集均值报告。采集器不生成或汇总 warm-start 数据。

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

运行创建全新的 `perf/runs/<utc-run-id>/`。每个 `fdN/` 都有独立且初始为空的
`save_root`。personalcare classification 在 case 开始时异步提交，与 reconstruction、matching
重叠，并在 analysis/dedup 前 join；之后继续 footprint、bundle export 和一次 cache-disabled
浏览器导航。不同数据集或不同 benchmark run 之间不复用 DA3/SAM3/classification 产物，
也不会写入 `Output/`。

同一次 case 内 reconstruction 生成、matching 消费的 cache 属于必要的阶段依赖，不是
warm start。若指定的 `--run-root` 已存在，采集器直接拒绝运行，避免复用旧产物。
`--datasets` 也不接受重复名称，防止同一 case 在一次调度中被第二次执行。

## 输出与解释

- `fdN/stages.json`：该数据集一次完整执行的阶段 receipts。
- `fdN/browser.json`：单次导航的 bundle-loaded、first-frame、第 10 帧、传输字节和 WebGL renderer。
- `summary.json` / `report.md`：fd2–4 完整 one-shot cold case 的均值、时间占比和显存峰值。

`stages.json` 同时记录各 stage elapsed time 和整个 case 的真实 `wall_seconds`。classification
与 reconstruction/matching 有重叠，所以阶段耗时占比与 GPU peaks 不可相加；端到端均值只使用
case wall time。

`nvidia-smi` peak 是 driver 可见显存，包含 CUDA context 与非 PyTorch 分配；它是本报告的
正式显存值。DA3 runner 尚未输出 `torch.cuda.max_memory_allocated()`，所以不能将 driver peak
解释为纯 tensor allocation。浏览器若报告 `SwiftShader`/`llvmpipe`/无 renderer，会标记为
`software_or_unavailable`；其资源和 JS 加载时延仍可用，但不能作为硬件 WebGL 或浏览器 GPU
显存结论。

当前正式基线是 [20260826T084815Z](runs/20260826T084815Z/FINAL_REPORT.md)：fd2–4 三个
one-shot cold case 全部完成，平均真实 wall time 为 396.943s。旧
`perf/runs/20260824T032553Z/` 只保留历史 cold 原始证据，不再作为当前结果。

## 验证

```bash
uv run --offline pytest perf/tests/test_benchmark.py -q
(cd perf && npm test)
(cd modules/viewer_web && npm test -- --run && npm run build)
```
