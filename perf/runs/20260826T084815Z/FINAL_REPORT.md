# fd2–4 DA3 到 Web Viewer one-shot cold 性能报告

## 结论

本报告来自全新 run `20260826T084815Z`。三个数据集各自使用不存在的新目录，从 DA3 reconstruction 开始重新生成 cache；personalcare classification 与 reconstruction/matching 并行，analysis/dedup 显式消费本次 enriched detections。三个 case 的 7 个 stage 均以 exit code 0 完成，没有 warm case、重复 browser navigation 或旧 run fallback。

- fd2：356.714s（5.95min）
- fd3：319.956s（5.33min）
- fd4：514.159s（8.57min）
- 三数据集平均真实 wall time：396.943s（6.62min）

当前主要瓶颈是 footprint 的 support-plane 选择，而不是 processed mask 加载。footprint 平均 239.435s，占真实 wall time 60.3%；其中 `load_self_exemplar_masks` 仅为 0.597s、0.327s、2.510s，`select_support_plane` 分别为 185.861s、190.465s、308.349s。

## 协议

- 数据集：`floor_display2`、`floor_display3`、`floor_display4`，串行执行。
- GPU：物理 GPU 2，NVIDIA GeForce RTX 4090 D，24,564MiB。
- 每个 dataset 只执行一次 cold case，使用独立 `save_root`。
- classification 在 case 开始时异步启动，在 analysis/dedup 前 join。
- reconstruction、matching、classification、footprint、viewer bundle 均来自本次 case。
- 浏览器仅执行一次 cache-disabled navigation。
- case 端到端时间使用整个 case 的 monotonic wall time；并行 stage 不重复累加。
- `nvidia-smi` 每 100ms 采样 driver-visible 显存。

## 逐数据集结果

| Dataset | Case wall | Classification | Reconstruction | Matching | Analysis + dedup | Footprint | Viewer export | Browser |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fd2 | 356.714s | 7.723s | 38.596s | 98.189s | 5.749s | 193.377s | 18.058s | 2.160s |
| fd3 | 319.956s | 7.719s | 35.177s | 42.569s | 7.985s | 206.458s | 20.376s | 6.742s |
| fd4 | 514.159s | 9.603s | 43.285s | 92.367s | 6.203s | 318.471s | 44.985s | 8.129s |
| **mean** | **396.943s** | **8.348s** | **39.019s** | **77.708s** | **6.646s** | **239.435s** | **27.807s** | **5.677s** |

Classification 在三个 case 中都早于 reconstruction/matching 完成，因此其 elapsed 没有增加临界路径。表中 stage 时间用于定位工作量，不能相加得到 case wall time。

## GPU 观测

| Stage | Mean absolute peak | Mean additional peak |
| --- | ---: | ---: |
| classification | 4,493.7MiB | 685.7MiB |
| reconstruction | 19,026.3MiB | 15,218.3MiB |
| matching | 23,831.0MiB | 18,413.0MiB |
| analysis + dedup | 5,774.0MiB | 1,561.3MiB |
| footprint | 7,354.0MiB | 1,580.0MiB |
| viewer export | 1,845.3MiB | 636.0MiB |
| browser | 3,245.3MiB | 2,034.7MiB |

最大 absolute peak 是 fd2 matching 的 23,989MiB，距离 24,564MiB 设备上限约 575MiB。该机器在运行期间存在其他 GPU resident allocation，baseline 在 4–9,034MiB 之间变化；因此这些数字是共享主机上的 driver-visible 峰值，不是单个 stage 的纯 tensor allocation。并行 classification/reconstruction 的两套 peak 也不能相加。

## Footprint 内部耗时

| Dataset | Load processed masks | Select support plane | OBB union | Shadow evidence | Formal status |
| --- | ---: | ---: | ---: | ---: | --- |
| fd2 | 0.597s | 185.861s | N/A | 0.000s | rejected |
| fd3 | 0.327s | 190.465s | 2.278s | 5.603s | rejected |
| fd4 | 2.510s | 308.349s | N/A | 0.000s | rejected |

三次 footprint 都成功发布了完整 rejected artifact，因此性能 stage 完成，但这不代表 formal 面积被接受。fd2/fd4 在 support-plane table compatibility gate 被拒绝；fd3 因部分 global IDs 被拒绝而整体 rejected。性能结果不能解释为面积质量验收。

## Browser 单次导航

| Dataset | Bundle loaded | First frame | Frame 10 | Transfer bytes | Renderer evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| fd2 | 129.6ms | 265.0ms | 1,265.0ms | 9,237,568 | software_or_unavailable |
| fd3 | 127.8ms | 207.7ms | 5,474.1ms | 11,098,410 | software_or_unavailable |
| fd4 | 215.0ms | 297.4ms | 6,563.8ms | 18,849,913 | software_or_unavailable |

fd3/fd4 使用 SwiftShader，fd2 没有暴露 renderer。浏览器数据可用于资源加载和 JavaScript 初始化观察，不能当作 NVIDIA WebGL 帧率或浏览器 GPU 显存结论。

## 下一优化点

1. 优先优化 support-plane 选择：它平均约 228.225s，是 footprint 的绝对主成本。
2. fd2/fd4 matching 接近 24GiB 上限；若继续使用 GPU 2，应减少 matching 峰值或改用 48GiB GPU。
3. fd4 viewer export 为 44.985s，适合进一步拆分点云过滤、标签传播和缩略图时间。
4. 在独占 GPU 与硬件 WebGL 环境重跑，才能获得可归因的 GPU peak 和浏览器渲染性能。

## Artifacts

- Machine summary：`summary.json`
- Generated report：`report.md`
- Resource preflight：`environment.preflight.json`
- Per-case receipts：`fd{2,3,4}/stages.json`
- GPU telemetry：`fd{2,3,4}/telemetry/*.csv`
- Browser receipts：`fd{2,3,4}/browser.json`
- Stage logs：`fd{2,3,4}/logs/*.log`
