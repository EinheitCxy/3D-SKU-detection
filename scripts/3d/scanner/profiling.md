# 选定 Scanner 候选的速度对比（2026-10-19）

以下记录属于早期两路均为 504×378 的实现，保留为历史结果。当前PromptDA默认长边1008，
最新对比与命令见[1008 profiling](profiling_1008.md)；此前原图实验见[原图历史 profiling](native_profiling.md)。
精确复现本页早期 PromptDA 策略需使用提交 `147330e` 的代码。

本机这组设置下，**DA3 + K/E 修复版的热态完整请求更快**：中位数 1.340 s，PromptDA full 为 1.674 s，DA3 延迟低约 20.0%。PromptDA 的显存占用明显更低。这里比较各自已接受的原生实现，没有更换模型或计算精度。

## 实测结果

共同输入：同一 capture `572dd9f6fb`，1 Hz 的 14 帧；RGB 1920×1440 → 504×378，PromptDA prompt 为原生 256×192 米制 LiDAR。物理 GPU 2，RTX 4090 D 24 GiB，容器通过 `--gpus device=2` 映射为 `cuda:0`。CPU/BLAS 线程为 8。

| 指标 | DA3 + K/E 修复版 | PromptDA full |
|---|---:|---:|
| 执行设置 | Native BF16 autocast，14 帧联合 | FP32，逐帧顺序 |
| 模型加载至 GPU 同步完成 / s（一次） | 8.990 | 14.423 |
| 冷态完整请求 / s（一次） | 2.261 | 2.299 |
| 热态完整请求中位数 / s（3 次） | 1.340 | 1.674 |
| 热态完整请求范围 / s | 1.323–1.460 | 1.672–1.675 |
| NPZ 压缩保存 / s（一次） | 1.600 | 0.944 |
| GLB 反投影、筛选、采样、写入及回读 / s（一次） | 0.843 | 1.224 |
| 冷请求开始 → NPZ + GLB 就绪 / s（一次，排除加载） | 4.708 | 4.471 |
| 加载 + 冷请求 → NPZ + GLB / s（上述两个窗口之和） | 13.698 | 18.894 |
| GPU allocated 峰值 / GiB（含模型） | 11.336 | 1.680 |
| GPU reserved 峰值 / GiB（含模型） | 15.123 | 1.865 |
| 本进程 sampled RSS 峰值 / GiB（含加载） | 13.648 | 3.424 |
| GLB 筛选后、限点前的点数 | 1,600,301 | 2,667,168 |
| GLB 实际点数 | 1,000,000 | 1,000,000 |

完整请求从模型已常驻、准备好的 PNG 路径及 sensor arrays 开始，到 dense depth 和 processed RGB 在 CPU 上就绪为止。包含 RGB 解码、预处理、H2D、模型计算、D2H 和输出打包，**不是纯 forward 时间**。模型加载窗口合并 CPU 构造/权重读取、一次性依赖导入和 CUDA 迁移。

单次冷请求加保存/导出时，PromptDA 观测值为 4.471 s、DA3 为 4.708 s；只有一次样本且 NPZ 字段、筛选点数不同，不据此宣称稳定优势。加入各自模型加载窗口后，本次 DA3 用时更少。DA3 使用仓库现有优化 loader；PromptDA 保持官方构造与加载方式，因此加载差异也不是纯网络计算差异。

## 测量范围和限制

- 每候选单独新进程，先 DA3 后 PromptDA；各一次 cold、三次 warm，不并行争用 GPU。cold 后只保存一份 NPZ 和 GLB，随后三个 warm 不重复落盘。
- Cold 指新进程首个请求；未清空文件页缓存。warm 保留模型与 allocator cache。三个样本只报告中位数、范围及 JSON 中的 sample SD，不估计尾延迟。
- 请求窗口前后同步 CUDA；GPU 峰值按窗口记录 baseline 后 reset。allocated/reserved 独立报告，包含模型且不能相加，不是整卡显存。采样 RSS 仅当前 PID、间隔 100 ms，可能漏短峰值。
- GPU2 运行前后已有约 5044 MiB 其他进程显存占用，均未更改；本次未建立独占 GPU 或主机隔离条件。
- 保留 DA3 native confidence 筛选与 PromptDA 全部有效预测，两者都固定 seed=42、上限 1M 点，不为统一点数前处理而改变已接受的视觉效果。NPZ 的 DA3 confidence/scale factor 字段也保留。
- 不含原视频解码/1 Hz 准备、matching、SAM、Viewer、网络传输或上传。“加载 + 冷请求”之和仍不包含进程启动、通用 Python imports 和准备 NPZ 的读取；不称完整进程延迟。

## 验证与复现

7 项 CPU tests 通过；新 Scanner 准备结果的全部数组与既有输入完全一致。本次新入口的两路 depth、RGB 和 frame IDs 与之前认可结果逐元素完全相同；DA3 的 K/E/conf/metric metadata 也完全相同；PromptDA 的 Scanner E 和 half-pixel K 已单独确认。两份 GLB 实际导出并单次回读通过。没有用完整 PromptDA 输入 LiDAR 重新宣称独立精度改善。

原始本地结果位于 `Output/scanner_candidates_profile_20261019/{da3,promptda}/profile.json`，对应同目录的 `run.log`、`container_wall_seconds.txt`、`predictions.npz`、`scene.glb`。同一输出根目录的 `launch.json` 记录实际 Docker argv 与 `--gpus device=2`，`gpu_before.json` / `gpu_after.json` 记录前后状态。原始输入、权重和大型生成文件不提交；[精简数字结果](profiling_results.json)保留实际各轮数值和验证结果。

使用[环境说明](README.md)中的依赖和模型，分别运行两个新 Python 进程：

```bash
CUDA_VISIBLE_DEVICES=2 uv run python scripts/3d/scanner/profile_candidates.py \
  --base Output/scanner_prepared --candidate da3 \
  --checkpoint /path/to/DA3NESTED-GIANT-LARGE --output Output/profile_da3 --device cuda:0

CUDA_VISIBLE_DEVICES=2 uv run python scripts/3d/scanner/profile_candidates.py \
  --base Output/scanner_prepared --candidate promptda \
  --promptda-repo runtime/PromptDA --checkpoint runtime/models/promptda/vitl/model.ckpt \
  --output Output/profile_promptda --device cuda:0
```

本次实际使用 `global-id-mapping:cos-background-20261013` 容器、Python 3.11.10、Torch 2.7.1+cu126；`--network none --gpus device=2 --shm-size 1g`，仓库与 prepared base 只读挂载，各候选输出目录独立可写。uv 通过 `run --no-project --offline --python /app/.venv/bin/python` 使用镜像既有环境；所有请求之外的完整 profiling 容器生命周期分别为 21.083 / 27.655 s，包含四轮请求及一次导出，不能当单请求耗时。
