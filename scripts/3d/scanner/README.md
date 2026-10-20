# Scanner 点云重建：选定的两个候选

独立入口只抽取每秒最近的一帧并重建彩色点云，不启动 SKU detection、SAM、matching 或 Viewer。
目前保留 **DA3 + K/E metric 修复版**和 **PromptDA full**。它们不修改 `main.py` 的默认 backend。

完整的单位、帧同步、相机约定和尺度修复见 [Stray Scanner 与 LiDAR 数据流](../../../docs/stray_scanner_lidar.md)。
需要将 PromptDA dense 预测接入 SKU matching、Viewer/纹理 GLB 或 footprint 时，
使用 [PromptDA 场景缓存接入](../../../docs/promptda_integration.md)；独立点云入口本身不运行这些阶段。

| 候选 | 网络输入 | 深度与相机策略 | 导出筛选 |
|---|---|---|---|
| `da3` | RGB、逐帧 K、Scanner w2c | 保留 DA3 Nested 原生 metric depth，显式恢复 Scanner E | 原 DA3 adaptive confidence |
| `promptda` | 长边1008的 RGB、完整 LiDAR depth | 官方 Large FP32 权重，默认 FP16 autocast、batch=2；K/E 仅用于投影 | 全部有效预测 |

DA3 保持既有 504×378 处理网格。PromptDA 默认 `--promptda-max-size 1008`：
本扫描 `1920×1440 → 1008×756`，使用官方 float32 `INTER_AREA` 规则，两边均为14的倍数，无需补边。
输出 K 与 source→processed affine 使用实际 resize 的半像素变换。
`--promptda-max-size 0` 保留此前原图 pad14/crop 模式；`504` 可复现早期分辨率。
共同使用 Scanner 首相机显示坐标、固定 seed 42，最多导出 1,000,000 个彩色 POINTS。

## 环境和模型

使用仓库根 `uv` 环境（Python 3.11、Torch 2.7.1、NumPy 1.26.4），系统需有 ffmpeg 和 CUDA。
DA3 使用本地 **Nested metric** 模型目录，包含 `config.json` 和 `model.safetensors`。
加载复用仓库 `src/da3_model_loading.py`，不下载权重，不导入 matching 流水线。

PromptDA 使用[官方代码](https://github.com/DepthAnything/PromptDA)，本次验证版本
`9161547ca7151e14fcc8f490ad3cc02fa9437483`。源码和权重单独准备，不提交到本仓库：

```bash
git clone https://github.com/DepthAnything/PromptDA.git runtime/PromptDA
git -C runtime/PromptDA checkout 9161547ca7151e14fcc8f490ad3cc02fa9437483
uv run hf download depth-anything/prompt-depth-anything-vitl model.ckpt \
  --revision 573ab70c94c27d96f0bc70f81f670d2b0a62d571 --local-dir runtime/models/promptda/vitl
```

代码通过 `--promptda-repo` 加载该 checkout，使用其中自带的 DINOv2 代码。
此路径在已有根依赖环境验证过，不需要把上游完整 requirements 覆盖到根环境。
相关模型使用条件以各自上游许可证为准。

## 准备 Scanner 数据

输入目录包含 `rgb.mp4`、`odometry.csv`、`depth/*.png`、`confidence/*.png`。

```bash
uv run python scripts/3d/scanner/prepare_scanner.py \
  fd-videos/572dd9f6fb Output/scanner_prepared
```

按照 odometry 相对首帧的时间，对每个整数秒选最近帧；保留原始 frame ID。
ffmpeg 使用 `-ignore_editlist 1 -noautorotate`，避免首帧被 edit list 丢弃以及显示旋转造成 K/depth 错配。
逐帧 K 来自 CSV 的 `fx/fy/cx/cy`，不用只记录末帧的 `camera_matrix.csv`。
CSV 的 `qxyzw` 已表示 OpenCV camera→world；直接生成 c2w，取逆得到 w2c，不再额外翻转坐标。
LiDAR PNG 是 uint16 毫米，除以 1000 后保留原生深度网格。
对应[官方数据格式](https://github.com/strayrobots/scanner/blob/ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb/docs/format.md)和
[位姿编码](https://github.com/strayrobots/scanner/blob/ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb/StrayScanner/Helpers/OdometryEncoder.swift)。

如 ffmpeg 与模型在不同环境，可先用 `--extract-only` 抽帧，再在有 NumPy/OpenCV 的环境用
`--metadata-only` 生成 `camera_inputs.npz`。完整准备时请使用新目录。

已有 `camera_inputs.npz` 和 `frames/{frame_id:06d}.png` 时可直接复用，下面的 `--base` 指向该目录。
两路输出目录必须是新目录：

```bash
uv run python scripts/3d/scanner/reconstruct.py \
  --base Output/scanner_prepared --candidate da3 \
  --checkpoint /path/to/DA3NESTED-GIANT-LARGE \
  --output Output/scanner_da3_metric --device cuda:0

uv run python scripts/3d/scanner/reconstruct.py \
  --base Output/scanner_prepared --candidate promptda \
  --promptda-repo runtime/PromptDA --checkpoint runtime/models/promptda/vitl/model.ckpt \
  --output Output/scanner_promptda_full --device cuda:0
```

每路生成 `predictions.npz`、`cloud.glb` 和阶段耗时 `run.json`。
PromptDA 默认不压缩 NPZ，减少 CPU 落盘时间；可加 `--compress-prediction` 换取更小文件。
DA3 保留原压缩策略。归档压缩不改变任何数组内容。
NPZ 保存 dense depth、processed RGB、K/E 和 frame IDs；DA3 额外保存 confidence 和 Nested scale factor。
PromptDA 额外记录 `preprocess_method=promptda_resize`、`preprocess_resolution=1008` 和逐帧
`network_image_sizes`。1008×756 下 LiDAR 保持原生256×192网格与米制值，各层插值由模型负责。
若选用原图模式而需要补边，LiDAR 才先按像素中心最近邻扩展到 RGB 网格，再同步复制右/底边。
GLB 仅是筛选并限制点数后的可视化，不替代 dense NPZ。

PromptDA 默认参数为 `--promptda-max-size 1008 --promptda-precision fp16 --promptda-batch-size 2`，
重建与profiling入口共用同一组默认值。FP32对照可显式指定`--promptda-precision fp32 --promptda-batch-size 1`。
本扫描batch2与8吞吐近似持平，batch14没有收益；默认选择batch2。
`--promptda-matmul-precision high` 是另一项 TF32 开关，不能视为乘法精度完全不变。
batch 取值应按实际分辨率测量，1008长边下的batch扫描与选择见 [1008 profiling](profiling_1008.md)。
此前原图batch2的结果只属于 [原图历史 profiling](native_profiling.md)，不能据此判断1008的最佳batch。

## 尺度修复的边界

DA3 官方 API 的默认最终对齐会用轨迹拟合出的比例再次缩放 depth。
本扫描中该比例约 3.051，而 Nested depth 已接近米制，导致点云过度缩小。
`candidates.predict_da3()` 设置 `align_to_input_ext_scale=False`，检查 `is_metric==1`，
然后显式把返回 E 复制回输入 Scanner w2c，保留 depth、K、RGB 和 confidence。
只设置 False 还不够：此时 API 仍返回对齐后的预测位姿。
不使用 LiDAR 拟合比例，也没有硬编码 3.051 或给非 metric 模型猜比例。

该策略面向本次采用的 metric 多帧模型与可信外部位姿。False 分支仍执行官方 Umeyama 对齐；
不承诺单帧、零基线或退化轨迹可用，也不将这次单扫描改善推广为所有场景。
DA3 保留 API 的 processed K；PromptDA 1008/504使用OpenCV resize的半像素 affine变换K，
原图模式保留原始K。缓存保留各自实际网格与策略，不会相互重标。

PromptDA full 使用所有 confidence 等级的有效原生 LiDAR；没有留出块、传感器点替换或事后尺度拟合。
完整 prompt 已作为模型输入，不能把模型与同一份输入 LiDAR 的误差当独立精度证据。

## 验证与 profiling

```bash
uv run python -m unittest discover -s test -p test_scanner_candidates.py -v
```

最小 CPU 检查覆盖 timestamp 抽帧、CV 相机约定、metric depth/E 策略、完整 prompt、
1008 RGB与K、原生LiDAR、原图padding/crop、批次顺序、极值归一化和真实POINTS GLB读写。
速度比较入口为 `profile_candidates.py`；早期 504 实测见 [profiling.md](profiling.md)，
原图历史对比见 [native_profiling.md](native_profiling.md)，当前1008对比见 [profiling_1008.md](profiling_1008.md)。
