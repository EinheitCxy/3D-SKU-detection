# Stray Scanner 与 LiDAR 数据流

本文说明 [Scanner 独立入口](../scripts/3d/scanner/README.md) 当前采用的数据与几何约定，适用于 **DA3 + K/E metric 修复版**和 **PromptDA full**。准备、推理和导出分别由 [prepare_scanner.py](../scripts/3d/scanner/prepare_scanner.py)、[candidates.py](../scripts/3d/scanner/candidates.py) 与 [scanner_io.py](../scripts/3d/scanner/scanner_io.py) 实现。

运行与模型准备见入口 README；接入共享 matching、footprint 和 Viewer 缓存见 [PromptDA 接入说明](promptda_integration.md)。独立入口完成 RGB-D 准备、深度预测和彩色 POINTS GLB 导出，主流水线的后续处理有各自的入口和验证范围。

## 输入文件与帧身份

项目版 [SKU Scanner](../sources/stray-scanner/README.md) 在采集端按真实时间1Hz保存同步 bundle，
格式与相机轴见 [sku_bundle.md](../sources/stray-scanner/docs/sku_bundle.md)。
使用 [prepare_bundle.py](../scripts/3d/scanner/prepare_bundle.py) 直接生成下文相同的 prepared 输入；
逐帧 K/c2w 保留，只求逆得到 w2c，并将毫米转米。原图与 LiDAR 都不在手机端缩放。
它选择每个经过秒内第一组完整且 tracking normal 的帧；历史 MP4 则使用整数秒最近帧。
以下 MP4/CSV/ffmpeg 说明继续适用于旧录制数据。新采集端尚待 Mac SDK 构建和 LiDAR 真机验证。

| 文件 | 数据含义 | 当前准备程序的用途 |
|---|---|---|
| `rgb.mp4` | 相机原始 RGB 视频，可能携带显示旋转与 MP4 edit list | 解码原始方向的 PNG，保留录制帧编号 |
| `odometry.csv` | 每帧时间戳、`frame`、平移、quaternion、逐帧 K | 选择采样时间，建立 RGB、depth、K/E 对应关系 |
| `depth/{frame:06d}.png` | 原生单通道 `uint16` 毫米深度，常见尺寸为 `192×256`（H×W） | 转为 float32 米制 `lidar_depth`，保留网格 |
| `confidence/{frame:06d}.png` | 与 depth 同尺寸的 `uint8` 传感器置信等级 | 检查并保存 `0/1/2`，供后续显式选择使用 |
| `camera_matrix.csv` | 最后一个录制帧的 K | 当前入口使用 CSV 逐帧 K，不从此文件替代 |
| `imu.csv` | IMU 采样 | 当前两路重建不消费 |
| `distortion/*.bin` | 可选的逐帧畸变表 | 当前准备程序不应用此表 |

上游 [format.md，固定版本](https://github.com/strayrobots/scanner/blob/ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb/docs/format.md#L27-L55) 定义了单位、文件对应和新增的逐帧标定字段。CSV 表头可包含空的 `distortion_center_x/y`；当前程序按字段名读取 K 与 pose，不依赖整行固定切片。存在畸变字段不代表本入口已经执行畸变校正。

上游 [DatasetEncoder.swift](https://github.com/strayrobots/scanner/blob/ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb/StrayScanner/Helpers/DatasetEncoder.swift#L79-L100) 为同一 `ARFrame` 保存 RGB、depth 和 odometry，并给已保存帧分配连续编号。准备程序要求原始 CSV 的 `frame` 从 0 连续递增；**采样后的 ID 可以不连续**。后续所有数组与 PNG 路径均按采样顺序和原始 ID 配对，不能重新编号后再与原始 depth 目录配对。

## 1 Hz 采样与视频解码

`read_selected()` 使用 odometry 时间戳而非视频名义 FPS。设首帧时间为 `t0`，最后一帧为 `t_last`，对整数秒

```text
k = 0, 1, ..., floor(t_last - t0)
```

选择使 `abs((t_i - t0) - k)` 最小的原始帧。等距时保留 CSV 中靠前的帧；若多个目标秒选择了同一帧，程序报错。它不会插值相机、复制 RGB 或合成 depth。

[ffmpeg_command()](../scripts/3d/scanner/prepare_scanner.py#L33-L57) 的三个关键约定是：

- `-ignore_editlist 1`：按保存的原始帧解码，避免 edit list 隐藏负 PTS 的首帧后，RGB 序号相对 CSV/depth 少一帧。
- `-noautorotate`：保留编码图像方向；播放器的竖屏显示旋转不自动应用到 RGB。否则还必须同步变换 depth、K 与相机坐标约定。
- `select=eq(n,原始frame_id)...` 与 `-fps_mode vfr`：按原解码序号选择帧，保留原始 ID 命名，不通过 `fps=1` 重采样建立对应关系。

视频 PTS 与 odometry 的采集时间可能不同；不能用视频播放时间重新匹配已经按保存帧对应的 PNG。所选帧最终保存为 `frames/{原始frame_id:06d}.png`。抽帧数量与目标数量不符会停止，不截短数组后继续。

准备结果 `camera_inputs.npz` 包含：

| 字段 | 形状与单位 |
|---|---|
| `frame_ids` | `[N]`，原始帧 ID，int64 |
| `target_seconds`、`timestamps_relative` | `[N]`，目标整数秒及实际采样的相对秒数，float64 |
| `intrinsics` | `[N,3,3]`，原始未旋转 RGB 网格的逐帧 K，float32 |
| `c2w`、`w2c` | `[N,4,4]`，齐次相机变换，float32，平移为米 |
| `image_size` | `[width,height]`，原始 RGB 尺寸 |
| `lidar_depth` | `[N,H_l,W_l]`，原生网格的 float32 米制深度 |
| `lidar_confidence` | `[N,H_l,W_l]`，原生传感器置信等级 |

## 相机坐标、K 与深度

### CSV quaternion 已经是 OpenCV camera → world

上游 [OdometryEncoder.swift](https://github.com/strayrobots/scanner/blob/ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb/StrayScanner/Helpers/OdometryEncoder.swift#L14-L51) 从 `frame.camera.transform` 获取 RGB 相机 pose，将旋转写为 `q_WA * q_AC`。其中 `q_AC=(1,0,0,0)`，即绕 x 轴旋转 π，已完成 ARKit camera 到 OpenCV camera 的轴转换。导出的顺序是 `qx,qy,qz,qw`。

[camera_from_row()](../scripts/3d/scanner/prepare_scanner.py#L79-L103) 归一化 quaternion 后直接构造：

```text
T_WC = c2w = [R(qxyzw), t]       X_world  = T_WC · X_camera
T_CW = w2c = inverse(T_WC)       X_camera = T_CW · X_world
```

OpenCV camera 的 `+x` 向右、`+y` 向下、`+z` 向前；平移来自 CSV `x/y/z`，单位为米。不要再次乘 `Rx(π)`，也不要把 camera pose 当成 IMU/body pose。DA3 的输入 E 使用 **w2c**；NPZ 中两种方向都保留。

### 原生 depth 是米制 camera Z-depth 的消费契约

Scanner 保存的是 ARKit `sceneDepth.depthMap`；这里的“原生”指它保存的 depth 网格。准备程序读取单通道 `uint16` PNG，执行 `depth.astype(float32) / 1000`。零是无效深度；有效值用于以下 Z-depth 反投影：

```text
X_camera = Z · inverse(K) · [u, v, 1]^T
```

这里的 `Z` 是相机坐标第三分量，不能作为沿单位射线的欧氏距离再乘一次归一化 ray，也不需要额外的 ray-length 修正。上游推荐的 [StrayVisualizer，固定版本](https://github.com/kekeblom/StrayVisualizer/blob/195c64091a13277b22734a20303079c91fa63c30/stray_visualize.py#L128-L144) 同样把毫米转米后的 depth 直接交给 Open3D RGB-D 反投影，并使用 `inverse(T_WC)`。

这是本仓库与推荐消费者采用的几何契约。Scanner 的格式说明明确了毫米单位，但没有单独展开 Z-depth 与 ray range 的底层传感器定义；本文不把消费契约当作独立物理标定，也不把 ARKit 输出视为无误差的外部真值。

需要在原生 LiDAR 网格反投影时，推荐消费者使用

```text
K_lidar = diag(W_l / W_rgb, H_l / H_rgb, 1) · K_source
```

其 [get_intrinsics()](https://github.com/kekeblom/StrayVisualizer/blob/195c64091a13277b22734a20303079c91fa63c30/stray_visualize.py#L75-L81) 对 `1920×1440` RGB 和 `256×192` depth 的两轴比例均为 `2/15`。这是原生传感器网格的标定缩放，与下节 RGB resize 的像素中心 affine 分开处理。

### RGB resize 的半像素约定

对于原始 RGB 尺寸 `(W_s,H_s)` 和实际处理网格 `(W_p,H_p)`，OpenCV resize 的 source→processed 像素中心映射为：

```text
sx = W_p / W_s,  sy = H_p / H_s
u_p = sx · u_s + (sx - 1) / 2
v_p = sy · v_s + (sy - 1) / 2

A_resize = [[sx, 0,  (sx - 1) / 2],
            [0,  sy, (sy - 1) / 2],
            [0,  0,  1]]
```

当前 PromptDA 将1920×1440 RGB按官方AREA规则缩到1008×756，无需补边。
`K_output = A_resize @ K_source`，其中`sx=sy=0.525`、两轴offset均为`−0.2375`。
此前原图模式保留为`--promptda-max-size 0`：右/底补边到1932×1442后将预测裁回原尺寸，保留source K和identity affine。
早期 504 模式的 K 由 [resize_intrinsics()](../scripts/3d/scanner/scanner_io.py) 计算
`K_processed = A_resize @ K_source`；`1920×1440 → 504×378` 时，`sx=sy=0.2625`，
两轴 offset 均为 `−0.36875`。不同分辨率须使用各自实际 affine，不能直接复用K。

DA3 路径则**保留原生 API 返回的 processed K**。传入 API 的仍是原图 K，让其输入处理只变换一次；本次尺度策略不额外改写 API 的 principal point。不要把 PromptDA 的半像素 affine 再乘到 DA3 返回的 K，也不要把两路 K 的构造方式混成同一个简单比例函数。

## 两路模型分别使用什么信息

| 阶段 | DA3 + K/E metric 修复版 | PromptDA full |
|---|---|---|
| 网络输入 | RGB、逐帧 source K、Scanner w2c | RGB、完整原生米制 LiDAR prompt |
| LiDAR 的作用 | 不参与网络或深度尺度拟合 | 进入网络，提供米制深度信息 |
| K/E 的作用 | 网络相机条件及最终反投影；输出 E 显式恢复 Scanner w2c | 不传入 PromptDA；仅用于最终反投影 |
| 保存的深度 | Nested 原生 metric depth | 官方网络直接输出的米制 depth |
| 传感器 confidence | 保存于准备数据，不参与此候选推理 | 不按等级筛 prompt |
| GLB 筛选 | DA3 预测 confidence 的既有 adaptive 规则 | 全部有限正值预测 |

传感器 `confidence` 的 `0/1/2` 表示递增的可信等级，与 DA3 网络的数值 confidence 是两种数据。DA3 导出阈值为 `min(max(1.05, percentile40), percentile90)`，分位数在有效预测的网络 confidence 上计算，不能把这个阈值用于传感器 `0/1/2`。

[predict_promptda()](../scripts/3d/scanner/candidates.py) 保留所有有限正值原生 depth，非有限或非正值置零；
不人为挖洞、不按 confidence 等级删点、不用传感器点替换网络输出。RGB默认长边1008，
本扫描无补边时LiDAR保持原生256×192网格。只有需要patch14补边时，才先把LiDAR按像素中心最近邻扩展到RGB尺寸，再同步复制右/底边；
这个步骤保留原米制值及零值，用于防止 prompt 的视场被拉伸到新增边框，不增加传感器测量。
模型内的多尺度 prompt 特征插值仍由官方实现负责。

官方 [PromptDA normalize()](https://github.com/DepthAnything/PromptDA/blob/9161547ca7151e14fcc8f490ad3cc02fa9437483/promptda/promptda.py#L89-L121) 使用整张 prompt 的 min/max，包括零值，没有单独的 invalid-mask 参数。当前候选拒绝恒值 prompt 以避免除零；这不表示任意稀疏零洞都属于已验证输入条件。生产 full 路径不采用实验中的 masked tiles。
当前适配器按样本用 `aminmax` 计算相同端点，避免 `quantile(q=0/1)` 的两次排序；
不跨 batch 合并深度范围，也不改变官方归一化公式。

### DA3 尺度修复机制与适用条件

Nested 模型先把深度对齐到其 metric 分支。原生 API 的最终轨迹对齐还可能按 Umeyama 比例 `s` 执行 `depth /= s`；可信外部轨迹与模型预测轨迹的尺度不一致时，这一步会改变已经得到的 metric depth。

[predict_da3()](../scripts/3d/scanner/candidates.py#L59-L86) 的显式策略是：

1. 使用 RGB、source K、Scanner w2c 调用原生 API，设置 `align_to_input_ext_scale=False`。
2. 要求 `prediction.is_metric == 1`；非 metric 输出直接报错，不自动估计比例。
3. 把返回 E 明确复制为原始 Scanner `w2c[:, :3, :]`，保留 native depth、processed K、RGB 和网络 confidence。

只设置 False 不够：API 仍执行原生 Umeyama，并返回 similarity-aligned 的预测位姿。该策略通过显式恢复 Scanner E，将 metric depth 与已知相机组合；没有硬编码某个扫描的比例，也没有利用 LiDAR 拟合新尺度。保存的 `scale_factor` 是 Nested 模型自己的 metric 对齐信息，不能当成这里额外应用的传感器修正比例。

适用条件是采用的模型已经输出 metric depth、外部 pose 可信，并且原生多帧 alignment 能正常执行。False 没有绕过该计算，因此本入口不承诺单帧、零基线或退化轨迹可用。它是已采用的特定相机/尺度策略，不是对所有场景精度提升的保证。

## 从 dense prediction 到 GLB

[reconstruct.py](../scripts/3d/scanner/reconstruct.py) 每路生成 `predictions.npz`、`cloud.glb` 和 `run.json`，并要求新的输出目录。`predictions.npz` 保留 `[N,H_p,W_p]` depth、对应 processed RGB、K/E、frame IDs 和 metric 标记；DA3 还保留网络 confidence 与 Nested scale factor。

共享反投影按每帧 K 和 Scanner c2w 计算 `X_world = R_WC · X_camera + t_WC`。模型输入或结果不经过 LiDAR 事后比例拟合。各帧点云直接汇集；该阶段没有时间融合、相机优化或表面重建。

GLB 的 [export_prediction()](../scripts/3d/scanner/scanner_io.py#L79-L136) 随后执行：

1. 应用对应候选的有效性/网络 confidence 筛选。
2. 超过 1,000,000 点时，用独立 RNG、seed=42 均匀随机保留点及其颜色。
3. 以 `D = diag(1,-1,-1,1) @ Scanner_w2c[0]` 变换到首相机的显示坐标，再写入彩色 POINTS GLB。

`D` 是所有帧共用的显示变换，长度仍以米计；它不参与模型输入，也不改写 NPZ 的世界坐标 E。GLB 的点数限制与 confidence 筛选同样只影响导出物，dense NPZ 不会因此被采样。

需要下游逐像素世界点时，应从 dense NPZ 按 K/Z/E 重建，不能从 GLB 恢复完整网格或把显示坐标当作原始世界坐标。[共享流水线适配](promptda_integration.md) 的有效性 confidence、像素 affine 和缓存边界另有明确约定。

## 评价与来源边界

PromptDA full 已把原始 LiDAR 作为网络输入。模型再与同一份输入 LiDAR 比较，衡量的是输入一致性，不能据此证明独立深度精度；也不能用这类分数证明它优于没有收到 LiDAR 的 DA3。独立精度评价需要独立参考深度及明确的相机/像素配准。

如果另行进行留出测量实验，必须在输入及 min/max 归一化之前移除对应测量，并将它标为该 masked 条件下的诊断。它与生产 full 的输入条件不同，所得结果不能直接赋给 full。视频相邻帧和相邻传感器像素存在相关性，像素数量不等于独立试验次数。

当前 [CPU tests](../test/test_scanner_candidates.py) 覆盖按时间采样、CV c2w、metric depth/E 策略、完整 prompt、half-pixel K 和小型 POINTS GLB 几何读写。它们验证实现约定，不能替代设备标定、真实重建质量或面积精度评价。当前准备代码也不应用可选 distortion lookup table，导出文件不能单独证明采集时的 tracking 质量。

本文使用的外部语义依据固定为 Scanner `ec3e1dc9d33f8df2289ede6a5c59f7991d1a6bbb`、StrayVisualizer `195c64091a13277b22734a20303079c91fa63c30` 和 PromptDA `9161547ca7151e14fcc8f490ad3cc02fa9437483`。旧 Visualizer 读取 quaternion 时使用整行尾部，不能直接适配新增 K/空 distortion 列；引用它是为了核对 pose 方向、单位与 Z-depth 消费方式，当前 CSV 解析以仓库实现为准。
