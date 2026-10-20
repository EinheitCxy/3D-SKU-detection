# SKU RGB-D bundle v1

每个目录是一次连续 ARKit session，所有字段由 DatasetEncoder.swift 生成。
一个采样点的 RGB、sceneDepth、confidence、camera.intrinsics 和 camera.transform 来自同一个 ARFrame。
记录开始后，按首个 ARFrame.timestamp 锚定 [k,k+1) 秒区间；
每个区间选择第一组 tracking normal 且 depth/confidence 齐全、写入队列有空位的帧。
不存在的区间不复制或补帧，这与历史离线视频“整数秒最近帧”的策略有区别。

```text
sku-<uuid>.rgbd/
  scan_bundle.json
  preview.png
  frames/
    000000/
      rgb.png
      depth.png
      confidence.png
      frame.json
    000061/
      ...
```

frame_id 是录制期间 ARSession 回调的原始序号，可不连续。
同组先写入 .pending-<frame_id>，四个文件成功后 rename 为正式目录。
manifest 最初为 recording，停止录制并排空队列后才原子写为 ready；
写入错误或没有有效帧时为 error。只有 ready 且 errors 为空的包可以从应用导出或在服务端准备。
每次最多保留两个待写入 ARFrame；写入繁忙时只拒绝整个候选，不会分开丢弃传感器分量。

## 固定约定

| 字段/数据 | 定义 |
|---|---|
| format / version | sku-rgbd / 1 |
| timestamp_clock | ARFrame.timestamp_seconds_since_boot |
| sample_interval_seconds | 1 |
| rgb.png | sRGB 8-bit PNG，可有全255的 alpha；capturedImage 原尺寸、原方向，无旋转、镜像或缩放 |
| depth.png | 原生 LiDAR 网格的单通道 uint16 PNG，毫米，0无效 |
| confidence.png | 同深度网格的单通道 uint8 PNG，数值严格为0/1/2，无灰度色彩映射 |
| depth_alignment | 与原图相同 normalized extent；当前版本要求 RGB 与 depth 长宽比一致 |
| intrinsics | 原始 RGB 网格的3×3 K，row-major |
| camera_to_world | 4×4 OpenCV camera→ARKit world，row-major，平移单位米 |
| tracking_state | normal；其他状态的候选会被拒绝 |

深度按原始 buffer 的 bytesPerRow 逐行读取，舍入为毫米；
非正、非有限或舍入后超过65535毫米的值置0。confidence 按原始行步幅复制，
超出0/1/2会报错。不会在手机端上采样 LiDAR、按 confidence 删点或拟合尺度。

相机轴变换为：

```text
T_world_cv = frame.camera.transform @ diag(1, -1, -1, 1)
X_world = T_world_cv @ X_cv
T_cv_world = inverse(T_world_cv)
```

OpenCV camera 为 x右/y下/z前。ARKit world 的坐标轴保持不变，y向上。
Swift 的 SIMD 矩阵按列存储，JSON 显式遍历 matrix[column][row] 输出 row-major；
服务端直接读取 camera_to_world，只取逆，不再乘第二次 Rx(pi)，不做轨迹或 LiDAR 尺度拟合。
深度沿用项目现有 Z-depth 反投影契约；本次未增加物理标定或额外畸变校正。

## manifest 与逐帧内容

scan_bundle.json 包含格式与坐标约定、UUID、status、source_frame_count、
rejected_candidate_frames、errors 和 frames 数组。rejected_candidate_frames 统计
tracking_not_normal、missing_depth_or_confidence、writer_busy 三种候选拒绝次数；
它们是候选回调次数，不等于缺失秒数。

每个 frames 元素也完整写入对应 frame.json：

```json
{
  "frame_id": 61,
  "timestamp_seconds": 101.02,
  "relative_timestamp_seconds": 1.02,
  "target_seconds": 1,
  "rgb_size": [1920, 1440],
  "depth_size": [256, 192],
  "intrinsics": [[1400, 0, 960], [0, 1400, 720], [0, 0, 1]],
  "camera_to_world": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
  "tracking_state": "normal"
}
```

上述数值仅是格式示例，不是设备标定。source_frame_count 包括没有被保存的回调。
relative_timestamp_seconds 相对录制期间第一个回调，target_seconds=floor(relative_timestamp_seconds)。
UUID、source_frame_count 和所有 frames 在最终 manifest 中提供，包之间不拼接轨迹。

服务端 prepare_bundle.py 核对 manifest/frame.json、时间、图像数值和矩阵约定后，
生成既有 frames/{frame_id:06d}.png + camera_inputs.npz。
RGB 原文件直接复制；K/c2w 原值保留；w2c 只求逆；LiDAR 毫米除1000转为 float32 米。
不调用 ffmpeg、不再次采样、不重估 pose，也不会把本目录当成历史 MP4 格式猜测解析。
