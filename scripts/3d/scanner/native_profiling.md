# 原图 PromptDA 与 DA3 + K/E profiling（2026-10-19）

本页是原图模式的历史记录。当前默认已改为长边1008，最新batch扫描见[1008 profiling](profiling_1008.md)；
复现本页PromptDA需显式加`--promptda-max-size 0`。

PromptDA 使用完整 1920×1440 RGB，仅右/底 edge padding 到1932×1442，网络输出裁回1920×1440；RGB像素、原始K/E与frame IDs已逐帧确认保持一致。DA3保留已接受的504×378、Nested metric深度和Scanner E修复策略。

这比较的是当前两套使用配置：PromptDA输出像素约为DA3的14.51倍，网络token数量约为14.62倍；两者输入信息、精度与多帧策略也不同，不能解释成相同分辨率的纯架构排名。

## 计时与环境

同一capture `572dd9f6fb`，1Hz的14帧；物理GPU2，RTX4090D 24GiB，Torch2.7.1+cu126，CPU/BLAS8线程。新进程按DA3、原图FP32 batch1、FP32 batch2、TF32 batch2、FP16 batch2顺序串行运行。其他进程约5044MiB整卡占用不计入本进程allocated峰，未占用GPU1。

每组一次模型加载、一次cold与3次warm。请求窗口从prepared PNG路径和常驻模型到CPU dense predictions，包含解码、RGB补边、LiDAR网格映射、H2D、网络、D2H、打包；窗口前后CUDA同步。排除模型加载、抽帧、NPZ/GLB落盘与matching。新进程不代表文件冷缓存；未清空页缓存、allocator或模型缓存。

## 请求实测

| 配置 | 输出尺寸 W×H | 加载 / s | Cold / s | Warm中位数 / s | Warm范围 / s | FPS | Allocated峰 / GiB | Reserved峰 / GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DA3 + K/E 修复版 | 504×378 | 9.304 | 2.471 | 1.340 | 1.333–1.478 | 10.447 | 11.336 | 15.123 |
| PromptDA 原图 FP32 / batch1 | 1920×1440 | 8.361 | 21.482 | 21.326 | 21.213–21.401 | 0.656 | 7.146 | 9.381 |
| PromptDA 原图 FP32 / batch2 | 1920×1440 | 8.010 | 21.569 | 21.169 | 21.104–21.175 | 0.661 | 13.679 | 17.908 |
| PromptDA 原图 TF32 / batch2 | 1920×1440 | 7.901 | 19.825 | 19.434 | 19.327–19.443 | 0.720 | 13.679 | 17.908 |
| PromptDA 原图 FP16 autocast / batch2 | 1920×1440 | 7.935 | 5.847 | 5.268 | 5.261–5.272 | 2.658 | 9.064 | 11.227 |

FP32扩大到batch2仅降低约0.74%请求延迟，却明显增加显存。TF32 batch2相对FP32 batch1降低约8.87%。FP16 autocast batch2相对FP32 batch1约快4.05倍，延迟降低75.30%；仍慢于当前低分辨率DA3配置。

Allocated/Reserved均为各次predict窗口reset后、包含常驻模型的绝对峰；两者不可相加，也不是整卡driver占用。RSS按当前PID每100ms采样，可能漏短峰。3个warm样本不用于推断尾延迟。所有组保留环境原有`cudnn.allow_tf32=True`，`highest/high`开关只控制FP32矩阵乘法，不改变卷积设置。完整加载/请求/RSS明细见每组profile.json。

## 数值变化与采用方式

本页原图实验以FP32、batch1、`--promptda-matmul-precision highest`为基线；当前默认为1008长边、FP16、batch2。同权重通过本地子类将官方逐样本端点`quantile(q=0/1)`替换为`aminmax`；有限输入的min/max与归一化公式相同，CPU测试做了零容差对照。所有模式均使用该归一化，不把其单独收益混入batch提升。

FP16为显式可选：FP32权重、CUDA FP16 autocast，保存的depth为float32；这不表示内部仍全程FP32。`--promptda-matmul-precision high`允许TF32内部乘法，含义见[PyTorch2.7说明](https://docs.pytorch.org/docs/2.7/generated/torch.set_float32_matmul_precision.html)。xFormers attention本来已启用，本轮没有重复添加，也没有做compile或CUDA Graph。

以下对完整14×1440×1920深度逐像素比较原图FP32 batch1，RGB/K/E/frame IDs与padding元数据均相同：

| 配置 | 平均绝对差 / mm | P99绝对差 / mm | 最大绝对差 / mm | 差值>1cm的像素占比 |
|---|---:|---:|---:|---:|
| PromptDA 原图 FP32 / batch2 | 0.0188 | 0.1869 | 23.099 | 0.000142% |
| PromptDA 原图 TF32 / batch2 | 0.0347 | 0.4406 | 40.210 | 0.000995% |
| PromptDA 原图 FP16 autocast / batch2 | 0.3500 | 2.0714 | 125.082 | 0.020084% |

这些是相对FP32输出的数值偏差，不是对独立参考深度的准确率。FP16在少量像素存在约125mm的最大差异，因此保留为可选加速；本轮未证明商品边缘、面积或matching精度完全等价。

## NPZ与GLB交付

| 配置 | NPZ方式 | NPZ保存 / s | NPZ / MiB | POINTS GLB导出及回读 / s | Cold请求→NPZ+GLB连续墙钟 / s |
|---|---|---:|---:|---:|---:|
| DA3 + K/E 修复版 | DEFLATE | 1.790 | 20.55 | 0.908 | 5.173 |
| PromptDA 原图 FP32 / batch1 | DEFLATE | 13.045 | 157.52 | 5.002 | 39.585 |
| PromptDA 原图 FP32 / batch2 | DEFLATE | 12.384 | 157.52 | 4.790 | 38.796 |
| PromptDA 原图 TF32 / batch2 | DEFLATE | 12.476 | 157.52 | 4.863 | 37.218 |
| PromptDA 原图 FP16 autocast / batch2 | 无压缩 | 0.362 | 258.40 | 4.896 | 11.161 |

前4组在改变存储策略前完成，仍用原DEFLATE；FP16组使用当前PromptDA默认无压缩。对相同已在CPU内存的FP32 arrays，另测无压缩保存0.465s、258.40MiB；其压缩保存为13.045s、157.52MiB。它是单独组件计时，不把阶段相加冒充连续墙钟。写入均含关闭文件但不强制fsync，可能由文件系统缓冲。

当前PromptDA默认无压缩NPZ，以更多磁盘换落盘时间；`--compress-prediction`恢复压缩，不改变数组内容。DA3仍保留原压缩行为。网络推理表不受NPZ方式影响，交付表须连同存储方式阅读。

所有GLB均回读PASS并保留原1,000,000彩色点上限。原图dense NPZ保留38,707,200个深度像素，未被GLB采样缩减；GLB仍为POINTS而非mesh。CPU转换cache使用原图网格与identity affine，不把显示坐标或GLB采样写回预测。

## 重现与产物

以下命令在仓库根、已有模型环境执行；prepared数据可复用，输出目录需新建：

```bash
CUDA_VISIBLE_DEVICES=2 uv run python scripts/3d/scanner/profile_candidates.py \
  --base Output/scanner_572dd9f6fb_1fps --candidate da3 \
  --checkpoint /path/to/DA3NESTED-GIANT-LARGE \
  --output Output/da3_ke_profile_new

CUDA_VISIBLE_DEVICES=2 uv run python scripts/3d/scanner/profile_candidates.py \
  --base Output/scanner_572dd9f6fb_1fps --candidate promptda \
  --promptda-max-size 0 \
  --promptda-repo runtime/PromptDA --checkpoint runtime/models/promptda/vitl/model.ckpt \
  --promptda-precision fp16 --promptda-batch-size 2 \
  --output Output/promptda_native_fp16_profile_new
```

FP32基线使用`--promptda-precision fp32 --promptda-batch-size 1 --compress-prediction`；FP32 batch2改batch为2；TF32再加`--promptda-matmul-precision high`。

实际Docker argv、GPU前后状态与进程墙钟在[launch.json](../../../Output/scanner_native_profile_20261019/launch.json)。每组目录包含profile.json、profiling.md、run.log、predictions.npz、scene.glb。指标与数值差异另存[native_profiling_results.json](native_profiling_results.json)。

[FP32原图GLB](../../../Output/scanner_native_profile_20261019/promptda_native_b1/scene.glb) · [FP16加速原图GLB](../../../Output/scanner_native_profile_20261019/promptda_native_fp16_b2/scene.glb) · [DA3+K/E GLB](../../../Output/scanner_native_profile_20261019/da3_ke/scene.glb)

本轮Scanner定向CPU测试10项，cache/footprint定向CPU测试22项通过。模型仅做独立重建与profiling，未运行SAM3/PVS或matching system；上述耗时和峰值不代表完整商品流水线。
